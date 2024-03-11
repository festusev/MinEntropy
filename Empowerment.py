import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, List, Dict, Any, Tuple
from ray.rllib.utils.typing import SampleBatchType
from ReplayBuffers import ActionReplayBuffer, SimpleBuffer

import numpy as np
from abc import ABC, abstractmethod, abstractproperty


def rowwise_cosine_similarity(t1: torch.tensor, t2: torch.tensor, temperature: nn.Parameter):
    dotted = torch.sum((t1 * t2).flatten(end_dim=-2), dim=-1)
    norm_dotted = dotted / (torch.norm(t1, dim=1) * torch.norm(t2, dim=1))
    return norm_dotted * temperature


class Empowerment(ABC):
    def __init__(self, in_channels, device):
        self.in_channels = in_channels
        self.device = device

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def eval(self) -> None:
        pass

    @abstractmethod
    def modelUpdate(self, sample_batches: List[SampleBatchType]) -> Tuple[float, float]:
        pass

    @abstractmethod
    def getLoss(self, *args, **kwargs):
        pass

    @abstractmethod
    def computeReward(self, obs, action, new_obs) -> Tuple[int, Dict[str, Any]]:
        pass

    @abstractmethod
    def update(self, empowerment):
        pass

    @abstractmethod
    def save_to_folder(self, folder):
        pass

    def convertObs(self, obs):
        return torch.tensor(obs[..., :self.in_channels].transpose((0, -1, 1, 2)), dtype=torch.float32,
                            device=self.device)

    def __call__(self, sample_batches: List[SampleBatchType]) -> Tuple[float, float]:
        return self.modelUpdate(sample_batches)


class ContrastiveEmpowerment(Empowerment):
    def __init__(self, num_actions, in_channels, obs_shape, device, prob=0.2, z_dim=16, batch_size: int = 1024,
                 buffer_max_size: int = 10_000):
        super().__init__(in_channels, device)

        self.num_actions = num_actions
        self.z_dim = z_dim
        self.sa_encoder: nn.Module = SAEncoder(in_channels, z_dim)
        self.g_encoder: nn.Module = SEncoder(in_channels, z_dim)
        self.temperature: nn.Parameter = nn.Parameter(torch.tensor(1, requires_grad=True, dtype=torch.float32))

        self.sa_encoder.to(device)
        self.g_encoder.to(device)

        self.optim = torch.optim.Adam([*self.sa_encoder.parameters(), *self.g_encoder.parameters()], lr=0.01)
        self.bce_loss = nn.BCELoss()

        self.info = {"empowerment_classifier_loss": 0, "contrastive_empowerment_rewards": 0}
        self.buffer = SimpleBuffer(max_size=buffer_max_size)

        self.prob = prob
        self.batch_size = batch_size

    def train(self):
        self.sa_encoder.train()
        self.g_encoder.train()

    def eval(self):
        self.sa_encoder.eval()
        self.g_encoder.eval()

    def sampleGeometricGoal(self, sa_idx: np.ndarray, max_idx):
        n = sa_idx.shape[0]

        g_idx = sa_idx + np.random.geometric(self.prob, size=n)
        g_idx = np.clip(g_idx, 0, max_idx - 1)

        return g_idx

    def getLoss(self, train_batch: Dict):
        self.train()

        obs, actions, goals = train_batch["obs"], train_batch["actions"], train_batch["goals"]
        n = obs.shape[0]

        null_g_idx = np.random.randint(0, high=n, size=n)

        g_z = self.g_encoder(self.convertObs(goals))
        sa_z = self.sa_encoder(self.convertObs(obs), torch.tensor(actions, device=self.device).long())

        dotted = rowwise_cosine_similarity(sa_z, g_z, self.temperature)
        null_dotted = rowwise_cosine_similarity(sa_z, g_z[null_g_idx], self.temperature)

        true_loss = self.bce_loss(nn.functional.sigmoid(dotted), torch.ones(n))
        null_loss = self.bce_loss(nn.functional.sigmoid(null_dotted), torch.zeros(n))
        loss = true_loss + null_loss

        return loss, {"true_loss": true_loss.item(), "null_loss": null_loss.item()}

    def rewardFromBatch(self, human_batch):
        self.eval()

        with torch.no_grad():
            obs, actions = human_batch["obs"], human_batch["actions"]
            obs = self.convertObs(obs)
            actions = torch.tensor(actions, device=self.device).long()

            n = obs.shape[0]

            sa_idx = np.arange(0, n)
            g_idx = self.sampleGeometricGoal(sa_idx, max_idx=n)

            g_z = self.g_encoder(obs[g_idx])
            sa_z = self.sa_encoder(obs, actions)

            dotted = rowwise_cosine_similarity(sa_z, g_z, self.temperature)

        return dotted

    def modelUpdate(self, sample_batches: Dict):
        self.train()

        empowerment_batch = None
        human_batch = None
        for key in sample_batches.keys():
            if "ppo" in key:
                human_batch = sample_batches[key]
            else:
                empowerment_batch = sample_batches[key]

        self.buffer.add(human_batch)

        empowerment_rewards = self.rewardFromBatch(human_batch)

        if empowerment_batch is not None:
            empowerment_batch["rewards"] = empowerment_rewards

        train_batch = self.getBatch(self.batch_size)  # 256
        loss, loss_info = self.getLoss(train_batch)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.g_encoder.parameters(), 250)
        torch.nn.utils.clip_grad_norm_(self.sa_encoder.parameters(), 250)

        self.optim.step()
        self.optim.zero_grad()

        self.info = {"empowerment_classifier_loss": loss.item(),
                     "contrastive_empowerment_rewards": empowerment_rewards.detach().numpy().mean()}
        self.info.update(loss_info)

        return loss

    def getBatch(self, batch_size) -> Dict:
        sa_idx = np.random.randint(0, high=len(self.buffer), size=batch_size)
        g_idx = self.sampleGeometricGoal(sa_idx, max_idx=len(self.buffer))

        obs = self.buffer.obs[sa_idx]
        actions = self.buffer.actions[sa_idx]
        goals = self.buffer.obs[g_idx]

        return {"obs": obs, "actions": actions, "goals": goals}

    def computeReward(self, obs, action, new_obs):
        # We compute reward later, after all the rollout workers have finished
        return torch.tensor(0), self.info

    def update(self, empowerment: "ContrastiveEmpowerment"):
        self.info = empowerment.info
        self.sa_encoder = empowerment.sa_encoder
        self.g_encoder = empowerment.g_encoder

    def save_to_folder(self, folder):
        import os
        os.makedirs(folder, exist_ok=True)
        torch.save(self.sa_encoder.state_dict(), os.path.join(folder, "sa_encoder.pt"))
        torch.save(self.g_encoder.state_dict(), os.path.join(folder, "g_encoder.pt"))


class SEncoder(nn.Module):
    # Encodes a state to a latent dimension
    def __init__(self, in_channels, z_dim):
        super().__init__()

        self.state_model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding='same'),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding='same'),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, 3, padding='same'),
            nn.LeakyReLU(),

            nn.Flatten(),

            nn.Linear(1120, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, z_dim)
        )

    def forward(self, state):
        z = self.state_model(state)

        return z

    # def __init__(self, in_channels):
    #     super().__init__()
    #
    #     self.state_model = nn.Sequential(
    #         nn.Conv2d(in_channels, 64, 3, padding='same'),
    #         nn.LeakyReLU(),
    #
    #         nn.Conv2d(64, 64, 3, padding='same'),
    #         nn.LeakyReLU(),
    #
    #         nn.Conv2d(64, 32, 3, padding='same'),
    #         nn.LeakyReLU(),
    #
    #         nn.Conv2d(32, 16, 3, padding='same'),
    #         nn.LeakyReLU(),
    #         nn.MaxPool2d(3, 1, padding=1),
    #
    #         nn.Conv2d(16, 8, 3, padding='same'),
    #         nn.LeakyReLU(),
    #         nn.Conv2d(8, 1, 3, padding='same'),
    #
    #         nn.Flatten()
    #     )
    #
    # def forward(self, state):
    #     return self.state_model(state)


class SAEncoder(nn.Module):
    # Encodes a state action pair to a latent dimension
    num_actions = 7

    def __init__(self, in_channels, z_dim):
        super().__init__()

        self.state_model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding='same'),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding='same'),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, 3, padding='same'),
            nn.LeakyReLU(),

            nn.Flatten()
        )

        self.action_model = nn.Sequential(
            nn.Linear(self.num_actions, 16),  # There are several actions, so we one-hot encode them
            nn.LeakyReLU(),
            nn.Linear(16, 64),
            nn.LeakyReLU()
        )

        self.combined_model = nn.Sequential(
            nn.Linear(1120 + 64, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, z_dim)
        )

    def forward(self, state, action):
        action = F.one_hot(action, num_classes=self.num_actions).to(torch.float32)

        z_state = self.state_model(state)
        z_action = self.action_model(action)

        combined = torch.hstack([z_state, z_action])

        return self.combined_model(combined)

    # def __init__(self, in_channels, z_dim):
    #     super().__init__()
    #
    #     self.state_model = nn.Sequential(
    #         nn.Conv2d(in_channels, 64, 3, padding='same'),
    #         nn.LeakyReLU(),
    #         nn.Conv2d(64, 64, 3, padding='same'),
    #         nn.LeakyReLU(),
    #     )
    #
    #     self.action_model = nn.Sequential(
    #         nn.Linear(self.num_actions, 16), # There are 6 actions, so we one-hot encode them
    #         nn.LeakyReLU(),
    #         nn.Linear(16, 64),
    #         nn.LeakyReLU()
    #     )
    #
    #     self.combined_model = nn.Sequential(
    #         nn.Conv2d(64, 32, 3, padding='same'),
    #         nn.LeakyReLU(),
    #         nn.Conv2d(32, 16, 3, padding='same'),
    #         nn.LeakyReLU(),
    #         nn.MaxPool2d(3, 1, padding=1),
    #
    #         nn.Conv2d(16, 8, 3, padding='same'),
    #         nn.LeakyReLU(),
    #         nn.Conv2d(8, 1, 3, padding='same'),
    #
    #         nn.Flatten()
    #     )
    #
    #
    # def forward(self, state, action):
    #     action = F.one_hot(action, num_classes=self.num_actions).to(torch.float32)
    #
    #     z_state = self.state_model(state)
    #     z_action = self.action_model(action)
    #     z_action = z_action.reshape(*z_action.shape[:2], 1, 1)
    #
    #     combined = z_state + z_action
    #
    #     return self.combined_model(combined)


class MIMIEmpowerment(Empowerment):
    batch_size = 128

    def __init__(self, num_actions, in_channels, obs_shape, device):
        super().__init__(in_channels, device)

        self.num_actions = num_actions
        self.humanBuffer = ActionReplayBuffer(num_actions)
        self.T: nn.Module = ActionConditional(num_actions, in_channels, obs_shape)
        self.a: nn.Module = JointState(in_channels, obs_shape)

        self.T.to(device)
        self.a.to(device)

        self.optim = torch.optim.Adam([*self.T.parameters(), *self.a.parameters()], lr=0.05)
        self.loss = nn.MSELoss()

        self.info = {"empowerment_tuba_loss": 0}

    def train(self):
        self.T.train()
        self.a.train()

    def eval(self):
        self.T.eval()
        self.a.eval()

    def get_batch(self, num_items) -> Tuple[List[SimpleBuffer], List[torch.tensor]]:
        batch = []
        null_actions = []

        for action in range(self.num_actions):
            action_batch: SimpleBuffer = self.humanBuffer.sample(num_items, action)
            batch.append(action_batch)
            null_actions.append(torch.randint(0, self.num_actions, (num_items,)))

        return batch, null_actions

    def modelUpdate(self, sample_batches: List[SampleBatchType]):
        for batch in sample_batches:
            self.humanBuffer.add(batch)

        self.train()

        batches, null_actions = self.get_batch(self.batch_size)
        loss = self.getLoss(batches, null_actions)

        loss.backward()
        self.optim.step()
        self.optim.zero_grad()

        self.info = {"empowerment_classifier_loss": loss.item()}

        return loss

    def getLoss(self, batches: List[SimpleBuffer], null_actions: List[torch.tensor]):
        loss = 0

        for i, batch in enumerate(batches):
            loss += -self.I_tuba(batch.obs, batch.actions, batch.new_obs, null_actions[i]).mean()

        return loss

    def I_tuba(self, obs, actions, new_obs, null_actions):
        obs = torch.tensor(obs[..., :self.in_channels].transpose((0, -1, 1, 2)), dtype=torch.float32,
                           device=self.device)
        new_obs = torch.tensor(new_obs[..., :self.in_channels].transpose((0, -1, 1, 2)), dtype=torch.float32,
                               device=self.device)

        actions = torch.tensor(actions, device=self.device)  # , dtype=torch.float32)
        null_actions = torch.tensor(null_actions, device=self.device)  # , dtype=torch.float32)

        T = self.T(obs, new_obs)
        a = self.a(obs, new_obs)

        T_true = T[np.arange(T.shape[0]), actions]  # T[:, actions]
        T_null = T[np.arange(T.shape[0]), null_actions]  # T[:, null_actions]

        rhs = torch.mean(torch.exp(T_null) / torch.exp(a) + a - 1, dim=1)

        I_tuba = T_true - rhs

        return I_tuba

    def computeReward(self, obs, action, new_obs):
        self.eval()

        with torch.no_grad():
            empowerment_reward = self.I_tuba(obs[:1], [action], new_obs[:1], [5])

        return empowerment_reward, self.info

    def update(self, empowerment: Empowerment):
        self.info = empowerment.info
        self.T = empowerment.T
        self.a = empowerment.a


class ClassifierEmpowerment(Empowerment):
    batch_size = 128

    def __init__(self, num_actions, in_channels, obs_shape, device):
        super().__init__(in_channels, device)

        self.num_actions = num_actions
        self.humanBuffer = ActionReplayBuffer(num_actions)
        self.classifier: nn.Module = TransitionClassifier(num_actions, in_channels, obs_shape)

        self.classifier.to(device)

        self.optim = torch.optim.Adam(self.classifier.parameters(), lr=0.05)
        self.loss = nn.MSELoss()

        self.info = {"empowerment_classifier_loss": 0, "true_classified": np.zeros(self.batch_size),
                     "null_classified": np.zeros(self.batch_size)}

    def train(self):
        self.classifier.train()

    def eval(self):
        self.classifier.eval()

    def get_batch(self, num_items) -> List[List[SimpleBuffer]]:
        batch = []
        for action in range(self.num_actions):
            action_batch: SimpleBuffer = self.humanBuffer.sample(num_items, action)

            null_indices = torch.randperm(
                num_items)  # np.roll(np.arange(num_items), np.random.randint(1, num_items - 1))
            null_obs = action_batch.new_obs[null_indices]
            null_batch = SimpleBuffer(action_batch.obs, action_batch.actions, null_obs)

            batch.append([action_batch, null_batch])

        return batch

    def modelUpdate(self, sample_batches: List[SampleBatchType]):
        for batch in sample_batches:
            self.humanBuffer.add(batch)

        self.train()

        batch: List[List[SimpleBuffer]] = self.get_batch(self.batch_size)
        loss, true_classified, null_classified = self.getLoss(batch)

        loss.backward()
        self.optim.step()
        self.optim.zero_grad()

        self.info = {"empowerment_classifier_loss": loss.item(),
                     "true_classified": true_classified.cpu().detach().numpy(),
                     "null_classified": null_classified.cpu().detach().numpy()}

        return loss

    def getLoss(self, batch: List[List[SimpleBuffer]]):
        loss = 0

        for true_batch, null_batch in batch:
            true_classified = self.classify(true_batch.obs, true_batch.actions, true_batch.new_obs)
            null_classified = self.classify(null_batch.obs, null_batch.actions, null_batch.new_obs)

            loss += -torch.mean(true_classified + torch.log1p(-torch.exp(null_classified)))

        return loss, true_classified, null_classified

    def classify(self, obs, actions, new_obs):
        obs = torch.tensor(obs[..., :self.in_channels].transpose((0, -1, 1, 2)), dtype=torch.float32,
                           device=self.device)
        new_obs = torch.tensor(new_obs[..., :self.in_channels].transpose((0, -1, 1, 2)), dtype=torch.float32,
                               device=self.device)

        actions = torch.tensor(actions, device=self.device)  # , dtype=torch.float32)

        return self.classifier(obs, actions, new_obs)

    def computeReward(self, obs, action, new_obs):
        self.eval()

        with torch.no_grad():
            empowerment_reward = self.classify(obs[:1], [action], new_obs[:1])

        return empowerment_reward, self.info

    def update(self, empowerment: Empowerment):
        self.info = empowerment.info
        self.classifier = empowerment.classifier


class JointState(nn.Module):
    # Learns (s_t s_{t+1}) -> 1
    def __init__(self, in_channels, obs_shape):
        super().__init__()
        self.in_channels = 2 * in_channels
        self.obs_shape = obs_shape

        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding='same'),
            nn.ReLU()
        )

        dummy_input = torch.rand(1, self.in_channels, *self.obs_shape)
        encoder_output_dim = self.conv(dummy_input).flatten().shape[0]

        self.mlp = nn.Sequential(
            nn.Linear(encoder_output_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, obs, new_obs):
        full_obs = torch.cat([obs, new_obs], dim=1)
        z = self.conv(full_obs)
        z = z.flatten(start_dim=1)

        return self.mlp(z)


class ActionConditional(nn.Module):
    # st, st+1 -> a
    def __init__(self, num_actions, in_channels, obs_shape):
        super().__init__()

        self.num_actions = num_actions
        self.in_channels = in_channels
        self.obs_shape = obs_shape

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 13, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(13, 5, 3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(5),
            nn.MaxPool2d(3, 2, padding=1)
        )

        dummy_input = torch.rand(1, self.in_channels, *self.obs_shape)
        encoder_output_dim = self.encoder(dummy_input).flatten().shape[0]

        self.mlp = nn.Sequential(
            nn.Linear(encoder_output_dim * 2, 16),
            nn.ReLU(),
            nn.Linear(16, num_actions)
        )

    def forward(self, obs, new_obs):
        obs_encoded = self.encoder(obs).flatten(start_dim=1)
        new_obs_encoded = self.encoder(new_obs).flatten(start_dim=1)

        full_encoded = torch.cat([obs_encoded, new_obs_encoded], dim=1)
        predicted = self.mlp(full_encoded)

        return predicted


class TransitionClassifier(nn.Module):
    def __init__(self, num_actions, in_channels, obs_shape):
        super().__init__()
        self.num_actions = num_actions
        self.in_channels = in_channels
        self.obs_shape = obs_shape

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 13, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(13, 5, 3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(5),
            nn.MaxPool2d(3, 2, padding=1)
        )

        dummy_input = torch.rand(1, self.in_channels, *self.obs_shape)
        encoder_output_dim = self.encoder(dummy_input).flatten().shape[0]

        self.action_encoder = nn.Sequential(
            nn.Linear(num_actions, 16),
            nn.ReLU(),
            nn.Linear(16, encoder_output_dim)
        )

    def forward(self, obs, action, new_obs):
        action = torch.nn.functional.one_hot(action, num_classes=self.num_actions)
        action = action.to(torch.float32)

        obs_encoded = self.encoder(obs).flatten(start_dim=1)
        new_obs_encoded = self.encoder(new_obs).flatten(start_dim=1)

        action_encoded = self.action_encoder(action)

        difference = torch.linalg.norm(obs_encoded + action_encoded - new_obs_encoded, dim=1)

        return -difference


class TwoHeadedEmpowerment(Empowerment):
    def __init__(self, in_channels, device):
        super().__init__(in_channels, device)

        self.state_marginal: nn.Module = StateMarginal(in_channels)
        self.transition: nn.Module = Transition(in_channels)

        self.state_marginal.to(device)
        self.transition.to(device)

        self.sm_optim = torch.optim.Adam(self.state_marginal.parameters(), lr=0.05)
        self.tr_optim = torch.optim.Adam(self.transition.parameters(), lr=0.05)
        self.loss = nn.CrossEntropyLoss()

    def train(self):
        self.state_marginal.train()
        self.transition.train()

    def eval(self):
        self.state_marginal.eval()
        self.transition.eval()

    def modelUpdate(self, sample_batches: List[SampleBatchType]):
        self.train()

        sm_loss, tr_loss = self.getLoss(sample_batches)

        sm_loss.backward()
        self.sm_optim.step()
        self.sm_optim.zero_grad()

        tr_loss.backward()
        self.tr_optim.step()
        self.tr_optim.zero_grad()

        return sm_loss.item(), tr_loss.item()

    def getLoss(self, sample_batches):
        sm_loss = 0
        tr_loss = 0
        for batch in sample_batches:
            obs = torch.tensor(batch["obs"][..., :self.in_channels].transpose((0, -1, 1, 2)), device=self.device,
                               dtype=torch.float32)
            new_obs = torch.tensor(batch["new_obs"][..., :self.in_channels].transpose((0, -1, 1, 2)),
                                   device=self.device, dtype=torch.float32)
            new_obs = new_obs.flatten(2)

            sm_prob = self.state_marginal(obs)
            sm_loss += self.loss(sm_prob.flatten(2), new_obs)

            action = torch.tensor(batch["actions"], dtype=torch.int64, device=self.device)  # .reshape(-1, 1)

            tr_prob = self.transition(obs, action)
            tr_loss += self.loss(tr_prob.flatten(2), new_obs)

        return sm_loss, tr_loss

    def computeReward(self, obs, action, new_obs):
        obs = torch.tensor(obs[..., :self.in_channels].transpose((0, -1, 1, 2)), dtype=torch.float32,
                           device=self.device)
        action = torch.tensor([action], device=self.device)  # , dtype=torch.float32)

        self.eval()

        with torch.no_grad():
            sm_prob = self.state_marginal(obs)
            tr_prob = self.transition(obs, action)

            sm_prob = torch.clamp(sm_prob, min=0.05, max=0.95)
            tr_prob = torch.clamp(tr_prob, min=0.05, max=0.95)

            H_sm = torch.log(sm_prob).sum()
            H_tr = torch.log(tr_prob).sum()

            I = H_sm - H_tr

        info = {"H_sm": H_sm.item(), "H_tr": H_tr.item()}
        return I.item(), info


class StateMarginal(nn.Module):
    # Learns P(s_{t+1} | s_t)
    def __init__(self, in_channels):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(3, 1, padding=1),

            nn.Conv2d(128, 128, 3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(3, 1, padding=1),

            nn.Conv2d(128, 64, 3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, in_channels, 3, padding='same'),
            nn.Sigmoid()
        )

    def forward(self, state):
        return self.model(state)


class Transition(nn.Module):
    # Learns P(s_{t+1} | a_t, s_t)
    num_actions = 6

    def __init__(self, in_channels):
        super().__init__()

        self.state_model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        self.action_model = nn.Sequential(
            nn.Linear(self.num_actions, 16),  # There are 6 actions, so we one-hot encode them
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU()
        )

        self.combined_model = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(3, 1, padding=1),

            nn.Conv2d(128, 128, 3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(3, 1, padding=1),

            nn.Conv2d(128, 64, 3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, in_channels, 3, padding='same'),
            nn.Sigmoid()
        )

    def forward(self, state, action):
        action = F.one_hot(action, num_classes=self.num_actions).to(torch.float32)

        z_state = self.state_model(state)
        z_action = self.action_model(action)
        z_action = z_action.reshape(*z_action.shape[:2], 1, 1)

        combined = z_state + z_action

        return self.combined_model(combined)
