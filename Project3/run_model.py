# Simplified DDPG with single actor and critic.
# Inputs concatenate states and actions
from unityagents import UnityEnvironment
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
import random
import time
import pandas as pd
import math
import heapq
import uuid

torch.set_printoptions(profile="full")

run = '80'
actor_local_file = 'checkpoint_actor_local_{}.pth'.format(run)
actor_target_file = 'checkpoint_actor_target_{}.pth'.format(run)
critic_local_file = 'checkpoint_critic_local_{}.pth'.format(run)
critic_target_file = 'checkpoint_critic_target_{}.pth'.format(run)

# Hyperparams
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 256  # minibatch size. 
GAMMA = 0.9999  # rewards discount factor
TAU = 0.06  # soft update rate
LR_ACTOR = 0.001  # initial learning rate for actor (upper bound)
LR_CRITIC = 0.001  # initial learning rate for critic (upper bound)
lr_schedule_episode_1 = -1  # episode designated for LR scheduled adjustment
lr_schedule_episode_2 = -1  # episode designated for LR scheduled adjustment
lr_schedule_episode_3 = -1  # episode designated for LR scheduled adjustment
lr_schedule_episode_4 = -1  # episode designated for LR scheduled adjustment
lr_schedule_factor = 0.1  # factor for LR scheduled adjustment
WEIGHT_DECAY = 0  # L2 weight decay for critic
UPDATE_EVERY = 5  # episodes between soft updates
LEARN_EVERY = 1  # timesteps between learning
learn = True  # learning flag
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
fc1_units = 256  # number of units in first hidden layer
fc2_units = 128  # number of units in output layer
n_episodes = 10000  # number of episodes
# max_t = 80  # max timesteps per episode
random_seed = 6142  # ensure reproducibility on a given machine
MU = 0
THETA = 0.15
SIGMA = 0.2  # initial value of noise parameter sigma
DECAY_RATE = 0.01  # sigma decay
SCALE = 1  # scale parameter for OU noise
sigma_schedule_1 = -1  # episode designated for sigma schedule adjustment
sigma_schedule_2 = -1
sigma_schedule_3 = -1
STATE_SIZE = 48  # both players
ACTION_SIZE = 4  # both players

def init_limits(layer):
    """Calculate bounds for weight initialization based on rule of thumb:
    limit = 1/sqrt(n) where n = number of inputs. Initial weights should be close to zero."""
    num_inputs = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(num_inputs)
    return (-lim, lim)

class Scores():
    def __init__(self):
        self.scores_deque = deque(maxlen=100)
        self.scores = []
        self.run_results = pd.DataFrame(columns=["100_episode_avg", "episode_score"])
        self.score = np.zeros(2)  # two players
        self.loss_results = pd.DataFrame(columns=["episode", 'player', 'actor_loss', "critic_loss"])
        self.actions = pd.DataFrame(columns=["episode", "action", "action_with_noise"])

    def add_rewards(self, rewards):
        self.score += rewards
        return

    def update_scores(self, episode):
        """Update score trackers at the end of an episode."""
        self.scores_deque.append(self.score[np.argmax(self.score)])
        self.scores.append(self.score[np.argmax(self.score)])
        self.run_results.loc[f"episode_{episode}", "100_episode_avg"] = np.mean(self.scores_deque)
        self.run_results.loc[f"episode_{episode}", "episode_score"] = self.score[np.argmax(self.score)]
        return

    def update_actions(self, t_episode, t_action, t_action_noise):
        "Record actions taken."
        self.actions = self.actions.append(
            {"episode": t_episode, "action": t_action, "action_with_noise": t_action_noise}, ignore_index=True)

    def reset(self):
        self.score = np.zeros(2)

    def save_results(self, results, filename):
        results.to_csv(f"{filename}.csv")

    def multiply_out(self, expression):
        """Multiply terms of expression"""

    def update_loss(self, t_episode, t_actor_loss, t_critic_loss):
        self.loss_results = self.loss_results.append(
            {"episode": t_episode, "actor_loss": t_actor_loss, "critic_loss": t_critic_loss},
            ignore_index=True)


class Actor(nn.Module):
    """Create an actor (policy) network that maps states -> actions."""

    def __init__(self, state_size, action_size, seed, fc1_units, fc2_units):
        """Initialize parameters and build model.
        Args:
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize layer weights."""
        self.fc1.weight.data.uniform_(*init_limits(self.fc1))
        self.fc2.weight.data.uniform_(*init_limits(self.fc2))
        self.fc3.weight.data.uniform_(*init_limits(self.fc3))
        return

    def forward(self, state):
        """Execute a forward pass of the model."""
        x = F.leaky_relu(self.fc1(state))
        x = F.leaky_relu(self.fc2(x))
        return F.tanh(self.fc3(x))


class Critic(nn.Module):
    """Build model to predict Q-value of state-action pairs."""

    def __init__(self, state_size, action_size, seed, fc1_units, fc2_units):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*init_limits(self.fc1))
        self.fc2.weight.data.uniform_(*init_limits(self.fc2))
        self.fc3.weight.data.uniform_(*init_limits(self.fc3))
        return

    def forward(self, states, actions):
        """Execute a forward pass of the model."""
        xs = F.relu(self.fc1(states))
        x = torch.cat((xs, actions), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class OUNoise():
    """Implement Ornstein-Uhlenbeck noise-generation process."""

    def __init__(self, action_dim, scale=0.1, mu=0.5, theta=0.15, sigma=2):
        self.action_dim = action_dim
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return torch.tensor(self.state * self.scale).float()


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class Agent():
    def __init__(self, state_size, action_size, random_seed, fc1_units, fc2_units, batch_size, device, lr_actor, lr_critic, weight_decay, buffer_size, gamma):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.gamma = gamma

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed, fc1_units, fc2_units).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed, fc1_units, fc2_units).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor, amsgrad=True)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed, fc1_units, fc2_units).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed, fc1_units, fc2_units).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=weight_decay, amsgrad=True)

        # hard update targets to match local weights
        self.soft_update(self.actor_local, self.actor_target, tau=1)
        self.soft_update(self.critic_local, self.critic_target, tau=1)

        # Initialize Replay Buffer
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, random_seed)

        # Initialize noise
        self.noise = OUNoise(action_dim=self.action_size)

    def explore(self, scale, mu, theta, sigma):
        # Noise process
        self.noise = OUNoise(action_dim=self.action_size, scale=scale, mu=mu, theta=theta, sigma=sigma)

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Separate and buffer experience from each player's perspective
        # player 0
        p0_state = np.reshape(state, (1, 48))
        p0_action = np.reshape(action, (1, 4))
        p0_next_state = np.reshape(next_state, (1, 48))
        self.memory.add(p0_state, p0_action, reward[0], p0_next_state, done[0])
        # player 1 (reverse order of states and actions)
        p1_state = np.hstack((state[1,:], state[0,:])).reshape(1,48)
        p1_action = np.hstack((p0_action[:, 2:], p0_action[:, 0:2])).reshape(1, 4)
        p1_next_state = np.hstack((next_state[1, :], next_state[0, :])).reshape(1, 48)
        self.memory.add(p1_state, p1_action, reward[1], p1_next_state, done[1])

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, self.gamma)

    def act(self, state):
        """Returns actions for given state as per current policy."""
        if state.shape[1] != 48:
            state = np.reshape(state, (1, 48))
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        action += self.noise.noise()
        return np.clip(action, -1, 1)

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)



if __name__ == "__main__":

    # Load the environment
    env = UnityEnvironment(file_name="../headless/Tennis.x86_64")

    # Get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # Initialize the agent
    agent = Agent(state_size=STATE_SIZE, action_size=ACTION_SIZE, random_seed=random_seed, fc1_units=fc1_units,
                  fc2_units=fc2_units, batch_size=BATCH_SIZE, device=device, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, buffer_size=BUFFER_SIZE, gamma=GAMMA)

    # Set OU Noise
    agent.explore(scale=SCALE, mu=MU, theta=THETA, sigma=SIGMA)

    # Initialize score tracking
    scores = Scores()

    # Execute episodes
    print('starting ddpg')
    for i_episode in range(1, n_episodes + 1):
        # Start run timer
        start = time.time()

        if i_episode in [sigma_schedule_1, sigma_schedule_2, sigma_schedule_3]:
            SIGMA = SIGMA * DECAY_RATE
            agent.explore(scale=SCALE, mu=MU, theta=THETA, sigma=SIGMA)

        # Reset scores
        scores.reset()

        # Reset the environment in train mode
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations

        # Adjust the learning rates
        if i_episode in [lr_schedule_episode_1, lr_schedule_episode_2, lr_schedule_episode_3, lr_schedule_episode_4]:
            for g in agent.actor_optimizer.param_groups:
                g['lr'] = g['lr'] * lr_schedule_factor
            for g in agent.critic_optimizer.param_groups:
                g['lr'] = g['lr'] * lr_schedule_factor

        # Soft update networks
        if i_episode % UPDATE_EVERY == 0:
            agent.soft_update(agent.critic_local, agent.critic_target, TAU)

        # Execute timesteps
        while True:
            # Set learn flag
            # learn = True if t % LEARN_EVERY == 0 else False

            # Get actions
            actions = agent.act(states)

            # Get environment response
            env_step = env.step(actions)[brain_name]
            next_states = env_step.vector_observations
            rewards = env_step.rewards
            dones = env_step.local_done
            actions = agent.act(states)
            agent.step(states, actions, rewards, next_states, dones)
            if np.any(dones):
                break

            # Prepare for next timestep
            states = next_states
            scores.add_rewards(rewards)

        # Post-episode tasks
        scores.update_scores(i_episode)
        scores.save_results(scores.run_results, f"run_{run}_results")
        end = time.time()
        print('\rEpisode {}\tAverage Score: {:.4f}\tScore: {:.4f}\t{:.2f}s'.format(i_episode, np.mean(scores.scores_deque), scores.scores[-1], end - start))
        if i_episode % 1000 == 0:
            torch.save(agent.actor_local.state_dict(),
                       f'checkpoint_actor_local_{run}.pth')
            torch.save(agent.actor_target.state_dict(),
                       f'checkpoint_actor_target_{run}.pth')
            torch.save(agent.critic_local.state_dict(),
                       f'checkpoint_critic_local_{run}.pth')
            torch.save(agent.critic_target.state_dict(),
                       f'checkpoint_critic_target_{run}.pth')


