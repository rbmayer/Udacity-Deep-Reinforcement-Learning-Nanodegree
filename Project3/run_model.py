# DDPG architecture with single Actor and separate Critics for each player.
# TODO: Add scores and test
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
torch.set_printoptions(profile="full")

run = '26'

# Hyperparams
BUFFER_SIZE = int(1e6)   # replay buffer size
BATCH_SIZE = 16           # minibatch size. must be even.
GAMMA = 0.99             # rewards discount factor
TAU = 0.001              # soft update rate
LR_ACTOR = 0.1           # learning rate for actor (upper bound)
LR_CRITIC = 0.1          # learning rate for critic (upper bound)
WEIGHT_DECAY = 0         # L2 weight decay for critic
UPDATE_EVERY = 200       # timesteps between soft updates
LEARN_EVERY = 1          # timesteps between learning
learn = False            # learning flag
device = "cpu"           # use CPU only
fc1_units = 1500         # number of units in first hidden layer
fc2_units = 1000         # number of units in second hidden layer
n_episodes = 2000        # number of episodes
max_t = 200              # max timesteps per episode
random_seed = 543        # ensure reproducibility on a given machine
SCALE = 1                # scale parameter for OU noise
SIGMA = 2                # initial value of noise parameter sigma
DECAY_RATE = 0.998       # sigma decay rate
STATE_SIZE = 24          # number of elements in player state returned by environment
ACTION_SIZE = 2          # number of elements in action space (up/down, forward/back)


### Utility functions ###
def init_limits(layer):
    """Calculate bounds for weight initialization based on rule of thumb:
    limit = 1/sqrt(n) where n = number of inputs. Initial weights should be close to zero."""
    num_inputs = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(num_inputs)
    return (-lim, lim)

def concatenate_inputs(inputs):
    """Concatenate player states into the following form:
        [[player_0_state player_1_state]
         [player_1_state player_0_state]]
        Args:
            len_24_states (numpy array)
        Return: Tensor with shape (num_rows, 48)
    """
    if type(inputs) == torch.Tensor:
        inputs = inputs.detach().numpy()
    num_rows = inputs.shape[0]
    new_size = 2 * inputs.shape[1]
    player_0 = inputs.flatten().reshape(num_rows//2, new_size)
    player_1 = np.flip(inputs, axis=0).flatten().reshape(num_rows//2, new_size)
    return torch.tensor(np.concatenate((player_0, player_1), axis=0), dtype=torch.float32)

def prep_critic_inputs(state_inputs, action_inputs):
    """Concatenate actions to player states
        Return: Critic input tensor with shape (num_rows, 52)
    """
    num_rows = state_inputs.shape[0]
    new_size = state_inputs.shape[1] + 2 * action_inputs.shape[1]
    catted_actions = concatenate_inputs(action_inputs)
    critic_inputs = np.concatenate((state_inputs, catted_actions), axis=1).flatten().reshape(num_rows, new_size)
    return torch.tensor(critic_inputs, dtype=torch.float32)

class Scores():
    def __init__(self):
        self.scores_deque = deque(maxlen=100)
        self.scores = []
        self.mean_scores = []
        self.run_results = pd.DataFrame(columns=["100_episode_avg", "episode_score"])
        self.score = np.zeros(2) # two players

    def add_rewards(self, rewards):
        self.score += rewards
        return

    def update_scores(self, episode):
        """Update score trackers at the end of an episode."""
        self.scores_deque.append(self.score[np.argmax(self.score)])
        self.scores.append(self.score[np.argmax(self.score)])
        self.mean_scores.append(np.mean(self.scores_deque))
        self.run_results.loc[f"episode_{episode}", "100_episode_avg"] = np.mean(self.scores_deque)
        self.run_results.loc[f"episode {episode}", "episode_score"] = self.score[np.argmax(self.score)]
        return

    def reset(self):
        self.score = np.zeros(2)

    def save_results(self):
        self.run_results.to_csv(f"run_{run}_results.csv")

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
        self.actor_input_size = 2 * state_size # input both players' states
        self.fc1 = nn.Linear(self.actor_input_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize layer weights."""
        self.fc1.weight.data.uniform_(*init_limits(self.fc1))
        self.fc2.weight.data.uniform_(*init_limits(self.fc2))
        self.fc3.weight.data.uniform_(-1e-3, 1e-3)
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
        self.critic_input_size = 2 * state_size + 2 * action_size # input both players' states and actions
        self.fc1 = nn.Linear(self.critic_input_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*init_limits(self.fc1))
        self.fc2.weight.data.uniform_(*init_limits(self.fc2))
        self.fc3.weight.data.uniform_(*init_limits(self.fc3))
        return

    def forward(self, state_action_inputs):
        """Execute a forward pass of the model."""
        x = F.leaky_relu(self.fc1(state_action_inputs))
        x = F.leaky_relu(self.fc2(x))
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

class ReplayBuffer():
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Args:
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, states, actions, rewards, next_states, dones):
        """Add experiences to ReplayBuffer"""
        e = self.experience(states, actions, rewards, next_states, dones)
        self.memory.append(e)
        return

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
    def __init__(self, state_size, action_size, random_seed, fc1_units, fc2_units, scale, batch_size, device, lr_actor=0.001, lr_critic=0.001, weight_decay=0, buffer_size=int(1e6)):
        # Initialize agent
        self.action_size = action_size
        self.batch_size = batch_size
        # Initialize Actor
        self.actor_local = Actor(state_size, action_size, random_seed, fc1_units, fc2_units).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed, fc1_units, fc2_units).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor, amsgrad=True)

        # Initialize Player 0 Critics
        self.critic_local_0 = Critic(state_size, action_size, random_seed, fc1_units, fc2_units).to(device)
        self.critic_target_0 = Critic(state_size, action_size, random_seed, fc1_units, fc2_units).to(device)
        self.critic_optimizer_0 = optim.Adam(self.critic_local_0.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY, amsgrad=True)

        # Initialize Player 1 Critics
        self.critic_local_1 = Critic(state_size, action_size, random_seed, fc1_units, fc2_units).to(device)
        self.critic_target_1 = Critic(state_size, action_size, random_seed, fc1_units, fc2_units).to(device)
        self.critic_optimizer_1 = optim.Adam(self.critic_local_1.parameters(), lr=lr_critic, weight_decay=weight_decay, amsgrad=True)

        # Replay buffer for player 0
        self.memory_player_0 = ReplayBuffer(action_size, buffer_size, batch_size, random_seed)
        # Replay buffer for player 1
        self.memory_player_1 = ReplayBuffer(action_size, buffer_size, batch_size, random_seed)

    def explore(self, scale, mu, theta, sigma):
        # Noise process
        self.noise = OUNoise(action_dim=self.action_size, scale=scale, mu=0, theta=0.15, sigma=sigma)

    def buffer(self, experience):
        """Add experience to replay buffers"""
        # Separate experience by player
        player_0 = (states[::2], actions[::2], rewards[::2], next_states[::2], dones[::2])
        player_1 = (states[1::2], actions[1::2], rewards[1::2], next_states[1::2], dones[1::2])

        # Add player experience to corresponding replay buffer
        self.memory_player_0.add(*player_0)
        self.memory_player_1.add(*player_1)
        return

    def act(self, input_states):
        """Return actions for a player given player's state. Optional: Clip actions after adding noise to keep racket height within range(-1,1).
        The action range exceeds (-1,1) but can be limited to this space to reduce training complexity.
        Args:
            input_states (array): player state concatenated with opposing player state. shape[-1]==48
        """
        catted_states = concatenate_inputs(input_states)
        self.actor_local.eval()
        with torch.no_grad():
            actions_pred = self.actor_local(catted_states).cpu().data.numpy()
        self.actor_local.train()
        player_0_noise = self.noise.noise().detach().numpy()
        player_1_noise = self.noise.noise().detach().numpy()
        # print(f"player_0_noise: {player_0_noise}")
        noise = [player_0_noise, player_1_noise]
        actions_pred += noise
        return np.clip(actions_pred, -1, 1)

    def train(self, gamma):
        """Train Actor and Critics using batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Args:
            gamma (float): discount factor
        """
        # Sample replay buffers
        if len(self.memory_player_0) < self.batch_size:
            return
        experiences = [self.memory_player_0.sample(), self.memory_player_1.sample()]
        local_critic = [self.critic_local_0, self.critic_local_1]
        target_critic = [self.critic_target_0, self.critic_target_1]
        critic_optimizer = [self.critic_optimizer_0, self.critic_optimizer_1]

        #### Train the Critics ####
        for player in range(2):
            states, actions, rewards, next_states, dones = experiences[player]
            catted_states = concatenate_inputs(states)
            catted_next_states = concatenate_inputs(next_states)

            # Compute Q targets
            next_actions = self.actor_target(catted_next_states)
            Q_targets_next_inputs = prep_critic_inputs(catted_next_states, next_actions)
            Q_targets_next = target_critic[player](Q_targets_next_inputs)
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

            # Compute critic loss
            Q_expected_inputs = prep_critic_inputs(catted_states, actions)
            Q_expected = local_critic[player](Q_expected_inputs)
            critic_loss = F.mse_loss(Q_expected, Q_targets)

            # Minimize the loss
            critic_optimizer[player].zero_grad() # reset gradients to 0
            critic_loss.backward() # compute the new gradients
            critic_optimizer[player].step() # update the parameters

            #### Train the Actor ####
            # Calculate the loss
            predicted_actions = self.actor_local(catted_states)
            states_and_predicted_actions = prep_critic_inputs(catted_states, predicted_actions)
            actor_loss = -local_critic[player](states_and_predicted_actions).mean()

            # Minimize the loss thru backprop
            self.actor_optimizer.zero_grad() # reset gradients to 0
            actor_loss.backward() # compute the new gradients
            self.actor_optimizer.step() # update the parameters

        return

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
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


if __name__ ==   "__main__":

    # Load the environment
    env = UnityEnvironment(file_name='../Tennis.x86_64')

    # Get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # Initialize the agent
    agent = Agent(state_size=STATE_SIZE, action_size=ACTION_SIZE, random_seed=random_seed, fc1_units=fc1_units, fc2_units=fc2_units, scale=SCALE, batch_size=BATCH_SIZE, device=device)

    # Initialize score tracking
    scores = Scores()

    # Execute episodes
    print('starting ddpg')
    for i_episode in range(1, n_episodes + 1):
        # Start run timer
        start = time.time()

        # Set OU Noise
        decaying_sigma = SIGMA * DECAY_RATE ** i_episode
        agent.noise = OUNoise(action_dim=ACTION_SIZE, scale=SCALE, mu=0, theta=0.15, sigma=decaying_sigma)

        # Reset scores
        scores.reset()

        # Reset the environment in train mode
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations

        # Execute timesteps
        for t in range(max_t):
            # Set update flags
            target_update = True if t % UPDATE_EVERY == 0 else False
            learn = True if t % LEARN_EVERY == 0 else False

            # Get actions
            actions = agent.act(states)

            # Get environment response
            env_step = env.step(actions)[brain_name]
            next_states = env_step.vector_observations
            rewards = env_step.rewards
            dones = env_step.local_done

            # Add experience to replay buffers
            reshaped_states = concatenate_inputs(states)
            reshaped_next_states = concatenate_inputs(next_states)
            experience = (reshaped_states, actions, rewards, reshaped_next_states, dones)
            agent.buffer(experience)

            # Train and update networks offline
            # TODO: train actor and critic networks
            if learn is True:
                agent.train(gamma=GAMMA)
            if target_update is True:
                agent.soft_update(agent.critic_local_0, agent.critic_target_0, TAU)
                agent.soft_update(agent.critic_local_1, agent.critic_target_1, TAU)

            # Prepare for next timestep
            if np.any(dones):
                break
            states = next_states
            scores.add_rewards(rewards)

        # Post-episode tasks
        scores.update_scores(i_episode)
        scores.save_results()
        end = time.time()
        print('\rEpisode {}\tAverage Score: {:.4f}\tScore: {:.4f}\t{:.2f}s'.format(i_episode,
                                                                                   np.mean(scores.scores_deque),
                                                                                   scores.scores[-1], end - start))


