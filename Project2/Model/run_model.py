from unityagents import UnityEnvironment
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
import random
import copy
import time
import pandas as pd
torch.set_printoptions(profile="full")

run = '79e'
actor_local_file = 'checkpoint_actor_local_{}.pth'.format(run)
actor_target_file = 'checkpoint_actor_target_{}.pth'.format(run)
critic_local_file = 'checkpoint_critic_local_{}.pth'.format(run)
critic_target_file = 'checkpoint_critic_target_{}.pth'.format(run)

def init_limits(layer):
    """Calculate bounds for weight initialization based on rule of thumb:
    limit = 1/sqrt(n) where n = number of inputs. Initial weights should be close to zero."""
    num_inputs = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(num_inputs)
    return (-lim, lim)


class Actor(nn.Module):
    """Create an actor (policy) network that maps states -> actions."""

    def __init__(self, state_size, action_size, seed, fc1_units, fc2_units):
        """Initialize parameters and build model.
        Args:
            state_size (int): Size of the state space
            action_size (int): Size of the action space
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize layer weights."""
        self.fc1.weight.data.uniform_(*init_limits(self.fc1))
        self.fc2.weight.data.uniform_(*init_limits(self.fc1))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        return

    def forward(self, state):
        """Execute a forward pass of the model."""
        x = F.leaky_relu(self.fc1(state))
        m1 = nn.BatchNorm1d(num_features=fc1_units).to(device)  # send to GPU if available
        xs = m1(x)
        x = F.leaky_relu(self.fc2(xs))
        return F.tanh(self.fc3(x))


class Critic(nn.Module):
    """Create a critic network that maps a (state, action) pair -> estimated Q-value."""

    def __init__(self, state_size, action_size, seed, fc1_units, fc2_units):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*init_limits(self.fc1))
        self.fc2.weight.data.uniform_(*init_limits(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        return

    def forward(self, state, action):
        """Execute a forward pass of the model."""
        xs = F.leaky_relu(self.fc1(state))
        # normalize the inputs to the first hidden layer
        m1 = nn.BatchNorm1d(num_features=fc1_units).to(device)  # send to GPU if available
        xs = m1(xs)
        # add action inputs in the last hidden layer when the state feature
        # representation is more developed (as per discussion in Udacity knowledgebase:
        # https://knowledge.udacity.com/questions/28877)
        x = torch.cat((xs, action), dim=1)
        # hidden layer
        x = F.leaky_relu(self.fc2(x))
        return self.fc3(x)


class OUNoise:
    """Implement Ornstein-Uhlenbeck noise-generation process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


class ReplayBuffer:
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
    def __init__(self, state_size, action_size, random_seed, fc1_units, fc2_units):
        """Initialize the Agent.
        Args:
            state_size (int): size of the state space
            action_size (int): size of the action space
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed, fc1_units, fc2_units).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed, fc1_units, fc2_units).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR, amsgrad=True)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed, fc1_units, fc2_units).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed, fc1_units, fc2_units).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY, amsgrad=True)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    def step(self, state, action, reward, next_state, done, target_update, learn):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        # and learn flag is True
        if len(self.memory) > BATCH_SIZE and learn is True:
            experiences = self.memory.sample()
            # print('experiences: {}'.format(experiences))
            self.learn(experiences, GAMMA, target_update)
        # ----------------------- update target networks ----------------------- #
        if target_update is True:
            self.soft_update(self.critic_local, self.critic_target, TAU)
            self.soft_update(self.actor_local, self.actor_target, TAU)

    def act(self, state):
        """Returns actions for given state as per current policy.
        Clip actions after adding noise to keep values within range(-1,1)."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset_noise(self):
        self.noise.reset()

    def learn(self, experiences, gamma, target_update=False):
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
        # print('size of dones: {}'.format(dones.shape))

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        # print('critic_target: {}'.format(self.critic_target))
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


def ddpg(env, n_episodes=200, max_t=700):
    """Train the agent.
    Args:
        n_episodes (int): number of episodes
        max_t: number of timesteps per episode
    """
    print('starting ddpg')
    scores_deque = deque(maxlen=100)
    scores = []
    mean_scores = []
    run_results = pd.DataFrame(
        columns=["scores_deque_mean", "episode_mean_score", *[f"agent_{i + 1}_score" for i in range(NUM_AGENTS)]])
    for i_episode in range(1, n_episodes+1):
        start = time.time()
        agent.reset_noise()
        score = np.zeros(20)
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        if i_episode in [lr_schedule_episode_1]:
            for g in agent.actor_optimizer.param_groups:
                g['lr'] = g['lr'] * lr_schedule_factor
            for g in agent.critic_optimizer.param_groups:
                g['lr'] = g['lr'] * lr_schedule_factor
        if i_episode == 28:
            agent.noise = OUNoise(size=action_size, seed=random_seed, sigma=0.10)
        elif i_episode == 56:
            agent.noise = OUNoise(size=action_size, seed=random_seed, sigma=0.05)
        for t in range(max_t):
            target_update = True if t%UPDATE_EVERY==0 else False
            actions = agent.act(states)
            brain_info = env.step(actions)[brain_name]
            next_states = brain_info.vector_observations
            rewards = brain_info.rewards
            dones = brain_info.local_done
            med = np.median(rewards)
            for i_agent in range(NUM_AGENTS):
                if rewards[i_agent] >= med:
                    learn = True
                else:
                    learn = False
                agent.step(states[i_agent], actions[i_agent], rewards[i_agent], next_states[i_agent], dones[i_agent],
                           target_update, learn)
            states = next_states
            score += rewards  # add each agent's reward to their existing tally
            if np.any(dones):
                break
        scores_deque.append(score.mean())
        scores.append(score)
        mean_scores.append(np.mean(scores_deque))
        # save gradients from first layers of actor and critic targets
        critic_fc1_gradient = pd.DataFrame(columns=["fc1_gradients"])
        actor_fc1_gradient = pd.DataFrame(columns=["fc1_gradients"])
        run_results.loc[f"episode_{i_episode}", "scores_deque_mean"] = np.mean(scores_deque)
        run_results.loc[f"episode_{i_episode}", "episode_mean_score"] = score.mean()
        run_results.loc[f"episode_{i_episode}", [f"agent_{i + 1}_score" for i in range(NUM_AGENTS)]] = score
        run_results.to_csv(f"run_{run}_results.csv")
        end = time.time()
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}\t{:.2f}s'.format(i_episode, np.mean(scores_deque), score.mean(), end - start))
        # save critic gradients
        current_critic_fc1_gradients = agent.critic_local.fc1.weight.grad.cpu()
        critic_fc1_gradient.loc[f"episode_{i_episode}", "fc1_gradients"] = current_critic_fc1_gradients
        # save actor gradients
        current_actor_fc1_gradients = agent.actor_local.fc1.weight.grad.cpu()
        actor_fc1_gradient.loc[f"episode_{i_episode}", "fc1_gradients"] = current_actor_fc1_gradients
        if i_episode % 25 == 0:
            torch.save(agent.actor_local.state_dict(),
                       f'checkpoint_actor_local_{run}.pth')
            torch.save(agent.actor_target.state_dict(),
                       f'checkpoint_actor_target_{run}.pth')
            torch.save(agent.critic_local.state_dict(),
                       f'checkpoint_critic_local_{run}.pth')
            torch.save(agent.critic_target.state_dict(),
                       f'checkpoint_critic_target_{run}.pth')
            critic_fc1_gradient.to_csv(f"run_{run}_critic_local_gradients.csv")
            actor_fc1_gradient.to_csv(f"run_{run}_actor_local_gradients.csv")
    return


if __name__=="__main__":
    # load the environment
    # select this option to load version 2 (with 20 agents) of the environment
    env = UnityEnvironment(file_name='../Reacher.x86_64')

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # get state_size and action_size
    env_info = env.reset(train_mode=True)[brain_name]
    state_size=env_info.vector_observations.shape[1]
    action_size=brain.vector_action_space_size

    # Hyperparams
    BUFFER_SIZE = int(1e5)      # replay buffer size
    BATCH_SIZE = 32             # minibatch size
    GAMMA = 0.99                # rewards discount factor
    TAU = 0.001                 # soft update interpolation parameter for target networks
    LR_ACTOR = 0.0007           # upper bound learning rate of the actor
    LR_CRITIC = 0.0003          # upper bound learning rate of the critic
    lr_schedule_episode_1 = 60  # episode designated for first LR scheduled adjustment
    lr_schedule_factor = 0.1    # factor for LR scheduled adjustment
    WEIGHT_DECAY = 0            # L2 weight decay
    UPDATE_EVERY = 20           # timesteps between soft updates
    learn = False               # whether to run backprop on agent inputs
    NUM_AGENTS = len(env_info.agents)   # number of simultaneous agents
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fc1_units = 200             # number of units in first hidden layer
    fc2_units = 100             # number of units in second hidden layer
    n_episodes = 130            # number of episodes
    max_t = 1000                # maximum timesteps per episode
    random_seed = 543           # random seed to ensure reproducibility on a given machine

    # Initialize the agent
    agent = Agent(state_size=state_size, action_size=action_size, random_seed=random_seed, fc1_units=fc1_units, fc2_units=fc2_units)

    # Train the model
    ddpg(env=env, n_episodes=n_episodes, max_t=max_t)
