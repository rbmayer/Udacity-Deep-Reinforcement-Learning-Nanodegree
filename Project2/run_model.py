import csv
from unityagents import UnityEnvironment
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torch.utils.data import DataLoader
from collections import namedtuple, deque
import random
import copy
# from workspace_utils import active_session # keep-alive url only works with Udacity workspace
import sdepy
import pandas as pd

run = 35

def init_limits(layer):
    """Calculate bounds for weight initialization based on rule of thumb:
    limit = 1/sqrt(n) where n = number of inputs to neurons in layer.
    Initial weights should be close to zero.
    """
    num_inputs = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(num_inputs)
    return (-lim, lim)
        
class Actor(nn.Module):
    """Create the actor network."""

    def __init__(self, state_size, action_size, seed, fc1_units, fc2_units):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
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
        self.fc1.weight.data.uniform_(*init_limits(self.fc1))
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)
        return

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        # normalize the inputs 
        m1 = nn.BatchNorm1d(num_features=state.shape[1]).to(device)  # send to GPU if available
        normalized_state = m1(state)
        # input layer
        x = F.leaky_relu(self.fc1(normalized_state))  
        # next layer
        x = F.leaky_relu(self.fc2(x))
        # output layer
        return torch.tanh(self.fc3(x)) # nn.functional.tanh is deprecated. Use torch.tanh instead.
                
class Critic(nn.Module):
    """Create the critic network."""

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
        """Build a critic (value) network that maps (state, action) pairs -> Q-values.
           Apply batch normalization between each layer.
           Add action inputs in the last hidden layer when the state feature 
           representation is more developed (as per discussion in Udacity knowledgebase: 
           https://knowledge.udacity.com/questions/28877).
        """
        # normalize the inputs 
        m1 = nn.BatchNorm1d(num_features=state.shape[1]).to(device)  # send to GPU if available
        normalized_state = m1(state)
        # input layer
        x = F.leaky_relu(self.fc1(normalized_state))          
        # add the action values
        x = torch.cat((x, action), dim=1)
        # next layer
        x = F.leaky_relu(self.fc2(x))
        # output layer
        return self.fc3(x)


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
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
                
class Agent():
    def __init__(self, state_size, action_size, random_seed, fc1_units, fc2_units):
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

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed, fc1_units, fc2_units).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed, fc1_units, fc2_units).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed, fc1_units, fc2_units).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed, fc1_units, fc2_units).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        # self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    def step(self, state, action, reward, next_state, done, target_update):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)
        # Learn, if enough samples are available in memory
        # print('len(memory): {}'.format(len(self.memory)))
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            # print('experiences: {}'.format(experiences))
            self.learn(experiences, GAMMA, target_update)

    def act(self, state, noise, timestep):
        """Returns actions for given state as per current policy and adds noise. 
        Clip actions after adding noise to keep values within range(-1,1)."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        action += noise
        # action += self.noise.sample()
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
        # add gradient clipping
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
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
        if target_update is True:
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
                        
if __name__=="__main__":
    # load the environment
    # select this option to load version 1 (with a single agent) of the environment
    # env = UnityEnvironment(file_name='/data/Reacher_One_Linux_NoVis/Reacher_One_Linux_NoVis.x86_64')

    # select this option to load version 2 (with 20 agents) of the environment
    # env = UnityEnvironment(file_name='/data/Reacher_Linux_NoVis/Reacher.x86_64')
    # P2:
    env = UnityEnvironment(file_name='Reacher.x86_64')

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    print(brain_name)

    # get state_size and action_size
    env_info = env.reset(train_mode=True)[brain_name]
    state_size=env_info.vector_observations.shape[1]
    action_size=brain.vector_action_space_size
    
    # Hyperparams
    BUFFER_SIZE = int(1e5)  # replay buffer size
    BATCH_SIZE = 32        # minibatch size
    GAMMA = 0.99            # discount factor
    TAU = 0.001             # for soft update of target parameters
    LR_ACTOR = 0.00007        # learning rate of the actor 
    LR_CRITIC = 0.00003       # learning rate of the critic
    WEIGHT_DECAY = 0        # L2 weight decay
    UPDATE_EVERY = 10      # steps between target network updates
    RANDOM_SEED = 543
    sigma = 0.2
    NOISE_DECAY = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device: {}'.format(device))

    fc1_units = 200
    fc2_units = 100

    NUM_AGENTS = len(env_info.agents)
    MAX_T = 1000            # max number of time steps per spisode
        
    agent = Agent(state_size=state_size, action_size=action_size, random_seed=RANDOM_SEED, fc1_units=fc1_units,
             fc2_units=fc2_units)
                         
    n_episodes=1000
    max_t=MAX_T
    print('starting ddpg')
    scores_deque = deque(maxlen=100)
    scores = []
    mean_scores = []
    run_results = pd.DataFrame(columns=["100_episode_mean", "episode_mean_score", *[f"agent_{i+1}_score" for i in range(NUM_AGENTS)]])
    for i_episode in range(1, n_episodes+1): 
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        # decay sd of the noise process 
        SIGMA = max(0.05, sigma*NOISE_DECAY**i_episode)
        @sdepy.integrate
        def ou_noise_process(t, x, theta=0.15, k=1, sigma=SIGMA):
            return {'dt': k*(theta - x), 'dw': sigma}   
        timeline = np.linspace(0, max_t-1, max_t)
        noise = ou_noise_process(x0=0, vshape=NUM_AGENTS, paths=action_size, steps=max_t)(timeline)
        score = np.zeros(NUM_AGENTS)
        for t in range(max_t):  
            target_update = True if t%UPDATE_EVERY==0 else False
            actions = agent.act(states, noise[t], t)
            brain_info = env.step(actions)[brain_name]
            next_states = brain_info.vector_observations
            rewards = brain_info.rewards
            dones = brain_info.local_done
            for a in range(NUM_AGENTS):
                agent.step(states[a], actions[a], rewards[a], next_states[a], dones[a], target_update)
            states = next_states
            score += rewards  # add each agent's reward to their existing tally
            if np.any(dones):
                break 
        scores_deque.append(score.mean())
        scores.append(score)
        mean_scores.append(np.mean(scores_deque))
        run_results.loc[f"episode_{i_episode}", "episode_mean_score"] = score.mean()
        run_results.loc[f"episode_{i_episode}", [f"agent_{i+1}_score" for i in range(NUM_AGENTS)]] = score
        run_results.loc[f"episode_{i_episode}", "100_episode_mean"] = run_results['episode_mean_score'][max(run_results.shape[0]-100, 0):].mean()
        run_results.to_csv(f"run_{run}_results.csv")
        print("\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}\tsigma: {}".format(i_episode, np.mean(scores_deque), score.mean(), SIGMA))
        if i_episode % 25 == 0:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor_local.pth')
            torch.save(agent.actor_target.state_dict(), 'checkpoint_actor_target.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic_local.pth')
            torch.save(agent.critic_target.state_dict(), 'checkpoint_critic_target.pth')