# Navigation Project Report

This project implements a deep Q-network (DQN) using the approach described in the 2015 paper by Mnih et al, [Human-level control through deep reinforcement learning](https://doi.org/10.1038/nature14236).

## Solution

This implementation solved the environment in 372 episodes. The suggested benchmark for completing the exercise was to solve the environment in fewer than 1800 episodes.  

The key to reducing the number of training episodes was to add a decaying epsilon-greedy policy to the basic DQN solution provided in earlier assignments. This ensured that early episodes focused on exploring the state space while transitioning fairly quickly (within a couple hundred episodes) to a policy based on experience.

## Reinforcement Learning Framework

The reinforcement learning (RL) framework models the interactions between:  

* an environment that provides rewards based on its state;
* an agent seeking to maximize those rewards;
* a policy that the agent uses to select an action in each state.

The RL solution generates the policy from an action-value function, also called a Q-function, that estimates the value of each action in a given state. The Q-function is used to estimate the action with the highest reward.

## Learning algorithm

The DQN uses two neural networks to estimate the action-value function: a local network representing the current action-value function estimate, and a target network that represents the supervised learning targets. Every n steps, the current reward is added to the target network's predicted value of the next state to generate the targets. The weights of the local network are trained to minimize loss relative to the targets.

The algorithm is an online method that updates the q-network incrementally while playing, rather than holding updates until the end of an episode. This implementation uses an epsilon-greedy policy in which the agent chooses a random action with probability ε and the greedy action with probability 1 - ε.

### Model architecture

Each neural network consist of three fully-connected layers. The input layer contains four units - one for each element of the state space - and the hidden layer contains 64 units. Both input and hidden layers have RELU activation functions.


### Hyperparameters

The solution utilizes the following hyperparameters:  

|Hyperparameter|Value|Description|
|-----|---|-----------|
|BUFFER_SIZE|100000|replay buffer size|
|BATCH_SIZE|64|minibatch size|
|GAMMA|0.99|discount rate|
|TAU|.001|weight for soft update of target q-network parameters|  
|LR|.0005|learning rate for backprop|  
|UPDATE_EVERY|4|number of time steps between backward passes on local q-network|  
|INITIAL EPSILON|0.99|initial value of ε, the probability of random action by the agent|
|DECAY RATE|0.005|rate at which epsilon decreases each episode|

## Ideas for Future Work

I experimented with a number of simple types of prioritized replay to see if training time could be reduced further. These efforts were unsuccessful, and resulted in longer training times.

My tests showed that for this particular game, almost 98% of steps yielded a reward of zero. The bananas are spread out in a relatively large space, such that the rewards from most steps don't yield much information; both positive and negative rewards are delayed.

I tried creating a second replay buffer and populating it with either positive experiences only, or positive and negative experiences. Then I tested learning with different proportions of replay experience drawn from each buffer. In hindsight, I suspect that this approach was unsuccessful because the key to learning the game is not what you do when you're close to a banana, but how learning to orient toward or away from the banana when you see it from far away.

For future work, I would like to understand how the key learning steps can be identified mathematically within the algorithm. It would be useful to abstract that information in order to be able to apply it to solutions of more complex problems.



## References

Mnih, V., Kavukcuoglu, K., Silver, D. et al. [Human-level control through deep reinforcement learning](https://doi.org/10.1038/nature14236). Nature 518, 529–533 (2015). https://doi.org/10.1038/nature14236
