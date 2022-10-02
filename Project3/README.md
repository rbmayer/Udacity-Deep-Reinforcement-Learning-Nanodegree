# Udacity Deep Reinforcement Learning Nanodegree
# Project 3: Collaboration and Competition

This project uses a deep reinforcement learning (deep RL) algorithm to solve a customized version of the Unity ML-Agents Tennis simulation. (The original Tennis instructions have been removed from Unity's Github repo, but a copy was still available in the [Chinese-language version](https://github.com/Unity-Technologies/ml-agents/blob/d34f3cd6ee078782b22341e4ceb958359069ab60/docs/localized/zh-CN/docs/Learning-Environment-Examples.md#tennis) as of October 2022.) The task in Tennis is to train a pair of agents to hit a tennis ball back and forth over a net and keep it in play for as long as possible.

## Environment

![](tennis.png)

The agents can move in two directions: up and down, and towards or away from the net. If the ball is hit over the net, the agent receives a reward of +0.1. If the ball hits the ground in the agent's space or the agent hits the ball out of bounds, the agent receives -0.1. These points are summed over the course of an episode to determine the scores.

The state space consists of 8 continuous variables. They represent the position and speed of the ball and racket. Each agent receives its own, local observation, such that the state space consists of 16 values. 

The environment returned two observations per step, each consisting of three "stacked" observations of 8 elements. Thus, the state vector returned by the environment contained 48 elements per step. It was not clear from the available documentation what each observation "stack" represented in terms of the game play.

## Definition of Solved

Each agent accrues points over the course of an episode without discounting. The higher of the two scores becomes the score for the episode. The environment is solved when an average score of +0.5 is achieved over 100 consecutive episodes. 

## How to Run

The code is written in python and must be run from the command line.
Following are instructions to set up and run the code in a Linux environment. 

1. Clone the project repository to a suitable machine. A workspace with a GPU is highly recommended. 
2. Create a virtual python environment based on Python 3.6. I strongly recommend using [conda](https://docs.conda.io/en/latest/miniconda.html), because the package requirements contain outdated packages that are not easily accessible using pip.  
    $ conda create -n drlnd
3. Activate the virtual environment and install the required packages.  
    $ conda activate drlnd
    $ conda install requirements.txt
4. Manually install torch version 0.4.1  
    $ conda install torch==0.4.1
5. Download the Unity environment provided by Udacity. For example:  
    $ wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip
6. Unzip the Tennis_Linux archive into a folder one level up in the hierarchy from where run_model.py is located.
7. Run the python script "run_model.py" from the command line to train the model.  
    $ python run_model.py
