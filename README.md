# Project : Colaboration and Competition

## Description 
For this project, we train a pair of agents to play tennis

![Agentes Playing Tennis][images/AgentsPlayingTennis.gif]

## Problem Statement 
In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. 
Thus, the goal of each agent is to keep the ball in play.


The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic. After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores. This yields a single score for each episode. 

The environment is considered solved  when the agents score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). 

## Files 
- `Tennis.ipynb`: Notebook used to control and train the agents 
- `TD3/td3_agent.py`: Agent class that interacts with and learns from the environment implemented by TD3 algorithm
- `TD3/td3_models.py`: Actor and Critic classes models used for TD3 algorithm
- `DDPG/ddpg_agent.py`: Agent class that interacts with and learns from the environment implemented by DDPG algorithm
- `DDPG/ddpg_models.py`: Actor and Critic classes models used for DDPG algorithm
- `utils/replay_buffer.py`: Internal class used to map states to action values using prioritized experience replay.
- `utils/ou_noise.py`: Internal class used to calculate the noise using Ornstein-Uhlenbeck method
- `utils/workspace_utils.py`: Internal functions to keep the workspace session alive during a longer run
- `saved_data/actor_solved.pth`: Saved actor weigths
- `saved_data/critic1_solved.pth`: Saved critic1 network weigths
- `saved_data/critic2_solved.pth`: Saved critic2 network weigths
- `README.md`: README file with project description
- `install_requirements.txt`: File with python packages install dependencies
- `report.pdf`: Technical report on the project. Algorithm choise, design decission, future improvements,etc... 

## Dependencies
To be able to run this code, you will need an environment with Python 3 and 
the dependencies are listed in the `requirements.txt` file so that you can install them
using the following command: 
```
pip install install_requirements.txt
``` 

Furthermore, you need to download the environment from one of the links below. You need only to select
the environment that matches your operating system:
- Linux : [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
- MAC OSX : [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- Windows : [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

## Running
Run the cells in the notebook `Tennis.ipynb` to train an agent that solves our required
task of playing tennis.