import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from TD3.td3_models import Actor, Critic
from utils.replay_buffer import ReplayBuffer
from utils.ou_noise import OUNoise
#variant with prioritized experience replay
#from utils.per import PER


##### HYPERPARAMETERS #####
# Replay buffer size 
BUFFER_SIZE = int(1e6)
# Minibatch size
BATCH_SIZE = 1024
# Discount factor
GAMMA = 0.99
# Target parameters soft update factor
TAU = 0.005
# Learning rate of the actor network
LR_ACTOR = 3e-4
# Learning rate of the critic network
LR_CRITIC = 3e-4
# L2 weight decay
WEIGHT_DECAY = 0.0
# The actor is updated after every so many times the critic is updated (Delayed Policy Updates)
UPDATE_ACTOR_EVERY = 2
# Std dev of Gaussian noise added to action policy (Target Policy Smoothing Regularization)
POLICY_NOISE = 0.2
# Clip boundaries of the noise added to action policy
POLICY_NOISE_CLIP = 0.5

actor_solved_model = 'saved_data/actor_solved.pth'
critic1_solved_model = 'saved_data/critic1_solved.pth'
critic2_solved_model = 'saved_data/critic2_solved.pth'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', device)

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed):
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
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        
        # Critic Network 1 (w/ Target Network1)
        self.critic1_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic1_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic1_optimizer = optim.Adam(self.critic1_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        
        # Critic Network 2 (w/ Target Network2)
        self.critic2_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic2_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic2_optimizer = optim.Adam(self.critic2_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        
        # Noise process
        self.noise = OUNoise(action_size)

        
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        
        # Initialize time step (for updating every UPDATE_EVERY and LEARN_EVERY steps)
        self.t_step = 0
    
    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory."""
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)
            
        # Learn, if enough samples are available in memory    
        if len(self.memory) > BATCH_SIZE:
            self.learn()
            
    def act(self, state):
        """Returns actions for given states as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        
        action += self.noise.sample()
        
        return np.clip(action, -1, 1)    
        
    
    def reset(self):
        self.noise.reset()
    
    def learn(self):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        """
        self.t_step+=1
        states, actions, rewards, next_states, dones =  self.memory.sample()
                
        # ---------------------------- update critic ---------------------------- #
        # Target Policy Smoothing Regularization: add a small amount of clipped random noises to the selected action
        if POLICY_NOISE > 0.0:
            noise = torch.empty_like(actions).data.normal_(0, POLICY_NOISE).to(device)
            noise = noise.clamp(-POLICY_NOISE_CLIP, POLICY_NOISE_CLIP)
            # Get predicted next-state actions and Q values from target models
            actions_next = (self.actor_target(next_states) + noise).clamp (-1., 1.)
        else:
            # Get predicted next-state actions and Q values from target models
            actions_next = self.actor_target(next_states)
            
        # Error Mitigation
        Q1_target = self.critic1_target(next_states, actions_next)
        Q2_target = self.critic2_target(next_states, actions_next)
        Q_targets_next = torch.min(Q1_target, Q2_target)
        
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + dones * GAMMA * Q_targets_next
            
        # Compute critic1 loss
        Q1_expected = self.critic1_local(states, actions)
        critic1_loss = F.mse_loss(Q1_expected, Q_targets)
        # Minimize the loss
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.critic1_local.parameters(), 1)
        self.critic1_optimizer.step()
                        
        # Compute critic2 loss
        Q2_expected = self.critic2_local(states, actions)
        critic2_loss = F.mse_loss(Q2_expected, Q_targets)
        # Minimize the loss
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.critic2_local.parameters(), 1)
        self.critic2_optimizer.step()
            
        # Delayed Policy Updates
        if self.t_step % UPDATE_ACTOR_EVERY == 0:
            # ---------------------------- update actor ---------------------------- #
            # Compute actor loss
            actions_pred = self.actor_local(states)
            actor_loss = -self.critic1_local(states, actions_pred).mean()
            # Minimize the loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
                
            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.critic1_local, self.critic1_target, TAU)
            self.soft_update(self.critic2_local, self.critic2_target, TAU)
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
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def save_models(self):
        torch.save(self.actor_local.state_dict(), actor_solved_model)
        torch.save(self.critic1_local.state_dict(), critic1_solved_model)
        torch.save(self.critic2_local.state_dict(), critic2_solved_model)
    



