"""
Neural network agents for large-scale RL training.
Optimized for efficient batch processing and distributed training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List
from .environment import AgentState, ObservationSystem


class ActorNetwork(nn.Module):
    """Actor network for policy-based RL agents."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Feature extraction
        self.feature_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Policy head
        self.policy_mean = nn.Linear(hidden_dim, action_dim)
        self.policy_std = nn.Linear(hidden_dim, action_dim)
        
        # Value head (for actor-critic)
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through actor network."""
        features = self.feature_net(observations)
        
        # Policy output
        action_mean = self.policy_mean(features)
        action_std = F.softplus(self.policy_std(features)) + 1e-5
        
        # Value output
        value = self.value_head(features)
        
        return action_mean, action_std, value


class CriticNetwork(nn.Module):
    """Critic network for value function estimation."""
    
    def __init__(self, obs_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass through critic network."""
        return self.value_net(observations)


class PPOAgent:
    """Proximal Policy Optimization agent optimized for batch processing."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = 'cpu'
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        
        # Hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Networks
        self.actor = ActorNetwork(obs_dim, action_dim, hidden_dim).to(device)
        self.critic = CriticNetwork(obs_dim, hidden_dim).to(device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr
        )
        
        # Training statistics
        self.training_stats = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy_loss': 0.0,
            'total_loss': 0.0,
            'clip_fraction': 0.0
        }
        
    def get_action(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action from policy."""
        with torch.no_grad():
            action_mean, action_std, value = self.actor(observations)
            
            # Sample action
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            
        return action, log_prob, value
        
    def evaluate_action(self, observations: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate action for training."""
        action_mean, action_std, value = self.actor(observations)
        
        # Compute log probability
        dist = torch.distributions.Normal(action_mean, action_std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, entropy, value
        
    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation."""
        batch_size = rewards.size(0)
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        # Compute advantages
        gae = 0
        for t in reversed(range(batch_size)):
            if t == batch_size - 1:
                next_value = 0
                next_done = 1
            else:
                next_value = values[t + 1]
                next_done = dones[t + 1]
                
            delta = rewards[t] + self.gamma * next_value * (1 - next_done) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - next_done) * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
            
        return advantages, returns
        
    def update(self, observations: torch.Tensor, actions: torch.Tensor, 
               rewards: torch.Tensor, dones: torch.Tensor, old_log_probs: torch.Tensor) -> Dict:
        """Update agent using PPO."""
        
        # Move to device
        observations = observations.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        
        # Evaluate current policy
        new_log_probs, entropy, values = self.evaluate_action(observations, actions)
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, dones)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute policy loss
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Compute value loss
        value_pred = self.critic(observations).squeeze()
        value_loss = F.mse_loss(value_pred, returns)
        
        # Compute entropy loss
        entropy_loss = -entropy.mean()
        
        # Total loss
        total_loss = (policy_loss + 
                     self.value_loss_coef * value_loss + 
                     self.entropy_coef * entropy_loss)
        
        # Update networks
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            self.max_grad_norm
        )
        self.optimizer.step()
        
        # Update statistics
        self.training_stats.update({
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item(),
            'clip_fraction': ((ratio - 1.0).abs() > self.clip_ratio).float().mean().item()
        })
        
        return self.training_stats.copy()


class BatchProcessor:
    """Efficient batch processing for large-scale training."""
    
    def __init__(self, batch_size: int = 1024, device: str = 'cpu'):
        self.batch_size = batch_size
        self.device = device
        
    def process_batch(self, agent: PPOAgent, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process a batch of observations efficiently."""
        num_samples = observations.size(0)
        
        all_actions = []
        all_log_probs = []
        all_values = []
        
        # Process in chunks
        for i in range(0, num_samples, self.batch_size):
            end_idx = min(i + self.batch_size, num_samples)
            batch_obs = observations[i:end_idx].to(self.device)
            
            actions, log_probs, values = agent.get_action(batch_obs)
            
            all_actions.append(actions.cpu())
            all_log_probs.append(log_probs.cpu())
            all_values.append(values.cpu())
            
        return (torch.cat(all_actions, dim=0),
                torch.cat(all_log_probs, dim=0),
                torch.cat(all_values, dim=0))


class DistributedAgentManager:
    """Manager for distributed agent training across multiple GPUs/processes."""
    
    def __init__(self, num_agents: int, obs_dim: int, action_dim: int, 
                 devices: List[str] = None, hidden_dim: int = 256):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Default to available devices
        if devices is None:
            devices = ['cpu']
            if torch.cuda.is_available():
                devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
                
        self.devices = devices
        self.num_devices = len(devices)
        
        # Distribute agents across devices
        agents_per_device = num_agents // self.num_devices
        self.agent_distribution = []
        
        for i, device in enumerate(devices):
            start_idx = i * agents_per_device
            end_idx = start_idx + agents_per_device if i < self.num_devices - 1 else num_agents
            
            agent = PPOAgent(obs_dim, action_dim, hidden_dim, device=device)
            self.agent_distribution.append({
                'device': device,
                'agent': agent,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'batch_processor': BatchProcessor(device=device)
            })
            
    def get_actions(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get actions from all agents across devices."""
        all_actions = []
        all_log_probs = []
        all_values = []
        
        for agent_info in self.agent_distribution:
            start_idx = agent_info['start_idx']
            end_idx = agent_info['end_idx']
            
            if start_idx >= observations.size(0):
                continue
                
            agent_obs = observations[start_idx:end_idx]
            actions, log_probs, values = agent_info['batch_processor'].process_batch(
                agent_info['agent'], agent_obs
            )
            
            all_actions.append(actions)
            all_log_probs.append(log_probs)
            all_values.append(values)
            
        return (torch.cat(all_actions, dim=0),
                torch.cat(all_log_probs, dim=0),
                torch.cat(all_values, dim=0))
                
    def update_agents(self, observations: torch.Tensor, actions: torch.Tensor,
                     rewards: torch.Tensor, dones: torch.Tensor, old_log_probs: torch.Tensor) -> Dict:
        """Update all agents across devices."""
        all_stats = []
        
        for agent_info in self.agent_distribution:
            start_idx = agent_info['start_idx']
            end_idx = agent_info['end_idx']
            
            if start_idx >= observations.size(0):
                continue
                
            # Extract data for this agent
            agent_obs = observations[start_idx:end_idx]
            agent_actions = actions[start_idx:end_idx]
            agent_rewards = rewards[start_idx:end_idx]
            agent_dones = dones[start_idx:end_idx]
            agent_old_log_probs = old_log_probs[start_idx:end_idx]
            
            # Update agent
            stats = agent_info['agent'].update(
                agent_obs, agent_actions, agent_rewards, agent_dones, agent_old_log_probs
            )
            all_stats.append(stats)
            
        # Average statistics across agents
        avg_stats = {}
        for key in all_stats[0].keys():
            avg_stats[key] = np.mean([stats[key] for stats in all_stats])
            
        return avg_stats
