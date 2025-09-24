"""
Distributed training system for large-scale RL simulations.
Supports thousands of parallel agents with efficient resource management.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional
import json
import os
from dataclasses import dataclass
import queue
import multiprocessing as mp

from .environment import DistributedSimulation, SimulationConfig
from .agents import PPOAgent, DistributedAgentManager, BatchProcessor


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    num_agents: int = 10000
    max_steps: int = 1000000
    update_frequency: int = 2048
    num_epochs: int = 4
    batch_size: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    save_frequency: int = 10000
    log_frequency: int = 100
    device: str = 'auto'
    num_workers: int = 4


class TrainingMetrics:
    """Track training metrics and performance statistics."""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset all metrics."""
        self.episode_rewards = []
        self.episode_lengths = []
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        self.total_losses = []
        self.clip_fractions = []
        self.fps_history = []
        self.agent_throughput = []
        
    def update(self, stats: Dict, fps: float, agent_throughput: float):
        """Update metrics with new statistics."""
        self.policy_losses.append(stats.get('policy_loss', 0))
        self.value_losses.append(stats.get('value_loss', 0))
        self.entropy_losses.append(stats.get('entropy_loss', 0))
        self.total_losses.append(stats.get('total_loss', 0))
        self.clip_fractions.append(stats.get('clip_fraction', 0))
        self.fps_history.append(fps)
        self.agent_throughput.append(agent_throughput)
        
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        return {
            'avg_policy_loss': np.mean(self.policy_losses[-100:]) if self.policy_losses else 0,
            'avg_value_loss': np.mean(self.value_losses[-100:]) if self.value_losses else 0,
            'avg_entropy_loss': np.mean(self.entropy_losses[-100:]) if self.entropy_losses else 0,
            'avg_total_loss': np.mean(self.total_losses[-100:]) if self.total_losses else 0,
            'avg_clip_fraction': np.mean(self.clip_fractions[-100:]) if self.clip_fractions else 0,
            'avg_fps': np.mean(self.fps_history[-100:]) if self.fps_history else 0,
            'avg_agent_throughput': np.mean(self.agent_throughput[-100:]) if self.agent_throughput else 0,
            'total_updates': len(self.policy_losses)
        }


class DistributedTrainer:
    """Distributed trainer for large-scale RL simulations."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # Setup device
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
            
        print(f"Using device: {self.device}")
        
        # Initialize simulation
        sim_config = SimulationConfig(
            num_agents=config.num_agents,
            max_steps=config.max_steps,
            num_workers=config.num_workers
        )
        self.simulation = DistributedSimulation(sim_config)
        
        # Initialize agents
        obs_dim = 8  # From ObservationSystem
        action_dim = 2  # [linear_velocity, angular_velocity]
        
        self.agent_manager = DistributedAgentManager(
            num_agents=config.num_agents,
            obs_dim=obs_dim,
            action_dim=action_dim,
            devices=[self.device] if self.device != 'cpu' else None,
            hidden_dim=256
        )
        
        # Training state
        self.step_count = 0
        self.update_count = 0
        self.start_time = time.time()
        self.metrics = TrainingMetrics()
        
        # Experience buffers
        self.observations_buffer = []
        self.actions_buffer = []
        self.rewards_buffer = []
        self.dones_buffer = []
        self.log_probs_buffer = []
        self.values_buffer = []
        
        # Threading for async operations
        self.experience_queue = queue.Queue(maxsize=1000)
        self.training_thread = None
        self.training_active = False
        
    def collect_experience(self) -> Dict:
        """Collect experience from simulation."""
        # Reset simulation
        observations = self.simulation.reset()
        
        episode_rewards = []
        episode_length = 0
        
        for step in range(self.config.update_frequency):
            # Get actions from agents
            actions, log_probs, values = self.agent_manager.get_actions(observations)
            
            # Step simulation
            next_observations, rewards, dones, info = self.simulation.step(actions)
            
            # Store experience
            self.observations_buffer.append(observations)
            self.actions_buffer.append(actions)
            self.rewards_buffer.append(rewards)
            self.dones_buffer.append(dones)
            self.log_probs_buffer.append(log_probs)
            self.values_buffer.append(values)
            
            # Update statistics
            episode_rewards.append(rewards.sum().item())
            episode_length += 1
            
            observations = next_observations
            self.step_count += 1
            
            # Check if episode is done
            if dones.all():
                break
                
        # Convert buffers to tensors
        experience = {
            'observations': torch.cat(self.observations_buffer, dim=0),
            'actions': torch.cat(self.actions_buffer, dim=0),
            'rewards': torch.cat(self.rewards_buffer, dim=0),
            'dones': torch.cat(self.dones_buffer, dim=0),
            'log_probs': torch.cat(self.log_probs_buffer, dim=0),
            'values': torch.cat(self.values_buffer, dim=0),
            'episode_rewards': episode_rewards,
            'episode_length': episode_length
        }
        
        # Clear buffers
        self.clear_buffers()
        
        return experience
        
    def clear_buffers(self):
        """Clear experience buffers."""
        self.observations_buffer.clear()
        self.actions_buffer.clear()
        self.rewards_buffer.clear()
        self.dones_buffer.clear()
        self.log_probs_buffer.clear()
        self.values_buffer.clear()
        
    def update_agents(self, experience: Dict) -> Dict:
        """Update agents using collected experience."""
        observations = experience['observations']
        actions = experience['actions']
        rewards = experience['rewards']
        dones = experience['dones']
        old_log_probs = experience['log_probs']
        
        # Update agents
        stats = self.agent_manager.update_agents(
            observations, actions, rewards, dones, old_log_probs
        )
        
        self.update_count += 1
        return stats
        
    def train_step(self) -> Dict:
        """Execute one training step."""
        # Collect experience
        experience = self.collect_experience()
        
        # Update agents
        stats = self.update_agents(experience)
        
        # Calculate performance metrics
        elapsed_time = time.time() - self.start_time
        fps = self.step_count / elapsed_time if elapsed_time > 0 else 0
        agent_throughput = (self.config.num_agents * self.step_count) / elapsed_time if elapsed_time > 0 else 0
        
        # Update metrics
        self.metrics.update(stats, fps, agent_throughput)
        
        # Create summary
        summary = {
            'step': self.step_count,
            'update': self.update_count,
            'episode_reward': np.mean(experience['episode_rewards']),
            'episode_length': experience['episode_length'],
            'fps': fps,
            'agent_throughput': agent_throughput,
            'elapsed_time': elapsed_time,
            **stats
        }
        
        return summary
        
    def train(self, max_updates: int = None) -> Dict:
        """Main training loop."""
        max_updates = max_updates or (self.config.max_steps // self.config.update_frequency)
        
        print(f"Starting training for {max_updates} updates...")
        print(f"Agents: {self.config.num_agents}, Update frequency: {self.config.update_frequency}")
        
        for update in range(max_updates):
            # Training step
            summary = self.train_step()
            
            # Logging
            if update % self.config.log_frequency == 0:
                self.log_progress(update, summary)
                
            # Save checkpoint
            if update % self.config.save_frequency == 0 and update > 0:
                self.save_checkpoint(update)
                
        # Final summary
        final_summary = self.metrics.get_summary()
        final_summary.update({
            'total_steps': self.step_count,
            'total_updates': self.update_count,
            'total_time': time.time() - self.start_time
        })
        
        return final_summary
        
    def log_progress(self, update: int, summary: Dict):
        """Log training progress."""
        print(f"\nUpdate {update}/{self.config.max_steps // self.config.update_frequency}")
        print(f"  Step: {summary['step']}")
        print(f"  Episode Reward: {summary['episode_reward']:.2f}")
        print(f"  Episode Length: {summary['episode_length']}")
        print(f"  FPS: {summary['fps']:.1f}")
        print(f"  Agent Throughput: {summary['agent_throughput']:.0f} agents/sec")
        print(f"  Policy Loss: {summary['policy_loss']:.4f}")
        print(f"  Value Loss: {summary['value_loss']:.4f}")
        print(f"  Total Loss: {summary['total_loss']:.4f}")
        print(f"  Clip Fraction: {summary['clip_fraction']:.3f}")
        
    def save_checkpoint(self, update: int):
        """Save training checkpoint."""
        checkpoint = {
            'update': update,
            'step_count': self.step_count,
            'config': self.config.__dict__,
            'metrics': self.metrics.get_summary(),
            'simulation_stats': self.simulation.get_performance_stats()
        }
        
        # Save agent states
        agent_states = {}
        for i, agent_info in enumerate(self.agent_manager.agent_distribution):
            agent_states[f'agent_{i}'] = {
                'actor_state_dict': agent_info['agent'].actor.state_dict(),
                'critic_state_dict': agent_info['agent'].critic.state_dict(),
                'optimizer_state_dict': agent_info['agent'].optimizer.state_dict()
            }
        checkpoint['agent_states'] = agent_states
        
        # Save to file
        checkpoint_path = f"checkpoint_update_{update}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load training state
        self.update_count = checkpoint['update']
        self.step_count = checkpoint['step_count']
        
        # Load agent states
        agent_states = checkpoint['agent_states']
        for i, agent_info in enumerate(self.agent_manager.agent_distribution):
            if f'agent_{i}' in agent_states:
                agent_info['agent'].actor.load_state_dict(agent_states[f'agent_{i}']['actor_state_dict'])
                agent_info['agent'].critic.load_state_dict(agent_states[f'agent_{i}']['critic_state_dict'])
                agent_info['agent'].optimizer.load_state_dict(agent_states[f'agent_{i}']['optimizer_state_dict'])
                
        print(f"Checkpoint loaded: {checkpoint_path}")


class BenchmarkTrainer:
    """Benchmark trainer for performance testing."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.trainer = DistributedTrainer(config)
        
    def benchmark_scalability(self, agent_counts: List[int], steps_per_test: int = 1000) -> Dict:
        """Benchmark performance across different agent counts."""
        results = {}
        
        for num_agents in agent_counts:
            print(f"\nBenchmarking {num_agents} agents...")
            
            # Update config
            self.config.num_agents = num_agents
            
            # Reinitialize trainer
            self.trainer = DistributedTrainer(self.config)
            
            # Benchmark
            start_time = time.time()
            
            for step in range(steps_per_test):
                summary = self.trainer.train_step()
                
            end_time = time.time()
            
            # Calculate metrics
            total_time = end_time - start_time
            fps = steps_per_test / total_time
            agent_throughput = (num_agents * steps_per_test) / total_time
            
            results[num_agents] = {
                'total_time': total_time,
                'fps': fps,
                'agent_throughput': agent_throughput,
                'time_per_step': total_time / steps_per_test,
                'time_per_agent_per_step': total_time / (num_agents * steps_per_test)
            }
            
            print(f"  FPS: {fps:.1f}")
            print(f"  Agent Throughput: {agent_throughput:.0f} agents/sec")
            print(f"  Time per step: {total_time/steps_per_test*1000:.2f}ms")
            
        return results
        
    def save_benchmark_results(self, results: Dict, filename: str = "benchmark_results.json"):
        """Save benchmark results to file."""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Benchmark results saved to {filename}")


if __name__ == "__main__":
    # Example usage
    config = TrainingConfig(
        num_agents=1000,
        max_steps=100000,
        update_frequency=2048,
        log_frequency=10
    )
    
    trainer = DistributedTrainer(config)
    summary = trainer.train(max_updates=50)
    
    print("\nTraining completed!")
    print(f"Final summary: {summary}")
