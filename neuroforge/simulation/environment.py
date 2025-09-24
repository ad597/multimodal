"""
Distributed simulation environment for large-scale RL training.
Supports thousands of parallel agents with efficient resource management.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
import multiprocessing as mp
from dataclasses import dataclass
import random


@dataclass
class AgentState:
    """State representation for individual agents."""
    position: np.ndarray
    velocity: np.ndarray
    orientation: float
    energy: float
    observations: Dict[str, Any]
    rewards: List[float]
    done: bool


@dataclass
class SimulationConfig:
    """Configuration for simulation parameters."""
    num_agents: int = 10000
    world_size: Tuple[int, int] = (1000, 1000)
    max_steps: int = 1000
    time_step: float = 0.1
    observation_range: float = 50.0
    max_speed: float = 10.0
    energy_decay: float = 0.01
    collision_penalty: float = -10.0
    goal_reward: float = 100.0
    num_workers: int = mp.cpu_count()


class PhysicsEngine:
    """Efficient physics simulation for multiple agents."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.world_size = config.world_size
        self.time_step = config.time_step
        self.max_speed = config.max_speed
        
    def update_physics(self, states: List[AgentState], actions: torch.Tensor) -> List[AgentState]:
        """Update physics for all agents in batch."""
        batch_size = len(states)
        
        # Extract positions and velocities
        positions = np.array([state.position for state in states])
        velocities = np.array([state.velocity for state in states])
        orientations = np.array([state.orientation for state in states])
        
        # Convert actions to numpy
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        
        # Apply actions (assume actions are [linear_velocity, angular_velocity])
        linear_actions = actions[:, 0]
        angular_actions = actions[:, 1]
        
        # Update orientation
        new_orientations = orientations + angular_actions * self.time_step
        
        # Update velocity based on orientation and linear action
        new_velocities = np.zeros_like(velocities)
        new_velocities[:, 0] = linear_actions * np.cos(new_orientations)
        new_velocities[:, 1] = linear_actions * np.sin(new_orientations)
        
        # Clamp velocity to max speed
        speed = np.linalg.norm(new_velocities, axis=1)
        mask = speed > self.max_speed
        new_velocities[mask] = new_velocities[mask] / speed[mask, None] * self.max_speed
        
        # Update positions
        new_positions = positions + new_velocities * self.time_step
        
        # Handle boundary conditions (wrap around)
        new_positions[:, 0] = np.mod(new_positions[:, 0], self.world_size[0])
        new_positions[:, 1] = np.mod(new_positions[:, 1], self.world_size[1])
        
        # Update states
        updated_states = []
        for i in range(batch_size):
            new_state = AgentState(
                position=new_positions[i],
                velocity=new_velocities[i],
                orientation=new_orientations[i],
                energy=states[i].energy - self.config.energy_decay,
                observations=states[i].observations,
                rewards=states[i].rewards,
                done=states[i].energy <= 0
            )
            updated_states.append(new_state)
            
        return updated_states


class ObservationSystem:
    """Efficient observation generation for multiple agents."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.observation_range = config.observation_range
        
    def generate_observations(self, states: List[AgentState]) -> torch.Tensor:
        """Generate observations for all agents."""
        batch_size = len(states)
        obs_dim = 8  # [x, y, vx, vy, orientation, energy, nearest_agent_dist, nearest_agent_angle]
        observations = torch.zeros(batch_size, obs_dim)
        
        positions = np.array([state.position for state in states])
        velocities = np.array([state.velocity for state in states])
        
        for i in range(batch_size):
            # Basic state information
            observations[i, 0] = states[i].position[0] / self.config.world_size[0]  # Normalized x
            observations[i, 1] = states[i].position[1] / self.config.world_size[1]  # Normalized y
            observations[i, 2] = states[i].velocity[0] / self.config.max_speed  # Normalized vx
            observations[i, 3] = states[i].velocity[1] / self.config.max_speed  # Normalized vy
            observations[i, 4] = states[i].orientation / (2 * np.pi)  # Normalized orientation
            observations[i, 5] = states[i].energy  # Energy level
            
            # Find nearest agent
            distances = np.linalg.norm(positions - positions[i], axis=1)
            distances[i] = np.inf  # Ignore self
            nearest_idx = np.argmin(distances)
            nearest_dist = distances[nearest_idx]
            
            if nearest_dist < self.observation_range:
                observations[i, 6] = nearest_dist / self.observation_range  # Normalized distance
                # Relative angle to nearest agent
                relative_pos = positions[nearest_idx] - positions[i]
                angle = np.arctan2(relative_pos[1], relative_pos[0])
                observations[i, 7] = (angle - states[i].orientation) / (2 * np.pi)
            else:
                observations[i, 6] = 1.0  # No agent in range
                observations[i, 7] = 0.0  # No angle information
                
        return observations


class RewardSystem:
    """Efficient reward computation for multiple agents."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.goals = self._generate_random_goals(config.num_agents)
        
    def _generate_random_goals(self, num_agents: int) -> np.ndarray:
        """Generate random goal positions for all agents."""
        goals = np.random.uniform(0, self.config.world_size[0], (num_agents, 2))
        return goals
        
    def compute_rewards(self, states: List[AgentState], actions: torch.Tensor) -> torch.Tensor:
        """Compute rewards for all agents."""
        batch_size = len(states)
        rewards = torch.zeros(batch_size)
        
        positions = np.array([state.position for state in states])
        
        for i in range(batch_size):
            if states[i].done:
                continue
                
            # Goal-reaching reward
            dist_to_goal = np.linalg.norm(positions[i] - self.goals[i])
            if dist_to_goal < 10.0:  # Goal reached
                rewards[i] += self.config.goal_reward
                # Generate new goal
                self.goals[i] = np.random.uniform(0, self.config.world_size[0], 2)
            else:
                # Distance-based reward (encourage moving toward goal)
                prev_dist = np.linalg.norm(states[i].position - self.goals[i])
                rewards[i] += (prev_dist - dist_to_goal) * 0.1
                
            # Collision penalty (simplified)
            other_distances = np.linalg.norm(positions - positions[i], axis=1)
            other_distances[i] = np.inf
            min_distance = np.min(other_distances)
            
            if min_distance < 5.0:  # Collision threshold
                rewards[i] += self.config.collision_penalty
                
            # Energy penalty
            if states[i].energy < 0.1:
                rewards[i] -= 1.0
                
        return rewards


class DistributedSimulation:
    """Large-scale distributed simulation environment."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.physics_engine = PhysicsEngine(config)
        self.observation_system = ObservationSystem(config)
        self.reward_system = RewardSystem(config)
        
        # Initialize agent states
        self.states = self._initialize_agents()
        
        # Performance tracking
        self.step_count = 0
        self.total_reward = 0.0
        self.start_time = time.time()
        
    def _initialize_agents(self) -> List[AgentState]:
        """Initialize random agent states."""
        states = []
        for i in range(self.config.num_agents):
            state = AgentState(
                position=np.random.uniform(0, self.config.world_size[0], 2),
                velocity=np.random.uniform(-1, 1, 2),
                orientation=np.random.uniform(0, 2 * np.pi),
                energy=1.0,
                observations={},
                rewards=[],
                done=False
            )
            states.append(state)
        return states
        
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Execute one simulation step for all agents."""
        # Update physics
        self.states = self.physics_engine.update_physics(self.states, actions)
        
        # Generate observations
        observations = self.observation_system.generate_observations(self.states)
        
        # Compute rewards
        rewards = self.reward_system.compute_rewards(self.states, actions)
        
        # Update total reward
        self.total_reward += rewards.sum().item()
        
        # Check if simulation is done
        done_flags = torch.tensor([state.done for state in self.states])
        
        self.step_count += 1
        
        # Performance info
        info = {
            'step': self.step_count,
            'total_reward': self.total_reward,
            'active_agents': (~done_flags).sum().item(),
            'avg_energy': np.mean([state.energy for state in self.states]),
            'fps': self.step_count / (time.time() - self.start_time) if time.time() - self.start_time > 0 else 0
        }
        
        return observations, rewards, done_flags, info
        
    def reset(self) -> torch.Tensor:
        """Reset simulation to initial state."""
        self.states = self._initialize_agents()
        self.step_count = 0
        self.total_reward = 0.0
        self.start_time = time.time()
        return self.observation_system.generate_observations(self.states)
        
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        elapsed_time = time.time() - self.start_time
        return {
            'total_steps': self.step_count,
            'total_reward': self.total_reward,
            'steps_per_second': self.step_count / elapsed_time if elapsed_time > 0 else 0,
            'agents_per_second': (self.config.num_agents * self.step_count) / elapsed_time if elapsed_time > 0 else 0,
            'active_agents': sum(1 for state in self.states if not state.done)
        }


class MultiProcessSimulation:
    """Multi-process simulation for even larger scale."""
    
    def __init__(self, config: SimulationConfig, num_processes: int = None):
        self.config = config
        self.num_processes = num_processes or mp.cpu_count()
        
        # Split agents across processes
        agents_per_process = config.num_agents // self.num_processes
        self.process_configs = []
        
        for i in range(self.num_processes):
            process_config = SimulationConfig(
                num_agents=agents_per_process,
                world_size=config.world_size,
                max_steps=config.max_steps,
                time_step=config.time_step,
                observation_range=config.observation_range,
                max_speed=config.max_speed,
                energy_decay=config.energy_decay,
                collision_penalty=config.collision_penalty,
                goal_reward=config.goal_reward
            )
            self.process_configs.append(process_config)
            
        # Initialize processes
        self.processes = []
        self.queues = []
        for i in range(self.num_processes):
            q = mp.Queue()
            self.queues.append(q)
            
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Execute step across all processes."""
        # Split actions across processes
        actions_per_process = actions.size(0) // self.num_processes
        process_actions = []
        
        for i in range(self.num_processes):
            start_idx = i * actions_per_process
            end_idx = start_idx + actions_per_process if i < self.num_processes - 1 else actions.size(0)
            process_actions.append(actions[start_idx:end_idx])
            
        # Execute on each process (simplified - would use actual multiprocessing)
        all_observations = []
        all_rewards = []
        all_dones = []
        all_infos = []
        
        for i, process_actions_i in enumerate(process_actions):
            # For demo purposes, run sequentially
            # In production, this would be parallelized
            sim = DistributedSimulation(self.process_configs[i])
            obs, rew, done, info = sim.step(process_actions_i)
            all_observations.append(obs)
            all_rewards.append(rew)
            all_dones.append(done)
            all_infos.append(info)
            
        # Concatenate results
        observations = torch.cat(all_observations, dim=0)
        rewards = torch.cat(all_rewards, dim=0)
        done_flags = torch.cat(all_dones, dim=0)
        
        # Combine info
        combined_info = {
            'total_steps': sum(info['step'] for info in all_infos),
            'total_reward': sum(info['total_reward'] for info in all_infos),
            'active_agents': sum(info['active_agents'] for info in all_infos),
            'avg_fps': np.mean([info['fps'] for info in all_infos])
        }
        
        return observations, rewards, done_flags, combined_info
