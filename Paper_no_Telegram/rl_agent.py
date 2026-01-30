"""
RL Agent Module - PPO-based Trading Agent

This module implements the RL agent that learns to optimize trading decisions
on top of the existing strategy. Uses Proximal Policy Optimization (PPO).

The agent outputs:
1. Entry decision (take signal or skip)
2. Position size multiplier
3. Stop loss adjustment factor
4. Target adjustment factor

Author: Algo Trading Bot
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
from typing import Dict, List, Tuple, Optional, Any
import logging
import os

from rl_config import (
    HIDDEN_LAYERS,
    LEARNING_RATE,
    GAMMA,
    GAE_LAMBDA,
    N_EPOCHS,
    CLIP_RANGE,
    ENTROPY_COEF,
    BATCH_SIZE,
    get_state_dimension,
    get_action_dimension,
    HARD_LIMITS,
    MIN_CONFIDENCE_TO_ACT,
    MODEL_SAVE_DIR,
    CONTINUOUS_ACTIONS,
    DISCRETE_ACTIONS
)

logger = logging.getLogger(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    """
    Actor-Critic neural network for PPO.
    
    Actor: Outputs action probabilities/parameters
    Critic: Estimates state value
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_layers: List[int] = None,
        continuous_action_dim: int = 4,
        discrete_action_dim: int = 2
    ):
        super(ActorCritic, self).__init__()
        
        if hidden_layers is None:
            hidden_layers = HIDDEN_LAYERS
        
        self.state_dim = state_dim
        self.continuous_action_dim = continuous_action_dim
        self.discrete_action_dim = discrete_action_dim
        
        # Build shared feature extractor
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        self.shared = nn.Sequential(*layers)
        
        # Actor head for discrete actions (entry/exit decision)
        self.actor_discrete = nn.Sequential(
            nn.Linear(hidden_layers[-1], 64),
            nn.ReLU(),
            nn.Linear(64, discrete_action_dim)
        )
        
        # Actor head for continuous actions (mean)
        self.actor_continuous_mean = nn.Sequential(
            nn.Linear(hidden_layers[-1], 64),
            nn.ReLU(),
            nn.Linear(64, continuous_action_dim),
            nn.Tanh()  # Output in [-1, 1]
        )
        
        # Actor head for continuous actions (log std)
        self.actor_continuous_log_std = nn.Parameter(
            torch.zeros(continuous_action_dim)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_layers[-1], 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through network.
        
        Returns:
            discrete_logits: Logits for discrete actions
            continuous_mean: Mean for continuous actions
            continuous_std: Std for continuous actions
            value: State value estimate
        """
        features = self.shared(state)
        
        # Discrete action logits
        discrete_logits = self.actor_discrete(features)
        
        # Continuous action mean and std
        continuous_mean = self.actor_continuous_mean(features)
        continuous_std = torch.exp(self.actor_continuous_log_std.clamp(-20, 2))
        
        # Value estimate
        value = self.critic(features)
        
        return discrete_logits, continuous_mean, continuous_std, value
    
    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
        """
        Get action from current policy.
        
        Args:
            state: Current state
            deterministic: If True, use mean action (no exploration)
            
        Returns:
            discrete_action: Discrete action (0 or 1)
            continuous_action: Continuous action values
            log_prob: Log probability of actions
        """
        discrete_logits, cont_mean, cont_std, value = self.forward(state)
        
        # Sample discrete action
        discrete_probs = F.softmax(discrete_logits, dim=-1)
        discrete_dist = Categorical(discrete_probs)
        
        if deterministic:
            discrete_action = torch.argmax(discrete_probs, dim=-1)
        else:
            discrete_action = discrete_dist.sample()
        
        # Sample continuous action
        continuous_dist = Normal(cont_mean, cont_std)
        
        if deterministic:
            continuous_action = cont_mean
        else:
            continuous_action = continuous_dist.sample()
        
        # Calculate log probabilities
        discrete_log_prob = discrete_dist.log_prob(discrete_action)
        continuous_log_prob = continuous_dist.log_prob(continuous_action).sum(dim=-1)
        total_log_prob = discrete_log_prob + continuous_log_prob
        
        return (
            discrete_action.cpu().numpy(),
            continuous_action.cpu().numpy(),
            total_log_prob
        )
    
    def evaluate_actions(
        self,
        states: torch.Tensor,
        discrete_actions: torch.Tensor,
        continuous_actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.
        
        Returns:
            log_probs: Log probabilities of actions
            values: State values
            entropy: Policy entropy
        """
        discrete_logits, cont_mean, cont_std, values = self.forward(states)
        
        # Discrete action evaluation
        discrete_probs = F.softmax(discrete_logits, dim=-1)
        discrete_dist = Categorical(discrete_probs)
        discrete_log_probs = discrete_dist.log_prob(discrete_actions.squeeze())
        discrete_entropy = discrete_dist.entropy()
        
        # Continuous action evaluation
        continuous_dist = Normal(cont_mean, cont_std)
        continuous_log_probs = continuous_dist.log_prob(continuous_actions).sum(dim=-1)
        continuous_entropy = continuous_dist.entropy().sum(dim=-1)
        
        # Total log prob and entropy
        log_probs = discrete_log_probs + continuous_log_probs
        entropy = discrete_entropy + continuous_entropy
        
        return log_probs, values.squeeze(), entropy


class PPOAgent:
    """
    PPO Agent for trading optimization.
    
    This agent learns to optimize your existing strategy by deciding:
    1. Whether to take a trade signal
    2. How to size the position
    3. How to adjust risk parameters
    """
    
    def __init__(
        self,
        state_dim: int = None,
        load_model: bool = False,
        model_path: str = None
    ):
        """
        Initialize PPO Agent.
        
        Args:
            state_dim: State dimension (auto-calculated if None)
            load_model: Whether to load pre-trained model
            model_path: Path to saved model
        """
        if state_dim is None:
            state_dim = get_state_dimension()
        
        self.state_dim = state_dim
        
        # Number of continuous actions
        self.continuous_action_dim = len(CONTINUOUS_ACTIONS)
        # Number of discrete actions (entry decision)
        self.discrete_action_dim = 2  # SKIP or TAKE
        
        # Initialize network
        self.network = ActorCritic(
            state_dim=state_dim,
            hidden_layers=HIDDEN_LAYERS,
            continuous_action_dim=self.continuous_action_dim,
            discrete_action_dim=self.discrete_action_dim
        ).to(device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=LEARNING_RATE
        )
        
        # Experience buffer
        self.buffer = ExperienceBuffer()
        
        # Training stats
        self.training_step = 0
        self.episodes_completed = 0
        
        # Load model if requested
        if load_model and model_path:
            self.load(model_path)
        
        logger.info(f"PPO Agent initialized on {device}")
        logger.info(f"State dim: {state_dim}, Continuous actions: {self.continuous_action_dim}")
    
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Dict[str, Any]:
        """
        Select action given current state.
        
        Args:
            state: Current state vector
            deterministic: If True, use deterministic policy
            
        Returns:
            Dict with action details:
                - take_signal: bool
                - position_size_mult: float
                - sl_adjustment: float
                - target_adjustment: float
                - trailing_factor: float
                - confidence: float
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            discrete_action, continuous_action, log_prob = self.network.get_action(
                state_tensor, deterministic
            )
        
        # Decode discrete action
        take_signal = discrete_action[0] == 1
        
        # Decode continuous actions and map to ranges
        action_names = list(CONTINUOUS_ACTIONS.keys())
        action_values = {}
        
        for i, name in enumerate(action_names):
            low, high = CONTINUOUS_ACTIONS[name]
            # Map from [-1, 1] to [low, high]
            raw_value = continuous_action[0][i] if len(continuous_action.shape) > 1 else continuous_action[i]
            mapped_value = low + (raw_value + 1) * (high - low) / 2
            action_values[name] = float(mapped_value)
        
        # Get confidence (probability of taking signal)
        _, _, _, value = self.network.forward(state_tensor)
        discrete_logits, _, _, _ = self.network.forward(state_tensor)
        probs = F.softmax(discrete_logits, dim=-1)
        confidence = float(probs[0, 1].cpu())  # Probability of TAKE_SIGNAL
        
        return {
            "take_signal": take_signal,
            "confidence": confidence,
            "position_size_multiplier": action_values.get("position_size_multiplier", 1.0),
            "sl_adjustment": action_values.get("sl_adjustment", 1.0),
            "target_adjustment": action_values.get("target_adjustment", 1.0),
            "trailing_stop_factor": action_values.get("trailing_stop_factor", 1.0),
            "log_prob": float(log_prob.cpu()),
            "value": float(value.cpu())
        }
    
    def store_experience(
        self,
        state: np.ndarray,
        discrete_action: int,
        continuous_action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        log_prob: float,
        value: float
    ):
        """Store experience in buffer."""
        self.buffer.add(
            state, discrete_action, continuous_action,
            reward, next_state, done, log_prob, value
        )
    
    def update(self) -> Dict[str, float]:
        """
        Perform PPO update.
        
        Returns:
            Dict with training metrics
        """
        if len(self.buffer) < BATCH_SIZE:
            return {}
        
        # Get data from buffer
        states, discrete_actions, continuous_actions, rewards, \
            next_states, dones, old_log_probs, old_values = self.buffer.get_all()
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(device)
        discrete_actions = torch.LongTensor(discrete_actions).to(device)
        continuous_actions = torch.FloatTensor(continuous_actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        dones = torch.FloatTensor(dones).to(device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(device)
        old_values = torch.FloatTensor(old_values).to(device)
        
        # Compute advantages using GAE
        advantages, returns = self._compute_gae(rewards, old_values, dones)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update loop
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for _ in range(N_EPOCHS):
            # Shuffle data
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), BATCH_SIZE):
                end = start + BATCH_SIZE
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_discrete = discrete_actions[batch_indices]
                batch_continuous = continuous_actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # Evaluate current policy
                log_probs, values, entropy = self.network.evaluate_actions(
                    batch_states, batch_discrete, batch_continuous
                )
                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - CLIP_RANGE, 1 + CLIP_RANGE) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, batch_returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + 0.5 * value_loss + ENTROPY_COEF * entropy_loss
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
        
        # Clear buffer
        self.buffer.clear()
        self.training_step += 1
        
        return {
            "policy_loss": total_policy_loss / N_EPOCHS,
            "value_loss": total_value_loss / N_EPOCHS,
            "entropy": total_entropy / N_EPOCHS
        }
    
    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation."""
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + GAMMA * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * last_gae
        
        returns = advantages + values
        
        return advantages, returns
    
    def save(self, path: str = None):
        """Save model to disk."""
        if path is None:
            os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
            path = os.path.join(MODEL_SAVE_DIR, "rl_trading_model.pt")
        
        torch.save({
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_step": self.training_step,
            "episodes_completed": self.episodes_completed
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model from disk."""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=device)
            self.network.load_state_dict(checkpoint["network_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.training_step = checkpoint.get("training_step", 0)
            self.episodes_completed = checkpoint.get("episodes_completed", 0)
            logger.info(f"Model loaded from {path}")
        else:
            logger.warning(f"Model not found at {path}")


class ExperienceBuffer:
    """Experience replay buffer for PPO."""
    
    def __init__(self):
        self.states = []
        self.discrete_actions = []
        self.continuous_actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def add(
        self,
        state, discrete_action, continuous_action,
        reward, next_state, done, log_prob, value
    ):
        self.states.append(state)
        self.discrete_actions.append(discrete_action)
        self.continuous_actions.append(continuous_action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def get_all(self):
        return (
            np.array(self.states),
            np.array(self.discrete_actions),
            np.array(self.continuous_actions),
            np.array(self.rewards),
            np.array(self.next_states),
            np.array(self.dones),
            np.array(self.log_probs),
            np.array(self.values)
        )
    
    def clear(self):
        self.states.clear()
        self.discrete_actions.clear()
        self.continuous_actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
    
    def __len__(self):
        return len(self.states)


class RLDecisionMaker:
    """
    High-level interface for RL-enhanced decision making.
    
    This class integrates the RL agent with your existing strategy,
    providing a simple interface for getting trading decisions.
    """
    
    def __init__(self, load_pretrained: bool = False, model_path: str = None):
        """Initialize RL decision maker."""
        from rl_state_builder import StateBuilder
        
        self.state_builder = StateBuilder()
        self.agent = PPOAgent(
            state_dim=self.state_builder.state_dim,
            load_model=load_pretrained,
            model_path=model_path
        )
        
        # Track last state for learning
        self.last_state = None
        self.last_action = None
        
        logger.info("RL Decision Maker initialized")
    
    def should_take_signal(
        self,
        df,
        signal_type: str,
        position: Optional[Dict] = None,
        session_stats: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Decide whether to take a trading signal.
        
        Args:
            df: DataFrame with indicators
            signal_type: "CE" or "PE"
            position: Current position (if any)
            session_stats: Today's trading stats
            
        Returns:
            Dict with decision and parameters
        """
        # Build state
        state = self.state_builder.build_state(df, position, session_stats)
        
        # Get RL action
        action = self.agent.select_action(state, deterministic=False)
        
        # Store for learning
        self.last_state = state
        self.last_action = action
        
        # Apply confidence threshold
        if action["confidence"] < MIN_CONFIDENCE_TO_ACT:
            action["take_signal"] = False
        
        # Apply hard limits
        action["position_size_multiplier"] = np.clip(
            action["position_size_multiplier"],
            0.5, HARD_LIMITS["max_position_size"] / 2
        )
        action["sl_adjustment"] = np.clip(
            action["sl_adjustment"],
            HARD_LIMITS["min_stop_loss_pct"] / 0.15,
            HARD_LIMITS["max_stop_loss_pct"] / 0.15
        )
        action["target_adjustment"] = np.clip(
            action["target_adjustment"],
            HARD_LIMITS["min_target_pct"] / 0.45,
            HARD_LIMITS["max_target_pct"] / 0.45
        )
        
        return action
    
    def should_exit_position(
        self,
        df,
        position: Dict,
        current_price: float,
        session_stats: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Decide whether to exit current position.
        
        Args:
            df: DataFrame with indicators
            position: Current position details
            current_price: Current market price
            session_stats: Today's trading stats
            
        Returns:
            Dict with exit decision
        """
        # Update position with current price
        position_copy = position.copy()
        position_copy["current_price"] = current_price
        
        # Build state
        state = self.state_builder.build_state(df, position_copy, session_stats)
        
        # Get RL action (reuse entry decision for exit)
        action = self.agent.select_action(state, deterministic=False)
        
        # For exit, we use inverse logic: if agent says "don't take signal" when in position, consider exit
        exit_signal = not action["take_signal"]
        
        return {
            "should_exit": exit_signal and action["confidence"] > MIN_CONFIDENCE_TO_ACT,
            "confidence": action["confidence"],
            "trailing_factor": action["trailing_stop_factor"]
        }
    
    def learn_from_trade(self, reward: float, done: bool = False):
        """
        Learn from completed trade outcome.
        
        Args:
            reward: Trade reward (P&L or risk-adjusted)
            done: Whether episode is done
        """
        if self.last_state is not None and self.last_action is not None:
            # Store experience
            discrete_action = 1 if self.last_action["take_signal"] else 0
            continuous_action = np.array([
                self.last_action["position_size_multiplier"],
                self.last_action["sl_adjustment"],
                self.last_action["target_adjustment"],
                self.last_action["trailing_stop_factor"]
            ])
            
            # Use current state as next_state (simplified)
            next_state = self.last_state
            
            self.agent.store_experience(
                self.last_state,
                discrete_action,
                continuous_action,
                reward,
                next_state,
                done,
                self.last_action["log_prob"],
                self.last_action["value"]
            )
            
            # Update if buffer is full
            if len(self.agent.buffer) >= BATCH_SIZE:
                metrics = self.agent.update()
                if metrics:
                    logger.info(f"RL Update - Policy Loss: {metrics['policy_loss']:.4f}, "
                               f"Value Loss: {metrics['value_loss']:.4f}")
    
    def save_model(self, path: str = None):
        """Save the trained model."""
        self.agent.save(path)
    
    def load_model(self, path: str):
        """Load a trained model."""
        self.agent.load(path)


if __name__ == "__main__":
    print("RL Agent Module")
    print("=" * 50)
    
    # Test agent initialization
    agent = PPOAgent()
    print(f"Agent initialized with state dim: {agent.state_dim}")
    
    # Test action selection with random state
    state = np.random.randn(agent.state_dim)
    action = agent.select_action(state)
    print(f"Sample action: {action}")
