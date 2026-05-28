"""
Reinforcement Learning Policy
A Deep Reinforcement Learning agent (SAC / PPO compatible) that learns
to race through gates.

Architecture
------------
    Observation  →  FeatureExtractor  →  PolicyNetwork  →  Action
    (state + gates + obstacles)           (Actor-Critic)    (velocity target)

The RL policy can run in two modes:
  - 'training'  : outputs actions + log-probs for training
  - 'inference' : outputs the best action only

This module is framework-agnostic — it defines the observation/action
spaces and a base class. Implement RLPolicyPPO or RLPolicySAC subclass
using PyTorch or similar.
"""

import math
import logging
import numpy as np
import random
import torch
from collections import deque
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, List

from core_gp.core.state_estimator import VehicleState
from core_gp.perception.perception import GateObservation, Obstacle
from core_gp.guidance.guidance import GuidanceOutput

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Observation Builder
# ---------------------------------------------------------------------------

@dataclass
class RLObservation:
    """
    Flat observation vector fed to the policy network.
    Designed to be informative yet compact.

    Dim breakdown:
      [0:3]   velocity NED (normalised)              3
      [3:6]   position error to next gate (NED)      3
      [6]     range to next gate                     1
      [7:9]   gate bearing (az, el) in body frame    2
      [9]     own speed normalised                   1
      [10:13] attitude (roll, pitch, yaw)            3
      [13:16] angular rates (p, q, r)                3
      [16:19] obstacle repulsion vector              3
      total                                         19
    """
    vector: np.ndarray   # shape (19,)
    dim: int = 19


class ObservationBuilder:
    """Converts VehicleState + perception outputs → flat RLObservation."""

    MAX_SPEED = 20.0       # for normalisation
    MAX_DIST  = 50.0

    def build(self, state: VehicleState,
              next_gate_pos: Optional[np.ndarray],
              gates: List[GateObservation],
              obstacles: List[Obstacle]) -> RLObservation:
        obs = np.zeros(19, dtype=np.float32)

        # Velocity (normalised)
        obs[0:3] = state.velocity_ned() / self.MAX_SPEED

        # Position error & range to next gate
        if next_gate_pos is not None:
            err = next_gate_pos - state.position_ned()
            dist = float(np.linalg.norm(err))
            obs[3:6] = np.clip(err / self.MAX_DIST, -1, 1)
            obs[6]   = min(dist / self.MAX_DIST, 1.0)

            # Gate bearing in body frame (azimuth, elevation)
            R_body_ned = state.body_to_ned_R()
            err_body = R_body_ned.T @ err
            az = math.atan2(err_body[1], err_body[0]) / math.pi
            el = math.atan2(-err_body[2], math.sqrt(err_body[0]**2 + err_body[1]**2)) / math.pi
            obs[7] = az
            obs[8] = el

        # Speed
        obs[9] = min(state.speed / self.MAX_SPEED, 1.0)

        # Attitude
        obs[10] = state.roll  / math.pi
        obs[11] = state.pitch / math.pi
        obs[12] = state.yaw   / math.pi

        # Angular rates
        obs[13] = np.clip(state.p / 5.0, -1, 1)
        obs[14] = np.clip(state.q / 5.0, -1, 1)
        obs[15] = np.clip(state.r / 5.0, -1, 1)

        # Aggregate obstacle repulsion
        if obstacles:
            repulsion = np.zeros(3)
            for o in obstacles:
                d = np.linalg.norm(o.position_body)
                if d > 0:
                    repulsion -= (o.position_body / d) / max(d, 0.5)
            repulsion = np.clip(repulsion / (len(obstacles) * 2), -1, 1)
            obs[16:19] = repulsion

        return RLObservation(vector=obs)


# ---------------------------------------------------------------------------
# Action Space
# ---------------------------------------------------------------------------

@dataclass
class RLAction:
    """
    Continuous action: desired velocity in body-NED frame (3 axes).
    Tanh-squashed to [-1, 1], then scaled.
    """
    raw: np.ndarray          # shape (3,), in [-1, 1]
    max_horiz_vel: float = 12.0
    max_vert_vel:  float = 4.0

    def to_velocity_ned(self, state: VehicleState) -> np.ndarray:
        """Convert action to NED velocity command."""
        R = state.body_to_ned_R()
        body_vel = np.array([
            self.raw[0] * self.max_horiz_vel,
            self.raw[1] * self.max_horiz_vel,
            self.raw[2] * self.max_vert_vel,
        ])
        return R @ body_vel


# ---------------------------------------------------------------------------
# Reward Function
# ---------------------------------------------------------------------------

class RewardFunction:
    """
    Shaped reward for gate racing.

    Positive:
      + Gate passage bonus
      + Progress toward next gate (potential-based)
      + Speed bonus (encourages fast flight)

    Negative:
      − Collision / obstacle proximity penalty
      − Time penalty (encourages completion)
      − Large attitude excursions
    """

    GATE_BONUS          = 100.0
    PROGRESS_SCALE      = 2.0
    SPEED_BONUS_SCALE   = 0.1
    OBSTACLE_PENALTY    = -20.0
    TIME_PENALTY        = -0.01
    ATTITUDE_PENALTY    = -0.5

    def __init__(self):
        self._prev_dist_to_gate: Optional[float] = None
        

    def compute(self, state: VehicleState,
                next_gate_pos: Optional[np.ndarray],
                gate_passed: bool,
                obstacle_hit: bool) -> float:
        reward = self.TIME_PENALTY

        # Gate passage
        if gate_passed:
            reward += self.GATE_BONUS

        # Progress reward (potential-based shaping)
        if next_gate_pos is not None:
            dist = float(np.linalg.norm(next_gate_pos - state.position_ned()))
            if self._prev_dist_to_gate is not None:
                progress = self._prev_dist_to_gate - dist
                reward += progress * self.PROGRESS_SCALE
            self._prev_dist_to_gate = dist

        # Speed bonus
        if next_gate_pos is not None:
            gate_vec = next_gate_pos - state.position_ned()
            norm = np.linalg.norm(gate_vec)
            
            if norm > 1e-6:
                gate_dir = gate_vec / norm
                forward_progress = np.dot(state.velocity_ned(), gate_dir)
                reward += forward_progress * self.SPEED_BONUS_SCALE

        # Obstacle
        if obstacle_hit:
            reward += self.OBSTACLE_PENALTY

        # Attitude penalty (excessive roll/pitch)
        att_penalty = (abs(state.roll) + abs(state.pitch)) / math.pi
        reward += att_penalty * self.ATTITUDE_PENALTY

        return reward

    def reset(self):
        self._prev_dist_to_gate = None


# ---------------------------------------------------------------------------
# Base Policy
# ---------------------------------------------------------------------------

class RLPolicyBase(ABC):
    """
    Abstract base for RL policies.
    Subclass with PyTorch (PPO/SAC) or any other framework.
    """

    def __init__(self, obs_dim: int = 19, action_dim: int = 3):
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.obs_builder = ObservationBuilder()
        self.reward_fn   = RewardFunction()
        self._step_count = 0

    @abstractmethod
    def get_action(self, obs: RLObservation) -> RLAction:
        """Return action given observation (inference)."""

    @abstractmethod
    def store_transition(self, obs: RLObservation, action: RLAction,
                         reward: float, next_obs: RLObservation, done: bool):
        """Store (s, a, r, s', done) for replay/rollout."""

    @abstractmethod
    def update(self) -> dict:
        """Run one gradient update step. Returns loss metrics."""

    @abstractmethod
    def save(self, path: str):
        """Persist weights."""

    @abstractmethod
    def load(self, path: str):
        """Load weights."""

    # ------------------------------------------------------------------
    # High-level step — called by the main loop
    # ------------------------------------------------------------------

    def step(self, state: VehicleState,
             next_gate_pos: Optional[np.ndarray],
             gates: List[GateObservation],
             obstacles: List[Obstacle],
             gate_passed: bool = False,
             obstacle_hit: bool = False,
             done: bool = False,
             training: bool = False) -> GuidanceOutput:
        """
        Full step: build observation → get action → compute reward → store.
        Returns a GuidanceOutput that the autopilot can consume directly.
        """
        obs = self.obs_builder.build(state, next_gate_pos, gates, obstacles)
        action = self.get_action(obs)
        vel_ned = action.to_velocity_ned(state)

        if training and hasattr(self, "_pending"):
            prev_obs, prev_actions, pre_reward, prev_done = self._pending
            self.store_transition(prev_obs, prev_actions, pre_reward, obs, prev_done)
            
        if training:
            reward = self.reward_fn.compute(state, next_gate_pos, gate_passed, obstacle_hit)
            self._pending = (obs, action, reward, done)
        if done: 
            self.reward_fn.reset()
            if hasattr(self, "_pending"):
                del self._pending
            


        target_pos = state.position_ned() + vel_ned * 0.5   # 0.5 s lookahead
        target_yaw = math.atan2(vel_ned[1],vel_ned[0]) if np.linalg.norm(vel_ned[:2]) > 0.1 else state.yaw

        self._step_count += 1
        return GuidanceOutput(
            target_position=target_pos,
            target_velocity=vel_ned,
            target_yaw=target_yaw,
            target_speed=float(np.linalg.norm(vel_ned)),
        )


# ---------------------------------------------------------------------------
# Placeholder / Random Policy (for testing without training)
# ---------------------------------------------------------------------------

class RandomPolicy(RLPolicyBase):
    """
    Random policy — useful to verify the pipeline runs before training.
    Replace with a trained PPO/SAC policy.
    """

    def get_action(self, obs: RLObservation) -> RLAction:
        raw = np.random.uniform(-0.3, 0.3, size=self.action_dim).astype(np.float32)
        # Bias forward
        raw[0] = abs(raw[0]) + 0.3
        return RLAction(raw=np.clip(raw, -1, 1))

    def store_transition(self, obs, action, reward, next_obs, done):
        pass  # No-op

    def update(self) -> dict:
        return {}

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass


# ---------------------------------------------------------------------------
# Stub: PyTorch SAC (implement with actual network)
# ---------------------------------------------------------------------------

class SACPolicy(RLPolicyBase):
    """
    Soft Actor-Critic stub. Requires PyTorch.
    Fill in Actor/Critic networks; the interface is already wired.
    """
    def _sample_batch(self):
        batch = random.sample(self._replay_buffer, self._batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return {
            'obs' : torch.FloatTensor(np.array(obs)),
            'actions': torch.FloatTensor(np.array(actions)),
            'rewards': torch.FloatTensor(rewards).unsqueeze(1),
            'next_obs': torch.FloatTensor(np.array(next_obs)),
            'dones': torch.FloatTensor(dones).unsqueeze(1),
        }

    def _sample_action(self, obs):
        mean = self.actor_mean(obs)
        log_std = torch.clamp(self.actor_log_std(obs), -5, 2)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob

    def __init__(self, obs_dim: int = 19, action_dim: int = 3,
                 hidden_dim: int = 256, lr: float = 3e-4):
        super().__init__(obs_dim, action_dim)
        self._built = False
        self._hidden_dim = hidden_dim
        self._lr = lr
        self._replay_buffer = deque(maxlen=100_000)
        self._batch_size = 256
        self._gamma = 0.99
        self._tau = 0.005
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            self._build_networks()
        except ImportError:
            logger.warning("PyTorch not available — SACPolicy will not train.")

    def _build_networks(self):
        import torch
        import torch.nn as nn

        class MLP(nn.Module):
            def __init__(self, in_d, out_d, hidden):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_d, hidden), nn.ReLU(),
                    nn.Linear(hidden, hidden), nn.ReLU(),
                    nn.Linear(hidden, out_d),
                )
            def forward(self, x):
                return self.net(x)

        H = self._hidden_dim
        # Actor: outputs mean + log_std of Gaussian policy
        self.actor_mean    = MLP(self.obs_dim, self.action_dim, H)
        self.actor_log_std = MLP(self.obs_dim, self.action_dim, H)
        # Twin Q-networks
        self.q1 = MLP(self.obs_dim + self.action_dim, 1, H)
        self.q2 = MLP(self.obs_dim + self.action_dim, 1, H)
        self.q1_target = MLP(self.obs_dim + self.action_dim, 1, H)
        self.q2_target = MLP(self.obs_dim + self.action_dim, 1, H)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        # ---------------------------------------------------
        # MOVE NETWORKS TO DEVICE
        # ---------------------------------------------------

        self.actor_mean.to(self.device)
        self.actor_log_std.to(self.device)

        self.q1.to(self.device)
        self.q2.to(self.device)

        self.q1_target.to(self.device)
        self.q2_target.to(self.device)

        # ---------------------------------------------------
        # COPY INITIAL TARGET WEIGHTS
        # ---------------------------------------------------

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self._built = True

        params = (list(self.actor_mean.parameters()) +
                  list(self.actor_log_std.parameters()))
        self.actor_opt = torch.optim.Adam(params, lr=self._lr)
        self.q_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=self._lr
        )
        logger.info("SAC networks built.")

    def get_action(self, obs: RLObservation) -> RLAction:
        if not self._built:
            return RLAction(raw=np.zeros(self.action_dim, dtype=np.float32))
        self.actor_mean.eval()
        self.actor_log_std.eval()
        
        with torch.no_grad():
            x = torch.FloatTensor(obs.vector).unsqueeze(0).to(self.device)
            mean = self.actor_mean(x)
            log_std = torch.clamp(self.actor_log_std(x), -5, 2)
            std = log_std.exp()
            action = torch.tanh(mean + std * torch.randn_like(std))
        return RLAction(raw=action.squeeze().cpu().numpy())

    def store_transition(self, obs, action, reward, next_obs, done):
        self._replay_buffer.append((obs.vector, action.raw, reward, next_obs.vector, done))


    def update(self) -> dict:
        if len(self._replay_buffer) < self._batch_size:
            return {}
        self.actor_mean.train()
        self.actor_log_std.train()
        self.q1.train()
        self.q2.train()
    
        import torch.nn.functional as F

        batch = self._sample_batch()

        obs = batch['obs'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_obs = batch['next_obs'].to(self.device)
        dones = batch['dones'].to(self.device)

        alpha = 0.2

        # --------------------------------------------------
        # TARGET Q
        # --------------------------------------------------

        with torch.no_grad():

            next_actions, next_log_probs = self._sample_action(next_obs)

            next_input = torch.cat(
                [next_obs, next_actions],
                dim=1
            )

            q1_next = self.q1_target(next_input)
            q2_next = self.q2_target(next_input)

            q_next = torch.min(q1_next, q2_next)

            target_q = rewards + (
                1 - dones
            ) * self._gamma * (
                q_next - alpha * next_log_probs
            )

        # --------------------------------------------------
        # Q LOSS
        # --------------------------------------------------

        q_input = torch.cat([obs, actions], dim=1)

        q1 = self.q1(q_input)
        q2 = self.q2(q_input)

        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)

        q_loss = q1_loss + q2_loss

        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        # --------------------------------------------------
        # ACTOR LOSS
        # --------------------------------------------------

        new_actions, log_probs = self._sample_action(obs)

        actor_input = torch.cat(
            [obs, new_actions],
            dim=1
        )

        q1_actor = self.q1(actor_input)
        q2_actor = self.q2(actor_input)

        q_actor = torch.min(q1_actor, q2_actor)

        actor_loss = (
            alpha * log_probs - q_actor
        ).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # --------------------------------------------------
        # SOFT TARGET UPDATE
        # --------------------------------------------------
        for p in self.q1_target.parameters():
            p.requires_grad_(False)
        
        for target_param, param in zip(
            self.q1_target.parameters(),
            self.q1.parameters()
        ):
            target_param.data.copy_(
                self._tau * param.data +
                (1 - self._tau) * target_param.data
            )
        for p in self.q2_target.parameters():
            p.requires_grad_(False)
        
        for target_param, param in zip(
            self.q2_target.parameters(),
            self.q2.parameters()
        ):
            target_param.data.copy_(
                self._tau * param.data +
                (1 - self._tau) * target_param.data
            )

        return {
            'q_loss': float(q_loss.item()),
            'actor_loss': float(actor_loss.item()),
            'buffer_size': len(self._replay_buffer),
        }

    def save(self, path: str):
        if not self._built:
            return
        import torch
        torch.save({
            'actor_mean': self.actor_mean.state_dict(),
            'actor_log_std': self.actor_log_std.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
        }, path)

    def load(self, path: str):
        if not self._built:
            return
        import torch
        ckpt = torch.load(path)
        self.actor_mean.load_state_dict(ckpt['actor_mean'])
        self.actor_log_std.load_state_dict(ckpt['actor_log_std'])
        self.q1.load_state_dict(ckpt['q1'])
        self.q2.load_state_dict(ckpt['q2'])
        logger.info(f"SAC weights loaded from {path}")
