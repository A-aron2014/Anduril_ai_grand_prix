# Fully Commented Autonomous Drone SAC Reinforcement Learning Policy


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

# ============================================================================
# IMPORTS
# ============================================================================

import math
import logging
import numpy as np
import random
import torch

# deque is used as an efficient replay buffer.
# Old experiences automatically fall off once maxlen is reached.
from collections import deque

# ABC + abstractmethod allow us to define a required interface
# that all RL policies must implement.
from abc import ABC, abstractmethod

# dataclass simplifies data container classes.
from dataclasses import dataclass

# Typing helpers improve readability and IDE support.
from typing import Optional, Tuple, List

# Internal project modules.
from state_estimator import VehicleState
from perception.perception import GateObservation, Obstacle
from guidance.guidance import GuidanceOutput

# Logger used for debugging and training information.
logger = logging.getLogger(__name__)


# ============================================================================
# Observation Builder
# ============================================================================

@dataclass
class RLObservation:
    """
    Flat observation vector fed directly into the neural network.

    The observation is intentionally compact while still containing
    enough information for the policy to learn racing behavior.

    Observation Layout
    ------------------
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

    # Final flattened observation vector.
    vector: np.ndarray

    # Total observation dimension.
    dim: int = 19


class ObservationBuilder:
    """
    Converts raw drone state + perception information into
    a compact neural-network-friendly observation vector.

    This class acts as the feature engineering stage of the RL pipeline.
    """

    # Expected maximum speed used for normalization.
    MAX_SPEED = 20.0

    # Expected maximum gate distance used for normalization.
    MAX_DIST = 50.0

    def build(
        self,
        state: VehicleState,
        next_gate_pos: Optional[np.ndarray],
        gates: List[GateObservation],
        obstacles: List[Obstacle]
    ) -> RLObservation:
        """
        Build the final 19-dimensional observation vector.

        Parameters
        ----------
        state:
            Current estimated drone state.

        next_gate_pos:
            NED position of the next race gate.

        gates:
            Current gate detections.

        obstacles:
            Current obstacle detections.

        Returns
        -------
        RLObservation
            Flattened observation vector for the policy network.
        """

        # Initialize empty observation vector.
        obs = np.zeros(19, dtype=np.float32)

        # ------------------------------------------------------------------
        # Velocity (normalized)
        # ------------------------------------------------------------------
        # Convert current drone velocity into normalized values.
        obs[0:3] = state.velocity_ned() / self.MAX_SPEED

        # ------------------------------------------------------------------
        # Relative Gate Position
        # ------------------------------------------------------------------
        if next_gate_pos is not None:

            # Vector from drone to next gate.
            err = next_gate_pos - state.position_ned()

            # Euclidean distance to gate.
            dist = float(np.linalg.norm(err))

            # Normalized relative gate position.
            obs[3:6] = np.clip(err / self.MAX_DIST, -1, 1)

            # Normalized scalar gate range.
            obs[6] = min(dist / self.MAX_DIST, 1.0)

            # --------------------------------------------------------------
            # Gate Bearing in Body Frame
            # --------------------------------------------------------------
            # Convert world-frame gate vector into body-frame coordinates.
            # This lets the network learn directional steering behavior.

            R_body_ned = state.body_to_ned_R()

            # Rotate NED vector into body frame.
            err_body = R_body_ned.T @ err

            # Left/right angle to gate.
            az = math.atan2(
                err_body[1],
                err_body[0]
            ) / math.pi

            # Up/down angle to gate.
            el = math.atan2(
                -err_body[2],
                math.sqrt(err_body[0]**2 + err_body[1]**2)
            ) / math.pi

            obs[7] = az
            obs[8] = el

        # ------------------------------------------------------------------
        # Speed
        # ------------------------------------------------------------------
        obs[9] = min(state.speed / self.MAX_SPEED, 1.0)

        # ------------------------------------------------------------------
        # Attitude
        # ------------------------------------------------------------------
        # Normalize Euler angles.

        obs[10] = state.roll / math.pi
        obs[11] = state.pitch / math.pi
        obs[12] = state.yaw / math.pi

        # ------------------------------------------------------------------
        # Angular Rates
        # ------------------------------------------------------------------
        # Normalize body angular rates.

        obs[13] = np.clip(state.p / 5.0, -1, 1)
        obs[14] = np.clip(state.q / 5.0, -1, 1)
        obs[15] = np.clip(state.r / 5.0, -1, 1)

        # ------------------------------------------------------------------
        # Obstacle Repulsion Vector
        # ------------------------------------------------------------------
        # Creates a simple obstacle avoidance feature.
        # Nearby obstacles contribute stronger repulsion.

        if obstacles:

            repulsion = np.zeros(3)

            for o in obstacles:

                # Distance to obstacle.
                d = np.linalg.norm(o.position_body)

                if d > 0:

                    # Push away from nearby obstacles.
                    repulsion -= (
                        o.position_body / d
                    ) / max(d, 0.5)

            # Average and clip repulsion signal.
            repulsion = np.clip(
                repulsion / (len(obstacles) * 2),
                -1,
                1
            )

            obs[16:19] = repulsion

        return RLObservation(vector=obs)


# ============================================================================
# Action Space
# ============================================================================

@dataclass
class RLAction:
    """
    Represents the continuous action output by the policy.

    The policy outputs values in the range [-1, 1].

    These values are later scaled into real drone velocity commands.
    """

    # Raw network action output.
    raw: np.ndarray

    # Maximum horizontal velocity command.
    max_horiz_vel: float = 12.0

    # Maximum vertical velocity command.
    max_vert_vel: float = 4.0

    def to_velocity_ned(self, state: VehicleState) -> np.ndarray:
        """
        Convert normalized action into a real NED velocity command.

        Steps
        -----
        1. Scale normalized action values.
        2. Convert body-frame velocity into world-frame NED velocity.
        """

        # Rotation matrix body -> NED.
        R = state.body_to_ned_R()

        # Scale normalized outputs into physical velocities.
        body_vel = np.array([
            self.raw[0] * self.max_horiz_vel,
            self.raw[1] * self.max_horiz_vel,
            self.raw[2] * self.max_vert_vel,
        ])

        # Rotate into world frame.
        return R @ body_vel


# ============================================================================
# Reward Function
# ============================================================================

class RewardFunction:
    """
    Computes the reinforcement learning reward signal.

    The reward is shaped to accelerate learning.

    Positive Rewards
    ----------------
    + Passing gates
    + Moving toward the next gate
    + High forward speed

    Negative Rewards
    ----------------
    - Crashing into obstacles
    - Taking too long
    - Excessive roll/pitch instability
    """

    # Large reward for passing a gate.
    GATE_BONUS = 100.0

    # Reward scaling for progress toward gate.
    PROGRESS_SCALE = 2.0

    # Reward for moving quickly toward target.
    SPEED_BONUS_SCALE = 0.1

    # Collision penalty.
    OBSTACLE_PENALTY = -20.0

    # Small per-step penalty encouraging fast completion.
    TIME_PENALTY = -0.01

    # Penalty for unstable attitudes.
    ATTITUDE_PENALTY = -0.5

    def __init__(self):
        """
        Track previous gate distance for progress reward calculation.
        """

        self._prev_dist_to_gate: Optional[float] = None

    def compute(
        self,
        state: VehicleState,
        next_gate_pos: Optional[np.ndarray],
        gate_passed: bool,
        obstacle_hit: bool
    ) -> float:
        """
        Compute reward for the current timestep.

        Returns
        -------
        float
            Final scalar reward.
        """

        # Base time penalty.
        reward = self.TIME_PENALTY

        # ------------------------------------------------------------------
        # Gate Reward
        # ------------------------------------------------------------------
        if gate_passed:
            reward += self.GATE_BONUS

        # ------------------------------------------------------------------
        # Progress Reward
        # ------------------------------------------------------------------
        # Encourages movement toward next gate.

        if next_gate_pos is not None:

            dist = float(
                np.linalg.norm(
                    next_gate_pos - state.position_ned()
                )
            )

            if self._prev_dist_to_gate is not None:

                progress = self._prev_dist_to_gate - dist

                reward += progress * self.PROGRESS_SCALE

            self._prev_dist_to_gate = dist

        # ------------------------------------------------------------------
        # Speed Bonus
        # ------------------------------------------------------------------
        # Encourages fast movement aligned toward gate direction.

        if next_gate_pos is not None:

            gate_vec = next_gate_pos - state.position_ned()

            norm = np.linalg.norm(gate_vec)

            if norm > 1e-6:

                gate_dir = gate_vec / norm

                forward_progress = np.dot(
                    state.velocity_ned(),
                    gate_dir
                )

                reward += (
                    forward_progress *
                    self.SPEED_BONUS_SCALE
                )

        # ------------------------------------------------------------------
        # Collision Penalty
        # ------------------------------------------------------------------
        if obstacle_hit:
            reward += self.OBSTACLE_PENALTY

        # ------------------------------------------------------------------
        # Attitude Penalty
        # ------------------------------------------------------------------
        # Penalizes aggressive unstable attitudes.

        att_penalty = (
            abs(state.roll) +
            abs(state.pitch)
        ) / math.pi

        reward += (
            att_penalty *
            self.ATTITUDE_PENALTY
        )

        return reward

    def reset(self):
        """
        Reset reward history at episode termination.
        """

        self._prev_dist_to_gate = None


# ============================================================================
# Base Policy
# ============================================================================

class RLPolicyBase(ABC):
    """
    Abstract base class for all reinforcement learning policies.

    Any derived policy (PPO, SAC, etc.) must implement:
        - action selection
        - transition storage
        - training update
        - save/load methods
    """

    def __init__(self, obs_dim: int = 19, action_dim: int = 3):

        # Observation dimension.
        self.obs_dim = obs_dim

        # Action dimension.
        self.action_dim = action_dim

        # Converts raw state into RL observations.
        self.obs_builder = ObservationBuilder()

        # Reward function object.
        self.reward_fn = RewardFunction()

        # Internal step counter.
        self._step_count = 0

    @abstractmethod
    def get_action(self, obs: RLObservation) -> RLAction:
        """
        Compute action from observation.
        Used during inference and training.
        """

    @abstractmethod
    def store_transition(
        self,
        obs: RLObservation,
        action: RLAction,
        reward: float,
        next_obs: RLObservation,
        done: bool
    ):
        """
        Store transition:
            (state, action, reward, next_state, done)

        Used for replay buffers or rollout storage.
        """

    @abstractmethod
    def update(self) -> dict:
        """
        Perform one policy update step.

        Returns
        -------
        dict
            Training metrics.
        """

    @abstractmethod
    def save(self, path: str):
        """
        Save model weights.
        """

    @abstractmethod
    def load(self, path: str):
        """
        Load model weights.
        """

    # ----------------------------------------------------------------------
    # Main RL Step Function
    # ----------------------------------------------------------------------

    def step(
        self,
        state: VehicleState,
        next_gate_pos: Optional[np.ndarray],
        gates: List[GateObservation],
        obstacles: List[Obstacle],
        gate_passed: bool = False,
        obstacle_hit: bool = False,
        done: bool = False,
        training: bool = False
    ) -> GuidanceOutput:
        """
        Full RL pipeline step.

        Flow
        ----
        1. Build observation.
        2. Query policy action.
        3. Convert action into velocity command.
        4. Compute reward.
        5. Store replay transition.
        6. Return autopilot guidance output.
        """

        # Build observation vector.
        obs = self.obs_builder.build(
            state,
            next_gate_pos,
            gates,
            obstacles
        )

        # Query policy.
        action = self.get_action(obs)

        # Convert policy output into NED velocity command.
        vel_ned = action.to_velocity_ned(state)

        # ------------------------------------------------------------------
        # Replay Transition Storage
        # ------------------------------------------------------------------
        # SAC stores transitions in the form:
        #   (s_t, a_t, r_t, s_t+1)
        #
        # Because next_obs is not available until the next timestep,
        # the previous step is temporarily cached.

        if training and hasattr(self, "_pending"):

            prev_obs, prev_actions, pre_reward, prev_done = self._pending

            self.store_transition(
                prev_obs,
                prev_actions,
                pre_reward,
                obs,
                prev_done
            )

        # ------------------------------------------------------------------
        # Reward Computation
        # ------------------------------------------------------------------

        if training:

            reward = self.reward_fn.compute(
                state,
                next_gate_pos,
                gate_passed,
                obstacle_hit
            )

            # Cache current transition until next observation exists.
            self._pending = (obs, action, reward, done)

        # ------------------------------------------------------------------
        # Episode Cleanup
        # ------------------------------------------------------------------

        if done:

            # Reset reward state.
            self.reward_fn.reset()

            # Clear pending transition.
            if hasattr(self, "_pending"):
                del self._pending

        # ------------------------------------------------------------------
        # Guidance Target Generation
        # ------------------------------------------------------------------
        # Convert velocity command into a short-horizon target position.
        # This smooths flight behavior.

        target_pos = state.position_ned() + vel_ned * 0.5

        # Face direction of motion if moving fast enough.
        target_yaw = (
            math.atan2(vel_ned[1], vel_ned[0])
            if np.linalg.norm(vel_ned[:2]) > 0.1
            else state.yaw
        )

        self._step_count += 1

        return GuidanceOutput(
            target_position=target_pos,
            target_velocity=vel_ned,
            target_yaw=target_yaw,
            target_speed=float(np.linalg.norm(vel_ned)),
        )


# ============================================================================
# Random Policy
# ============================================================================

class RandomPolicy(RLPolicyBase):
    """
    Simple random-action policy.

    Useful for:
        - Pipeline testing
        - Simulator validation
        - Debugging

    Not intended for real racing performance.
    """

    def get_action(self, obs: RLObservation) -> RLAction:
        """
        Generate random action.
        """

        # Generate random control inputs.
        raw = np.random.uniform(
            -0.3,
            0.3,
            size=self.action_dim
        ).astype(np.float32)

        # Add slight forward bias.
        raw[0] = abs(raw[0]) + 0.3

        return RLAction(raw=np.clip(raw, -1, 1))

    def store_transition(self, obs, action, reward, next_obs, done):
        """
        No replay storage needed.
        """

        pass

    def update(self) -> dict:
        """
        Random policy does not train.
        """

        return {}

    def save(self, path: str):
        """
        No weights to save.
        """

        pass

    def load(self, path: str):
        """
        No weights to load.
        """

        pass


# ============================================================================
# Soft Actor-Critic Policy
# ============================================================================

class SACPolicy(RLPolicyBase):
    """
    Soft Actor-Critic implementation.

    SAC is an off-policy actor-critic algorithm that combines:
        - entropy maximization
        - twin Q-networks
        - stochastic policy learning

    Benefits
    --------
    - Stable continuous control
    - Excellent sample efficiency
    - Strong exploration behavior
    - Well suited for drone racing
    """

    def _sample_batch(self):
        """
        Sample a minibatch from replay buffer.

        Returns tensors ready for neural network training.
        """

        batch = random.sample(
            self._replay_buffer,
            self._batch_size
        )

        obs, actions, rewards, next_obs, dones = zip(*batch)

        return {
            'obs': torch.FloatTensor(np.array(obs)),
            'actions': torch.FloatTensor(np.array(actions)),
            'rewards': torch.FloatTensor(rewards).unsqueeze(1),
            'next_obs': torch.FloatTensor(np.array(next_obs)),
            'dones': torch.FloatTensor(dones).unsqueeze(1),
        }

    def _sample_action(self, obs):
        """
        Sample stochastic action from policy distribution.

        SAC uses a Gaussian policy followed by tanh squashing.

        Returns
        -------
        action:
            Sampled continuous action.

        log_prob:
            Log probability used for entropy regularization.
        """

        # Mean action prediction.
        mean = self.actor_mean(obs)

        # Clamp standard deviation for stability.
        log_std = torch.clamp(
            self.actor_log_std(obs),
            -5,
            2
        )

        # Convert log std -> std.
        std = log_std.exp()

        # Gaussian distribution.
        normal = torch.distributions.Normal(mean, std)

        # Reparameterization trick.
        z = normal.rsample()

        # Squash into [-1, 1].
        action = torch.tanh(z)

        # Log probability correction for tanh squashing.
        log_prob = (
            normal.log_prob(z)
            - torch.log(1 - action.pow(2) + 1e-6)
        )

        # Sum action dimensions.
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob

    def __init__(
        self,
        obs_dim: int = 19,
        action_dim: int = 3,
        hidden_dim: int = 256,
        lr: float = 3e-4
    ):
        """
        Initialize SAC policy.
        """

        super().__init__(obs_dim, action_dim)

        # Whether networks were built successfully.
        self._built = False

        # Hidden layer width.
        self._hidden_dim = hidden_dim

        # Learning rate.
        self._lr = lr

        # Replay buffer.
        self._replay_buffer = deque(maxlen=100_000)

        # Training minibatch size.
        self._batch_size = 256

        # Discount factor.
        self._gamma = 0.99

        # Soft target update rate.
        self._tau = 0.005

        # Select CUDA if available.
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        try:
            self._build_networks()

        except ImportError:
            logger.warning(
                "PyTorch not available — SACPolicy will not train."
            )

    def _build_networks(self):
        """
        Construct actor + critic neural networks.
        """

        import torch
        import torch.nn as nn

        class MLP(nn.Module):
            """
            Simple fully-connected multilayer perceptron.
            """

            def __init__(self, in_d, out_d, hidden):
                super().__init__()

                self.net = nn.Sequential(
                    nn.Linear(in_d, hidden),
                    nn.ReLU(),

                    nn.Linear(hidden, hidden),
                    nn.ReLU(),

                    nn.Linear(hidden, out_d),
                )

            def forward(self, x):
                """
                Forward neural network pass.
                """

                return self.net(x)

        H = self._hidden_dim

        # ------------------------------------------------------------------
        # Actor Networks
        # ------------------------------------------------------------------
        # SAC models policy as Gaussian distribution.

        self.actor_mean = MLP(
            self.obs_dim,
            self.action_dim,
            H
        )

        self.actor_log_std = MLP(
            self.obs_dim,
            self.action_dim,
            H
        )

        # ------------------------------------------------------------------
        # Twin Q Networks
        # ------------------------------------------------------------------
        # Two critics reduce overestimation bias.

        self.q1 = MLP(
            self.obs_dim + self.action_dim,
            1,
            H
        )

        self.q2 = MLP(
            self.obs_dim + self.action_dim,
            1,
            H
        )

        # ------------------------------------------------------------------
        # Target Networks
        # ------------------------------------------------------------------
        # Slowly updated copies stabilize training.

        self.q1_target = MLP(
            self.obs_dim + self.action_dim,
            1,
            H
        )

        self.q2_target = MLP(
            self.obs_dim + self.action_dim,
            1,
            H
        )

        # Copy initial weights.
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # ------------------------------------------------------------------
        # Move Networks To GPU / CPU
        # ------------------------------------------------------------------

        self.actor_mean.to(self.device)
        self.actor_log_std.to(self.device)

        self.q1.to(self.device)
        self.q2.to(self.device)

        self.q1_target.to(self.device)
        self.q2_target.to(self.device)

        # Copy initial target weights again after device move.
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self._built = True

        # ------------------------------------------------------------------
        # Optimizers
        # ------------------------------------------------------------------

        params = (
            list(self.actor_mean.parameters()) +
            list(self.actor_log_std.parameters())
        )

        # Actor optimizer.
        self.actor_opt = torch.optim.Adam(
            params,
            lr=self._lr
        )

        # Critic optimizer.
        self.q_opt = torch.optim.Adam(
            list(self.q1.parameters()) +
            list(self.q2.parameters()),
            lr=self._lr
        )

        logger.info("SAC networks built.")

    def get_action(self, obs: RLObservation) -> RLAction:
        """
        Generate action from policy network.

        Used during:
            - training
            - inference
        """

        # If networks failed to initialize.
        if not self._built:
            return RLAction(
                raw=np.zeros(self.action_dim, dtype=np.float32)
            )

        # Switch networks into inference mode.
        self.actor_mean.eval()
        self.actor_log_std.eval()

        with torch.no_grad():

            # Convert observation to tensor.
            x = torch.FloatTensor(obs.vector)
            x = x.unsqueeze(0).to(self.device)

            # Gaussian policy parameters.
            mean = self.actor_mean(x)

            log_std = torch.clamp(
                self.actor_log_std(x),
                -5,
                2
            )

            std = log_std.exp()

            # Sample stochastic action.
            action = torch.tanh(
                mean + std * torch.randn_like(std)
            )

        return RLAction(
            raw=action.squeeze().cpu().numpy()
        )

    def store_transition(
        self,
        obs,
        action,
        reward,
        next_obs,
        done
    ):
        """
        Store replay transition in replay buffer.
        """

        self._replay_buffer.append(
            (
                obs.vector,
                action.raw,
                reward,
                next_obs.vector,
                done
            )
        )

    def update(self) -> dict:
        """
        Perform one SAC training update.

        Training Stages
        ---------------
        1. Sample replay batch
        2. Compute target Q values
        3. Update critic networks
        4. Update actor policy
        5. Soft-update target critics
        """

        # Wait until replay buffer contains enough samples.
        if len(self._replay_buffer) < self._batch_size:
            return {}

        # Enable training mode.
        self.actor_mean.train()
        self.actor_log_std.train()
        self.q1.train()
        self.q2.train()

        import torch.nn.functional as F

        # Sample replay minibatch.
        batch = self._sample_batch()

        # Move tensors to GPU/CPU.
        obs = batch['obs'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_obs = batch['next_obs'].to(self.device)
        dones = batch['dones'].to(self.device)

        # SAC entropy coefficient.
        alpha = 0.2

        # ------------------------------------------------------------------
        # Target Q Computation
        # ------------------------------------------------------------------

        with torch.no_grad():

            # Sample next-state actions.
            next_actions, next_log_probs = self._sample_action(next_obs)

            # Concatenate state + action.
            next_input = torch.cat(
                [next_obs, next_actions],
                dim=1
            )

            # Evaluate target critics.
            q1_next = self.q1_target(next_input)
            q2_next = self.q2_target(next_input)

            # Conservative Q estimate.
            q_next = torch.min(q1_next, q2_next)

            # Bellman backup.
            target_q = rewards + (
                1 - dones
            ) * self._gamma * (
                q_next - alpha * next_log_probs
            )

        # ------------------------------------------------------------------
        # Critic Update
        # ------------------------------------------------------------------

        # Concatenate current state + action.
        q_input = torch.cat([obs, actions], dim=1)

        # Current Q estimates.
        q1 = self.q1(q_input)
        q2 = self.q2(q_input)

        # Mean-squared Bellman error.
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)

        # Combined critic loss.
        q_loss = q1_loss + q2_loss

        # Gradient descent step.
        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        # ------------------------------------------------------------------
        # Actor Update
        # ------------------------------------------------------------------

        # Sample fresh policy actions.
        new_actions, log_probs = self._sample_action(obs)

        # Evaluate actor through critics.
        actor_input = torch.cat(
            [obs, new_actions],
            dim=1
        )

        q1_actor = self.q1(actor_input)
        q2_actor = self.q2(actor_input)

        q_actor = torch.min(q1_actor, q2_actor)

        # SAC objective:
        # maximize Q while maximizing entropy.
        actor_loss = (
            alpha * log_probs - q_actor
        ).mean()

        # Actor gradient step.
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ------------------------------------------------------------------
        # Soft Target Updates
        # ------------------------------------------------------------------
        # Slowly move target networks toward live networks.

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

        # Return training statistics.
        return {
            'q_loss': float(q_loss.item()),
            'actor_loss': float(actor_loss.item()),
            'buffer_size': len(self._replay_buffer),
        }

    def save(self, path: str):
        """
        Save model weights to disk.
        """

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
        """
        Load model weights from disk.
        """

        if not self._built:
            return

        import torch

        ckpt = torch.load(path)

        self.actor_mean.load_state_dict(ckpt['actor_mean'])
        self.actor_log_std.load_state_dict(ckpt['actor_log_std'])
        self.q1.load_state_dict(ckpt['q1'])
        self.q2.load_state_dict(ckpt['q2'])

        logger.info(f"SAC weights loaded from {path}")

