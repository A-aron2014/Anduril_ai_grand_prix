"""
Autonomy Stack — Main Orchestrator
Wires all components together and runs the control loop.

    Vision (30 Hz) ──────────────┐
    MAVLink telemetry (120 Hz) ──┤→ StateEstimator → Guidance → Autopilot → MAVLink cmd
                                 └→ PerceptionManager (gates, obstacles, VO)
                                         │
                                    RLPolicy (optional, replaces guidance)

Usage
-----
    stack = AutonomyStack(config=StackConfig())
    stack.run()          # blocking
    # or
    await stack.run_async()
"""

import asyncio
import logging
import time
import threading
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np

from comms.mavlink_bridge  import MAVLinkBridge, TelemetryState, ControlCommand
from comms.vision_stream   import VisionStreamReceiver
from core.state_estimator  import StateEstimator, VehicleState
from perception.perception import PerceptionManager
from guidance.guidance     import GuidanceAlgorithm, CourseMap, GuidanceOutput
from control.autopilot     import Autopilot, AutopilotConfig
from rl.policy             import RLPolicyBase, RandomPolicy

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)s %(message)s')


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class StackConfig:
    # Network
    mavlink_host:   str = '127.0.0.1'
    mavlink_port:   int = 14550
    vision_host:    str = '0.0.0.0'
    vision_port:    int = 5600

    # Control
    control_rate_hz: float = 50.0      # autopilot command rate (≤ 100 Hz)
    use_rl_policy:   bool  = False     # True = RL replaces guidance
    training_mode:   bool  = False     # True = store transitions + update

    # Course (pre-loaded gate positions in NED, metres)
    # Leave empty to build the map live from vision
    gate_positions: List = field(default_factory=list)
    gate_speed_mps: float = 8.0

    # Weights
    policy_weights: Optional[str] = None   # path to .pt file


# ---------------------------------------------------------------------------
# Autonomy Stack
# ---------------------------------------------------------------------------

class AutonomyStack:

    def __init__(self, config: Optional[StackConfig] = None):
        self.cfg = config or StackConfig()

        # --- Communication ---
        self.mavlink = MAVLinkBridge(
            host=self.cfg.mavlink_host,
            port=self.cfg.mavlink_port,
            telemetry_callback=self._on_telemetry,
        )
        self.vision = VisionStreamReceiver(
            host=self.cfg.vision_host,
            port=self.cfg.vision_port,
            on_frame_callback=self._on_frame,
        )

        # --- Core ---
        self.estimator  = StateEstimator()
        self.perception = PerceptionManager()

        # --- Guidance & Control ---
        course_map = CourseMap()
        if self.cfg.gate_positions:
            course_map.load_from_list(self.cfg.gate_positions, self.cfg.gate_speed_mps)
        self.guidance  = GuidanceAlgorithm(course_map=course_map)
        self.autopilot = Autopilot(AutopilotConfig())

        # --- RL Policy ---
        self.rl_policy: RLPolicyBase = RandomPolicy()
        if self.cfg.policy_weights:
            try:
                from rl.policy import SACPolicy
                self.rl_policy = SACPolicy()
                self.rl_policy.load(self.cfg.policy_weights)
                logger.info("Loaded SAC policy weights.")
            except Exception as e:
                logger.warning(f"Failed to load policy: {e} — using RandomPolicy.")

        self._running = False
        self._latest_guidance: Optional[GuidanceOutput] = None

    # ------------------------------------------------------------------
    # Callbacks (called from background threads)
    # ------------------------------------------------------------------

    def _on_telemetry(self, telem: TelemetryState):
        """Called ~120 Hz from MAVLink reader thread."""
        self.estimator.update_from_mavlink(telem)

    def _on_frame(self, frame, sim_time_ns: int):
        """Called ~30 Hz from vision receiver thread."""
        state = self.estimator.get_state()
        body_vel = state.velocity_ned()
        speed = state.speed

        self.perception.update(frame, body_vel, speed)

        # Feed VO back to estimator
        if self.perception.latest_vo and self.perception.latest_vo.valid:
            self.estimator.update_from_vo(self.perception.latest_vo)

    # ------------------------------------------------------------------
    # Main Loop
    # ------------------------------------------------------------------

    def run(self):
        """Blocking run. Ctrl-C to stop."""
        self._startup()
        dt = 1.0 / self.cfg.control_rate_hz
        try:
            while self._running:
                t0 = time.monotonic()
                self._control_tick()
                elapsed = time.monotonic() - t0
                sleep = dt - elapsed
                if sleep > 0:
                    time.sleep(sleep)
                elif sleep < -0.005:
                    logger.debug(f"Control loop overrun by {-sleep*1000:.1f} ms")
        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            self._shutdown()

    async def run_async(self):
        """Async version for integration with asyncio event loops."""
        self._startup()
        dt = 1.0 / self.cfg.control_rate_hz
        try:
            while self._running:
                t0 = time.monotonic()
                self._control_tick()
                elapsed = time.monotonic() - t0
                await asyncio.sleep(max(0, dt - elapsed))
        finally:
            self._shutdown()

    # ------------------------------------------------------------------
    # Control Tick (one iteration of the guidance+autopilot loop)
    # ------------------------------------------------------------------

    def _control_tick(self):
        state   = self.estimator.get_state()
        gates   = list(self.perception.latest_gates)
        obstacles = list(self.perception.latest_obstacles)

        # --- Choose guidance source ---
        if self.cfg.use_rl_policy:
            next_gate_pos = self._next_gate_pos()
            guidance = self.rl_policy.step(
                state, next_gate_pos, gates, obstacles,
                training=self.cfg.training_mode,
            )
            if self.cfg.training_mode:
                metrics = self.rl_policy.update()
                if metrics:
                    logger.debug(f"RL update: {metrics}")
        else:
            guidance = self.guidance.compute(state, gates, obstacles)

        self._latest_guidance = guidance

        if guidance.course_complete:
            logger.info("Course complete — holding position.")
            self._hover(state)
            return

        # --- Autopilot → MAVLink ---
        cmd = self.autopilot.compute(state, guidance)
        self.mavlink.send_command(cmd)

    def _next_gate_pos(self) -> Optional[np.ndarray]:
        wp = self.guidance.sequencer.current_waypoint()
        return wp.position_ned if wp else None

    def _hover(self, state: VehicleState):
        """Command a position hold at current location."""
        hold_pos = state.position_ned()
        cmd = ControlCommand(
            target_north=hold_pos[0],
            target_east=hold_pos[1],
            target_down=hold_pos[2],
            target_vn=0, target_ve=0, target_vd=0,
            target_yaw=state.yaw,
            type_mask=0b0000_1111_1000_0000,  # position only
        )
        self.mavlink.send_command(cmd)

    # ------------------------------------------------------------------
    # Startup / Shutdown
    # ------------------------------------------------------------------

    def _startup(self):
        logger.info("Starting Autonomy Stack...")
        self.mavlink.connect()
        self.vision.start()
        self._running = True

        # Wait for first heartbeat
        logger.info("Waiting for simulator heartbeat...")
        for _ in range(30):
            if self.mavlink.is_connected():
                logger.info("Heartbeat received — ready.")
                break
            time.sleep(0.5)
        else:
            logger.warning("No heartbeat after 15 s — proceeding anyway.")

    def _shutdown(self):
        logger.info("Shutting down...")
        self._running = False
        self.vision.stop()
        self.mavlink.close()


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    config = StackConfig(
        mavlink_host='127.0.0.1',
        mavlink_port=14550,
        vision_host='0.0.0.0',
        vision_port=5600,
        control_rate_hz=50.0,
        use_rl_policy=False,       # flip to True when policy is trained
        training_mode=False,
        gate_positions=[
            # Example: fill in actual gate NED positions from course briefing
            # (north, east, down)   — down is negative for altitude
            (20,  0, -5),
            (40, 10, -5),
            (60,  0, -5),
            (80, -5, -5),
        ],
        gate_speed_mps=8.0,
    )
    stack = AutonomyStack(config=config)
    stack.run()

if __name__ == '__main__':
    main()
