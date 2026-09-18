"""A small audio-only navigation environment on the bundled test scene.

The simulator owns privileged world state. ``reset`` and ``step`` expose only
the raw acoustic impulse response under the ``audio`` observation key. Debug
state is returned in ``info`` for manual inspection and is never part of the
agent observation.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import quaternion  # noqa: F401  (must precede habitat_sim on this branch)
import habitat_sim


class Actions(str, Enum):
    MOVE_FORWARD = "MOVE_FORWARD"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    STOP = "STOP"


@dataclass(frozen=True)
class AudioNavConfig:
    """Explicit mechanics and episode settings."""

    scene_path: Optional[Path] = None
    max_steps: int = 50
    success_distance: float = 1.0
    step_penalty: float = -0.01
    move_distance: float = 0.25
    turn_angle_degrees: float = 30.0
    audio_sample_rate: int = 16000
    random_seed: int = 7


ActionLike = Union[Actions, str]


class AudioNavEnv:
    """Minimal POMDP environment with raw acoustic observations."""

    def __init__(self, config: Optional[AudioNavConfig] = None):
        self.config = config or AudioNavConfig()
        scene = self.config.scene_path or (
            Path.home() / "third_party/habitat-sim/data/test_assets/scenes/simple_room.glb"
        )
        self.scene_path = Path(scene).expanduser().resolve()
        if not self.scene_path.is_file():
            raise FileNotFoundError(self.scene_path)
        if self.config.max_steps < 1 or self.config.move_distance <= 0:
            raise ValueError("max_steps and move_distance must be positive")
        if self.config.success_distance <= 0:
            raise ValueError("success_distance must be positive")

        backend = habitat_sim.SimulatorConfiguration()
        backend.scene_id = str(self.scene_path)
        backend.enable_physics = False
        backend.load_semantic_mesh = False
        backend.gpu_device_id = -1
        agent_config = habitat_sim.agent.AgentConfiguration()

        # Required internally by this pinned branch to initialize/load geometry.
        depth = habitat_sim.CameraSensorSpec()
        depth.uuid = "_internal_geometry_depth"
        depth.sensor_type = habitat_sim.SensorType.DEPTH
        depth.resolution = [16, 16]
        depth.position = [0.0, 1.5, 0.0]
        agent_config.sensor_specifications = [depth]
        self.sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent_config]))

        settings = habitat_sim.NavMeshSettings()
        settings.set_defaults()
        if not self.sim.recompute_navmesh(self.sim.pathfinder, settings):
            self.sim.close()
            raise RuntimeError("Could not build a navigation mesh")
        # Seed once per environment. Pathfinder RNG state then advances across
        # reset() calls, while equivalent environments reproduce the sequence.
        self.sim.pathfinder.seed(self.config.random_seed)
        self._audio_spec = habitat_sim.AudioSensorSpec()
        # Habitat-Sim 0.2.2's audio binding looks up this historical UUID
        # internally, so the policy-facing key is mapped separately below.
        self._audio_spec.uuid = "audio_sensor"
        self._audio_spec.enableMaterials = False
        self._audio_spec.position = [0.0, 1.5, 0.0]
        self._audio_spec.channelLayout.type = (
            habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Binaural
        )
        self._audio_spec.channelLayout.channelCount = 2
        self._audio_spec.acousticsConfig.sampleRate = self.config.audio_sample_rate
        self._audio_spec.acousticsConfig.indirect = True
        self.sim.add_sensor(self._audio_spec)
        self._source_floor: Optional[np.ndarray] = None
        self._agent_floor: Optional[np.ndarray] = None
        self._yaw = 0.0
        self._steps = 0
        self._done = False
        self._last_audio: Optional[np.ndarray] = None

    def reset(self) -> Dict[str, np.ndarray]:
        """Start a connected episode and return its audio-only observation."""
        self._done = False
        self._steps = 0
        self._agent_floor, self._source_floor = self._sample_episode_points()
        self._yaw = 0.0
        state = habitat_sim.AgentState()
        state.position = self._agent_floor.copy()
        state.rotation = self._rotation_for_yaw(self._yaw)
        self.sim.get_agent(0).set_state(state, True)
        self._last_audio = self._observe_audio()
        return self._observation(self._last_audio)

    def step(self, action: ActionLike) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """Apply one action and return ``(audio_observation, reward, done, info)``."""
        if self._done:
            raise RuntimeError("Episode is done; call reset() before step()")
        action = self._coerce_action(action)
        self._steps += 1
        if action == Actions.STOP:
            success = self.geodesic_distance <= self.config.success_distance
            self._done = True
            reward = 1.0 if success else 0.0
        else:
            if action == Actions.TURN_LEFT:
                self._yaw += np.deg2rad(self.config.turn_angle_degrees)
                self._set_pose(self._agent_floor, self._yaw)
            elif action == Actions.TURN_RIGHT:
                self._yaw -= np.deg2rad(self.config.turn_angle_degrees)
                self._set_pose(self._agent_floor, self._yaw)
            elif action == Actions.MOVE_FORWARD:
                direction = self._forward_direction()
                candidate = self._agent_floor + self.config.move_distance * direction
                filtered = np.asarray(self.sim.pathfinder.try_step(self._agent_floor, candidate))
                if not np.array_equal(filtered, self._agent_floor):
                    self._agent_floor = filtered
                    self._set_pose(self._agent_floor, self._yaw)
            reward = self.config.step_penalty
            if self._steps >= self.config.max_steps:
                self._done = True
        self._last_audio = self._observe_audio()
        return self._observation(self._last_audio), reward, self._done, self._info(action)

    @property
    def geodesic_distance(self) -> float:
        """Privileged shortest-path distance, for success/debugging only."""
        if self._agent_floor is None or self._source_floor is None:
            return float("inf")
        path = habitat_sim.ShortestPath()
        path.requested_start = self._agent_floor
        path.requested_end = self._source_floor
        if not self.sim.pathfinder.find_path(path):
            return float("inf")
        return float(path.geodesic_distance)

    @property
    def agent_position(self) -> np.ndarray:
        return self._agent_floor.copy()

    @property
    def source_position(self) -> np.ndarray:
        return self._source_floor.copy()

    @property
    def agent_yaw(self) -> float:
        return self._yaw

    def close(self) -> None:
        if self.sim is not None:
            self.sim.close()
            self.sim = None

    def _sample_episode_points(self) -> Tuple[np.ndarray, np.ndarray]:
        start = np.asarray(self.sim.pathfinder.get_random_navigable_point())
        for _ in range(100):
            candidate = np.asarray(self.sim.pathfinder.get_random_navigable_point())
            path = habitat_sim.ShortestPath()
            path.requested_start = start
            path.requested_end = candidate
            if 1.0 < np.linalg.norm(candidate - start) < 3.0 and self.sim.pathfinder.find_path(path):
                return start, candidate
        raise RuntimeError("Could not find two connected navigable episode points")

    def _set_pose(self, floor_position: np.ndarray, yaw: float) -> None:
        state = self.sim.get_agent(0).get_state()
        state.position = floor_position.copy()
        state.rotation = self._rotation_for_yaw(yaw)
        self.sim.get_agent(0).set_state(state, True)

    def _rotation_for_yaw(self, yaw: float):
        return habitat_sim.utils.common.quat_from_angle_axis(yaw, habitat_sim.geo.UP)

    def _forward_direction(self) -> np.ndarray:
        """Return the world-space direction of the agent's local FRONT (-Z)."""
        rotation = self._rotation_for_yaw(self._yaw)
        return np.asarray(
            habitat_sim.utils.common.quat_rotate_vector(rotation, habitat_sim.geo.FRONT),
            dtype=np.float32,
        )

    def _observe_audio(self) -> np.ndarray:
        source = self._source_floor + np.array([0.0, 1.5, 0.0])
        sensor = self.sim.get_agent(0)._sensors[self._audio_spec.uuid]
        sensor.setAudioSourceTransform(source)
        audio = np.asarray(self.sim.get_sensor_observations()[self._audio_spec.uuid], dtype=np.float32)
        if audio.ndim != 2 or audio.shape[0] != 2 or audio.shape[1] == 0:
            raise RuntimeError(f"Invalid audio observation shape: {audio.shape}")
        if not np.isfinite(audio).all() or not np.any(audio != 0):
            raise RuntimeError("Audio observation must be finite and non-empty")
        return audio.copy()

    @staticmethod
    def _observation(audio: np.ndarray) -> Dict[str, np.ndarray]:
        # Deliberately only expose the raw acoustic observation.
        return {"audio": audio.copy()}

    def _info(self, action: Actions) -> Dict:
        # This is debug/metrics data, explicitly outside the policy observation.
        return {
            "action": action.value,
            "steps": self._steps,
            "success": self._done and action == Actions.STOP and self.geodesic_distance <= self.config.success_distance,
            "timeout": self._done and action != Actions.STOP and self._steps >= self.config.max_steps,
            "debug": {
                "agent_position": self.agent_position.copy(),
                "source_position": self.source_position.copy(),
                "agent_yaw": self.agent_yaw,
                "geodesic_distance": self.geodesic_distance,
            },
        }

    @staticmethod
    def _coerce_action(action: ActionLike) -> Actions:
        try:
            return action if isinstance(action, Actions) else Actions(action)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unknown action {action!r}; expected one of {[a.value for a in Actions]}") from exc
