import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from audio_nav import Actions, AudioNavConfig, AudioNavEnv


class AudioNavEnvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = AudioNavEnv(AudioNavConfig(max_steps=3, move_distance=0.1))

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        self.env.reset()

    def test_observation_is_audio_only_and_valid(self):
        observation = self.env.reset()
        self.assertEqual(set(observation), {"audio"})
        self.assertEqual(observation["audio"].shape[0], 1)
        self.assertGreater(observation["audio"].shape[1], 0)
        self.assertTrue(np.isfinite(observation["audio"]).all())

    def test_turn_changes_orientation_and_source_stays_fixed(self):
        source = self.env.source_position.copy()
        yaw = self.env.agent_yaw
        self.env.step(Actions.TURN_LEFT)
        self.assertNotEqual(self.env.agent_yaw, yaw)
        self.assertTrue(np.array_equal(source, self.env.source_position))

    def test_forward_changes_position_when_possible(self):
        before = self.env.agent_position.copy()
        observation, _, _, _ = self.env.step(Actions.MOVE_FORWARD)
        self.assertEqual(set(observation), {"audio"})
        # This bundled room has navigable space in the default forward direction.
        self.assertFalse(np.array_equal(before, self.env.agent_position))

    def test_stop_failure_and_success_are_distinguished(self):
        _, reward, done, info = self.env.step(Actions.STOP)
        self.assertTrue(done)
        self.assertFalse(info["success"])
        self.assertEqual(reward, 0.0)

        self.env.reset()
        self.env._agent_floor = self.env.source_position.copy()  # privileged test setup
        self.env._set_pose(self.env._agent_floor, self.env.agent_yaw)
        _, reward, done, info = self.env.step(Actions.STOP)
        self.assertTrue(done)
        self.assertTrue(info["success"])
        self.assertEqual(reward, 1.0)

    def test_timeout_terminates(self):
        self.env.reset()
        _, _, done, info = self.env.step(Actions.TURN_LEFT)
        self.assertFalse(done)
        _, _, done, info = self.env.step(Actions.TURN_RIGHT)
        self.assertFalse(done)
        _, _, done, info = self.env.step(Actions.TURN_LEFT)
        self.assertTrue(done)
        self.assertTrue(info["timeout"])


if __name__ == "__main__":
    unittest.main()
