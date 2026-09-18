#!/usr/bin/env python3
"""Run a short debug episode; privileged values are printed for inspection only."""
import os
import sys
from pathlib import Path

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio_nav import Actions, AudioNavConfig, AudioNavEnv


def main():
    env = AudioNavEnv(AudioNavConfig(max_steps=8))
    try:
        observation = env.reset()
        print("reset audio shape:", observation["audio"].shape)
        print("DEBUG ONLY initial position:", env.agent_position.tolist())
        print("DEBUG ONLY source position:", env.source_position.tolist())
        for action in (Actions.TURN_LEFT, Actions.MOVE_FORWARD, Actions.TURN_RIGHT):
            before = env.agent_position.copy()
            observation, reward, done, info = env.step(action)
            print(action.value, "audio shape:", observation["audio"].shape,
                  "pose changed:", not (before == env.agent_position).all(),
                  "reward:", reward, "done:", done)
            print("DEBUG ONLY:", info["debug"])
            if done:
                break
        if not done:
            _, reward, done, info = env.step(Actions.STOP)
            print("STOP reward:", reward, "successful:", info["success"], "done:", done)
            print("DEBUG ONLY final geodesic distance:", info["debug"]["geodesic_distance"])
        print("manual episode terminated:", done)
    finally:
        env.close()


if __name__ == "__main__":
    main()
