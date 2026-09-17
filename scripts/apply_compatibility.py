"""Apply narrowly scoped fixes to the pinned external SoundSpaces stack."""
import argparse
from pathlib import Path


def replace(path, old, new):
    text = path.read_text()
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"Unexpected source version: {path}")
    path.write_text(text.replace(old, new, 1))
    print(f"Patched {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--third-party", type=Path, default=Path.home() / "third_party")
    root = parser.parse_args().third_party
    replace(root / "habitat-sim/src/deps/corrade/src/Corrade/Utility/Arguments.h",
            "#include <string>", "#include <cstdint>\n#include <string>")
    # Required by the official SoundSpaces installation guide for this branch.
    replace(root / "habitat-lab/habitat/tasks/rearrange/rearrange_sim.py",
            "from habitat_sim.robots import FetchRobot, FetchRobotNoWheels",
            "# SoundSpaces audio branch does not provide the Fetch robot classes.")
    # Importing the simulator must not eagerly import the training packages.
    path = root / "sound-spaces/soundspaces/benchmark.py"
    text = path.read_text()
    old = "from ss_baselines.av_nav.config import get_task_config\n"
    new = "        from ss_baselines.av_nav.config import get_task_config\n\n"
    if new not in text:
        if old not in text or "        config_env = get_task_config(config_paths)" not in text:
            raise RuntimeError(f"Unexpected source version: {path}")
        text = text.replace(old, "", 1).replace(
            "        config_env = get_task_config(config_paths)",
            new + "        config_env = get_task_config(config_paths)", 1)
        path.write_text(text)
        print(f"Patched {path}")


if __name__ == "__main__":
    main()
