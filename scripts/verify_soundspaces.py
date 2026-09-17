"""Verify the pinned SoundSpaces 2 runtime with a bundled, tiny 3D scene."""
import argparse
import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path

# Use Mesa software rendering for the geometry context on WSL without CUDA.
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
# Upstream recommends this import order to avoid native invalid-pointer errors.
import quaternion  # noqa: F401
import habitat_sim
import habitat
import soundspaces
import numpy as np
from scipy.io import wavfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--third-party", type=Path, default=Path.home() / "third_party")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    scene = args.third_party / "habitat-sim/data/test_assets/scenes/simple_room.glb"
    # /tmp is writable in managed WSL sessions; pass --output-dir for another
    # external location when running outside the managed environment.
    output = args.output_dir or Path("/tmp/audio-only-navigation-data/verification")
    if not scene.is_file():
        raise FileNotFoundError(scene)
    output.mkdir(parents=True, exist_ok=True)

    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(scene.resolve())
    backend.enable_physics = False
    backend.load_semantic_mesh = False
    backend.gpu_device_id = -1  # Select an EGL device without CUDA matching.
    agent_config = habitat_sim.agent.AgentConfiguration()
    # This branch only loads geometry when a renderer is initialized.
    depth = habitat_sim.CameraSensorSpec()
    depth.uuid = "geometry_depth"
    depth.sensor_type = habitat_sim.SensorType.DEPTH
    depth.resolution = [16, 16]
    depth.position = [0.0, 1.5, 0.0]
    agent_config.sensor_specifications = [depth]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent_config]))
    try:
        settings = habitat_sim.NavMeshSettings()
        settings.set_defaults()
        if not sim.recompute_navmesh(sim.pathfinder, settings):
            raise RuntimeError("Could not build navigation mesh from scene geometry")
        sim.pathfinder.seed(7)
        listener_floor = np.asarray(sim.pathfinder.get_random_navigable_point())
        source_floor = None
        for _ in range(100):
            candidate = np.asarray(sim.pathfinder.get_random_navigable_point())
            distance = np.linalg.norm(candidate - listener_floor)
            path = habitat_sim.ShortestPath()
            path.requested_start = listener_floor
            path.requested_end = candidate
            if 1.0 < distance < 3.0 and sim.pathfinder.find_path(path):
                source_floor = candidate
                break
        if source_floor is None:
            raise RuntimeError("Could not find two connected, separated scene positions")
        state = habitat_sim.AgentState()
        state.position = listener_floor
        sim.get_agent(0).set_state(state)

        spec = habitat_sim.AudioSensorSpec()
        spec.uuid = "audio_sensor"
        spec.enableMaterials = False  # Bundled scene has no semantic material labels.
        spec.position = [0.0, 1.5, 0.0]
        spec.channelLayout.type = habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Mono
        spec.channelLayout.channelCount = 1
        spec.acousticsConfig.sampleRate = 16000
        spec.acousticsConfig.indirect = True
        sim.add_sensor(spec)
        source = source_floor + np.array([0.0, 1.5, 0.0])
        sensor = sim.get_agent(0)._sensors[spec.uuid]
        sensor.setAudioSourceTransform(source)
        ir = np.asarray(sim.get_sensor_observations()[spec.uuid], dtype=np.float32)
        if ir.ndim != 2 or ir.shape[0] != 1 or ir.shape[1] == 0:
            raise RuntimeError(f"Invalid impulse response shape: {ir.shape}")
        if not np.isfinite(ir).all() or not np.any(ir != 0):
            raise RuntimeError("Impulse response must be finite and nonzero")
        wav_path = output / "impulse_response.wav"
        wavfile.write(str(wav_path), 16000, ir.T)
        rate, saved = wavfile.read(str(wav_path))
        if rate != 16000 or not np.array_equal(saved, ir[0]):
            raise RuntimeError("Saved WAV did not round-trip correctly")
        report = {
            "status": "PASS",
            "versions": {name: version(name) for name in ("habitat-sim", "habitat", "sound-spaces", "numpy")},
            "imports": {"habitat_sim": habitat_sim.__file__, "habitat": habitat.__file__, "soundspaces": soundspaces.__file__},
            "scene": str(scene.resolve()),
            "scene_bytes": scene.stat().st_size,
            "scene_sha256": hashlib.sha256(scene.read_bytes()).hexdigest(),
            "navmesh_area_m2": sim.pathfinder.navigable_area,
            "agent_position": listener_floor.tolist(),
            "listener_position": (listener_floor + [0.0, 1.5, 0.0]).tolist(),
            "source_position": source.tolist(),
            "sample_rate_hz": 16000,
            "channels": "mono",
            "materials": "default (semantic materials disabled)",
            "indirect_propagation": True,
            "ir_shape": list(ir.shape),
            "ir_peak": float(np.max(np.abs(ir))),
            "ir_energy": float(np.sum(ir.astype(np.float64) ** 2)),
            "ir_nonzero_samples": int(np.count_nonzero(ir)),
            "wav": str(wav_path.resolve()),
        }
        (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        sim.close()


if __name__ == "__main__":
    main()
