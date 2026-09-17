# SoundSpaces 2 runtime setup

This setup is limited to installing the simulator and verifying runtime impulse
responses. It does not configure training, CL1, preprocessing, encoding/decoding,
or a navigation experiment.

## Locations and pinned sources

Project source, configs, scripts and documentation: `~/projects/audio-only-navigation/`.
External source repositories: `~/third_party/`.
Large scenes, audio, generated outputs and experiment data: `~/data/audio-only-navigation/`.
Python environment: existing `~/miniconda3/envs/ss` (Python 3.9.25).
The project repository never contains upstream checkouts, native build trees,
large assets or generated audio.

| Component | Repository URL | Branch/tag | Commit | External directory |
| --- | --- | --- | --- |
| Habitat-Sim | `https://github.com/facebookresearch/habitat-sim.git` | `RLRAudioPropagationUpdate` | `4f61e321477708fa606fbd8f42b4bef41d67c672` | `~/third_party/habitat-sim` |
| Habitat-Lab | `https://github.com/facebookresearch/habitat-lab.git` | `v0.2.2` | `0f454f62e41050bc90ca468c62db35d7484923ff` | `~/third_party/habitat-lab` |
| SoundSpaces | `https://github.com/facebookresearch/sound-spaces.git` | `main` | `287184fd7067a0385558492716355c54875500ee` | `~/third_party/sound-spaces` |
| RLR audio propagation | Habitat-Sim submodule | `4fd446b4abb5c71fb7a232a083bbddd65f25fc6f` | `habitat-sim/src/deps/rlr-audio-propagation` |

Upstream instructions: [installation](https://github.com/facebookresearch/sound-spaces/blob/main/INSTALLATION.md),
[runtime API](https://github.com/facebookresearch/sound-spaces/blob/main/SoundSpaces2.md),
[minimal example](https://github.com/facebookresearch/sound-spaces/blob/main/examples/minimal_example.py).

## System prerequisites

Verified WSL2 / Ubuntu 24.04.2 LTS, Git 2.43.0, CMake 3.14.0 in `ss`, and
Ubuntu GCC 13. The following Ubuntu packages were already installed by the user:

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  build-essential ninja-build pkg-config \
  libjpeg-dev libglm-dev libgl1 libegl1-mesa-dev \
  mesa-utils xorg-dev freeglut3-dev
```

`zlib1g-dev` and `libpng-dev` are also present. No system Python changes are needed.

## Reproduce the external source checkouts

The commands below document reproduction; the original environment already has
these clones. Do not clone again over existing directories.

```bash
conda activate ss
mkdir -p ~/third_party
git clone --depth 1 --branch RLRAudioPropagationUpdate \
  https://github.com/facebookresearch/habitat-sim.git ~/third_party/habitat-sim
git -C ~/third_party/habitat-sim fetch --depth 1 origin 4f61e321477708fa606fbd8f42b4bef41d67c672
git -C ~/third_party/habitat-sim checkout 4f61e321477708fa606fbd8f42b4bef41d67c672
git -C ~/third_party/habitat-sim submodule update --init --recursive --depth 1 --jobs 4
git clone --depth 1 --branch v0.2.2 \
  https://github.com/facebookresearch/habitat-lab.git ~/third_party/habitat-lab
git clone --depth 1 https://github.com/facebookresearch/sound-spaces.git ~/third_party/sound-spaces
git -C ~/third_party/sound-spaces fetch --depth 1 origin 287184fd7067a0385558492716355c54875500ee
git -C ~/third_party/sound-spaces checkout 287184fd7067a0385558492716355c54875500ee
```

## Compatibility choices

Exactly three source patches are applied by `scripts/apply_compatibility.py`:

1. Corrade `src/Corrade/Utility/Arguments.h`: add `<cstdint>` to fix GCC 13 errors
   for standard integer types.
2. Habitat-Lab `habitat/tasks/rearrange/rearrange_sim.py`: remove the Fetch robot
   import, following upstream SoundSpaces instructions. Fetch rearrangement is
   unsupported by this setup.
3. SoundSpaces `soundspaces/benchmark.py`: move `get_task_config` into
   `Benchmark.__init__`. Ordinary simulator imports no longer load training
   modules eagerly. Instantiating benchmarks still requires their dependencies.

Additional build configuration, without further source patches:

- `-include cstdint` supplies the same missing standard header to other pinned
  C++ sources (the next observed failure was Corrade `String.h`).
- A Linux-only `PATH` prevents CMake from selecting Windows MinGW headers/libraries
  through WSL's inherited path. The build wrapper clears cached Zlib/OpenGL/X11
  discovery because the first configure found `/mnt/c/msys64/mingw64`.
- Two compiler jobs limit memory use on the 8 GB WSL instance.
- Build headless, with audio enabled; CUDA, Bullet, and GUI viewers are disabled.
- The smoke test uses `gpu_device_id=-1` and Mesa software rendering
  (`LIBGL_ALWAYS_SOFTWARE=1`) to avoid CUDA/EGL device matching. A 16×16 depth
  sensor initializes the renderer because this branch otherwise skips loading
  geometry. This is an internal geometry requirement, not a sensing pipeline.
- Import `quaternion` before `habitat_sim`, following upstream's native pointer
  error workaround.

Python libraries are pinned for the older stack: NumPy 1.23.5, Numba 0.57.1,
numpy-quaternion 2022.4.3, SciPy 1.10.1, Gym 0.22.0, librosa 0.9.2,
NetworkX 2.8.8, and CPU-only PyTorch 1.13.1+cpu. Packaging tools are pip 24.0,
setuptools 65.5.0 and wheel 0.38.4 for legacy upstream setup scripts.

SoundSpaces is installed with `--no-deps` and explicit runtime dependencies.
Its broad metadata also lists TensorFlow, notebook, and astropy; those are omitted
from this simulation-only environment. `pip check` therefore reports those three
missing distributions. This is not a complete environment for upstream training
or notebook workflows.

## Build and install

The Conda package snapshot is `requirements/conda-linux-64-explicit.txt`.
On a fresh machine it can create `ss` with
`conda create -n ss --file requirements/conda-linux-64-explicit.txt`.
For this installation the existing `ss` was reused. Conda's snapshot records its
original packaging tools; the following pip commands deliberately replace those
with the legacy-compatible versions before building.

From the project root, after activating `ss`:

```bash
python -m pip install pip==24.0 setuptools==65.5.0 wheel==0.38.4
python -m pip install -r requirements/runtime-py39-linux.txt
python scripts/apply_compatibility.py
bash scripts/build_habitat_sim.sh
python -m pip install --no-deps --no-build-isolation \
  -e ~/third_party/habitat-lab -e ~/third_party/sound-spaces
```

## Minimal acoustic verification

```bash
PATH="$HOME/miniconda3/envs/ss/bin:/usr/local/bin:/usr/bin:/bin" \
LIBGL_ALWAYS_SOFTWARE=1 NUMBA_CACHE_DIR=/tmp/numba-cache \
  "$HOME/miniconda3/envs/ss/bin/python" scripts/verify_soundspaces.py
```

The script imports all three packages and uses Habitat-Sim's SoundSpaces 2 audio
sensor directly. It loads `~/third_party/habitat-sim/data/test_assets/scenes/simple_room.glb`,
which is bundled with the pinned source checkout; no scene download, credentials,
or pre-rendered RIR dataset is required. Its SHA-256 is
`8ff7ad895a5cd62780f2950955966804e0acf00de1d469f51e91de3a5aefce20`.
It reconstructs a navigation mesh and
chooses two connected floor positions 1–3 meters apart, then places the listener
and source 1.5 meters above them. Audio is mono at 16 kHz with indirect propagation
enabled and default acoustic materials (the asset has no semantic annotations).

The test rejects missing geometry/navigation, invalid positions, empty or malformed
audio, nonfinite values, all-zero responses, and a failed float-WAV round trip.
It writes temporary `impulse_response.wav` and `verification.json` under
`/tmp/audio-only-navigation-data/verification` (or an external directory supplied
with `--output-dir`). The JSON records the asset
hash, positions, package versions, shape, peak, energy and nonzero sample count.
For persistent experiment data, use `--output-dir "$HOME/data/audio-only-navigation/..."`.
The repository `.gitignore` excludes generated audio, arrays, scenes, datasets,
assets, outputs and third-party directories.

This is a synthetic-scene runtime smoke test. It does not validate perceptual
accuracy, binaural cues, dataset-specific materials, robot physics, visual
rendering, or a trained navigation policy. Acoustic ray tracing is stochastic;
exact impulse response samples need not match across runs.

Expected warnings are the archived Gym deprecation notice, Mesa's inability to
write its optional shader cache under the managed filesystem, and Habitat's
informational messages about absent semantic annotations. The `NUMBA_CACHE_DIR`
setting avoids a known Numba cache locator error when librosa is imported from
the Conda site-packages installation.
