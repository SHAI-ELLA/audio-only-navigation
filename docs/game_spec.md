# Minimal audio-only navigation game

This milestone defines a small interactive task: an agent moves through the
bundled `simple_room.glb` scene toward one stationary sound source. The agent
receives audio only. It can repeat the cycle

```python
observation = env.reset()
observation, reward, done, info = env.step(action)
```

The interface uses the four-value step convention from the pinned Gym-era
ecosystem (`observation`, `reward`, `done`, `info`). It intentionally does not
depend on Gym spaces or a learning framework; future preprocessing can consume
the raw audio independently.

## Why this is partially observable

The simulator knows the whole world: where the agent is, where the source is,
the scene geometry, and paths through the scene. A policy does not receive that
information. It hears an acoustic impulse response from its current listener
pose. Different places can produce similar audio, so one observation does not
uniquely identify the complete state. This is a partially observable Markov
decision process (POMDP).

The distinction is explicit in the implementation:

- Simulator/internal state is held by Habitat-Sim and the environment: agent
  position and orientation, fixed source position, scene, navmesh and geodesic
  path calculations.
- Agent observation is a dictionary with exactly one key, `audio`. Its value is
  the raw finite, non-empty two-channel binaural impulse response as a NumPy
  array. It contains
  no coordinates, distance, map, GPS, compass, RGB or depth.
- Debug/info/metrics are returned separately in `info`. They include positions,
  yaw and geodesic distance so a human can inspect an episode. They are not part
  of the policy observation and must not be passed to a policy.

The pinned Habitat-Sim branch also needs a tiny internal depth sensor to load
scene geometry and initialize its renderer. The environment creates that sensor
at 16×16 resolution, but never returns its value.

## Actions and mechanics

The action is one of:

- `MOVE_FORWARD`: attempt to move forward by `move_distance` meters. Habitat-Sim
  filters the whole attempted segment through the navmesh (`PathFinder.try_step`),
  so the action cannot cross non-navigable geometry. The returned position may
  be clipped or slide along collision geometry according to that API.
- `TURN_LEFT`: rotate counter-clockwise by `turn_angle_degrees`.
- `TURN_RIGHT`: rotate clockwise by `turn_angle_degrees`.
- `STOP`: end the episode immediately.

Both distances and angles are fields of `AudioNavConfig`; the defaults are 0.25
meters and 30 degrees. There is no backward movement, strafing or continuous
control.

## Episode lifecycle and outcomes

`reset()` samples a navigable start and a different connected navigable source
point in the bundled room, initializes the agent orientation, fixes the source,
and returns the first audio observation. The environment seeds the pathfinder
once at construction; successive resets advance its random sequence, while
equivalent environments with the same seed reproduce that sequence. Every
non-terminal action updates the pose if needed and produces a fresh audio
observation.

`STOP` succeeds when the internally computed geodesic distance to the source is
at most `success_distance` (default 1.0 meter). A successful STOP returns reward
`+1.0`. A STOP outside the radius returns reward `0.0` and fails. Every ordinary
step has the configurable `step_penalty` (default `-0.01`). Reaching
`max_steps` (default 50) ends the episode as a timeout failure. There is no
sound-intensity reward or geodesic reward shaping in this milestone.

## Deliberate scope limits

The current audio value is a two-channel binaural acoustic impulse response, not a microphone-like
source waveform. Source-audio convolution is deliberately deferred to a later
design step.

This is an environment smoke test, not an agent or dataset pipeline. It does
not include PPO, reinforcement-learning training, neural networks, CL1,
spectrograms, waveform preprocessing, learned audio representations, RGB/depth
policy observations, large scene downloads, or a navigation episode dataset.
The `simple_room.glb` asset is tiny and synthetic, lacks semantic acoustic
materials, and is useful for API and control verification rather than realistic
research conclusions. Before generating a larger dataset, revisit scene and
material coverage, source placement distributions, episode reproducibility,
collision/movement semantics, observation length conventions, and whether
success and reward definitions should vary across experimental conditions.
