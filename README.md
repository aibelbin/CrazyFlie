# Drone follow-and-avoid

A simulated quadrotor that follows a walking person and steers around debris, pedestrians
and rival drones. The avoidance behaviour is a **PPO policy trained in simulation**; the
person is found with a **pretrained YOLO** model. Physics come from
[gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones) (vendored in `vendor/`).

> **Hardware note:** training and other heavy jobs do not run on the laptop (a fanless
> MacBook Air). Training is done on a remote machine and only the results come back.
> Running, evaluating a few episodes and the live demo are light and fine locally.

## Live demo (3D window)

```
source .venv/bin/activate
python -m drone_follow.live --model runs/ppo_ft/final.zip
```

The drone flies in real time with a walking person to follow, drifting debris, walking
pedestrians and rival drones. Click the window, then:

| Key | Action |
|---|---|
| `o` | drop debris between the drone and the person |
| `r` | launch a rival drone at the drone |
| `p` | add a walking pedestrian |
| `1` / `2` / `3` | trained policy / hand-written reactive policy / naive follower |
| `n` | new episode |
| `c` | toggle the auto-follow camera |

Colours: person green, drone blue (with trail), debris grey, rival drones red.
A label over the drone shows `FOLLOW TARGET`, `AVOIDING` or `COLLISION`.

## How it works

```
webcam / video --> YOLO (person) --> world position --+
                                                      v
 scenario (person, debris, pedestrians, rivals) --> FollowAvoidAviary --> reward
     24-ray range strip ----------------------------> obs (64) --> PPO policy
                                              velocity command <------+
                                              PID controller --> motor RPMs (real quadrotor physics)
```

- **Observation (64 numbers):** direction/distance to the person, person velocity, own
  velocity and tilt, altitude, last action, a 24-ray range strip (8 azimuths x 3 elevations,
  6 m) and its rate of change.
- **Action:** a 3-D velocity command (the library's PID controller flies the motors).
- **Reward:** stay about 2 m from the person, `-50` and end the episode on a collision,
  small penalties for getting close to obstacles and for jerky commands.
- **Hazards:** spheres for debris and rival drones, three-sphere columns for pedestrians
  (drawn as walking figures in the live demo). The person and scenery are visual only.
- **Difficulty levels:** 0 clear sky, 1 static debris, 2 drifting debris + pedestrians + a
  rival drone, 3 stress (denser debris, more people, two faster rivals).

## Results

200 unseen scenarios per level, pedestrians included (`results/`):

| Collision rate | Level 2 | Level 3 | Follow error |
|---|---|---|---|
| Naive follow (ignores obstacles) | 70% | 78% | 0.26 m |
| Hand-written reactive avoidance | 2% | 12% | 0.33 m |
| Trained PPO, first model (`ppo_v1`) | 13.5% | 17.5% | 0.37-0.39 m |
| Trained PPO, fine-tuned (`ppo_ft`) | 12.5% | 21.5% | 0.29 m |
| Trained PPO, +3M more steps at mixed levels 2-3 (`ppo_mix`) | 34.5% | 43.5% | 0.36-0.38 m |

The trained policy avoids obstacles far better than not avoiding them, but a simple
potential-field rule using the same range strip still beats it. At 200 episodes each
estimate carries roughly +-5 points of noise, so `ppo_v1` and `ppo_ft` are tied on
collisions; `ppo_ft` follows more tightly. Continuing training (`ppo_mix`) made the policy
clearly worse, so `ppo_ft` is the model to use; the cause was not investigated.

## Layout

| Path | What |
|---|---|
| `drone_follow/env.py` | Gymnasium environment on `BaseRLAviary` |
| `drone_follow/scenario.py`, `walker.py` | the world: person, debris, pedestrians, rivals, walking figures, scenery |
| `drone_follow/sensing.py` | ray-cast range strip |
| `drone_follow/policies.py` | naive and reactive baselines |
| `drone_follow/train.py` | PPO training (curriculum, `--resume`, `--mix`) |
| `drone_follow/evaluate.py` | collision rate / follow error on fixed seeds |
| `drone_follow/live.py` | live 3-D demo |
| `drone_follow/demo_sim.py` | side-by-side clips of policies on the same scenario |
| `drone_follow/demo_webcam.py`, `person_tracker.py` | webcam or video -> YOLO -> simulated drone |
| `docs/superpowers/specs/` | design spec |

## Setup

Python 3.11. On macOS `pip install pybullet` fails to compile (the bundled zlib redefines
`fdopen`); download the sdist, change line 127 of
`examples/ThirdPartyLibs/zlib/zutil.h` to `#if 0`, and `pip install --no-build-isolation`
the patched folder. On Linux/WSL without a compiler, `conda-forge`/`micromamba` ships a
prebuilt `pybullet`. Then:

```
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
echo "$PWD/vendor/gym-pybullet-drones" > .venv/lib/python3.11/site-packages/gpd.pth
python -m pytest tests -q
```

## Commands

```
python -m drone_follow.evaluate --episodes 50 --difficulties 1 2 3 --model runs/ppo_ft/final.zip
python -m drone_follow.demo_sim --policies naive_follow ppo --model runs/ppo_ft/final.zip --out videos/compare.mp4
python -m drone_follow.demo_webcam --model runs/ppo_ft/final.zip --source 0      # or a video path
python -m drone_follow.train --out runs/ppo --resume runs/ppo_ft/final.zip --lr 1e-4 --envs 6 --mix 2,3 --steps 3000000   # run on a remote machine
```

## Limits

- Debris and rival drones are detected by the simulator's range sensing, not by a vision
  model looking at pixels; only the person is found by a real detector (YOLO).
- Camera-to-world depth from the person's box height is approximate.
- Sensing is noiseless and there is no wind, so expect a sim-to-real gap on hardware.
- The palm-gesture trigger from the original idea is not built; MediaPipe's `Open_Palm`
  class could add it.
