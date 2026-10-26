# Drone follow-and-avoid: design

## Goal
A vision-driven drone brain that follows a person and steers around debris and rival
drones, demonstrated entirely in simulation on a Mac, reusing existing models and
physics wherever possible.

## Decisions (from brainstorming)
- **Simulation only** for v1; the control interface is velocity commands, which is how
  PX4/ArduPilot offboard control works, so hardware is a later swap.
- **Palm gesture dropped from v1.** The on-screen "FOLLOW TARGET" appears when YOLO
  locks onto a person. (MediaPipe's built-in `Open_Palm` class could add the trigger later.)
- **Webcam is the drone's camera.** Pretrained YOLO finds the person; the box is mapped to a
  position in the simulated world, which the drone follows.
- **Avoidance is a trained RL policy** (PPO), not hand-written rules. The reward trades off
  following distance against collisions, so following and avoiding are blended.
- **Compact-state observations**: target direction/distance, a 24-ray range strip (+ its rate
  of change), own velocity, previous action. Trains in minutes on a laptop CPU.
- **3D world, real quadrotor physics** from gym-pybullet-drones (Crazyflie model + PID
  controller), commanded in velocity mode.

## Reused components
| Need | Reused |
|---|---|
| Quadrotor physics, controller, 3D view | gym-pybullet-drones (vendored) on PyBullet |
| RL algorithm | Stable-Baselines3 PPO |
| Person detection | Ultralytics YOLOv8n (pretrained, COCO) |

No pretrained obstacle-avoidance policy was reused: none fits this observation/action
interface, so the PPO policy is trained here.

## Architecture
```
webcam -> PersonTracker (YOLO) -> box_to_world -> Scenario.person
                                                      |
Scenario (person, debris, rival) --> FollowAvoidAviary (Gymnasium env on BaseRLAviary)
      range_scan (PyBullet rays) -->   obs(64) -> policy -> velocity cmd -> PID -> motors
```
- `scenario.py`: NumPy world state mirrored into PyBullet bodies. Debris and rival drones are
  spheres (rays and the analytic collision check agree); the person is visual-only.
- `sensing.py`: 8 azimuths x 3 elevations, 6 m range.
- `env.py`: observation, reward, termination (collision, ground, flip), 3-D velocity action.
- `train.py`: curriculum 0 clear sky -> 1 static debris -> 2 drifting debris + rival drone -> 3 stress.
- `policies.py`: baselines. `naive_follow` (ignores obstacles), `reactive_follow` (potential field).
- `evaluate.py`: collision rate / follow error / survival on fixed seeds.
- `demo_sim.py`, `demo_webcam.py`: recorded side-by-side clips and the live demo.

## Reward (per step)
`+0.3 * (1 - |dist - 2 m| / 4)` follow, `-0.3 * ((1 - nearest_ray)/0.7)^2` proximity,
`-0.02 * |a - a_prev|` smoothness, `-50` and episode end on collision.

## Success criteria
On fixed unseen seeds, the trained policy's collision rate is far below `naive_follow` and
competitive with `reactive_follow`, while keeping the mean follow-distance error small.
Results are reported by `evaluate.py`, not claimed here.

## Known limits
- Obstacles are spheres; the detector interface for them is the simulator itself (no
  vision model detects debris or drones from pixels yet).
- Camera-to-world depth from box height is approximate (pinhole + assumed visible height).
- Simulated sensing is noiseless; no wind. Expect a sim-to-real gap on hardware.
