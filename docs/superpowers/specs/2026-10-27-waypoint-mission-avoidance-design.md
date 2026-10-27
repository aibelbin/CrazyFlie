# Waypoint missions with obstacle avoidance: design

## Goal
The real drone flies distant waypoint missions under ArduPilot. An avoidance layer runs alongside
it and steers around obstacles the mission did not plan for. This round builds that task in
simulation and finds out which avoider works best: a classical one or a trained RL policy.

## Why the change
The follow-the-person F450 runs (`runs/main_s*`, 2026-10-27) plateaued within 4-9M steps at
~20-30% collisions on levels 3 and 5, and the earlier Crazyflie runs showed a hand-written
potential-field rule beating PPO (2% vs 12.5% collisions). So RL has to earn its place against
classical avoiders on the task the drone will actually fly.

## Decisions (from brainstorming)
- **Task:** waypoint missions with legs far longer than the sensor range, not following a person.
- **Obstacles:** static only (trees, poles, rocks, buildings, terrain). No moving obstacles this round.
- **Manoeuvre:** go around or climb over, whichever is shorter.
- **Cruise speed:** 5 m/s (ArduPilot's default `WPNAV_SPEED`).
- **Sensor:** not chosen yet. A generic range ring stands in for it, extended to 20 m: at 5 m/s,
  stopping alone needs ~4-5 m and the sensor and command delays add more.
- **Approach:** build classical and RL avoiders on the same scenarios and compare them. RL is only
  kept if it clearly beats the best classical one.

## 1. Sim task: `MissionAviary`
- New environment beside `FollowAvoidAviary`, sharing the F450 model and velocity controller
  (`f450.py`), wind (`WindField`), terrain and static scenery (`world.py`), and sensor noise and delay.
  No person, pedestrians, rival drones or falling debris.
- **Episode = one leg:** start, then a waypoint 80-150 m away. The world is a corridor of about
  200 m x 60 m along the leg, with terrain sized to match.
- **Difficulty levels** (obstacles placed near the straight-line path):
  - 0: none
  - 1: sparse trees
  - 2: dense trees and poles
  - 3: level 2 plus buildings and walls, some of which are shorter to climb over than to go around
- **Speed:** the commanded speed limit rises from 2.5 to 6 m/s for this task (cruise 5 m/s).
- **Sensor:** the existing horizontal range ring out to 20 m, plus a few rays angled up and down,
  so the drone can tell whether climbing over is possible. Noise, dropouts and delay as now.
- **Outcome:**
  - Success: within 2 m of the waypoint.
  - Failure: a collision, or more than 3x the straight-line time at cruise speed.

## 2. Mirroring ArduPilot: the supervisor
- A supervisor flies straight at the waypoint at cruise speed (ArduPilot AUTO) until the nearest
  range reading drops below a trigger distance (~12 m).
- It then hands control to the avoider (ArduPilot GUIDED, velocity commands over MAVLink) and takes
  control back once the direct line to the waypoint is clear again.
- Every avoider runs under the same supervisor, so the comparison is fair and the handover logic is
  tested in sim before it meets ArduPilot.

## 3. Controllers
All share one interface: observation in, velocity command out.

| Controller | What it does |
|---|---|
| Straight line | Flies at the waypoint and ignores obstacles. The floor. |
| Potential field | `reactive_follow` from `policies.py`, retargeted from the person to the waypoint. |
| BendyRuler-style | Samples candidate headings and climb angles, rejects any that the ranges show blocked within a lookahead distance plus margin, and picks the clear one closest to the waypoint direction. Modelled on ArduPilot's BendyRuler. |
| RL (PPO) | Sees ranges, direction and distance to the waypoint, own velocity and previous action, and outputs a velocity command. Reward: progress toward the waypoint, a large crash penalty and a small smoothness penalty. |

## 4. Evaluation
- One fixed, seeded set of 500 legs per difficulty level, the same for every controller.
- Metrics, in priority order: crash rate, success rate, extra distance and time against the
  straight line, smoothness (command changes and jerk).
- RL training: short runs of 5M steps, 2 seeds, on gpu-sys (never the Mac), with `progress.json`
  memory logging on. Evaluation runs on gpu-sys too.
- Results are written to `results/` as JSON plus a short table in the README.

## 5. Testing
- Unit tests for scenario generation (obstacles respect the levels and keep start and waypoint clear),
  the supervisor's handover rules and each classical controller on a hand-built case (for example a
  single wall ahead makes BendyRuler turn or climb, not stop).
- A quick smoke test on the Mac may build the env and step it a few times; no training there.

## Out of scope this round
ArduPilot SITL, real hardware, moving obstacles, choosing the real sensor.
