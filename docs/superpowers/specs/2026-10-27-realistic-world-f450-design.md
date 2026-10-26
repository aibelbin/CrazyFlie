# Realistic world, F450 drone and better viewer: design

Goal: take the follow-and-avoid policy from a toy sim toward a real DIY F450 (ArduPilot, depth camera,
YOLO person detection), and make the demo look like a real environment.

## Decisions
- Physics and training stay in PyBullet (headless, CPU, the PC's 16 cores). No GPU-engine port for now.
- Visuals are a separate job: a three.js viewer in the browser, fed with poses from the sim.
- Sensing stays as ray casts (8 azimuth x 3 elevation, 6 m). On the real drone a depth image is
  downsampled to the same grid, so the sim never renders depth.
- Real hardware target: DIY F450 (about 1.2-1.6 kg), ArduPilot GUIDED mode, velocity setpoints over
  MAVLink at 30 Hz from a companion computer, depth camera (e.g. RealSense / OAK-D).
- Baseline model for demos until the new one is better: `runs/exp_ctrl/final.zip` (collisions 0/1/6%
  on levels 1/2/3, follow error 0.18 m).

## Stage 1: freeze and view
- Copy the winning model into the repo, commit everything outstanding.
- three.js viewer: realistic drone with spinning props, trees, sky, shadows, free orbit camera. Served
  by the existing web server, reachable over Tailscale. Sim streams poses; viewer interpolates.

## Stage 2: the new world
- Terrain heightfield (hills, mountains). Policy sees height above terrain and a downward ray.
- Obstacles: forests with tree colliders, rocks, walls, poles.
- Sudden hazards: falling branches, tumbling debris, swaying trees, a rival cutting in.
- Wind: steady wind plus gusts, stronger near hills; felt only through the drone's motion.
- Drone: F450 mass, thrust and motor lag; ArduPilot-style velocity controller; sensor noise and delay.
- Larger map; the person and pedestrians walk longer routes.

## Stage 3: retrain and validate
- Curriculum: flat and calm -> hills -> wind -> sudden hazards.
- 4 seeds on the PC, evaluate on fresh episodes (200 per level), keep the best by collisions.
- Final check against ArduPilot SITL.

## Safety (real flights)
Tethered or netted first flights, away from people, manual kill switch, a hard stop outside the
learned policy (geofence and minimum-distance stop).

## Out of scope for now
GPU physics engines, camera-image policies, bigger networks (maybe 2 of the 4 seeds later).
