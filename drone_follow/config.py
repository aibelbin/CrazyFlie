"""Shared constants for the follow-and-avoid environment (SI units, world frame)."""

# Simulation rates
PYB_FREQ = 240
CTRL_FREQ = 30

# Flight
SPEED_LIMIT = 1.5          # m/s, magnitude of the commanded velocity at |action| = 1
FOLLOW_ALT = 1.2           # m, altitude of the point the drone tracks above the person
FOLLOW_DIST = 2.0          # m, desired 3D distance to the follow point
DRONE_RADIUS = 0.09        # m, collision radius used for sphere obstacles
GROUND_Z = 0.06            # m, below this the drone counts as crashed
MAX_TILT = 1.2             # rad, beyond this the drone counts as flipped

# World
ARENA_PERSON = 3.5         # m, person walks inside +-this square
ARENA_BOUND = 8.0          # m, drone leaving +-this square truncates the episode
Z_MAX = 3.5

# Range sensing (a simple depth/lidar-style strip around the drone)
N_AZ = 8
ELEVS_DEG = (-30.0, 0.0, 30.0)
MAX_RANGE = 6.0
RAY_START_OFFSET = 0.12

# Episode
EPISODE_SEC = 20.0

# Reward
R_FOLLOW = 0.3
R_COLLISION = -50.0
R_PROXIMITY = 0.3
PROXIMITY_START = 1.0      # m, penalty begins when the nearest ray hit is closer than this
R_SMOOTH = 0.02
