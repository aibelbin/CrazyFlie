"""Shared constants for the follow-and-avoid environment (SI units, world frame)."""
import os

# Simulation rates
PYB_FREQ = 240
CTRL_FREQ = 30

# Flight
DRONE = os.environ.get("DF_DRONE", "cf2x")     # "cf2x" (Crazyflie, library controller) or "f450" (1.5 kg DIY quad)
F450 = DRONE == "f450"
WIND_MEAN = float(os.environ.get("DF_WIND", "0"))          # m/s steady wind (f450 only)
WIND_GUST = float(os.environ.get("DF_GUST", "0"))          # m/s gust strength (f450 only)
SPEED_LIMIT = 2.5 if F450 else 1.5    # m/s, magnitude of the commanded velocity at |action| = 1
FOLLOW_ALT = 1.2           # m, altitude of the point the drone tracks above the person
FOLLOW_DIST = 2.0          # m, desired 3D distance to the follow point
DRONE_RADIUS = 0.32 if F450 else 0.09   # m, collision radius used for sphere obstacles (F450: frame + props)
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
R_SMOOTH = 0.08
R_TTC = 0.6                # penalty for a hazard on a collision course (scaled by time-to-collision)
TTC_START = 2.0            # s, penalty begins when predicted time to collision drops below this
R_CLEAR = 0.4              # penalty for being near any hazard surface
CLEARANCE_START = 0.8      # m
REWARD_SCALE = 0.1         # all rewards are multiplied by this, which keeps value targets small

# Experiment switches (environment variables, so a trained model can be run with the same settings).
# Both default to off, which reproduces the original behaviour.
ACTION_RATE_LIMIT = float(os.environ.get("DF_ACTION_RATE_LIMIT", "0"))   # max change of the commanded action per control step (0 = off)
R_TILT = float(os.environ.get("DF_R_TILT", "0"))                         # penalty for leaning toward a flip (0 = off)
TILT_START = 0.35                                                        # rad, tilt penalty starts here (a flip is MAX_TILT)
