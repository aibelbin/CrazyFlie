# CLAUDE.md

Read this first. It says what this project is, where it stands, and exactly what to do next.

## The goal right now

**A presentation is due, with no time left for training.** The job is to produce a small, clear
visualization (and a few numbers) of a drone flying a waypoint mission and avoiding obstacles.
Do not start long training runs. Everything below runs on a normal Windows laptop in minutes.

## What the project is

A simulated drone (a 1.5 kg DIY F450 flown by an ArduPilot-style velocity controller, in PyBullet)
flies a **mission leg**: 80-150 m from a start to a waypoint at 5 m/s, through static obstacles
(trees, poles, rocks, buildings, a wall). It mirrors how the real drone will work:

- ArduPilot flies the mission (AUTO). A **supervisor** watches the depth sensor and hands control
  to an **avoider** (GUIDED, velocity commands) when something is within 12 m ahead, then hands back.
- Avoiders compared on the same seeded legs: `straight` (no avoidance, the floor), `potential`
  (2D potential field), `bendy` (simplified ArduPilot BendyRuler, can go around or climb over),
  `rl` / `rl_alone` (a PPO policy with / without the supervisor; needs a trained model).
- Difficulty levels: 0 empty, 1 sparse trees, 2 dense trees + poles + rocks, 3 adds buildings and a wall.

Main files (all in `drone_follow/`):

| File | What |
|---|---|
| `mission_world.py` | the leg: terrain, obstacles, exact clearance |
| `mission_env.py` | `MissionAviary` gym env: track-frame observations, 80-sector depth sensor (20 m), noise, 1-step command delay |
| `mission_policies.py` | `straight_line`, `potential_field`, `BendyRuler`, `Supervisor`, `load_ppo` |
| `mission_evaluate.py` | score a policy on seeded legs; `--table` prints a markdown table |
| `mission_viz.py` | **the presentation picture / GIF** (top + side view, all policies on one leg) |
| `mission_train.py` | PPO trainer (only if a tiny run is really needed; see below) |

Design and plan: `docs/superpowers/specs/2026-10-27-waypoint-mission-avoidance-design.md` and
`docs/superpowers/plans/2026-10-27-waypoint-mission-avoidance.md` (Tasks 1-6 done; Task 7 = runs and results).
The older follow-a-person task (`env.py`, `train*.py`, `web.py` viewer) still works but is not the focus.

## Setup on Windows (PowerShell, from the repo root)

pybullet has no reliable pip wheel on Windows (pip tries to compile it), so take it from conda-forge.
Install Miniforge (https://github.com/conda-forge/miniforge) if `conda` is not available, then:

```powershell
conda create -y -n df -c conda-forge python=3.11 pybullet
conda activate df
pip install -r requirements-windows.txt
# make the vendored gym-pybullet-drones importable (a .pth file in site-packages)
python -c "import sysconfig,os; open(os.path.join(sysconfig.get_paths()['purelib'],'gpd.pth'),'w').write(os.path.abspath('vendor/gym-pybullet-drones'))"
python -m pytest tests -q
```

Always run modules from the repo root as `python -m drone_follow.<module>`. This code has been run on
macOS and Linux (WSL), not yet on native Windows: if a test fails there, fix the Windows issue (paths,
encodings, multiprocessing) rather than skipping it. Known Windows points already handled: matplotlib
uses the file-only `Agg` backend, memory logging (`/proc`) quietly returns nothing on Windows, and
`mission_evaluate --table` expands `*.json` itself because PowerShell does not.

## Do this next (in order)

1. **Pictures for the slides** (seconds each):

   ```powershell
   python -m drone_follow.mission_viz --level 3 --pick --out videos/mission_L3.png --gif videos/mission_L3.gif
   python -m drone_follow.mission_viz --level 2 --pick --out videos/mission_L2.png --gif videos/mission_L2.gif
   python -m drone_follow.mission_viz --level 1 --pick --out videos/mission_L1.png
   ```
   `--pick` finds a leg where the straight line crashes and BendyRuler gets through. Use `--seed N` to
   choose a leg yourself. Thick line = avoider in control, X = crash. A sample is in `docs/img/mission_L3.png`.

2. **A few numbers** (about 5-10 minutes total; 50 legs per policy and level, not the planned 500):

   ```powershell
   foreach ($p in "straight","potential","bendy") { foreach ($l in 0..3) {
     python -m drone_follow.mission_evaluate --policy $p --level $l --legs 50 --out "results/mission/${p}_L$l.json" } }
   python -m drone_follow.mission_evaluate --table "results/mission/*.json"
   ```
   Put the table in `README.md` (section "Waypoint missions") with one line on the noise: with 50 legs a
   crash rate near 10% is only good to about +-8 points.

3. **RL is optional.** Two 5M-step PPO runs were training on the owner's PC (`gpu-sys`, WSL, folder
   `~/DetectionAndAvoidance/runs/mission_s1` and `mission_s2`) when it lost power; their `best.zip` /
   `latest.zip` may be on that disk. If that PC is back, copy a `best.zip` into `runs/` here and add
   `rl` to the commands above (`--policies straight bendy rl --model runs/mission_s1/best.zip`).
   If not, either leave RL out of the presentation (the classical comparison stands on its own) or, only
   if asked, do a **tiny** run, clearly labelled as such (about 10-20 minutes on a laptop; expect a weak policy):

   ```powershell
   python -m drone_follow.mission_train --out runs/mission_tiny --seed 1 --total-steps 300000 --envs 4 --threads 2 --eval-every 100000 --eval-legs 10 --max-hours 1
   ```
   Never change `mission_env.py`'s observation layout after training: saved models depend on it.

## What is known so far (be honest about it in the slides)

- A quick check with 6 legs per level (not significant) gave: straight line crashes on every leg at
  levels 1-3; BendyRuler under the supervisor had no crashes at levels 0-3; the potential field crashed on
  1-2 of 6 legs and often timed out at level 3 (it does not climb and gets stuck in front of walls).
- The depth sensor is modelled as 80 sectors (16 directions x 5 elevations), each reporting the nearest
  surface in that sector up to 20 m, with noise, dropouts and 66 ms delay. The real drone's sensor is not
  chosen yet, so this is a generic stand-in.
- Not done: ArduPilot SITL, real hardware, moving obstacles.

## House rules

- Tests: `python -m pytest tests -q` must pass before committing.
- Do not commit `runs/` or `videos/` (git-ignored); commit `results/mission/*.json` and README changes.
- Keep replies short and direct: the owner prefers that.
