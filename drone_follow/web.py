"""Web viewer.

    python -m drone_follow.web --model runs/ppo_ft/final.zip --port 8765 \
        --runs-dir runs --runs long_a,long_b
    # then open http://<this machine>:8765

  /       training dashboard: each run's live training episode plus progress and score charts
  /play   a finished model flying in a 3D browser scene (three.js), with buttons to add hazards.
          The simulation runs here and streams poses (server-sent events); the browser draws them.

The training frames come from one worker per run, which only renders while this page is open
(it checks a `watch` file this server touches). The /play simulation pauses a few seconds after
its last viewer leaves, so an idle server costs almost nothing. There is no login: only expose
the port on a network you trust (e.g. a tailnet).
"""
import argparse
import json
import os
import queue
import threading
import time

import numpy as np
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from drone_follow import config as C
from drone_follow.env import FollowAvoidAviary, SingleDroneWrapper
from drone_follow.policies import naive_follow, reactive_follow, smoothed
from drone_follow.render import status_from

IDLE_AFTER_S = 8.0
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
STATIC_TYPES = {"three.min.js": "text/javascript", "OrbitControls.js": "text/javascript",
                "viewer.js": "text/javascript", "viewer.html": "text/html; charset=utf-8"}

STYLE = """
 :root{color-scheme:dark light;--bg:#0f1115;--fg:#e8eaed;--card:#1a1d24;--acc:#4c8dff;--mut:#9aa0a6;
  --a:#ff7a59;--b:#4cc38a;--c:#b58cff;--d:#f2c94c}
 @media (prefers-color-scheme:light){:root{--bg:#f4f5f7;--fg:#1c1e22;--card:#fff;--acc:#1a5fe0;--mut:#5f6368;
  --a:#d4491f;--b:#1a8f5a;--c:#7a4fd0;--d:#b7860b}}
 body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.4 system-ui,sans-serif}
 main{max-width:1100px;margin:0 auto;padding:16px}
 h1{font-size:18px;margin:0 0 4px} h2{font-size:16px;margin:0 0 8px}
 a{color:var(--acc)} .mut{color:var(--mut)}
 img{width:100%;border-radius:10px;background:#000;display:block}
 .row{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0}
 button,select{background:var(--card);color:var(--fg);border:1px solid #8884;border-radius:8px;
   padding:9px 12px;font:inherit}
 button:active{background:var(--acc);color:#fff} .on{outline:2px solid var(--acc)}
 label{color:var(--mut);display:flex;align-items:center;gap:6px}
 .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:16px}
 @media (max-width:800px){.grid{grid-template-columns:1fr}}
 .card{background:var(--card);border-radius:12px;padding:12px}
 .bar{height:10px;border-radius:6px;background:#8883;overflow:hidden;margin:6px 0}
 .bar>i{display:block;height:100%;background:var(--acc)}
 .kv{display:grid;grid-template-columns:auto auto;gap:2px 14px;justify-content:start;margin:8px 0}
 .kv b{font-weight:600} canvas{width:100%;height:170px;display:block}
"""

DASH_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Drone training dashboard</title><style>%STYLE%</style></head><body><main>
<h1>Drone follow-and-avoid: training</h1>
<div class="mut">Live episodes from one training worker per run (random exploration early on, sped up),
 plus scores on fixed unseen scenarios. <a href="/play">Watch a finished model in 3D</a></div>
<div class="grid" id="runs" style="margin-top:12px"></div>
<script>
const RUNS=%RUNS%, COL=['var(--a)','var(--b)','var(--c)','var(--d)'];
const BASE={L2:0.02,L3:0.12};   // hand-written reactive baseline, collision rate
const fmt=(x,d=1)=>x==null?'–':x.toFixed(d), pct=x=>x==null?'–':(100*x).toFixed(0)+'%';
const hrs=h=>h==null?'–':(h<1?Math.round(h*60)+' min':h.toFixed(1)+' h');
document.getElementById('runs').innerHTML=RUNS.map((r,i)=>`
 <div class="card"><h2 style="color:${COL[i%4]}">${r.title}</h2>
  <img src="/train/${r.id}/stream" alt="live training episode">
  <div class="bar"><i id="bar-${r.id}" style="width:0%"></i></div>
  <div class="kv" id="kv-${r.id}"></div>
  <canvas id="ch-${r.id}" width="600" height="220"></canvas>
  <div class="mut" id="best-${r.id}"></div></div>`).join('');

function chart(id,evals,total,color){
 const c=document.getElementById('ch-'+id),g=c.getContext('2d'),W=c.width,H=c.height,P=34;
 g.clearRect(0,0,W,H);g.font='12px system-ui';g.fillStyle=getComputedStyle(document.body).color;
 const ymax=Math.max(0.5,...evals.map(e=>Math.max(e.L2_collision,e.L3_collision)))*1.05;
 const X=s=>P+(W-P-8)*s/total,Y=v=>H-P+4-(H-P-14)*v/ymax;
 g.globalAlpha=.35;g.strokeStyle=g.fillStyle;g.beginPath();g.moveTo(P,10);g.lineTo(P,H-P+4);g.lineTo(W-6,H-P+4);g.stroke();
 for(let v=0;v<=ymax;v+=0.1){g.fillText(Math.round(v*100)+'%',2,Y(v)+4)}
 g.globalAlpha=1;g.fillText('training steps',W/2-30,H-6);g.fillText((total/1e6)+'M',W-34,H-P+18);
 for(const [k,key,dash] of [['L2','L2_collision',[]],['L3','L3_collision',[5,4]]]){
  g.strokeStyle=color;g.lineWidth=2;g.setLineDash(dash);g.beginPath();
  evals.forEach((e,j)=>{const x=X(e.steps),y=Y(e[key]);j?g.lineTo(x,y):g.moveTo(x,y)});g.stroke();
  g.setLineDash([2,5]);g.globalAlpha=.5;g.lineWidth=1;g.beginPath();g.moveTo(P,Y(BASE[k]));g.lineTo(W-6,Y(BASE[k]));g.stroke();g.globalAlpha=1}
 g.setLineDash([]);g.fillText('solid = level 2 · dashed = level 3 · dotted = hand-written baseline',P+4,H-18)}

async function poll(){
 for(const r of RUNS){
  try{const d=await (await fetch('/train/'+r.id+'/progress')).json(),p=d.progress||{};
   document.getElementById('bar-'+r.id).style.width=(p.pct||0)+'%';
   const lag=p.updated_at?Math.round(Date.now()/1000-p.updated_at):null;
   document.getElementById('kv-'+r.id).innerHTML=
    `<span class="mut">steps</span><b>${fmt((p.steps||0)/1e6,2)}M / ${fmt((p.total||0)/1e6,0)}M (${fmt(p.pct,1)}%)</b>
     <span class="mut">speed</span><b>${fmt(p.sps,0)} steps/s</b>
     <span class="mut">time</span><b>${hrs(p.elapsed_h)} in · ETA ${hrs(p.eta_h)} ${p.budget_h>1000?'(no time limit)':'(limit '+hrs(p.budget_h)+')'}</b>
     <span class="mut">stage</span><b>${p.phase||'–'}</b>
     <span class="mut">training collisions</span><b>${pct(p.train_collision_rate)} of last ${p.n_recent||0} episodes</b>
     <span class="mut">avg reward / length</span><b>${fmt(p.ep_reward,1)} / ${fmt(p.ep_len_s,1)} s</b>
     <span class="mut">status</span><b>${lag==null?'waiting for first update':(lag>90?'STALE: no update for '+lag+' s':'running')}</b>`;
   chart(r.id,d.evals||[],p.total||65e6,COL[RUNS.indexOf(r)%4]);
   const b=d.best;document.getElementById('best-'+r.id).textContent=b?
    `best so far (step ${fmt(b.steps/1e6,1)}M): level 2 ${pct(b.L2_collision)}, level 3 ${pct(b.L3_collision)} collisions`:'no evaluation yet (first at 1M steps)';
  }catch(e){}
 }
 setTimeout(poll,3000)}
poll();
</script></main></body></html>"""


def _r(v, n=3):
    return [round(float(x), n) for x in v]


def world_snapshot(sc, episode_id: int) -> bytes:
    """Static scenery for the viewer (fetched once per episode): terrain heights, trees and rocks."""
    terrain = None
    if sc.terrain is not None:
        from drone_follow.world import GRID, HALF
        terrain = {"half": HALF, "n": GRID, "z": [round(float(v), 2) for v in sc.terrain.grid().flatten()]}
    statics = [{"k": str(k), "p": _r(pp, 2), "r": round(float(r), 2), "g": int(g)}
               for pp, r, k, g in zip(sc.static_pos, sc.static_r, sc.static_kind, sc.static_group)]
    return json.dumps({"ep": episode_id, "terrain": terrain, "statics": statics}, separators=(",", ":")).encode()


class Sim:
    """Owns the /play simulation. Only its own thread touches PyBullet; handlers send commands.
    Each control step it publishes a JSON snapshot of every pose for the browser to draw."""

    def __init__(self, policies: dict, current: str, difficulty: int, seed: int):
        self.policies, self.current = policies, current
        self.difficulty, self.seed = difficulty, seed
        self.cmds: queue.Queue = queue.Queue()
        self.cond = threading.Condition()
        self.state, self.frame_no = "", 0
        self.last_seen = 0.0                      # starts idle until someone opens /play
        self.status, self.episodes, self.collisions = "STARTING", 0, 0
        self.episode_id = 0
        self.world_json = b"{}"

    def touch(self):
        self.last_seen = time.time()

    def publish(self, sim):
        sc = sim.scenario
        obs = [{"id": i, "k": o.kind, "p": _r(o.pos), "v": _r(o.vel), "r": round(float(o.radius), 3),
                "a": int(o.active), "g": o.group} for i, o in enumerate(sc.obstacles)]
        wind = sim._wind.mean + sim._wind.dev if sim._wind is not None else np.zeros(3)
        msg = {"ep": self.episode_id, "drone": {"p": _r(sim.pos[0]), "q": _r(sim.quat[0], 4), "v": _r(sim.vel[0]),
                                                  "gz": round(sc.ground(sim.pos[0][:2]), 3)},
               "person": {"p": _r(sc.person), "v": _r(sc.person_vel), "gz": round(sc.ground(sc.person), 3)},
               "wind": _r(wind, 2), "obs": obs,
               "status": self.status, "policy": self.current, "difficulty": self.difficulty,
               "episodes": self.episodes, "collisions": self.collisions}
        with self.cond:
            self.state, self.frame_no = json.dumps(msg, separators=(",", ":")), self.frame_no + 1
            self.cond.notify_all()

    def run(self):
        env = SingleDroneWrapper(FollowAvoidAviary(
            gui=False, difficulty=self.difficulty, seed=self.seed, episode_sec=1e9))
        sim = env.unwrapped

        def new_episode():
            self.episode_id += 1
            obs, _ = env.reset()
            self.status = "FOLLOW TARGET"
            self.world_json = world_snapshot(sim.scenario, self.episode_id)
            return obs

        obs = new_episode()
        hold = 0
        while True:
            t0 = time.perf_counter()
            if time.time() - self.last_seen > IDLE_AFTER_S:      # nobody watching: sleep
                time.sleep(0.25)
                continue
            while not self.cmds.empty():
                cmd = self.cmds.get_nowait()
                if cmd == "debris":
                    sim.scenario.spawn_debris_ahead(sim.pos[0])
                elif cmd == "rival":
                    sim.scenario.spawn_rival(sim.pos[0])
                elif cmd == "pedestrian":
                    sim.scenario.add_pedestrian()
                elif cmd == "new":
                    obs, hold = new_episode(), 0
                elif cmd.startswith("difficulty/"):
                    env.set_difficulty(int(cmd.split("/")[1]))
                    obs, hold = new_episode(), 0
                elif cmd.startswith("policy/") and cmd.split("/")[1] in self.policies:
                    self.current = cmd.split("/")[1]
            if hold > 0:
                hold -= 1
                if hold == 0:
                    obs = new_episode()
            else:
                obs, _, term, trunc, info = env.step(self.policies[self.current](obs))
                self.status = status_from(info, term)
                if term or trunc:
                    self.episodes += 1
                    self.collisions += int(info["collision"])
                    hold = C.CTRL_FREQ * 3 // 2                   # linger on the crash for 1.5 s
            self.publish(sim)
            time.sleep(max(0.0, 1.0 / C.CTRL_FREQ - (time.perf_counter() - t0)))


def read_json(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def read_jsonl(path: str) -> list:
    out = []
    try:
        with open(path) as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    except OSError:
        pass
    return out


def make_handler(sim: Sim, runs_dir: str, runs: list):
    def current_runs() -> list:
        """The runs to show: the fixed list, or (if empty) every exp_*/main_* folder found."""
        if runs:
            return runs
        try:
            names = sorted(d for d in os.listdir(runs_dir)
                           if d.startswith(("exp_", "main_")) and os.path.isdir(os.path.join(runs_dir, d)))
        except OSError:
            names = []
        return [{"id": n, "title": n.replace("_", " ")} for n in names]

    def dash_page() -> str:
        return DASH_PAGE.replace("%STYLE%", STYLE).replace("%RUNS%", json.dumps(current_runs()))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _mjpeg_headers(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

        def _frame(self, jpeg: bytes):
            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                             + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")

        def _static(self, name: str):
            if name not in STATIC_TYPES:
                self._send(404, b"not found", "text/plain")
                return
            with open(os.path.join(STATIC_DIR, name), "rb") as f:
                self._send(200, f.read(), STATIC_TYPES[name])

        def do_GET(self):
            path = self.path.split("?")[0]
            parts = path.strip("/").split("/")
            run_ids = {r["id"] for r in current_runs()}
            if path == "/":
                self._send(200, dash_page().encode(), "text/html; charset=utf-8")
            elif path == "/play":
                sim.touch()
                self._static("viewer.html")
            elif len(parts) == 2 and parts[0] == "static":
                self._static(parts[1])
            elif path == "/api/state":
                sim.touch()
                self._send(200, json.dumps({
                    "status": sim.status, "policy": sim.current, "episodes": sim.episodes,
                    "collisions": sim.collisions}).encode())
            elif path == "/api/world":
                self._send(200, sim.world_json)
            elif path == "/events":                       # server-sent events: one pose snapshot per step
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()
                last = -1
                try:
                    while True:
                        with sim.cond:
                            sim.cond.wait_for(lambda: sim.frame_no != last, timeout=5)
                            data, last = sim.state, sim.frame_no
                        sim.touch()
                        self.wfile.write(b"data: " + data.encode() + b"\n\n" if data else b": keepalive\n\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
            elif len(parts) == 3 and parts[0] == "train" and parts[1] in run_ids and parts[2] == "progress":
                d = os.path.join(runs_dir, parts[1])
                self._send(200, json.dumps({
                    "progress": read_json(os.path.join(d, "progress.json")),
                    "evals": read_jsonl(os.path.join(d, "eval_log.jsonl")),
                    "best": read_json(os.path.join(d, "best.json"))}).encode())
            elif len(parts) == 3 and parts[0] == "train" and parts[1] in run_ids and parts[2] == "stream":
                d = os.path.join(runs_dir, parts[1])
                flag, live = os.path.join(d, "watch"), os.path.join(d, "live.jpg")
                self._mjpeg_headers()
                last_m, last_flag = 0.0, 0.0
                try:
                    while True:
                        now = time.time()
                        if now - last_flag > 2.0:          # tells the training worker to render
                            try:
                                with open(flag, "w") as f:
                                    f.write(str(now))
                            except OSError:
                                pass
                            last_flag = now
                        try:
                            m = os.path.getmtime(live)
                        except OSError:
                            m = 0.0
                        if m and m != last_m:
                            with open(live, "rb") as f:
                                data = f.read()
                            last_m = m
                            if data:
                                self._frame(data)
                        time.sleep(0.08)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            sim.touch()
            path = self.path.partition("?")[0]
            if path.startswith("/api/"):
                cmd = path[len("/api/"):]
                if cmd.startswith("difficulty/"):
                    sim.difficulty = min(max(int(cmd.split("/")[1] or 2), 0), 5)
                    cmd = f"difficulty/{sim.difficulty}"
                sim.cmds.put(cmd)
            self._send(200, b"{}")

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="trained PPO .zip for /play")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--smooth", type=float, default=0.0,
                    help="low-pass the trained policy's commands (0 = off, 0.5 = recommended)")
    ap.add_argument("--difficulty", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs-dir", default="runs", help="folder containing the training run folders")
    ap.add_argument("--runs", default="auto", help="comma list of run folders to show, or auto")
    args = ap.parse_args()

    policies = {"reactive": reactive_follow, "naive": naive_follow}
    if args.model:
        from drone_follow.evaluate import load_ppo_policy
        policies["trained"] = load_ppo_policy(args.model)
        if args.smooth > 0:
            policies["trained"] = smoothed(policies["trained"], args.smooth)
    sim = Sim(policies, "trained" if "trained" in policies else "reactive", args.difficulty, args.seed)
    threading.Thread(target=sim.run, daemon=True).start()
    names = [r for r in args.runs.split(",") if r and r != "auto"]      # "auto" = discover exp_*/main_*
    runs = [{"id": r, "title": r.replace("_", " ")} for r in names]
    server = ThreadingHTTPServer((args.host, args.port), make_handler(sim, args.runs_dir, runs))
    print(f"serving on http://{args.host}:{args.port}  runs={[r['id'] for r in runs]}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
