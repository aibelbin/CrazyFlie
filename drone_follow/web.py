"""Web viewer: the simulation streamed to a browser, with buttons to add hazards.

    python -m drone_follow.web --model runs/ppo_ft/final.zip --port 8080
    # then open http://<this machine>:8080  (e.g. its Tailscale address)

The simulation runs headless and is rendered on the CPU. It pauses a few seconds after the last
viewer leaves, so an idle server costs almost nothing. There is no login: only expose the port
on a network you trust (e.g. a tailnet).
"""
import argparse
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

from drone_follow import config as C
from drone_follow.drone_visual import DroneVisual
from drone_follow.env import FollowAvoidAviary, SingleDroneWrapper
from drone_follow.policies import naive_follow, reactive_follow
from drone_follow.render import render_frame, status_from

IDLE_AFTER_S = 8.0
FRAME_W, FRAME_H = 800, 450
RENDER_EVERY = 2                      # render every 2nd 30 Hz control step (15 fps)

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Drone follow and avoid</title>
<style>
 :root{color-scheme:dark light;--bg:#0f1115;--fg:#e8eaed;--card:#1a1d24;--acc:#4c8dff;--mut:#9aa0a6}
 @media (prefers-color-scheme:light){:root{--bg:#f4f5f7;--fg:#1c1e22;--card:#fff;--acc:#1a5fe0;--mut:#5f6368}}
 body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.4 system-ui,sans-serif}
 main{max-width:920px;margin:0 auto;padding:16px}
 h1{font-size:18px;margin:0 0 10px}
 img{width:100%;border-radius:10px;background:#000;display:block}
 .row{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0}
 button,select{background:var(--card);color:var(--fg);border:1px solid #8884;border-radius:8px;
   padding:9px 12px;font:inherit}
 button:active{background:var(--acc);color:#fff}
 .on{outline:2px solid var(--acc)}
 label{color:var(--mut);display:flex;align-items:center;gap:6px}
 #st{color:var(--mut);margin-top:6px}
</style></head><body><main>
<h1>Drone follow-and-avoid <span id="st"></span></h1>
<img id="v" src="/stream" alt="live simulation">
<div class="row">
 <button onclick="act('debris')">Drop debris</button>
 <button onclick="act('rival')">Rival drone</button>
 <button onclick="act('pedestrian')">Add pedestrian</button>
 <button onclick="act('new')">New episode</button>
</div>
<div class="row" id="pol">
 <button data-p="trained" onclick="pol('trained')">Trained</button>
 <button data-p="reactive" onclick="pol('reactive')">Reactive (rules)</button>
 <button data-p="naive" onclick="pol('naive')">Naive (crashes)</button>
 <label>Difficulty <select id="diff" onchange="act('difficulty/'+this.value)">
  <option>0</option><option>1</option><option selected>2</option><option>3</option></select></label>
</div>
<div class="row">
 <label>Orbit <input id="az" type="range" min="0" max="360" value="225" oninput="cam()"></label>
 <label>Height <input id="el" type="range" min="8" max="80" value="27" oninput="cam()"></label>
 <label>Zoom <input id="di" type="range" min="3" max="16" value="7" step="0.5" oninput="cam()"></label>
</div>
<script>
const post=p=>fetch('/api/'+p,{method:'POST'});
const act=p=>post(p);
const pol=p=>post('policy/'+p);
let t=null;
function cam(){clearTimeout(t);t=setTimeout(()=>post('cam?az='+az.value+'&el='+el.value+'&d='+di.value),60)}
async function poll(){
 try{const s=await (await fetch('/api/state')).json();
  st.textContent='· '+s.status+' · '+s.policy+' · '+s.episodes+' episodes, '+s.collisions+' collisions';
  document.querySelectorAll('#pol button').forEach(b=>b.classList.toggle('on',b.dataset.p===s.policy));
 }catch(e){}
 setTimeout(poll,1000)}
poll();
</script></main></body></html>"""


class Sim:
    """Owns the simulation. Only its own thread touches PyBullet; handlers send commands."""

    def __init__(self, policies: dict, current: str, difficulty: int, seed: int):
        self.policies, self.current = policies, current
        self.difficulty, self.seed = difficulty, seed
        self.cmds: queue.Queue = queue.Queue()
        self.cond = threading.Condition()
        self.jpeg, self.frame_no = b"", 0
        self.last_seen = time.time()
        self.cam = (225.0, 27.0, 7.0)
        self.status, self.episodes, self.collisions = "STARTING", 0, 0

    def touch(self):
        self.last_seen = time.time()

    def run(self):
        env = SingleDroneWrapper(FollowAvoidAviary(
            gui=False, difficulty=self.difficulty, seed=self.seed, episode_sec=1e9, realistic=True))
        sim = env.unwrapped
        visual = None

        def new_episode():
            nonlocal visual
            obs, _ = env.reset()
            visual = DroneVisual(sim.CLIENT, int(sim.DRONE_IDS[0]))
            visual.update()
            return obs

        obs = new_episode()
        step, hold = 0, 0
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
                visual.update()
                self.status = status_from(info, term)
                if term or trunc:
                    self.episodes += 1
                    self.collisions += int(info["collision"])
                    hold = C.CTRL_FREQ * 3 // 2                   # linger on the crash for 1.5 s
            step += 1
            if step % RENDER_EVERY == 0:
                img = render_frame(sim, FRAME_W, FRAME_H, status=self.status,
                                   title=f"policy: {self.current}   level {self.difficulty}", cam=self.cam)
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 72])
                if ok:
                    with self.cond:
                        self.jpeg, self.frame_no = buf.tobytes(), self.frame_no + 1
                        self.cond.notify_all()
            time.sleep(max(0.0, 1.0 / C.CTRL_FREQ - (time.perf_counter() - t0)))


def make_handler(sim: Sim):
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

        def do_GET(self):
            sim.touch()
            path = self.path.split("?")[0]
            if path == "/":
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/api/state":
                self._send(200, json.dumps({
                    "status": sim.status, "policy": sim.current, "episodes": sim.episodes,
                    "collisions": sim.collisions}).encode())
            elif path == "/snapshot.jpg":
                with sim.cond:
                    sim.cond.wait_for(lambda: sim.frame_no > 0, timeout=15)
                self._send(200 if sim.jpeg else 503, sim.jpeg, "image/jpeg")
            elif path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                last = -1
                try:
                    while True:
                        with sim.cond:
                            sim.cond.wait_for(lambda: sim.frame_no != last, timeout=5)
                            jpeg, last = sim.jpeg, sim.frame_no
                        sim.touch()
                        if jpeg:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                             + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            sim.touch()
            path, _, query = self.path.partition("?")
            if path == "/api/cam":
                q = dict(kv.split("=") for kv in query.split("&") if "=" in kv)
                try:
                    sim.cam = (float(q["az"]), min(max(float(q["el"]), 5.0), 85.0),
                               min(max(float(q["d"]), 2.0), 20.0))
                except (KeyError, ValueError):
                    pass
            elif path.startswith("/api/"):
                cmd = path[len("/api/"):]
                if cmd.startswith("difficulty/"):
                    sim.difficulty = min(max(int(cmd.split("/")[1] or 2), 0), 3)
                    cmd = f"difficulty/{sim.difficulty}"
                sim.cmds.put(cmd)
            self._send(200, b"{}")

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="trained PPO .zip")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--difficulty", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    policies = {"reactive": reactive_follow, "naive": naive_follow}
    if args.model:
        from drone_follow.evaluate import load_ppo_policy
        policies["trained"] = load_ppo_policy(args.model)
    sim = Sim(policies, "trained" if "trained" in policies else "reactive", args.difficulty, args.seed)
    threading.Thread(target=sim.run, daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(sim))
    print(f"serving on http://{args.host}:{args.port}  (policy: {sim.current})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
