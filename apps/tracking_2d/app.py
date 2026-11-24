# -*- coding: utf-8 -*-
"""
Tracking 2D y comparación con odometría visual (Google Colab)
Ventana 1: por qué la odometría visual / Pose Tracking 3D es más difícil que el tracking 2D (animaciones 3D).
Ventana 2: tracking 2D con YOLO + ByteTrack / BoT-SORT, conteo por línea virtual y análisis de trayectorias.
Uso en Colab:  !pip -q install -r apps/tracking_2d/requirements.txt   y luego   !python apps/tracking_2d/app.py
"""
import asyncio, os, csv, json, math, time, uuid, queue, shutil, subprocess, threading, re
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2, numpy as np, torch
from fastapi import FastAPI, UploadFile, File, Form, WebSocket
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse

W = Path(os.environ.get("JOBS_DIR", "/content/jobs")); W.mkdir(parents=True, exist_ok=True)
NPOOL = int(os.environ.get("NPOOL", 6))               # instancias de YOLO en paralelo
WEIGHTS = os.environ.get("WEIGHTS", "yolo11l.pt")
pool, jobs = queue.Queue(), {}
ex = ThreadPoolExecutor(NPOOL)
app = FastAPI()
stopping = threading.Event()

def load_models():
    from ultralytics import YOLO
    for _ in range(NPOOL):
        m = YOLO(WEIGHTS); m.to("cuda" if torch.cuda.is_available() else "cpu")
        m.predict(np.zeros((640, 640, 3), np.uint8), verbose=False, half=torch.cuda.is_available())
        pool.put(m)

def color(tid):
    return tuple(int(v) for v in cv2.cvtColor(np.uint8([[[(tid * 47) % 180, 190, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])

def run(job, tracker, conf, line, classes):
    d = W / job["id"]; job["state"] = "queued"; m = pool.get(); job["state"] = "running"
    vw = None
    gen = None
    try:
        if stopping.is_set(): raise InterruptedError("Análisis interrumpido al cerrar el servidor.")
        cap = cv2.VideoCapture(str(d / "in.mp4")); fps = cap.get(5) or 30; w, h = int(cap.get(3)), int(cap.get(4)); n = max(int(cap.get(7)), 1)
        ok, f0 = cap.read(); cap.release(); cv2.imwrite(str(d / "poster.jpg"), f0)
        vw = cv2.VideoWriter(str(d / "raw.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        x1, y1, x2, y2 = line[0] * w, line[1] * h, line[2] * w, line[3] * h
        side = lambda x, y: (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        last, done, tail, tr, series, pc = {}, set(), {}, [], [], {}
        cin = cout = 0; names = m.names; t0 = time.time()
        gen = m.track(str(d / "in.mp4"), stream=True, tracker=tracker + ".yaml", conf=conf, half=torch.cuda.is_available(),
                      persist=False, verbose=False, classes=classes or None, imgsz=960)
        for i, r in enumerate(gen):
            if stopping.is_set(): raise InterruptedError("Análisis interrumpido al cerrar el servidor.")
            fr = r.orig_img.copy(); b = r.boxes; act = 0
            if b is not None and b.id is not None:
                for tid, bx, c in zip(b.id.int().tolist(), b.xyxy.cpu().numpy(), b.cls.int().tolist()):
                    cx, cy = float((bx[0] + bx[2]) / 2), float((bx[1] + bx[3]) / 2); act += 1
                    tr.append((tid, c, i, round(cx, 1), round(cy, 1)))
                    s = side(cx, cy)
                    if tid in last and last[tid] * s < 0:
                        dr = "in" if s > 0 else "out"
                        if (tid, dr) not in done:
                            done.add((tid, dr)); pc.setdefault(names[c], [0, 0])[0 if dr == "in" else 1] += 1
                            if dr == "in": cin += 1
                            else: cout += 1
                    last[tid] = s
                    col = color(tid); tail.setdefault(tid, deque(maxlen=45)).append((int(cx), int(cy)))
                    cv2.rectangle(fr, (int(bx[0]), int(bx[1])), (int(bx[2]), int(bx[3])), col, 2)
                    cv2.putText(fr, f"#{tid} {names[c]}", (int(bx[0]), max(18, int(bx[1]) - 6)), 0, .6, col, 2)
                    cv2.polylines(fr, [np.array(tail[tid], np.int32)], False, col, 2)
            cv2.line(fr, (int(x1), int(y1)), (int(x2), int(y2)), (80, 180, 255), 3)
            cv2.rectangle(fr, (0, 0), (260, 70), (30, 20, 16), -1)
            cv2.putText(fr, f"Entran: {cin}", (10, 28), 0, .8, (160, 240, 160), 2); cv2.putText(fr, f"Salen: {cout}", (10, 58), 0, .8, (120, 120, 255), 2)
            vw.write(fr); series.append(act); job["p"] = min((i + 1) / n, .99)
        vw.release(); dt = time.time() - t0
        subprocess.run(["ffmpeg", "-y", "-i", str(d / "raw.mp4"), "-vf", "scale='min(960,iw)':-2", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "28", "-movflags", "+faststart", str(d / "video.mp4"), "-loglevel", "error"])
        per = {}
        for tid, c, f, x, y in tr: per.setdefault(tid, [c, []])[1].append((f, x, y))
        rows = []
        for tid, (c, p) in per.items():
            dist = sum(math.dist(p[k][1:], p[k + 1][1:]) for k in range(len(p) - 1)); dur = (p[-1][0] - p[0][0] + 1) / fps
            rows.append(dict(id=tid, cls=names[c], frames=len(p), dur=round(dur, 2), dist=round(dist), vel=round(dist / dur, 1) if dur else 0))
        rows.sort(key=lambda r: -r["dist"])
        heat = np.zeros((18, 32), int)
        for _, _, _, x, y in tr: heat[min(17, int(y / h * 18)), min(31, int(x / w * 32))] += 1
        with open(d / "trayectorias.csv", "w", newline="") as f:
            wr = csv.writer(f); wr.writerow(["id", "clase", "frame", "tiempo_s", "cx", "cy"])
            for tid, c, fi, x, y in tr: wr.writerow([tid, names[c], fi, round(fi / fps, 3), x, y])
        st = max(1, len(series) // 300)
        top = [r["id"] for r in rows[:300]]
        res = dict(w=w, h=h, fps=round(fps, 1), frames=len(series), ids=len(per), cin=cin, cout=cout, pc=pc, line=line,
                   inf_fps=round(len(series) / dt, 1), mb=round((d / "video.mp4").stat().st_size / 1e6, 1), series=series[::st], st=st, rows=rows[:50], heat=heat.tolist(),
                   tr=[dict(id=t, p=[[round(a), round(b_)] for _, a, b_ in per[t][1][::max(1, len(per[t][1]) // 150)]]) for t in top])
        json.dump(res, open(d / "resultados.json", "w")); job.update(res=res, p=1, state="done")
    except Exception as e:
        job.update(state="error", err=str(e))
    finally:
        try:
            if gen is not None: gen.close()
        finally:
            if vw is not None: vw.release()
            pool.put(m)

@app.post("/api/track")
async def track(file: UploadFile = File(...), tracker: str = Form("bytetrack"), conf: float = Form(.3),
                line: str = Form("[0,0.5,1,0.5]"), classes: str = Form("")):
    for p in W.iterdir():                                     # limpieza de trabajos con más de 2 h
        if p.is_dir() and time.time() - p.stat().st_mtime > 7200: shutil.rmtree(p, ignore_errors=True)
    jid = uuid.uuid4().hex[:10]; d = W / jid; d.mkdir()
    with open(d / "in.mp4", "wb") as f: shutil.copyfileobj(file.file, f)
    job = dict(id=jid, state="queued", p=0); jobs[jid] = job
    ex.submit(run, job, tracker if tracker in ("bytetrack", "botsort") else "bytetrack", conf, json.loads(line),
              [int(c) for c in classes.split(",") if c != ""])
    return {"id": jid}

@app.get("/api/job/{jid}")
def job_state(jid: str): return jobs.get(jid, {"state": "error", "err": "Trabajo no encontrado"})

@app.get("/api/info")
def info():
    return {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", "queue": sum(j["state"] == "queued" for j in jobs.values()),
            "running": sum(j["state"] == "running" for j in jobs.values()), "slots": NPOOL}

@app.get("/files/{jid}/{name}")
def files(jid: str, name: str):
    if name not in ("video.mp4", "poster.jpg", "trayectorias.csv", "resultados.json") or not re.fullmatch(r"[0-9a-f]{10}", jid): return JSONResponse({}, 404)
    return FileResponse(W / jid / name)

@app.websocket("/ws/live")
async def live(ws: WebSocket):
    """Tracking en tiempo real: el navegador envía fotogramas JPEG y recibe cajas, IDs y conteos."""
    await ws.accept()
    try: m = pool.get_nowait()
    except queue.Empty:
        await ws.send_json({"error": "Todos los espacios de GPU están ocupados. Inténtalo de nuevo en unos segundos."}); await ws.close(); return
    cfg = dict(tracker="bytetrack", conf=.3, line=[0, .5, 1, .5], classes=[]); first = True
    last, done, pc, cin, cout = {}, set(), {}, 0, 0
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect": break
            if msg.get("text"):
                cm = json.loads(msg["text"])
                if first and cm.get("tracker") in ("bytetrack", "botsort"): cfg["tracker"] = cm["tracker"]
                if "line" in cm and cm["line"] != cfg["line"]: last.clear()
                for k in ("conf", "line", "classes"):
                    if k in cm: cfg[k] = cm[k]
                continue
            frame = cv2.imdecode(np.frombuffer(msg.get("bytes") or b"", np.uint8), cv2.IMREAD_COLOR)
            if frame is None: continue
            h, w = frame.shape[:2]; t0 = time.time(); pers = not first
            r = (await asyncio.to_thread(lambda: m.track(frame, persist=pers, tracker=cfg["tracker"] + ".yaml", conf=float(cfg["conf"]),
                 half=torch.cuda.is_available(), verbose=False, classes=cfg["classes"] or None, imgsz=640)))[0]
            first = False
            x1, y1, x2, y2 = cfg["line"][0] * w, cfg["line"][1] * h, cfg["line"][2] * w, cfg["line"][3] * h
            out, b = [], r.boxes
            if b is not None and b.id is not None:
                for tid, bx, cl in zip(b.id.int().tolist(), b.xyxy.cpu().numpy(), b.cls.int().tolist()):
                    cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
                    sd = (x2 - x1) * (cy - y1) - (y2 - y1) * (cx - x1)
                    if tid in last and last[tid] * sd < 0:
                        dr = "in" if sd > 0 else "out"
                        if (tid, dr) not in done:
                            done.add((tid, dr)); pc.setdefault(m.names[cl], [0, 0])[0 if dr == "in" else 1] += 1
                            if dr == "in": cin += 1
                            else: cout += 1
                    last[tid] = sd
                    out.append([int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3]), tid, cl, m.names[cl]])
            await ws.send_json(dict(b=out, cin=cin, cout=cout, pc=pc, ms=round((time.time() - t0) * 1000), w=w, h=h))
    except Exception:
        pass
    finally:
        pool.put(m)

@app.get("/", response_class=HTMLResponse)
def home(): return HTML

HTML = r"""<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tracking 2D y trayectorias</title>
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;800&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{--bg:#0f1226;--pn:#171c3a;--ln:#283060;--tx:#e8ebff;--mu:#9aa3d6;--ac:#7cc4ff;--am:#ffb86b;--ok:#7be0a8;--bad:#ff7a8a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);font:15px/1.55 Manrope,system-ui,sans-serif;display:flex;min-height:100vh}
aside{width:260px;padding:28px 20px;border-right:1px solid var(--ln);display:flex;flex-direction:column;gap:8px;position:sticky;top:0;height:100vh}
h1{font-size:20px;font-weight:800;margin:0 0 4px;line-height:1.2}.sub{color:var(--mu);font-size:13px;margin-bottom:22px}
.nav button{all:unset;display:block;width:100%;padding:12px 14px;border-radius:12px;cursor:pointer;color:var(--mu);font-weight:600;margin-bottom:6px;box-sizing:border-box}
.nav button:hover{background:#1d2347}.nav button.on{background:var(--ac);color:#0c1230}
#gpu{margin-top:auto;font-size:12px;color:var(--mu);border:1px solid var(--ln);border-radius:10px;padding:10px}
main{flex:1;padding:28px 32px;min-width:0}.view{display:none}.view.on{display:block}
h2{font-size:26px;margin:0 0 4px;font-weight:800}.lead{color:var(--mu);margin:0 0 20px;max-width:75ch}
.grid1{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(300px,1fr);gap:18px}
.card{background:var(--pn);border:1px solid var(--ln);border-radius:16px;padding:16px}
.stage{height:480px;padding:0;overflow:hidden;position:relative}.stage canvas{width:100%;height:100%;display:block}
#hud{position:absolute;left:14px;bottom:12px;right:14px;display:grid;grid-template-columns:1fr 1fr;gap:10px;font-size:12px;pointer-events:none}
.bar{height:8px;background:#0c1030;border-radius:6px;overflow:hidden;margin-top:4px}.bar i{display:block;height:100%;width:0;border-radius:6px}
.rs{display:flex;flex-direction:column;gap:8px}.rs button{all:unset;cursor:pointer;padding:11px 14px;border:1px solid var(--ln);border-radius:12px;font-weight:600;background:#121733}
.rs button:hover{border-color:var(--ac)}.rs button.on{border-color:var(--am);background:#241f33;color:var(--am)}
#why h3{margin:0 0 6px;font-size:18px}#why p{margin:0 0 10px;color:#cfd5ff}.tag{display:inline-block;font-size:12px;padding:3px 9px;border-radius:20px;border:1px solid var(--ln);color:var(--mu);margin-right:6px}
.btn{all:unset;cursor:pointer;background:var(--am);color:#2a1700;font-weight:800;padding:12px 20px;border-radius:12px;text-align:center}.btn:disabled{opacity:.4;cursor:default}
.ctl{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px;align-items:end}label{font-size:12px;color:var(--mu);display:block;margin-bottom:4px}
select,input[type=range]{width:100%;background:#0f1330;color:var(--tx);border:1px solid var(--ln);border-radius:10px;padding:9px}
.drop{border:2px dashed var(--ln);border-radius:14px;padding:22px;text-align:center;color:var(--mu);cursor:pointer}.drop.h{border-color:var(--ac)}
.pv{position:relative;margin-top:12px}.pv video,.pv img{width:100%;border-radius:12px;display:block;background:#000;max-height:420px}.pv canvas{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px;margin:18px 0}.kpi b{display:block;font-size:30px;font-weight:800}.kpi span{color:var(--mu);font-size:12px}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:18px;margin-bottom:18px}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:7px 8px;border-bottom:1px solid var(--ln);text-align:left}th{color:var(--mu);font-weight:600}
.chk label{display:inline-flex;gap:5px;margin-right:12px;color:var(--tx);font-size:13px}.prog{height:10px;background:#0c1030;border-radius:8px;overflow:hidden;margin-top:12px}.prog i{display:block;height:100%;width:0;background:var(--ac);transition:width .4s}
a.dl{color:var(--ac);font-weight:600;margin-right:16px}.ctrlbar{margin-top:12px;display:flex;flex-wrap:wrap;gap:10px 16px;align-items:center;font-size:13px}.ctrlbar .btn{padding:8px 14px;font-size:13px}.ctrlbar .sec,.btn.sec{background:#283060;color:var(--tx)}.ctrlbar input[type=range]{width:140px;padding:0;border:0;background:none;accent-color:var(--am)}.ctrlbar label{margin:0;color:var(--tx)}.tabs{display:flex;gap:8px;margin:16px 0 6px}.tabs button{all:unset;cursor:pointer;padding:9px 16px;border-radius:10px;border:1px solid var(--ln);font-weight:600;color:var(--mu)}.tabs button.on{background:var(--ac);color:#0c1230;border-color:var(--ac)}
@media(max-width:900px){body{flex-direction:column}aside{width:auto;height:auto;position:static}.grid1{grid-template-columns:1fr}main{padding:18px}}
</style></head><body>
<aside><h1>Tracking 2D y trayectorias</h1><div class="sub">Etapa 02 – Aplicaciones y conceptos</div>
<div class="nav"><button class="on" data-v="v1">Por qué la odometría visual es más difícil</button><button data-v="v2">Tracking 2D con YOLO</button></div>
<div id="gpu">Conectando con la GPU…</div></aside>
<main>
<section class="view on" id="v1"><h2>Odometría visual 3D frente a tracking 2D</h2>
<p class="lead">Elige una razón. La escena 3D simula el proceso y compara la carga computacional de cada enfoque.</p>
<div class="grid1"><div><div class="card stage"><canvas id="cv"></canvas>
<div id="hud"><div><span>Tracking 2D (YOLO + Kalman)</span><div class="bar"><i id="b2" style="background:var(--ok)"></i></div></div><div><span>Odometría visual / Pose 3D</span><div class="bar"><i id="b3" style="background:var(--am)"></i></div></div><div id="mt" style="grid-column:1/3;color:var(--ac);font-weight:600"></div></div></div><div class="card ctrlbar"><button class="btn" id="pp">Pausar</button><button class="btn sec" id="rst">Reiniciar</button><span>Velocidad <b id="sv">1.00x</b></span><input type="range" id="sp" min="0.1" max="2" step="0.05" value="1"><span>Línea de tiempo</span><input type="range" id="tl" min="0" max="1000" value="0" style="width:200px"><label><input type="checkbox" id="lp" checked> Repetir en bucle</label><label><input type="checkbox" id="ar"> Rotación automática</label><button class="btn sec" id="vr">Restablecer vista</button><div style="color:var(--mu);width:100%;font-size:12px">Arrastra para rotar, usa la rueda para acercar o alejar y el clic derecho para desplazar. En el móvil, un dedo rota y dos dedos acercan.</div></div></div>
<div><div class="rs card" id="rs"></div><div class="card" id="why" style="margin-top:14px"></div></div></div></section>

<section class="view" id="v2"><h2>Tracking 2D: conteo y trayectorias</h2>
<p class="lead">Sube un video, ubica la línea virtual y analiza qué objetos la cruzan. El procesamiento se ejecuta en el servidor con GPU, si está disponible, o CPU.</p>
<div class="card"><div class="ctl">
<div><label>Algoritmo de seguimiento</label><select id="trk"><option value="bytetrack">YOLO + ByteTrack</option><option value="botsort">YOLO + BoT-SORT</option></select></div>
<div><label>Confianza mínima: <b id="cv_">0.30</b></label><input type="range" id="conf" min="0.1" max="0.8" step="0.05" value="0.3"></div>
<div><label>Orientación de la línea</label><select id="ori"><option value="h">Horizontal</option><option value="v">Vertical</option></select></div>
<div><label>Posición de la línea</label><input type="range" id="pos" min="5" max="95" value="50"></div></div>
<div class="chk" style="margin-top:14px"><label><input class="cls" type="checkbox" value="0" checked>Persona</label><label><input class="cls" type="checkbox" value="1">Bicicleta</label><label><input class="cls" type="checkbox" value="2" checked>Auto</label><label><input class="cls" type="checkbox" value="3" checked>Moto</label><label><input class="cls" type="checkbox" value="5" checked>Bus</label><label><input class="cls" type="checkbox" value="7" checked>Camión</label></div>
<div class="tabs"><button class="on" data-m="mvid">Archivo de video</button><button data-m="mlive">Cámara en vivo</button></div><div id="mvid"><div class="drop" id="drop" style="margin-top:14px">Arrastra un video aquí o haz clic para elegirlo<input type="file" id="file" accept="video/*" hidden></div>
<div class="pv" id="pvw" style="display:none"><video id="pv" muted controls></video><canvas id="pc"></canvas></div>
<div style="display:flex;gap:14px;align-items:center;margin-top:14px"><button class="btn" id="go" disabled>Analizar video</button><span id="st" style="color:var(--mu)"></span></div><div class="prog"><i id="pg"></i></div></div><div id="mlive" style="display:none"><div class="ctl" style="margin-top:14px"><div><label>Cámara (webcam o cámara externa)</label><select id="cam"><option value="">Cámara predeterminada</option></select></div><div style="display:flex;gap:10px"><button class="btn" id="lgo">Iniciar cámara</button><button class="btn sec" id="lst" disabled>Detener y guardar</button></div></div><div id="lerr" style="color:var(--bad);margin-top:8px"></div><div class="pv"><canvas id="lc" style="position:static;height:auto;width:100%;border-radius:12px;background:#000;min-height:220px"></canvas></div><div class="kpis" id="lk"></div><div class="two"><div><b>Objetos activos en vivo</b><canvas id="lch"></canvas></div><div><b>Resumen por ID</b><table id="ltb"></table><p id="ldl"></p></div></div><video id="lv" playsinline muted style="display:none"></video></div></div>
<div id="out" style="display:none">
<div class="kpis" id="kp"></div>
<div class="two"><div class="card"><video id="rv" controls style="width:100%;border-radius:12px"></video></div>
<div class="card"><div style="display:flex;justify-content:space-between"><b>Trayectorias</b><label style="margin:0"><input type="checkbox" id="hm"> Mapa de calor</label></div><div class="pv"><canvas id="tc" style="position:static;height:auto;width:100%;border-radius:12px"></canvas></div></div></div>
<div class="two"><div class="card"><b>Objetos activos por instante</b><canvas id="c1"></canvas></div><div class="card"><b>Cruces de la línea por clase</b><canvas id="c2"></canvas></div></div>
<div class="card"><b>Objetos con mayor recorrido</b><table id="tb"></table><p id="dls" style="margin-bottom:0"></p></div></div></section></main>
<script>
const $=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)];
$$('.nav button').forEach(b=>b.onclick=()=>{$$('.nav button').forEach(x=>x.classList.toggle('on',x==b));$$('.view').forEach(v=>v.classList.toggle('on',v.id==b.dataset.v));resize()});
const gi=()=>fetch('/api/info').then(r=>r.json()).then(i=>$('#gpu').textContent=i.gpu+' – '+i.running+'/'+i.slots+' análisis activos, '+i.queue+' en cola').catch(()=>{});gi();setInterval(gi,5000);
/* ---------- Ventana 1: escenas 3D ---------- */
let R,S,C,G,AL,OC,cur=null,DUR=8,tt=0,playing=true,spd=1,LT=performance.now();
const M=(c)=>new THREE.MeshStandardMaterial({color:c,roughness:.4,metalness:.2});
const LN=(pts,c)=>{const l=new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),new THREE.LineBasicMaterial({color:c,transparent:true}));G.add(l);return l};
const sph=(r,c,p)=>{const m=new THREE.Mesh(new THREE.SphereGeometry(r,14,14),M(c));m.position.set(...p);G.add(m);return m};
const box=(w,h,d,c,p)=>{const m=new THREE.Mesh(new THREE.BoxGeometry(w,h,d),M(c));m.position.set(...p);G.add(m);return m};
function lab(s,p,sc=3.2){const c=document.createElement('canvas');c.width=512;c.height=96;const x=c.getContext('2d');x.font='600 42px Manrope,sans-serif';x.fillStyle='#e8ebff';x.textAlign='center';x.fillText(s,256,62);const sp=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(c),transparent:true}));sp.position.set(...p);sp.scale.set(sc,sc*96/512,1);G.add(sp)}
const cone=c=>{const m=new THREE.Mesh(new THREE.ConeGeometry(.5,1.2,4),M(c));m.rotation.z=-Math.PI/2;return m};
const rnd=(()=>{let s=7;return()=>(s=(s*16807)%2147483647)/2147483647})();
const ez=t=>t*t*(3-2*t);
const SC=[
{n:'Seis grados de libertad',l:[8,38],
 why:'En 2D basta con ubicar cada objeto en la imagen: dos coordenadas (x, y). En odometría visual hay que estimar la pose completa de la cámara: tres traslaciones y tres rotaciones. Un error pequeño de rotación se amplifica con la distancia y desplaza todo el mapa estimado.',
 run(){box(5,.05,4,0x232a55,[-4.5,0,0]);const a=sph(.3,0x7cc4ff,[0,.3,0]);lab('2D: 2 parámetros',[-4.5,2.6,0]);lab('3D: 6 parámetros',[4.5,3.4,0]);
  const cam=new THREE.Group();cam.add(cone(0xffb86b),new THREE.AxesHelper(1.6));G.add(cam);
  return t=>{const u=t*6.28;a.position.set(-4.5+Math.cos(u)*1.8,.3,Math.sin(u)*1.4);cam.position.set(4.5+Math.cos(u)*1.8,1.5+Math.sin(u*2)*.8,Math.sin(u)*1.4);cam.rotation.set(t*5,t*7,t*3);return 'Parámetros a estimar por cuadro: '+(t<.1?'…':'2 frente a 6')}}},
{n:'Ambigüedad de escala',l:[10,45],
 why:'Con una sola cámara, un objeto pequeño y cercano produce la misma imagen que uno grande y lejano. La escala del movimiento no se puede observar sin estéreo, IMU, sensor de profundidad o conocimiento previo. En tracking 2D no hace falta: se trabaja directamente en píxeles.',
 run(){const c=cone(0xffb86b);c.position.set(-6,0,0);G.add(c);LN([new THREE.Vector3(-6,0,0),new THREE.Vector3(5,2.2,0)],0x7cc4ff);LN([new THREE.Vector3(-6,0,0),new THREE.Vector3(5,-2.2,0)],0x7cc4ff);
  box(.03,.8,.8,0x7cc4ff,[-4.5,0,0]);lab('Misma imagen',[-4.5,1.4,0],2.4);const o=box(1,1,1,0xff7a8a,[0,0,0]);
  return t=>{const d=3+8*(.5+.5*Math.sin(t*6.28-1.57));o.position.x=-6+d;o.scale.setScalar(.4*d);return 'Distancia '+d.toFixed(1)+' m, tamaño '+(.8*d).toFixed(1)+' m: indistinguibles en la imagen'}}},
{n:'Deriva acumulada',l:[6,70],
 why:'La odometría suma movimientos relativos cuadro a cuadro, y cada pequeño error se acumula sin límite. Para frenar la deriva se necesitan cierres de bucle y optimización global (pose graph, bundle adjustment). En tracking 2D, un error en un cuadro se corrige con la detección del cuadro siguiente.',
 run(){const N=200,gt=[],es=[];let ex=0,ey=0;for(let i=0;i<N;i++){const a=i/N*6.28;ex+=Math.sin(i*1.7)*.02+.013;ey+=Math.cos(i*2.3)*.02+.009;gt.push(new THREE.Vector3(Math.cos(a)*4,0,Math.sin(a)*4));es.push(new THREE.Vector3(Math.cos(a)*4+ex,0,Math.sin(a)*4+ey))}
  const g=LN(gt,0x7be0a8),e=LN(es,0xff7a8a);lab('Trayectoria real',[-5,.5,5.2],3);lab('Trayectoria estimada',[5,.5,5.2],3);const h=sph(.2,0xffb86b,[4,0,0]);
  return t=>{const k=Math.max(2,Math.floor(t*N));g.geometry.setDrawRange(0,k);e.geometry.setDrawRange(0,k);h.position.copy(es[k-1]);return 'Error de posición acumulado: '+gt[k-1].distanceTo(es[k-1]).toFixed(2)+' m'}}},
{n:'Correspondencias y outliers',l:[12,62],
 why:'Hay que detectar puntos característicos (ORB, SIFT, SuperPoint), emparejarlos entre cuadros y descartar los falsos con RANSAC sobre la geometría epipolar. Unos pocos emparejamientos erróneos bastan para arruinar la pose. El tracking 2D solo compara cajas por IoU y apariencia.',
 run(){box(3.6,4.2,.05,0x232a55,[-4,0,0]);box(3.6,4.2,.05,0x232a55,[4,0,0]);lab('Cuadro t',[-4,2.7,0],2.4);lab('Cuadro t+1',[4,2.7,0],2.4);
  const N=36,ls=[],bad=[];for(let i=0;i<N;i++){const x=(rnd()-.5)*3,y=(rnd()-.5)*3.6,o=rnd()<.3;bad.push(o);const q=o?[(rnd()-.5)*3,(rnd()-.5)*3.6]:[x+.2,y+.1];ls.push(LN([new THREE.Vector3(-4+x,y,.1),new THREE.Vector3(4+q[0],q[1],.1)],0x7cc4ff));ls[i].visible=false}
  return t=>{const v=Math.floor(Math.min(1,t*3)*N);let ok=0;ls.forEach((l,i)=>{l.visible=i<v;const o=bad[i];l.material.color.set(t>.45?(o?0xff7a8a:0x7be0a8):0x7cc4ff);l.material.opacity=(t>.75&&o)?Math.max(0,1-(t-.75)*8):1;if(!o)ok++});return t<.45?'Emparejando puntos…':t<.75?'RANSAC: iterando hipótesis':'Inliers conservados: '+ok+' de '+N}}},
{n:'Cámara en movimiento',l:[9,55],
 why:'En seguridad la cámara está fija: el fondo no cambia y todo lo que se mueve es un objeto. En odometría la cámara se mueve, así que todo el fondo se desplaza (ego-motion) y hay que separar ese flujo del movimiento propio de personas y vehículos.',
 run(){const n=500,pos=new Float32Array(n*3);for(let i=0;i<n;i++){pos[i*3]=(rnd()-.5)*16;pos[i*3+1]=(rnd()-.5)*6;pos[i*3+2]=(rnd()-.5)*16}const z0=Float32Array.from({length:n},(_,i)=>pos[i*3+2]);
  const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(pos,3));G.add(new THREE.Points(g,new THREE.PointsMaterial({color:0x9aa3d6,size:.09})));
  const cs=[0xffb86b,0xff7a8a,0x7be0a8].map((c,i)=>box(.8,.8,.8,c,[0,0,0]));
  return t=>{const ego=t>.45;for(let i=0;i<n;i++)pos[i*3+2]=((z0[i]+8+(ego?(t-.45)*40:0))%16+16)%16-8;g.attributes.position.needsUpdate=true;
   cs.forEach((c,i)=>c.position.set(Math.sin(t*9+i*2)*5,.5+i*.8-1,Math.cos(t*7+i)*3));return ego?'Cámara móvil: todo el fondo se mueve':'Cámara fija: solo se mueven los objetos'}}},
{n:'Carga computacional',l:[14,95],
 why:'El tracking 2D ejecuta un detector, un filtro de Kalman y una asociación húngara: unas decenas de milisegundos en GPU. La odometría añade extracción de puntos, emparejamiento, RANSAC, PnP o triangulación, bundle adjustment local, mapa y cierre de bucle: optimización no lineal con miles de variables que debe cumplirse en tiempo real.',
 run(){const a=['Detector','Kalman','Asociación'],b=['Features','Matching','RANSAC','PnP','Bundle Adj.','Mapa'],ta=[1.4,.7,.5],tb=[1.4,1.2,2.8,1.6,4.2,2.6],bs=[];
  a.forEach((s,i)=>{bs.push([box(1.4,1,1.4,0x7be0a8,[-2+i*2,0,-3]),ta[i],i]);lab(s,[-2+i*2,4.6,-3],2.4)});b.forEach((s,i)=>{bs.push([box(1.4,1,1.4,0xffb86b,[-5+i*2,0,3]),tb[i],i+3]);lab(s,[-5+i*2,5.2,3],2.4)});
  return t=>{const k=ez(Math.min(1,t*2));bs.forEach(([m,h,i])=>{const H=Math.max(.05,h*k*(.88+.12*Math.sin(t*50+i)));m.scale.y=H;m.position.y=H/2});return 'Tiempo por cuadro estimado: 2D '+(12*k).toFixed(0)+' ms, 3D '+(85*k).toFixed(0)+' ms'}}},
{n:'Condiciones adversas',l:[18,85],
 why:'La odometría depende de textura e iluminación estables. La poca luz, el desenfoque por movimiento, las paredes lisas, los reflejos y el rolling shutter hacen perder puntos y la estimación se rompe. Los detectores profundos son más robustos porque aprenden la apariencia del objeto, no la geometría del entorno.',
 run(){const P=[];for(let i=0;i<60;i++){P.push([sph(.14,0x7be0a8,[-4.5+(i%10),Math.floor(i/10)-2,0]),rnd()*.85])}box(10,6,.05,0x1b2145,[0,0,-.2]);
  return t=>{const L=1-t*.95;AL.intensity=.15+.55*L;let n=0;P.forEach(([m,th],i)=>{const ok=L>th;if(ok)n++;m.material.color.set(ok?0x7be0a8:0xff7a8a);m.scale.setScalar(ok?1:.4);m.position.x=-4.5+(i%10)+Math.sin(i*12.9+t*80)*.05*t});return 'Puntos rastreados: '+n+' de 60'+(n<15?'. Seguimiento perdido':'')}}}
];
function init3(){const el=$('#cv');R=new THREE.WebGLRenderer({canvas:el,antialias:true});S=new THREE.Scene();S.background=new THREE.Color(0x101426);C=new THREE.PerspectiveCamera(50,1,.1,100);C.position.set(0,6,13);OC=THREE.OrbitControls?new THREE.OrbitControls(C,el):{update(){},target:new THREE.Vector3()};OC.target.set(0,1,0);OC.enableDamping=true;OC.dampingFactor=.08;OC.minDistance=3;OC.maxDistance=45;OC.autoRotateSpeed=1.2;
 AL=new THREE.AmbientLight(0xffffff,.7);S.add(AL);const d=new THREE.DirectionalLight(0xffffff,.8);d.position.set(5,8,5);S.add(d);G=new THREE.Group();S.add(G);
 const rs=$('#rs');SC.forEach((s,i)=>{const b=document.createElement('button');b.textContent=s.n;b.onclick=()=>play(i);rs.appendChild(b)});resize();bind3();play(0);loop()}
function resize(){if(!R)return;const p=$('#cv').parentElement,w=p.clientWidth,h=p.clientHeight;if(!w)return;R.setSize(w,h,false);C.aspect=w/h;C.updateProjectionMatrix()}
addEventListener('resize',resize);
function bind3(){$('#pp').onclick=()=>{if(!playing&&tt>=1)tt=0;playing=!playing;$('#pp').textContent=playing?'Pausar':'Reproducir'};$('#rst').onclick=()=>{tt=0;playing=true;$('#pp').textContent='Pausar'};$('#sp').oninput=e=>{spd=+e.target.value;$('#sv').textContent=spd.toFixed(2)+'x'};$('#tl').oninput=e=>{tt=e.target.value/1000;playing=false;$('#pp').textContent='Reproducir'};$('#ar').onchange=e=>OC.autoRotate=e.target.checked;$('#vr').onclick=()=>{C.position.set(0,6,13);OC.target.set(0,1,0);OC.update()}}
function play(i){[...$('#rs').children].forEach((b,k)=>b.classList.toggle('on',k==i));while(G.children.length)G.remove(G.children[0]);AL.intensity=.7;G.add(new THREE.GridHelper(20,20,0x2a3160,0x1d2347));
 const s=SC[i];cur={tick:s.run(),l:s.l};tt=0;playing=true;$('#pp').textContent='Pausar';$('#why').innerHTML='<h3>'+s.n+'</h3><p>'+s.why+'</p><span class="tag">2D: carga '+s.l[0]+'%</span><span class="tag">3D: carga '+s.l[1]+'%</span><p style="margin-top:12px"><button class="btn" onclick="play('+i+')">Repetir animación</button></p>'}
function loop(){requestAnimationFrame(loop);const now=performance.now(),dt=Math.min((now-LT)/1000,.1);LT=now;if(cur){if(playing){tt=Math.min(1,tt+dt*spd/DUR);if(tt>=1){if($('#lp').checked)tt=0;else{playing=false;$('#pp').textContent='Reproducir'}}}const t=tt;$('#tl').value=t*1000;$('#mt').textContent=cur.tick(t);const k=ez(Math.min(1,t*2.2));$('#b2').style.width=cur.l[0]*k+'%';$('#b3').style.width=cur.l[1]*k+'%'}
 OC.update();R.render(S,C)}
init3();
/* ---------- Ventana 2: tracking ---------- */
let FILE=null,RES=null,JID=null;
$('#conf').oninput=e=>$('#cv_').textContent=(+e.target.value).toFixed(2);
$('#drop').onclick=()=>$('#file').click();$('#file').onchange=e=>e.target.files[0]&&setFile(e.target.files[0]);
$('#drop').ondragover=e=>{e.preventDefault();$('#drop').classList.add('h')};$('#drop').ondragleave=()=>$('#drop').classList.remove('h');
$('#drop').ondrop=e=>{e.preventDefault();$('#drop').classList.remove('h');e.dataTransfer.files[0]&&setFile(e.dataTransfer.files[0])};
function setFile(f){FILE=f;$('#pvw').style.display='block';const v=$('#pv');v.src=URL.createObjectURL(f);v.onloadeddata=drawLine;$('#go').disabled=false;$('#drop').firstChild.textContent=f.name}
const lineP=()=>{const p=$('#pos').value/100;return $('#ori').value=='h'?[0,p,1,p]:[p,0,p,1]};
function drawLine(){const v=$('#pv'),c=$('#pc');c.width=v.clientWidth;c.height=v.clientHeight;const x=c.getContext('2d'),l=lineP();x.setLineDash([10,6]);x.strokeStyle='#ffb86b';x.lineWidth=3;x.beginPath();x.moveTo(l[0]*c.width,l[1]*c.height);x.lineTo(l[2]*c.width,l[3]*c.height);x.stroke()}
$('#pos').oninput=$('#ori').onchange=drawLine;
$('#go').onclick=async()=>{const fd=new FormData();fd.append('file',FILE);fd.append('tracker',$('#trk').value);fd.append('conf',$('#conf').value);fd.append('line',JSON.stringify(lineP()));fd.append('classes',$$('.cls:checked').map(c=>c.value).join(','));
 $('#go').disabled=true;$('#st').textContent='Subiendo video…';$('#out').style.display='none';
 try{const r=await(await fetch('/api/track',{method:'POST',body:fd})).json();JID=r.id;poll()}catch(e){$('#st').textContent='No se pudo subir el video. Revisa tu conexión e inténtalo de nuevo.';$('#go').disabled=false}};
async function poll(){const j=await(await fetch('/api/job/'+JID)).json();$('#pg').style.width=(j.p||0)*100+'%';
 if(j.state=='done'){$('#st').textContent='Análisis completo';$('#go').disabled=false;show(j.res)}
 else if(j.state=='error'){$('#st').textContent='Error: '+j.err;$('#go').disabled=false}
 else{$('#st').textContent=j.state=='queued'?'En cola, esperando un espacio de procesamiento…':'Procesando en el servidor: '+Math.round((j.p||0)*100)+'%';setTimeout(poll,800)}}
let ch=[];
function show(r){RES=r;$('#out').style.display='block';$('#rv').src='/files/'+JID+'/video.mp4';
 const k=[['Entran',r.cin],['Salen',r.cout],['Objetos únicos (IDs)',r.ids],['Cuadros analizados',r.frames],['FPS de inferencia',r.inf_fps],['FPS del video',r.fps],['Tamaño del video (MB)',r.mb]];
 $('#kp').innerHTML=k.map(x=>'<div class="card kpi"><b>'+x[1]+'</b><span>'+x[0]+'</span></div>').join('');
 ch.forEach(c=>c.destroy());const o={plugins:{legend:{labels:{color:'#9aa3d6'}}},scales:{x:{ticks:{color:'#9aa3d6'},grid:{color:'#283060'}},y:{ticks:{color:'#9aa3d6'},grid:{color:'#283060'}}}};
 ch=[new Chart($('#c1'),{type:'line',data:{labels:r.series.map((_,i)=>((i*r.st)/r.fps).toFixed(1)+' s'),datasets:[{label:'Objetos activos',data:r.series,borderColor:'#7cc4ff',pointRadius:0,tension:.3,fill:true,backgroundColor:'#7cc4ff22'}]},options:o}),
 new Chart($('#c2'),{type:'bar',data:{labels:Object.keys(r.pc),datasets:[{label:'Entran',data:Object.values(r.pc).map(v=>v[0]),backgroundColor:'#7be0a8'},{label:'Salen',data:Object.values(r.pc).map(v=>v[1]),backgroundColor:'#ff7a8a'}]},options:o})];
 $('#tb').innerHTML='<tr><th>ID</th><th>Clase</th><th>Duración (s)</th><th>Recorrido (px)</th><th>Velocidad media (px/s)</th></tr>'+r.rows.slice(0,15).map(x=>'<tr><td>#'+x.id+'</td><td>'+x.cls+'</td><td>'+x.dur+'</td><td>'+x.dist+'</td><td>'+x.vel+'</td></tr>').join('');
 $('#dls').innerHTML=['trayectorias.csv','resultados.json'].map(n=>'<a class="dl" href="/files/'+JID+'/'+n+'" download>Descargar '+n+'</a>').join('');drawT()}
$('#hm').onchange=()=>drawT();
function drawT(){const r=RES,im=new Image();im.onload=()=>{const c=$('#tc');c.width=r.w;c.height=r.h;const x=c.getContext('2d');x.drawImage(im,0,0);x.fillStyle='#0f1226aa';x.fillRect(0,0,r.w,r.h);
 if($('#hm').checked){const mx=Math.max(...r.heat.flat(),1),cw=r.w/32,chh=r.h/18;r.heat.forEach((row,j)=>row.forEach((v,i)=>{if(v){x.fillStyle='rgba(255,184,107,'+(.15+.8*v/mx)+')';x.fillRect(i*cw,j*chh,cw,chh)}}))}
 else r.tr.forEach(t=>{x.strokeStyle='hsl('+(t.id*47%360)+',90%,65%)';x.lineWidth=Math.max(2,r.w/500);x.beginPath();t.p.forEach((q,i)=>i?x.lineTo(q[0],q[1]):x.moveTo(q[0],q[1]));x.stroke()});
 const l=r.line;x.strokeStyle='#ffb86b';x.lineWidth=Math.max(3,r.w/400);x.setLineDash([16,10]);x.beginPath();x.moveTo(l[0]*r.w,l[1]*r.h);x.lineTo(l[2]*r.w,l[3]*r.h);x.stroke()};im.src='/files/'+JID+'/poster.jpg'}
/* ---------- Cámara en vivo ---------- */
$$('.tabs button').forEach(b=>b.onclick=()=>{$$('.tabs button').forEach(x=>x.classList.toggle('on',x==b));$('#mvid').style.display=b.dataset.m=='mvid'?'block':'none';$('#mlive').style.display=b.dataset.m=='mlive'?'block':'none';$('#out').style.display=(b.dataset.m=='mvid'&&RES)?'block':'none';if(b.dataset.m=='mlive')listCams()});
let WS=null,STRM=null,LAST=null,TRK={},ROWS=[],LF=0,LCH=null,LT0=0,LP=0,LFPS=0,RUN=false;const OFF=document.createElement('canvas');
const cfg=()=>({tracker:$('#trk').value,conf:+$('#conf').value,line:lineP(),classes:$$('.cls:checked').map(c=>+c.value)});
['conf','pos','ori'].forEach(id=>$('#'+id).addEventListener('input',()=>{if(WS&&WS.readyState==1)WS.send(JSON.stringify(cfg()))}));
async function listCams(){try{const d=(await navigator.mediaDevices.enumerateDevices()).filter(x=>x.kind=='videoinput'),s=$('#cam'),v=s.value;s.innerHTML='<option value="">Cámara predeterminada</option>'+d.map((x,i)=>'<option value="'+x.deviceId+'">'+(x.label||'Cámara '+(i+1))+'</option>').join('');s.value=v}catch(e){}}
$('#lgo').onclick=async()=>{$('#lerr').textContent='';
 try{STRM=await navigator.mediaDevices.getUserMedia({video:{deviceId:$('#cam').value?{exact:$('#cam').value}:undefined,width:{ideal:1280},height:{ideal:720}},audio:false})}catch(e){$('#lerr').textContent='No se pudo abrir la cámara. Permite el acceso en el navegador o elige otra cámara.';return}
 const v=$('#lv');v.srcObject=STRM;await v.play();listCams();
 TRK={};ROWS=[];LF=0;LAST=null;LP=0;LFPS=0;RUN=true;LT0=performance.now();$('#lgo').disabled=true;$('#lst').disabled=false;$('#ltb').innerHTML='';$('#ldl').innerHTML='';
 if(LCH)LCH.destroy();LCH=new Chart($('#lch'),{type:'line',data:{labels:[],datasets:[{label:'Objetos activos',data:[],borderColor:'#7cc4ff',pointRadius:0,tension:.3,fill:true,backgroundColor:'#7cc4ff22'}]},options:{animation:false,plugins:{legend:{display:false}},scales:{x:{ticks:{color:'#9aa3d6',maxTicksLimit:6},grid:{color:'#283060'}},y:{beginAtZero:true,ticks:{color:'#9aa3d6',precision:0},grid:{color:'#283060'}}}}});
 WS=new WebSocket((location.protocol=='https:'?'wss://':'ws://')+location.host+'/ws/live');
 WS.onopen=()=>{WS.send(JSON.stringify(cfg()));sendF()};
 WS.onmessage=e=>{const r=JSON.parse(e.data);if(r.error){$('#lerr').textContent=r.error;stopLive(false);return}onRes(r);sendF()};
 WS.onclose=()=>{if(RUN)stopLive(false)};drawL()};
$('#lst').onclick=()=>stopLive(true);
function sendF(){if(!RUN)return;const v=$('#lv');if(WS.readyState!=1||!v.videoWidth)return setTimeout(sendF,50);const w=Math.min(640,v.videoWidth);OFF.width=w;OFF.height=Math.round(w*v.videoHeight/v.videoWidth);OFF.getContext('2d').drawImage(v,0,0,OFF.width,OFF.height);OFF.toBlob(b=>{if(RUN&&WS.readyState==1)WS.send(b)},'image/jpeg',.7)}
function onRes(r){LAST=r;LF++;const v=$('#lv'),s=v.videoWidth/r.w,n=performance.now(),t=(n-LT0)/1000;LFPS=LP?LFPS*.85+.15*(1000/(n-LP)):1;LP=n;
 r.b.forEach(b=>{const cx=(b[0]+b[2])/2*s,cy=(b[1]+b[3])/2*s;(TRK[b[4]]=TRK[b[4]]||{n:b[6],p:[]}).p.push([cx,cy]);ROWS.push([b[4],b[6],LF,t.toFixed(3),cx.toFixed(1),cy.toFixed(1)])});
 $('#lk').innerHTML=[['Entran',r.cin],['Salen',r.cout],['Activos',r.b.length],['IDs únicos',Object.keys(TRK).length],['FPS de extremo a extremo',LFPS.toFixed(1)],['Latencia de la GPU (ms)',r.ms]].map(x=>'<div class="card kpi"><b>'+x[1]+'</b><span>'+x[0]+'</span></div>').join('');
 if(LF%3==0){const d=LCH.data;d.labels.push(t.toFixed(0)+' s');d.datasets[0].data.push(r.b.length);if(d.labels.length>150){d.labels.shift();d.datasets[0].data.shift()}LCH.update('none')}}
function drawL(full){const v=$('#lv'),c=$('#lc');if(!v.videoWidth){if(RUN)requestAnimationFrame(()=>drawL());return}
 c.width=v.videoWidth;c.height=v.videoHeight;const x=c.getContext('2d'),W=c.width,H=c.height,lw=Math.max(2,W/400);x.drawImage(v,0,0,W,H);
 const l=lineP();x.strokeStyle='#ffb86b';x.lineWidth=lw*1.5;x.setLineDash([14,8]);x.beginPath();x.moveTo(l[0]*W,l[1]*H);x.lineTo(l[2]*W,l[3]*H);x.stroke();x.setLineDash([]);
 for(const id in TRK){const p=full?TRK[id].p:TRK[id].p.slice(-60);x.strokeStyle='hsl('+(id*47%360)+',90%,65%)';x.lineWidth=lw;x.beginPath();p.forEach((q,i)=>i?x.lineTo(q[0],q[1]):x.moveTo(q[0],q[1]));x.stroke()}
 if(LAST&&!full){const s=W/LAST.w;x.font='600 '+Math.max(14,W/50)+'px Manrope,sans-serif';LAST.b.forEach(b=>{const col='hsl('+(b[4]*47%360)+',90%,65%)';x.strokeStyle=col;x.fillStyle=col;x.lineWidth=lw*1.2;x.strokeRect(b[0]*s,b[1]*s,(b[2]-b[0])*s,(b[3]-b[1])*s);x.fillText('#'+b[4]+' '+b[6],b[0]*s,Math.max(14,b[1]*s-6))})}
 if(LAST){x.fillStyle='#0f1226cc';x.fillRect(0,0,W/4,H/6);x.font='700 '+Math.max(16,W/40)+'px Manrope,sans-serif';x.fillStyle='#7be0a8';x.fillText('Entran: '+LAST.cin,12,H/20);x.fillStyle='#ff7a8a';x.fillText('Salen: '+LAST.cout,12,H/10)}
 if(RUN&&!full)requestAnimationFrame(()=>drawL())}
function stopLive(save){if(!RUN)return;RUN=false;if(save)drawL(true);try{WS.close()}catch(e){}STRM&&STRM.getTracks().forEach(t=>t.stop());$('#lgo').disabled=false;$('#lst').disabled=true;if(save)summary()}
function summary(){const rows=Object.entries(TRK).map(([id,t])=>{let d=0;for(let i=1;i<t.p.length;i++)d+=Math.hypot(t.p[i][0]-t.p[i-1][0],t.p[i][1]-t.p[i-1][1]);return{id,n:t.n,f:t.p.length,d:Math.round(d)}}).sort((a,b)=>b.d-a.d).slice(0,15);
 $('#ltb').innerHTML='<tr><th>ID</th><th>Clase</th><th>Detecciones</th><th>Recorrido (px)</th></tr>'+rows.map(x=>'<tr><td>#'+x.id+'</td><td>'+x.n+'</td><td>'+x.f+'</td><td>'+x.d+'</td></tr>').join('');
 const csv='id,clase,frame,tiempo_s,cx,cy\n'+ROWS.map(r=>r.join(',')).join('\n');$('#ldl').innerHTML='<a class="dl" download="trayectorias_en_vivo.csv" href="'+URL.createObjectURL(new Blob([csv],{type:'text/csv'}))+'">Descargar trayectorias_en_vivo.csv</a>'}
</script></body></html>"""

def main():
    import uvicorn
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    tunnel = None
    try:
        print("Cargando", NPOOL, "instancias de", WEIGHTS, "en", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", flush=True)
        load_models()
        thread.start()
        deadline = time.monotonic() + 30
        while not server.started:
            if not thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("No se pudo iniciar el servidor en el puerto 8000.")
            time.sleep(0.1)
        binary = Path("cloudflared")
        if not binary.is_file():
            temporary = binary.with_suffix(".download")
            try:
                subprocess.run(["wget", "-q", "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", "-O", str(temporary)], check=True)
                temporary.chmod(0o700)
                temporary.replace(binary)
            finally:
                temporary.unlink(missing_ok=True)
        tunnel = subprocess.Popen([str(binary.resolve()), "tunnel", "--url", "http://127.0.0.1:8000", "--no-autoupdate"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        for line in tunnel.stderr:
            match = re.search(r"https://[-a-z0-9]+\.trycloudflare\.com", line)
            if match:
                print("\nLINK PÚBLICO:", match.group(0), "\nDescarga los resultados antes de interrumpir esta celda.\n", flush=True)
            elif "error" in line.lower() or "failed" in line.lower():
                print(line.strip(), flush=True)
        raise RuntimeError("El túnel terminó. Ejecuta de nuevo la celda para obtener otro enlace.")
    except KeyboardInterrupt:
        print("\nCerrando el servidor y el túnel…", flush=True)
    finally:
        stopping.set()
        if tunnel is not None:
            tunnel.terminate()
            try:
                tunnel.wait(timeout=5)
            except subprocess.TimeoutExpired:
                tunnel.kill()
                tunnel.wait()
            if tunnel.stderr:
                tunnel.stderr.close()
        server.should_exit = True
        if thread.ident is not None:
            thread.join(timeout=5)
        ex.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    main()
