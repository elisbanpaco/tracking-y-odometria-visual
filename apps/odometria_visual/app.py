# -*- coding: utf-8 -*-
"""
Odometría visual con cámara en movimiento (Google Colab).
Pipeline por fotograma:
  YOLO + ByteTrack (vehículos/personas, se enmascaran como objetos dinámicos)
  -> ORB (FAST orientado + BRIEF rotado) -> emparejamiento Hamming + test de razón
  -> matriz esencial + RANSAC -> recoverPose (R, dirección de t)
  -> escala métrica con el plano del suelo (homografía + altura de la cámara)
  -> pose acumulada (trayectoria) y velocidad de la cámara
  -> velocidad de cada vehículo: "aparente" (cámara supuesta fija) vs "real" (compensada con la odometría).
Uso en Colab:  !pip -q install -r apps/odometria_visual/requirements.txt   y luego   !python apps/odometria_visual/app.py
"""
import asyncio, os, json, math, queue, re, struct, subprocess, threading, time
from collections import deque
from pathlib import Path
import cv2, numpy as np, torch
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

NPOOL = int(os.environ.get("NPOOL", 3)); WEIGHTS = os.environ.get("WEIGHTS", "yolo11m.pt")
pool = queue.Queue(); app = FastAPI()

def load_models():
    from ultralytics import YOLO
    for _ in range(NPOOL):
        m = YOLO(WEIGHTS); m.to("cuda" if torch.cuda.is_available() else "cpu")
        m.predict(np.zeros((640, 640, 3), np.uint8), verbose=False, half=torch.cuda.is_available()); pool.put(m)

class Odo:
    """Odometría visual monocular con escala por plano del suelo."""
    def __init__(s, nf=1500, fov=70., hcam=1.4):
        s.orb = cv2.ORB_create(nfeatures=nf, fastThreshold=12); s.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        s.fov, s.h, s.K, s.Ki, s.prev, s.tp = fov, hcam, None, None, None, None
        s.T = np.eye(4); s.nrm = np.array([0., 1., 0.]); s.mag = 0.; s.dist = 0.; s.sp = 0.; s.tr = {}

    def step(s, gray, mask, t):
        h, w = gray.shape
        if s.K is None:
            f = w / (2 * math.tan(math.radians(s.fov) / 2)); s.K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.]]); s.Ki = np.linalg.inv(s.K)
        kp, des = s.orb.detectAndCompute(gray, mask)
        info = dict(nk=len(kp), nm=0, ni=0, st="inicializando", flow=[])
        dt = (t - s.tp) if s.tp is not None and t - s.tp > 1e-3 else None
        if s.prev is not None and des is not None and s.prev[1] is not None and len(kp) > 20:
            pk, pd = s.prev; ok = False
            pr = [x for x in s.bf.knnMatch(pd, des, k=2) if len(x) == 2]
            ms = [a for a, b in pr if a.distance < .75 * b.distance]; info["nm"] = len(ms)
            if len(ms) >= 30:
                p1 = np.float32([pk[a.queryIdx].pt for a in ms]); p2 = np.float32([kp[a.trainIdx].pt for a in ms])
                E, mk = cv2.findEssentialMat(p1, p2, s.K, cv2.RANSAC, .999, 1.0)
                if E is not None and E.shape[0] >= 3 and E.shape[1] == 3:
                    _, R, tv, mk2 = cv2.recoverPose(E[:3], p1, p2, s.K, mask=mk)
                    inl = mk2.ravel() > 0; info["ni"] = int(inl.sum())
                    if info["ni"] >= 25:
                        ok = True
                        sel = inl & (p1[:, 1] > h * .45)                      # mitad inferior = suelo
                        if sel.sum() >= 15:
                            H, hm = cv2.findHomography(p1[sel], p2[sel], cv2.RANSAC, 2.0)
                            if H is not None and hm.sum() >= 12:
                                k, _, ts, ns = cv2.decomposeHomographyMat(H, s.K)
                                j = max(range(k), key=lambda i: abs(ns[i][1, 0])); nv = ns[j].ravel(); nv = nv if nv[1] > 0 else -nv
                                s.nrm = .85 * s.nrm + .15 * nv; s.nrm /= np.linalg.norm(s.nrm)   # normal del suelo
                                mg = float(np.linalg.norm(ts[j])) * s.h                          # |t| en metros
                                if mg < 3: s.mag = (.6 * s.mag + .4 * mg) if s.mag else mg
                        tm = tv.ravel() / (np.linalg.norm(tv) + 1e-9) * s.mag
                        Tr = np.eye(4); Tr[:3, :3] = R.T; Tr[:3, 3] = -R.T @ tm
                        s.T = s.T @ Tr; d = float(np.linalg.norm(tm)); s.dist += d
                        if dt: s.sp = .7 * s.sp + .3 * d / dt
                    o = np.argsort(-inl.astype(int))[:120]
                    info["flow"] = [[int(p1[i, 0]), int(p1[i, 1]), int(p2[i, 0]), int(p2[i, 1]), int(inl[i])] for i in o]
            info["st"] = ("ok" if info["ni"] >= 40 else "débil") if ok else "perdido"
        s.prev, s.tp = (kp, des), t
        fw, P = s.T[:3, 2], s.T[:3, 3]
        info["cam"] = dict(p=[round(float(v), 3) for v in P], yaw=round(math.degrees(math.atan2(fw[0], fw[2])), 1), v=round(s.sp * 3.6, 2), d=round(s.dist, 2))
        return info

    def ground(s, u, v):
        """Punto 3D (marco de la cámara) del píxel (u,v) suponiendo que está sobre el suelo."""
        r = s.Ki @ np.array([u, v, 1.]); den = float(s.nrm @ r)
        if den < .08: return None
        X = r * (s.h / den)
        return X if X[2] < 80 else None

    def track(s, tid, t, u, v):
        """Devuelve (v_aparente, v_real, posición mundo) en km/h y metros."""
        X = s.ground(u, v)
        if X is None: return None, None, None
        P = s.T[:3, :3] @ X + s.T[:3, 3]
        q = s.tr.setdefault(tid, deque(maxlen=40)); q.append((t, P, X))
        old = next((e for e in q if t - e[0] <= 1.2), None); pw = [round(float(a), 2) for a in P]
        if old is None or t - old[0] < .4: return None, None, pw
        dt = t - old[0]
        return round(float(np.linalg.norm(X - old[2]) / dt * 3.6), 1), round(float(np.linalg.norm(P - old[1]) / dt * 3.6), 1), pw

def process(m, od, frame, t, cfg, pers):
    t0 = time.time(); h, w = frame.shape[:2]
    r = m.track(frame, persist=pers, tracker="bytetrack.yaml", conf=.3, classes=[0, 2, 3, 5, 7], imgsz=640,
                half=torch.cuda.is_available(), verbose=False)[0]
    mask = np.full((h, w), 255, np.uint8); dets = []; b = r.boxes
    if b is not None and b.id is not None:
        for tid, bx, cl in zip(b.id.int().tolist(), b.xyxy.cpu().numpy(), b.cls.int().tolist()):
            x1, y1, x2, y2 = [int(v) for v in bx]; cv2.rectangle(mask, (x1 - 8, y1 - 8), (x2 + 8, y2 + 8), 0, -1); dets.append((tid, bx, m.names[cl]))
    info = od.step(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), mask, t); out = []
    for tid, bx, name in dets:
        vn, vc, p = od.track(tid, t, (bx[0] + bx[2]) / 2, bx[3])
        out.append(dict(bx=[int(v) for v in bx], id=tid, n=name, vn=vn, vc=vc, p=p, mv=bool(vc is not None and vc > 5), al=bool(vc is not None and vc > float(cfg["limit"]))))
    info.update(b=out, ms=round((time.time() - t0) * 1000), w=w, h=h); return info

@app.websocket("/ws/vo")
async def vo(ws: WebSocket):
    await ws.accept()
    try: m = pool.get_nowait()
    except queue.Empty:
        await ws.send_json({"error": "Todos los espacios de GPU están ocupados. Inténtalo de nuevo en unos segundos."}); await ws.close(); return
    cfg = dict(fov=70, hcam=1.4, nf=1500, limit=30); od = None; first = True
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect": break
            if msg.get("text"):
                cfg.update(json.loads(msg["text"]))
                if od: od.h = float(cfg["hcam"])
                continue
            data = msg.get("bytes") or b""
            if len(data) < 16: continue
            t = struct.unpack("<d", data[:8])[0]; frame = cv2.imdecode(np.frombuffer(data[8:], np.uint8), cv2.IMREAD_COLOR)
            if frame is None: continue
            if od is None: od = Odo(int(cfg["nf"]), float(cfg["fov"]), float(cfg["hcam"]))
            res = await asyncio.to_thread(process, m, od, frame, t, cfg, not first); first = False
            await ws.send_json(res)
    except Exception:
        pass
    finally:
        pool.put(m)

@app.get("/api/info")
def info(): return {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", "free": pool.qsize(), "slots": NPOOL}

@app.get("/", response_class=HTMLResponse)
def home(): return HTML

HTML = r"""<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Odometría visual con cámara en movimiento</title>
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;800&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{--bg:#0f1226;--pn:#171c3a;--ln:#283060;--tx:#e8ebff;--mu:#9aa3d6;--ac:#7cc4ff;--am:#ffb86b;--ok:#7be0a8;--bad:#ff7a8a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);font:15px/1.55 Manrope,system-ui,sans-serif;padding:28px 32px;max-width:1500px;margin:auto}
h1{font-size:28px;font-weight:800;margin:0 0 4px}.lead{color:var(--mu);margin:0 0 20px;max-width:80ch}.card{background:var(--pn);border:1px solid var(--ln);border-radius:16px;padding:16px}
.ctl{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px;align-items:end}label{font-size:12px;color:var(--mu);display:block;margin-bottom:4px}
select,input[type=number],input[type=file]{width:100%;background:#0f1330;color:var(--tx);border:1px solid var(--ln);border-radius:10px;padding:9px}input[type=range]{width:100%;accent-color:var(--am)}
.btn{all:unset;cursor:pointer;background:var(--am);color:#2a1700;font-weight:800;padding:12px 20px;border-radius:12px;text-align:center}.btn:disabled{opacity:.4;cursor:default}.btn.sec{background:#283060;color:var(--tx)}
.tabs{display:flex;gap:8px;margin:14px 0 10px}.tabs button{all:unset;cursor:pointer;padding:9px 16px;border-radius:10px;border:1px solid var(--ln);font-weight:600;color:var(--mu)}.tabs button.on{background:var(--ac);color:#0c1230;border-color:var(--ac)}
.g2{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px;margin:18px 0}canvas.v{width:100%;display:block;border-radius:12px;background:#000}#cv3{width:100%;height:100%;min-height:340px;display:block;border-radius:12px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(125px,1fr));gap:12px;margin:18px 0}.kpi b{display:block;font-size:26px;font-weight:800}.kpi span{color:var(--mu);font-size:12px}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:7px 8px;border-bottom:1px solid var(--ln);text-align:left}th{color:var(--mu);font-weight:600}
a.dl{color:var(--ac);font-weight:600;margin-right:16px}.chk label{display:inline-flex;gap:5px;margin-right:14px;color:var(--tx);font-size:13px}ol{margin:6px 0 0;padding-left:20px;color:#cfd5ff}li{margin-bottom:4px}
@media(max-width:700px){body{padding:16px}.g2{grid-template-columns:1fr}}
</style></head><body>
<h1>Odometría visual con cámara en movimiento</h1>
<p class="lead">Etapa 03, Nivel 3. Mueve la cámara (o sube un video caminando) y la app estima su trayectoria con ORB y RANSAC. Con esa pose estima velocidades compensadas de los vehículos. La precisión depende de la calibración y de la escena.</p>
<div class="card"><div class="tabs"><button class="on" data-m="cam">Cámara en vivo</button><button data-m="file">Video grabado</button></div>
<div class="ctl"><div id="pcam"><label>Cámara (en el celular se usa la trasera por defecto)</label><select id="cam"><option value="">Cámara predeterminada</option></select></div>
<div id="pfile" style="display:none"><label>Video caminando por el pasillo o la calle</label><input type="file" id="file" accept="video/*"><label style="margin-top:6px">Velocidad de reproducción</label><select id="rate"><option value="0.5">0.5x (más preciso)</option><option value="1">1x</option></select></div>
<div><label>Campo de visión horizontal: <b id="fv_">70</b>°</label><input type="range" id="fov" min="40" max="110" value="70"></div>
<div><label>Altura de la cámara sobre el suelo (m)</label><input type="number" id="hc" min="0.3" max="3" step="0.05" value="1.4"></div>
<div><label>Puntos ORB por cuadro: <b id="nf_">1500</b></label><input type="range" id="nf" min="500" max="3000" step="100" value="1500"></div>
<div><label>Límite de velocidad (km/h)</label><input type="number" id="lim" min="5" max="120" value="30"></div></div>
<div style="display:flex;gap:12px;margin-top:16px;flex-wrap:wrap;align-items:center"><button class="btn" id="go">Iniciar odometría</button><button class="btn sec" id="stp" disabled>Detener y guardar</button>
<div class="chk"><label><input type="checkbox" id="sf" checked>Mostrar puntos y flujo</label><label><input type="checkbox" id="fl" checked>Seguir cámara en el mapa</label></div><span id="err" style="color:var(--bad)"></span><span id="gpu" style="color:var(--mu);font-size:12px"></span></div></div>
<video id="src" playsinline muted style="display:none"></video>
<div class="kpis" id="kp"></div>
<div class="g2"><div class="card"><b>Cámara y seguimiento</b><canvas class="v" id="lc" style="margin-top:10px;min-height:220px"></canvas></div>
<div class="card" style="display:flex;flex-direction:column"><div style="display:flex;justify-content:space-between;margin-bottom:10px"><b>Mapa 3D de la trayectoria</b><button class="btn sec" id="vr" style="padding:6px 12px;font-size:12px">Restablecer vista</button></div><div style="flex:1;min-height:340px"><canvas id="cv3"></canvas></div>
<div style="color:var(--mu);font-size:12px;margin-top:8px">Naranja: trayectoria de la cámara. Esferas: vehículos (verde = estacionado, naranja = en movimiento, rojo = excede el límite). Arrastra para rotar y usa la rueda para acercar.</div></div></div>
<div class="g2"><div class="card"><b>Velocidad de la cámara (km/h)</b><canvas id="c1"></canvas></div><div class="card"><b>Inliers de RANSAC por cuadro</b><canvas id="c2"></canvas></div></div>
<div class="g2"><div class="card"><b>Vehículos: velocidad aparente y compensada estimada</b><table id="tb" style="margin-top:8px"></table><p id="dls" style="margin-bottom:0"></p></div>
<div class="card"><b>Cómo funciona</b><ol><li>YOLO detecta vehículos y personas; sus cajas se enmascaran porque son objetos dinámicos.</li><li>ORB (FAST orientado + BRIEF rotado) extrae puntos y se emparejan entre cuadros.</li><li>RANSAC sobre la matriz esencial descarta emparejamientos falsos y entrega rotación y dirección del movimiento.</li><li>La escala sale del plano del suelo (homografía de la mitad inferior y la altura de la cámara), porque una sola cámara no la puede medir.</li><li>La pose se acumula para dibujar la trayectoria.</li><li>Cada vehículo se ubica sobre el suelo en 3D. Velocidad aparente: supone la cámara fija. Velocidad compensada estimada: usa la pose para reducir el movimiento aparente causado por la cámara; no garantiza 0 km/h en un objeto estacionado.</li></ol></div></div>
<script>
const $=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)];
let SRC='cam',WS,STRM,RUN=false,LAST=null,TJ=[],CR=[],LF=0,LP=0,LFPS=0,ch1,ch2,LT0=0;const OFF=document.createElement('canvas');
$$('.tabs button').forEach(b=>b.onclick=()=>{$$('.tabs button').forEach(x=>x.classList.toggle('on',x==b));SRC=b.dataset.m;$('#pcam').style.display=SRC=='cam'?'block':'none';$('#pfile').style.display=SRC=='file'?'block':'none'});
$('#fov').oninput=e=>$('#fv_').textContent=e.target.value;$('#nf').oninput=e=>$('#nf_').textContent=e.target.value;
fetch('/api/info').then(r=>r.json()).then(i=>$('#gpu').textContent=i.gpu+' - '+i.free+'/'+i.slots+' espacios libres').catch(()=>{});
async function cams(){try{const d=(await navigator.mediaDevices.enumerateDevices()).filter(x=>x.kind=='videoinput'),s=$('#cam'),v=s.value;s.innerHTML='<option value="">Cámara predeterminada</option>'+d.map((x,i)=>'<option value="'+x.deviceId+'">'+(x.label||'Cámara '+(i+1))+'</option>').join('');s.value=v}catch(e){}}cams();
const cfg=()=>({fov:+$('#fov').value,hcam:+$('#hc').value,nf:+$('#nf').value,limit:+$('#lim').value});
['hc','lim'].forEach(id=>$('#'+id).addEventListener('input',()=>{if(WS&&WS.readyState==1)WS.send(JSON.stringify(cfg()))}));
/* ---- mapa 3D ---- */
let R3,S3,C3,OC,TL,CM,TP=new Float32Array(3*30000),TN=0,CMK={};
function init3(){const el=$('#cv3');R3=new THREE.WebGLRenderer({canvas:el,antialias:true});S3=new THREE.Scene();S3.background=new THREE.Color(0x101426);C3=new THREE.PerspectiveCamera(55,1,.1,600);C3.position.set(0,14,12);
 OC=THREE.OrbitControls?new THREE.OrbitControls(C3,el):{update(){},target:new THREE.Vector3()};OC.enableDamping=true;OC.dampingFactor=.08;
 S3.add(new THREE.AmbientLight(0xffffff,.9));const d=new THREE.DirectionalLight(0xffffff,.6);d.position.set(5,10,5);S3.add(d);S3.add(new THREE.GridHelper(200,100,0x2a3160,0x1d2347));
 const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(TP,3));g.setDrawRange(0,0);TL=new THREE.Line(g,new THREE.LineBasicMaterial({color:0xffb86b}));TL.frustumCulled=false;S3.add(TL);
 CM=new THREE.Mesh(new THREE.ConeGeometry(.5,1.4,4),new THREE.MeshStandardMaterial({color:0x7cc4ff}));CM.geometry.rotateX(-Math.PI/2);S3.add(CM);
 new ResizeObserver(rs3).observe(el.parentElement);rs3();(function l(){requestAnimationFrame(l);OC.update();R3.render(S3,C3)})()}
function rs3(){const p=$('#cv3').parentElement,w=p.clientWidth,h=p.clientHeight;if(w&&h){R3.setSize(w,h,false);C3.aspect=w/h;C3.updateProjectionMatrix()}}
$('#vr').onclick=()=>{const p=CM.position;C3.position.set(p.x,p.y+14,p.z+12);OC.target.copy(p)};
function map3(r){const[x,y,z]=r.cam.p,nx=x,ny=-y,nz=-z,old=CM.position.clone();CM.position.set(nx,ny,nz);CM.rotation.y=-r.cam.yaw*Math.PI/180;
 if(TN<30000){TP.set([nx,ny,nz],TN*3);TN++;TL.geometry.setDrawRange(0,TN);TL.geometry.attributes.position.needsUpdate=true}
 if($('#fl').checked){const d=CM.position.clone().sub(old);C3.position.add(d);OC.target.add(d)}
 const seen=new Set();r.b.forEach(b=>{if(!b.p)return;seen.add(b.id);let m=CMK[b.id];if(!m){m=CMK[b.id]=new THREE.Mesh(new THREE.SphereGeometry(.4,12,12),new THREE.MeshStandardMaterial({color:0x7be0a8}));S3.add(m)}
  m.position.set(b.p[0],-b.p[1],-b.p[2]);m.userData.l=LF;m.material.color.set(b.al?0xff7a8a:b.mv?0xffb86b:0x7be0a8)});
 for(const id in CMK)if(LF-CMK[id].userData.l>150){S3.remove(CMK[id]);delete CMK[id]}}
init3();
/* ---- captura y envío ---- */
$('#go').onclick=async()=>{$('#err').textContent='';const v=$('#src');
 try{if(SRC=='cam'){STRM=await navigator.mediaDevices.getUserMedia({video:{deviceId:$('#cam').value?{exact:$('#cam').value}:undefined,facingMode:$('#cam').value?undefined:{ideal:'environment'},width:{ideal:1280},height:{ideal:720}},audio:false});v.srcObject=STRM}
  else{const f=$('#file').files[0];if(!f){$('#err').textContent='Elige un video primero.';return}STRM=null;v.srcObject=null;v.src=URL.createObjectURL(f);v.playbackRate=+$('#rate').value;v.onended=()=>stop(true)}
  await v.play()}catch(e){$('#err').textContent='No se pudo abrir la fuente. Permite el acceso a la cámara o prueba con otra.';return}
 cams();TJ=[];CR=[];LF=0;LAST=null;LP=0;LFPS=0;RUN=true;LT0=performance.now();TN=0;TL.geometry.setDrawRange(0,0);for(const id in CMK){S3.remove(CMK[id]);delete CMK[id]}
 $('#go').disabled=true;$('#stp').disabled=false;$('#dls').innerHTML='';$('#tb').innerHTML='';
 [ch1,ch2].forEach(c=>c&&c.destroy());const mk=(id,col)=>new Chart($(id),{type:'line',data:{labels:[],datasets:[{data:[],borderColor:col,pointRadius:0,tension:.3,fill:true,backgroundColor:col+'22'}]},options:{animation:false,plugins:{legend:{display:false}},scales:{x:{ticks:{color:'#9aa3d6',maxTicksLimit:6},grid:{color:'#283060'}},y:{beginAtZero:true,ticks:{color:'#9aa3d6'},grid:{color:'#283060'}}}}});ch1=mk('#c1','#ffb86b');ch2=mk('#c2','#7cc4ff');
 WS=new WebSocket((location.protocol=='https:'?'wss://':'ws://')+location.host+'/ws/vo');
 WS.onopen=()=>{WS.send(JSON.stringify(cfg()));sendF()};
 WS.onmessage=e=>{const r=JSON.parse(e.data);if(r.error){$('#err').textContent=r.error;stop(false);return}onRes(r);sendF()};WS.onclose=()=>{if(RUN)stop(false)};draw()};
$('#stp').onclick=()=>stop(true);
function sendF(){if(!RUN)return;const v=$('#src');if(WS.readyState!=1||!v.videoWidth)return setTimeout(sendF,50);
 const w=Math.min(640,v.videoWidth);OFF.width=w;OFF.height=Math.round(w*v.videoHeight/v.videoWidth);OFF.getContext('2d').drawImage(v,0,0,OFF.width,OFF.height);
 const t=SRC=='cam'?performance.now()/1000:v.currentTime;OFF.toBlob(b=>{if(RUN&&WS.readyState==1)WS.send(new Blob([new Float64Array([t]),b]))},'image/jpeg',.75)}
function onRes(r){LAST=r;LF++;const n=performance.now(),t=(n-LT0)/1000,c=r.cam,tv=SRC=='cam'?t:$('#src').currentTime;LFPS=LP?LFPS*.85+.15*(1000/(n-LP)):1;LP=n;
 TJ.push([tv.toFixed(3),c.p[0],c.p[1],c.p[2],c.yaw,c.v,r.ni,r.st]);r.b.forEach(b=>{if(b.p)CR.push([tv.toFixed(3),b.id,b.n,b.p[0],b.p[1],b.p[2],b.vn??'',b.vc??'',b.al?'excede':b.mv?'en movimiento':'estacionado'])});
 const col=r.st=='ok'?'var(--ok)':r.st=='débil'?'var(--am)':'var(--bad)';
 $('#kp').innerHTML=[['Velocidad de la cámara (km/h)',c.v],['Distancia recorrida (m)',c.d],['Rumbo (°)',c.yaw],['Puntos ORB',r.nk],['Coincidencias',r.nm],['Inliers RANSAC',r.ni],['Estado','<span style="color:'+col+'">'+r.st+'</span>'],['FPS',LFPS.toFixed(1)]].map(x=>'<div class="card kpi"><b>'+x[1]+'</b><span>'+x[0]+'</span></div>').join('');
 map3(r);
 if(LF%3==0){[[ch1,c.v],[ch2,r.ni]].forEach(([ch,v])=>{const d=ch.data;d.labels.push(tv.toFixed(0)+' s');d.datasets[0].data.push(v);if(d.labels.length>200){d.labels.shift();d.datasets[0].data.shift()}ch.update('none')})}
 if(LF%4==0)$('#tb').innerHTML='<tr><th>ID</th><th>Clase</th><th>Aparente (km/h)</th><th>Compensada estimada (km/h)</th><th>Estado</th></tr>'+r.b.filter(b=>b.vc!=null).map(b=>'<tr><td>#'+b.id+'</td><td>'+b.n+'</td><td style="color:var(--mu)">'+b.vn+'</td><td><b>'+b.vc+'</b></td><td>'+(b.al?'excede el límite':b.mv?'en movimiento':'estacionado')+'</td></tr>').join('')}
function draw(){const v=$('#src'),c=$('#lc');if(!v.videoWidth){if(RUN)requestAnimationFrame(draw);return}
 c.width=v.videoWidth;c.height=v.videoHeight;const x=c.getContext('2d'),W=c.width,lw=Math.max(2,W/400);x.drawImage(v,0,0,W,c.height);
 if(LAST){const s=W/LAST.w;if($('#sf').checked)LAST.flow.forEach(f=>{x.strokeStyle=f[4]?'#7be0a8':'#ff7a8a';x.lineWidth=lw*.8;x.beginPath();x.moveTo(f[0]*s,f[1]*s);x.lineTo(f[2]*s,f[3]*s);x.stroke()});
  x.font='600 '+Math.max(13,W/55)+'px Manrope,sans-serif';
  LAST.b.forEach(b=>{const k=b.al?'#ff5a6e':b.mv?'#ffb86b':'#7be0a8',[a,e,g,h]=b.bx.map(q=>q*s);x.strokeStyle=k;x.fillStyle=k;x.lineWidth=lw*1.5;x.strokeRect(a,e,g-a,h-e);
   x.fillText('#'+b.id+' '+b.n,a,Math.max(16,e-6-(b.vc!=null?W/50:0)));if(b.vc!=null){x.fillText('aparente '+b.vn+' | compensada estimada '+b.vc+' km/h',a,Math.max(16,e-6))}})}
 if(RUN)requestAnimationFrame(draw)}
function stop(save){if(!RUN)return;RUN=false;try{WS.close()}catch(e){}STRM&&STRM.getTracks().forEach(t=>t.stop());$('#src').pause();$('#go').disabled=false;$('#stp').disabled=true;
 if(save){const f=(n,h,rows)=>'<a class="dl" download="'+n+'" href="'+URL.createObjectURL(new Blob([h+'\n'+rows.map(r=>r.join(',')).join('\n')],{type:'text/csv'}))+'">Descargar '+n+'</a>';
  $('#dls').innerHTML=f('trayectoria_camara.csv','tiempo_s,x_m,y_m,z_m,rumbo_deg,vel_kmh,inliers,estado',TJ)+f('vehiculos.csv','tiempo_s,id,clase,x_m,y_m,z_m,v_aparente_kmh,v_real_kmh,estado',CR)}}
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


if __name__ == "__main__":
    main()
