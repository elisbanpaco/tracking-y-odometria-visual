# -*- coding: utf-8 -*-
"""
Tracking 2D y trayectorias - Etapa 02 (Google Colab)
Ventana 1: por qué la odometría visual / Pose Tracking 3D es más difícil que el tracking 2D (animaciones 3D).
Ventana 2: tracking 2D con YOLO + ByteTrack / BoT-SORT, conteo por línea virtual y análisis de trayectorias.
Ventana 3: velocidad, zonas y calibración métrica.
Ventana 4: solución de tracking con cámara móvil mediante odometría.
Uso en Colab: !pip -q install uv; !uv pip install --system -r apps/tracking_2d/requirements.txt
Luego: !python apps/tracking_2d/app.py
Comprobar cálculos sin cargar YOLO: python app.py --autoprueba
Comprobar compensación con OpenCV: python app.py --autoprueba-planar
Comprobar video y exportaciones sin YOLO: JOBS_DIR=/tmp/prueba-video python app.py --autoprueba-video
"""
import asyncio, os, csv, json, math, time, uuid, queue, shutil, subprocess, threading, re, sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

def medir(previos, tid, t, x, y, ancho, alto):
    """Velocidad aparente del centro de cada ID y zona en una cuadrícula 3 × 3."""
    zona = f"F{min(2, max(0, int(3 * y / alto))) + 1}C{min(2, max(0, int(3 * x / ancho))) + 1}"
    anterior = previos.get(tid)
    velocidad = None
    if anterior is not None and t > anterior[0]:
        velocidad = math.hypot(x - anterior[1], y - anterior[2]) / (t - anterior[0])
    previos[tid] = (t, x, y)
    return velocidad, zona

if __name__ == "__main__" and "--autoprueba" in sys.argv:
    previos = {}
    assert medir(previos, 1, 0, 0, 0, 300, 300) == (None, "F1C1")
    assert medir(previos, 1, .5, 3, 4, 300, 300)[0] == 10
    assert medir(previos, 2, .5, 100, 100, 300, 300) == (None, "F2C2")
    assert medir(previos, 1, 2.5, 9, 12, 300, 300)[0] == 5
    assert medir(previos, 1, 3, 9, 12, 300, 300)[0] == 0
    assert medir(previos, 1, 3, 9, 12, 300, 300)[0] is None
    assert medir({}, 1, 0, 300, 300, 300, 300)[1] == "F3C3"
    assert medir({}, 1, 0, -1, -1, 300, 300)[1] == "F1C1"
    print("Autoprueba correcta: velocidad, IDs, intervalos y zonas.")
    raise SystemExit

import cv2, numpy as np

def mapa_calor(imagen, conteos):
    """Superponer densidad suavizada; cero detecciones conserva solo el fondo."""
    alto, ancho = imagen.shape[:2]
    escala = min(1, 960 / max(ancho, alto))
    fondo = cv2.resize(imagen, (max(1, round(ancho*escala)), max(1, round(alto*escala))))
    densidad = cv2.GaussianBlur(np.asarray(conteos, np.float32), (0, 0), .8)
    densidad = np.log1p(densidad)
    if densidad.max() > 0:
        densidad /= densidad.max()
    densidad = cv2.resize(densidad, (fondo.shape[1], fondo.shape[0]), interpolation=cv2.INTER_LINEAR)
    colores = cv2.applyColorMap(np.uint8(densidad*255), cv2.COLORMAP_TURBO)
    alpha = (np.sqrt(densidad)*.88)[..., None]
    return np.uint8(fondo*.55*(1-alpha) + colores*alpha)

def proyectar(homografia, puntos):
    p = np.column_stack((np.asarray(puntos, dtype=float).reshape(-1, 2), np.ones(len(puntos))))
    q = p @ homografia.T
    if not np.isfinite(q).all() or np.any(np.abs(q[:, 2]) < 1e-8):
        raise ValueError("Proyección planar inválida")
    return q[:, :2] / q[:, 2:]

class MovimientoPlanar:
    """Compensación planar respecto a una referencia capturada explícitamente."""
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=1600)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.referencia = None

    def actualizar(self, frame, cajas, capturar=False):
        alto, ancho = frame.shape[:2]
        mask = np.full((alto, ancho), 255, np.uint8)
        for x1, y1, x2, y2 in cajas:
            cv2.rectangle(mask, (max(0, int(x1)-8), max(0, int(y1)-8)),
                          (min(ancho-1, int(x2)+8), min(alto-1, int(y2)+8)), 0, -1)
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        puntos, des = self.orb.detectAndCompute(gris, mask)
        xy = np.float32([p.pt for p in puntos]).reshape(-1, 2)
        cobertura = cv2.contourArea(cv2.convexHull(xy)) / mask.size if len(xy) >= 3 else 0
        lista = des is not None and len(puntos) >= 30 and cobertura >= .03
        res = dict(ok=False, lista=bool(lista), nueva=False, inliers=0, error_px=None,
                   num_puntos=len(puntos), fondo_pct=round(100*np.count_nonzero(mask)/mask.size),
                   puntos=xy[::max(1, len(xy)//150)].round(1).tolist())
        try:
            if not lista:
                raise ValueError("Muestra más fondo con textura fuera de las cajas amarillas")
            if capturar:
                self.referencia = (puntos, des, gris.shape)
                return dict(res, ok=True, nueva=True, etapa="activa", h=np.eye(3).tolist(),
                            estado="Referencia capturada. Ahora mueve suavemente la cámara")
            if self.referencia is None:
                return dict(res, etapa="preparar", estado="Fondo listo. Pulsa 2. Capturar referencia")
            base, des_base, forma = self.referencia
            if gris.shape != forma:
                raise ValueError("Cambió la resolución; captura otra referencia")
            pares = self.matcher.knnMatch(des, des_base, k=2)
            candidatos = sorted((par[0] for par in pares if len(par)==2 and par[0].distance < .7*par[1].distance),
                                key=lambda m: m.distance)
            buenos, usados = [], set()
            for m in candidatos:
                if m.trainIdx not in usados:
                    buenos.append(m)
                    usados.add(m.trainIdx)
            if len(buenos) < 20:
                raise ValueError("Pocas correspondencias; vuelve a mirar el fondo de referencia")
            origen = np.float32([puntos[m.queryIdx].pt for m in buenos])
            destino = np.float32([base[m.trainIdx].pt for m in buenos])
            homografia, valido = cv2.findHomography(origen, destino, cv2.RANSAC, 3.0)
            res["inliers"] = int(valido.sum()) if valido is not None else 0
            if homografia is None or res["inliers"] < 15 or res["inliers"]/len(buenos) < .6:
                raise ValueError("Correspondencias no fiables; evita desenfoque y cambios de profundidad")
            aceptados = valido.ravel().astype(bool)
            for soporte in (origen[aceptados], destino[aceptados]):
                if cv2.contourArea(cv2.convexHull(soporte)) / mask.size < .03:
                    raise ValueError("Correspondencias demasiado concentradas en una región")
            esquinas = np.float32([[0,0],[ancho,0],[ancho,alto],[0,alto]])
            denominadores = np.column_stack((esquinas, np.ones(4))) @ homografia[2]
            if not (np.all(denominadores > 1e-8) or np.all(denominadores < -1e-8)):
                raise ValueError("La homografía cruza una singularidad")
            borde = proyectar(homografia, esquinas).astype(np.float32)
            area = abs(cv2.contourArea(borde))/mask.size
            if not cv2.isContourConvex(borde) or not .3 < area < 3 or np.max(np.abs(borde)) > 20*max(ancho,alto):
                raise ValueError("Transformación inestable; reduce el movimiento de cámara")
            error = np.linalg.norm(proyectar(homografia, origen[aceptados])-destino[aceptados], axis=1)
            res["error_px"] = round(float(np.median(error)), 2)
            if res["error_px"] > 2.5:
                raise ValueError("Error geométrico demasiado alto")
            # shortcut: un solo plano y solapamiento con la referencia; usar VO 3D para escenas generales.
            return dict(res, ok=True, etapa="activa", h=homografia.tolist(), estado="Compensación activa")
        except (ValueError, cv2.error) as e:
            etapa = "preparar" if self.referencia is None else "perdida"
            return dict(res, etapa=etapa, estado=("Captura rechazada. " if capturar else "") + str(e)
                        + (". Sin referencia: mediciones suspendidas" if etapa=="preparar" else
                           ". Mediciones suspendidas; recupera el encuadre o captura otra referencia"))

if __name__ == "__main__" and "--autoprueba-planar" in sys.argv:
    rng = np.random.default_rng(7)
    inicial = cv2.cvtColor(rng.integers(0,256,(360,640),dtype=np.uint8), cv2.COLOR_GRAY2BGR)
    vo = MovimientoPlanar()
    previa = vo.actualizar(inicial, [])
    assert previa["lista"] and not previa["ok"] and vo.referencia is None
    assert not vo.actualizar(inicial, [[0,0,640,360]], capturar=True)["lista"]
    assert vo.referencia is None
    assert vo.actualizar(inicial, [], capturar=True)["nueva"]
    referencia = vo.referencia
    for dx in (12,24,36):
        frame = cv2.warpAffine(inicial,np.float32([[1,0,dx],[0,1,6]]),(640,360))
        estado = vo.actualizar(frame, [])
        assert estado["ok"], estado["estado"]
        assert np.linalg.norm(proyectar(np.array(estado["h"]),[[200+dx,156]])[0]-[200,150]) < 1
    perdida = vo.actualizar(np.zeros_like(inicial), [])
    assert not perdida["ok"] and perdida["etapa"]=="perdida"
    assert vo.actualizar(inicial, [])["ok"]  # Recuperación en la misma referencia, sin mezclar coordenadas.
    assert not vo.actualizar(np.zeros_like(inicial), [], capturar=True)["ok"]
    assert vo.referencia is referencia  # Una captura rechazada no elimina la referencia anterior.
    perspectiva = np.array([[1,.01,10],[-.01,1,5],[.00002,-.00001,1]])
    movida = cv2.warpPerspective(inicial,perspectiva,(640,360))
    estado = vo.actualizar(movida, [])
    assert estado["ok"], estado["estado"]
    observado = proyectar(perspectiva,[[200,150]])
    assert np.linalg.norm(proyectar(np.array(estado["h"]),observado)[0]-[200,150]) < 1
    assert vo.actualizar(movida, [], capturar=True)["nueva"]
    assert np.allclose(vo.actualizar(movida, [])["h"],np.eye(3),atol=.01)
    con_objeto = inicial.copy()
    cv2.rectangle(con_objeto, (230,130), (310,250), (255,255,255), -1)
    prueba_objeto = MovimientoPlanar()
    assert prueba_objeto.actualizar(con_objeto, [[230,130,310,250]], capturar=True)["ok"]
    desplazada = cv2.warpAffine(inicial,np.float32([[1,0,20],[0,1,0]]),(640,360))
    cv2.rectangle(desplazada, (300,130), (380,250), (255,255,255), -1)
    estado = prueba_objeto.actualizar(desplazada, [[300,130,380,250]])
    assert estado["ok"], estado["estado"]
    # El centro se mueve 70 px en imagen: 20 de cámara y 50 de movimiento propio.
    assert np.linalg.norm(proyectar(np.array(estado["h"]),[[340,190]])[0]-[320,190]) < 1
    print("Autoprueba planar correcta: preparación, captura, compensación, rechazo y recuperación.")
    raise SystemExit

import torch
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse

W = Path(os.environ.get("JOBS_DIR", "/content/jobs")); W.mkdir(parents=True, exist_ok=True)
NPOOL = int(os.environ.get("NPOOL", 6))               # instancias de YOLO en paralelo en GPU o CPU
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

def nueva_sesion(m):
    # Conservar IDs entre frames, pero no entre videos o sesiones de webcam.
    m.reset_callbacks()
    m.predictor = None

def referencia_metrica(ref, ancho, alto):
    if ref is None:
        return None
    if not isinstance(ref, dict):
        raise ValueError("Referencia métrica inválida")
    puntos, metros = ref.get("puntos"), ref.get("metros")
    if (not isinstance(puntos, list) or len(puntos) != 2
            or type(ancho) not in (int, float) or type(alto) not in (int, float)
            or not math.isfinite(ancho) or not math.isfinite(alto) or ancho <= 0 or alto <= 0
            or any(not isinstance(p, list) or len(p) != 2 for p in puntos)
            or not all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for p in puntos for v in p)
            or type(metros) not in (int, float) or not math.isfinite(metros) or metros <= 0
            or ref.get("ancho") != ancho or ref.get("alto") != alto):
        raise ValueError("Referencia métrica inválida o de otra resolución")
    pixeles = math.hypot((puntos[1][0]-puntos[0][0])*ancho, (puntos[1][1]-puntos[0][1])*alto)
    if pixeles < 5:
        raise ValueError("La referencia debe abarcar al menos 5 píxeles")
    # shortcut: escala uniforme local; usar homografía métrica para perspectiva apreciable.
    return dict(puntos=puntos, metros=metros, pixeles=pixeles, metros_por_pixel=metros/pixeles,
                ancho=ancho, alto=alto, metodo="escala local aproximada")

def run(job, tracker, conf, line, classes, referencia=None):
    d = W / job["id"]; job["state"] = "queued"; m = pool.get(); job["state"] = "running"
    vw = gen = None
    try:
        if stopping.is_set():
            raise InterruptedError("Análisis interrumpido al cerrar el servidor.")
        if not shutil.which("ffmpeg"):
            raise ValueError("Falta FFmpeg en el servidor para generar el video reproducible")
        nueva_sesion(m)
        cap = cv2.VideoCapture(str(d / "in.mp4")); fps = cap.get(5); w, h = int(cap.get(3)), int(cap.get(4)); n = max(int(cap.get(7)), 1)
        ok, f0 = cap.read(); cap.release()
        if not ok or not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video ilegible o FPS inválidos")
        referencia = referencia_metrica(referencia, w, h)
        factor = referencia["metros_por_pixel"] if referencia else None
        cv2.imwrite(str(d / "poster.jpg"), f0)
        vw = cv2.VideoWriter(str(d / "raw.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not vw.isOpened():
            raise ValueError("No se pudo crear el video de salida; revisa el espacio disponible")
        x1, y1, x2, y2 = line[0] * w, line[1] * h, line[2] * w, line[3] * h
        side = lambda x, y: (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        last, done, tail, tr, series, pc = {}, set(), {}, [], [], {}
        cin = cout = 0; names = m.names; t0 = time.time(); previos, medidas = {}, []
        gen = m.track(str(d / "in.mp4"), stream=True, tracker=tracker + ".yaml", conf=conf, half=torch.cuda.is_available(),
                      persist=True, verbose=False, classes=classes or None, imgsz=960)
        for i, r in enumerate(gen):
            if stopping.is_set():
                raise InterruptedError("Análisis interrumpido al cerrar el servidor.")
            fr = r.orig_img.copy(); b = r.boxes; act = 0
            if b is not None and b.id is not None:
                for tid, bx, c in zip(b.id.int().tolist(), b.xyxy.cpu().numpy(), b.cls.int().tolist()):
                    cx, cy = float((bx[0] + bx[2]) / 2), float((bx[1] + bx[3]) / 2); act += 1
                    tr.append((tid, c, i, round(cx, 1), round(cy, 1)))
                    vel, zona = medir(previos, tid, i / fps, cx, cy, w, h)
                    medidas.append((None if vel is None else round(vel, 1), zona))
                    s = side(cx, cy)
                    if tid in last and last[tid] * s < 0:
                        dr = "in" if s > 0 else "out"
                        if (tid, dr) not in done:
                            done.add((tid, dr)); pc.setdefault(names[c], [0, 0])[0 if dr == "in" else 1] += 1
                            if dr == "in": cin += 1
                            else: cout += 1
                    if s:
                        last[tid] = s
                    col = color(tid); tail.setdefault(tid, deque(maxlen=45)).append((int(cx), int(cy)))
                    cv2.rectangle(fr, (int(bx[0]), int(bx[1])), (int(bx[2]), int(bx[3])), col, 2)
                    etiqueta = f"#{tid} {names[c]} {vel:.1f} px/s {zona}" if vel is not None else f"#{tid} {names[c]} -- px/s {zona}"
                    if factor is not None and vel is not None:
                        etiqueta += f" ~{vel*factor:.2f} m/s"
                    cv2.putText(fr, etiqueta, (int(bx[0]), max(18, int(bx[1]) - 6)), 0, .6, col, 2)
                    cv2.polylines(fr, [np.array(tail[tid], np.int32)], False, col, 2)
            cv2.line(fr, (int(x1), int(y1)), (int(x2), int(y2)), (80, 180, 255), 3)
            if referencia:
                a, b = [(round(p[0]*w), round(p[1]*h)) for p in referencia["puntos"]]
                cv2.line(fr, a, b, (255,255,0), 3)
                cv2.putText(fr, f"Referencia: {referencia['metros']:g} m", (a[0], max(18,a[1]-8)), 0, .6, (255,255,0), 2)
            cv2.rectangle(fr, (0, 0), (260, 70), (30, 20, 16), -1)
            cv2.putText(fr, f"Entran: {cin}", (10, 28), 0, .8, (160, 240, 160), 2); cv2.putText(fr, f"Salen: {cout}", (10, 58), 0, .8, (120, 120, 255), 2)
            vw.write(fr); series.append(act); job.update(p=min((i + 1) / n, .99), frames=i+1, total=n)
        vw.release(); dt = time.time() - t0
        if not series:
            raise ValueError("No se procesó ningún fotograma del video")
        if stopping.is_set():
            raise InterruptedError("Análisis interrumpido al cerrar el servidor.")
        job["state"] = "encoding"
        subprocess.run(["ffmpeg", "-y", "-i", str(d / "raw.mp4"), "-vf", "scale='min(960,iw)':-2", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "28", "-movflags", "+faststart", str(d / "video.mp4"), "-loglevel", "error"], check=True, capture_output=True, text=True)
        per = {}
        for tid, c, f, x, y in tr: per.setdefault(tid, [c, []])[1].append((f, x, y))
        rows = []
        for tid, (c, p) in per.items():
            dist = sum(math.dist(p[k][1:], p[k + 1][1:]) for k in range(len(p) - 1)); dur = (p[-1][0] - p[0][0]) / fps
            rows.append(dict(id=tid, cls=names[c], frames=len(p), dur=round(dur, 2), dist=round(dist), vel=round(dist / dur, 1) if dur else None,
                             velocidad_m_s=round(dist/dur*factor, 3) if dur and factor is not None else None))
        rows.sort(key=lambda r: -r["dist"])
        heat = np.zeros((18, 32), int)
        for _, _, _, x, y in tr: heat[min(17, max(0, int(y / h * 18))), min(31, max(0, int(x / w * 32)))] += 1
        if not cv2.imwrite(str(d / "mapa_calor.png"), mapa_calor(f0, heat)):
            raise ValueError("No se pudo guardar el mapa de calor")
        with open(d / "trayectorias.csv", "w", newline="") as f:
            wr = csv.writer(f); wr.writerow(["id", "clase", "frame", "tiempo_s", "cx", "cy", "velocidad_px_s", "zona", "velocidad_m_s_aprox", "metros_por_pixel", "referencia_m"])
            for (tid, c, fi, x, y), (vel, zona) in zip(tr, medidas):
                wr.writerow([tid, names[c], fi, round(fi / fps, 3), x, y, vel, zona,
                             round(vel*factor,3) if vel is not None and factor is not None else None,
                             factor, referencia["metros"] if referencia else None])
        st = max(1, len(series) // 300)
        top = [r["id"] for r in rows[:300]]
        res = dict(w=w, h=h, fps=round(fps, 1), frames=len(series), ids=len(per), cin=cin, cout=cout, pc=pc, line=line, referencia=referencia,
                   inf_fps=round(len(series) / dt, 1), mb=round((d / "video.mp4").stat().st_size / 1e6, 1), series=series[::st], st=st, rows=rows[:50], heat=heat.tolist(),
                   tr=[dict(id=t, p=[[round(a), round(b_)] for _, a, b_ in per[t][1][::max(1, len(per[t][1]) // 150)]]) for t in top])
        json.dump(res, open(d / "resultados.json", "w")); job.update(res=res, p=1, state="done")
    except subprocess.CalledProcessError as e:
        job.update(state="error", err="FFmpeg no pudo convertir el video: " + (e.stderr or str(e))[-600:])
    except Exception as e:
        job.update(state="error", err=str(e))
    finally:
        try:
            if gen is not None:
                gen.close()
        finally:
            if vw is not None:
                vw.release()
            pool.put(m)

if __name__ == "__main__" and "--autoprueba-video" in sys.argv:
    import tempfile
    from types import SimpleNamespace

    class DeteccionesSinteticas:
        names = {2: "auto"}

        def reset_callbacks(self):
            pass

        def track(self, source, **kwargs):
            cap = cv2.VideoCapture(source)
            try:
                i = 0
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    x = 32 + 32*i
                    boxes = SimpleNamespace(id=torch.tensor([1]), cls=torch.tensor([2]),
                                            xyxy=torch.tensor([[x-15,70,x+15,110]], dtype=torch.float32))
                    yield SimpleNamespace(orig_img=frame, boxes=boxes)
                    i += 1
            finally:
                cap.release()

    with tempfile.TemporaryDirectory(prefix="actividad-video-") as temporal:
        W = Path(temporal)
        d = W / "0123456789"; d.mkdir()
        writer = cv2.VideoWriter(str(d / "in.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320,180))
        assert writer.isOpened()
        for i in range(8):
            frame = np.full((180,320,3), 60, np.uint8)
            cv2.rectangle(frame, (17+32*i,70), (47+32*i,110), (220,220,220), -1)
            writer.write(frame)
        writer.release()
        pool.put(DeteccionesSinteticas())
        prueba = dict(id=d.name)
        run(prueba, "bytetrack", .3, [.5,0,.5,1], [2])
        assert prueba["state"] == "done", prueba
        with open(d / "trayectorias.csv", newline="") as f:
            muestras = list(csv.DictReader(f))
        assert len(muestras) == 8 and muestras[0]["velocidad_px_s"] == ""
        assert all(float(r["velocidad_px_s"]) == 320 for r in muestras[1:])
        assert {r["zona"] for r in muestras} == {"F2C1","F2C2","F2C3"}
        assert prueba["res"]["cin"] + prueba["res"]["cout"] == 1
        assert np.array(prueba["res"]["heat"]).sum() == 8
        calor = cv2.imread(str(d / "mapa_calor.png"))
        assert calor is not None and np.any(calor[:,:,2].astype(int)-calor[:,:,0] > 50)
        vacio = mapa_calor(np.full((180,320,3),60,np.uint8), np.zeros((18,32)))
        assert np.array_equal(vacio[:,:,0],vacio[:,:,2])  # No inventar zonas calientes sin detecciones.
        salida = cv2.VideoCapture(str(d / "video.mp4"))
        assert salida.isOpened() and int(salida.get(cv2.CAP_PROP_FRAME_COUNT)) == 8
        salida.release()
        (d / "in.mp4").write_bytes(b"archivo invalido")
        run(prueba, "bytetrack", .3, [.5,0,.5,1], [2])
        assert prueba["state"] == "error" and "Video ilegible" in prueba["err"]
        assert pool.qsize() == 1
    print("Autoprueba de video correcta: MP4 reproducible, velocidad, zonas, cruces, CSV, calor visible y error de archivo inválido. Detecciones simuladas; no prueba YOLO.")
    raise SystemExit

@app.post("/api/track")
def track(file: UploadFile = File(...), tracker: str = Form("bytetrack"), conf: float = Form(.3),
                line: str = Form("[0,0.5,1,0.5]"), classes: str = Form(""), referencia: str = Form("null")):
    try:
        linea = json.loads(line)
        ref = json.loads(referencia)
        if ref is not None:
            referencia_metrica(ref, ref.get("ancho") if isinstance(ref, dict) else 0, ref.get("alto") if isinstance(ref, dict) else 0)
        filtros = [int(c) for c in classes.split(",") if c.strip()]
        if (not isinstance(linea, list) or len(linea) != 4
                or not all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in linea)
                or linea[:2] == linea[2:] or not math.isfinite(conf) or not 0 <= conf <= 1
                or any(c < 0 for c in filtros)):
            raise ValueError
    except (ValueError, TypeError):
        raise HTTPException(400, "Línea, confianza, clases o referencia métrica inválidas")
    for p in W.iterdir():                                     # limpieza de trabajos con más de 2 h
        if p.is_dir() and time.time() - p.stat().st_mtime > 7200: shutil.rmtree(p, ignore_errors=True)
    jid = uuid.uuid4().hex[:10]; d = W / jid; d.mkdir()
    with open(d / "in.mp4", "wb") as f: shutil.copyfileobj(file.file, f)
    job = dict(id=jid, state="queued", p=0); jobs[jid] = job
    ex.submit(run, job, tracker if tracker in ("bytetrack", "botsort") else "bytetrack", conf, linea, filtros, ref)
    return {"id": jid}

@app.get("/api/job/{jid}")
def job_state(jid: str): return jobs.get(jid, {"state": "error", "err": "Trabajo no encontrado"})

@app.get("/api/info")
def info():
    return {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", "queue": sum(j["state"] == "queued" for j in jobs.values()),
            "running": sum(j["state"] == "running" for j in jobs.values()), "slots": NPOOL}

@app.get("/files/{jid}/{name}")
def files(jid: str, name: str):
    if name not in ("video.mp4", "poster.jpg", "mapa_calor.png", "trayectorias.csv", "resultados.json") or not re.fullmatch(r"[0-9a-f]{10}", jid): return JSONResponse({}, 404)
    return FileResponse(W / jid / name)

@app.websocket("/ws/live")
async def live(ws: WebSocket):
    """Tracking en tiempo real: el navegador envía fotogramas JPEG y recibe cajas, IDs y conteos."""
    await ws.accept()
    try: m = pool.get_nowait()
    except queue.Empty:
        await ws.send_json({"error": "Todos los espacios de GPU están ocupados. Inténtalo de nuevo en unos segundos."}); await ws.close(); return
    cfg = dict(tracker="bytetrack", conf=.3, line=[0, .5, 1, .5], classes=[]); first = True
    planar = None; capturar_planar = False
    last, done, pc, cin, cout = {}, set(), {}, 0, 0
    try:
        nueva_sesion(m)
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect": break
            if msg.get("text"):
                cm = json.loads(msg["text"])
                if first and cm.get("planar") is True:
                    planar = MovimientoPlanar()
                if planar is not None and cm.get("capturar_referencia") is True:
                    capturar_planar = True
                if first and cm.get("tracker") in ("bytetrack", "botsort"): cfg["tracker"] = cm["tracker"]
                if "line" in cm and cm["line"] != cfg["line"]: last.clear()
                for k in ("conf", "line", "classes"):
                    if k in cm: cfg[k] = cm[k]
                continue
            frame = cv2.imdecode(np.frombuffer(msg.get("bytes") or b"", np.uint8), cv2.IMREAD_COLOR)
            if frame is None: continue
            h, w = frame.shape[:2]; t0 = time.time()
            r = (await asyncio.to_thread(lambda: m.track(frame, persist=True, tracker=cfg["tracker"] + ".yaml", conf=float(cfg["conf"]),
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
                    if sd:
                        last[tid] = sd
                    out.append([int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3]), tid, cl, m.names[cl]])
            extra = {}
            if planar is not None:
                cajas = b.xyxy.cpu().numpy() if b is not None else []
                extra["planar"] = await asyncio.to_thread(planar.actualizar, frame, cajas, capturar_planar)
                capturar_planar = False
            await ws.send_json(dict(b=out, cin=cin, cout=cout, pc=pc, ms=round((time.time() - t0) * 1000), w=w, h=h, **extra))
    except Exception:
        pass
    finally:
        pool.put(m)

@app.get("/", response_class=HTMLResponse)
def home(): return HTML

HTML = r"""<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tracking 2D · Etapa 02</title>
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
<div class="nav"><button class="on" data-v="v1">a. Tracking 2D frente a odometría</button><button data-v="v2">b. Tracking 2D con YOLO</button><button data-v="v3">c. Velocidad y zonas</button><button data-v="v4">d. Cámara móvil: compensación planar</button></div>
<div id="gpu">Conectando con la GPU…</div></aside>
<main>
<section class="view on" id="v1"><h2>a. Tracking 2D frente a odometría visual</h2>
<p class="lead">Elige una razón. La escena 3D simula el proceso y ilustra sus diferencias. Las cargas y tiempos de la animación son ilustrativos, no mediciones.</p>
<div class="grid1"><div><div class="card stage"><canvas id="cv"></canvas>
<div id="hud"><div><span>Tracking 2D (YOLO + Kalman)</span><div class="bar"><i id="b2" style="background:var(--ok)"></i></div></div><div><span>Odometría visual / Pose 3D</span><div class="bar"><i id="b3" style="background:var(--am)"></i></div></div><div id="mt" style="grid-column:1/3;color:var(--ac);font-weight:600"></div></div></div><div class="card ctrlbar"><button class="btn" id="pp">Pausar</button><button class="btn sec" id="rst">Reiniciar</button><span>Velocidad <b id="sv">1.00x</b></span><input type="range" id="sp" min="0.1" max="2" step="0.05" value="1"><span>Línea de tiempo</span><input type="range" id="tl" min="0" max="1000" value="0" style="width:200px"><label><input type="checkbox" id="lp" checked> Repetir en bucle</label><label><input type="checkbox" id="ar"> Rotación automática</label><button class="btn sec" id="vr">Restablecer vista</button><div style="color:var(--mu);width:100%;font-size:12px">Arrastra para rotar, usa la rueda para acercar o alejar y el clic derecho para desplazar. En el móvil, un dedo rota y dos dedos acercan.</div></div></div>
<div><div class="rs card" id="rs"></div><div class="card" id="why" style="margin-top:14px"></div></div></div></section>

<section class="view" id="v2"><h2>b. Tracking 2D: conteo y trayectorias</h2>
<p class="lead">Sube un video, ubica la línea virtual y analiza qué objetos la cruzan. El procesamiento usa GPU si está disponible.</p>
<div id="btools"><div id="trackingWorkspace">
<div class="card"><div class="ctl">
<div><label>Algoritmo de seguimiento</label><select id="trk"><option value="bytetrack">YOLO + ByteTrack</option><option value="botsort">YOLO + BoT-SORT</option></select></div>
<div><label>Confianza mínima: <b id="cv_">0.30</b></label><input type="range" id="conf" min="0.1" max="0.8" step="0.05" value="0.3"></div>
<div><label>Orientación de la línea</label><select id="ori"><option value="h">Horizontal</option><option value="v">Vertical</option></select></div>
<div><label>Posición de la línea</label><input type="range" id="pos" min="5" max="95" value="50"></div></div>
<div class="chk" style="margin-top:14px"><label><input class="cls" type="checkbox" value="0" checked>Persona</label><label><input class="cls" type="checkbox" value="1">Bicicleta</label><label><input class="cls" type="checkbox" value="2" checked>Auto</label><label><input class="cls" type="checkbox" value="3" checked>Moto</label><label><input class="cls" type="checkbox" value="5" checked>Bus</label><label><input class="cls" type="checkbox" value="7" checked>Camión</label></div>
<div class="tabs"><button class="on" data-m="mvid">Archivo de video</button><button data-m="mlive">Cámara en vivo</button></div><div id="mvid"><div class="drop" id="drop" style="margin-top:14px">Arrastra un video aquí o haz clic para elegirlo<input type="file" id="file" accept="video/*" hidden></div>
<div class="pv" id="pvw" style="display:none"><video id="pv" muted controls></video><canvas id="pc"></canvas></div>
<div style="display:flex;gap:14px;align-items:center;margin-top:14px"><button class="btn" id="go" disabled>Analizar video</button><span id="st" role="status" style="color:var(--mu)"></span><button class="btn sec" id="retryvideo" style="display:none">Consultar resultado</button></div><div class="prog"><i id="pg"></i></div></div><div id="mlive" style="display:none"><div class="ctl" style="margin-top:14px"><div><label>Cámara (webcam o cámara externa)</label><select id="cam"><option value="">Cámara predeterminada</option></select></div><div style="display:flex;gap:10px"><button class="btn" id="lgo">Iniciar cámara</button><button class="btn sec" id="lst" disabled>Detener y guardar</button></div></div><div id="lerr" style="color:var(--bad);margin-top:8px"></div><div class="pv"><canvas id="lc" style="position:static;height:auto;width:100%;border-radius:12px;background:#000;min-height:220px"></canvas></div><label><input type="checkbox" id="lhm"> Mapa de calor en vivo (ocupación por detecciones)</label><div class="kpis" id="lk"></div><div class="two"><div><b>Objetos activos en vivo</b><canvas id="lch"></canvas></div><div><b>Resumen por ID</b><table id="ltb"></table><p id="ldl"></p></div></div><video id="lv" playsinline muted style="display:none"></video></div></div>
<div id="out" style="display:none">
<div class="kpis" id="kp"></div>
<div class="two"><div class="card"><video id="rv" controls style="width:100%;border-radius:12px"></video></div>
<div class="card"><div style="display:flex;justify-content:space-between"><b>Trayectorias / calor de ocupación por detecciones</b><label style="margin:0"><input type="checkbox" id="hm" checked> Mapa de calor</label></div><div class="pv"><canvas id="tc" style="position:static;height:auto;width:100%;border-radius:12px"></canvas></div>
<div id="heatlegend"><div style="height:14px;border-radius:6px;margin-top:10px;background:linear-gradient(90deg,#30123b,#4662d7,#1ae4b6,#a4fc3c,#f9ba38,#a50026)"></div><div style="display:flex;justify-content:space-between;font-size:12px"><span>Menor presencia</span><span>Mayor presencia</span></div></div><p id="heatinfo" role="status" style="font-size:12px;margin-bottom:0"></p></div></div>
<div class="two"><div class="card"><b>Objetos activos por instante</b><canvas id="c1"></canvas></div><div class="card"><b>Cruces de la línea por clase</b><canvas id="c2"></canvas></div></div>
<div class="card"><b>Objetos con mayor recorrido</b><table id="tb"></table><p id="dls" style="margin-bottom:0"></p></div></div></div></div></section>
<section class="view" id="v3"><h2>c. Velocidad y zonas</h2>
<p class="lead">Analiza un video o usa la cámara para medir velocidad aparente en px/s, registrar zonas y visualizar el mapa de calor. Las zonas F1C1 a F3C3 van de arriba abajo y de izquierda a derecha. Se conserva el análisis al cambiar entre b y c.</p>
<div class="card"><h3>Referencia real para convertir a m/s</h3>
<p>Selecciona el video abajo o abre la cámara en vivo. Pulsa «Marcar referencia», toca los dos extremos de una distancia que hayas medido e ingresa su longitud en metros. Después aplica la referencia. Para video, hazlo antes de analizar.</p>
<div style="display:flex;gap:12px;flex-wrap:wrap;align-items:end">
<button class="btn sec" id="mref">Marcar referencia</button>
<div><label for="metros">Longitud real del segmento (metros)</label><input id="metros" type="number" min="0.001" step="any" placeholder="Ej.: 2" style="color:#111;background:white;padding:10px;border-radius:8px;width:170px"></div>
<button class="btn" id="maplicar">Aplicar referencia</button><button class="btn sec" id="mborrar">Quitar referencia</button></div>
<canvas id="mcanvas" style="display:none;width:100%;max-width:960px;height:auto;touch-action:none;margin-top:12px;border:2px solid #3de2ef;border-radius:10px" aria-label="Toca dos extremos para medir la referencia"></canvas>
<p id="mestado" role="status">Sin referencia: las velocidades se muestran en px/s.</p>
<p>La velocidad en m/s es una estimación con escala local uniforme. Mantén la cámara fija y usa una referencia en el mismo plano y a una profundidad similar a los objetos. Si hay perspectiva apreciable, hace falta una homografía métrica con cuatro puntos; una sola distancia no calibra toda la escena. Al cambiar de archivo o reiniciar la cámara, marca otra referencia.</p></div>
<div id="ctools"></div>
<div class="card"><h3>c. Qué se mide</h3>
<p>v = √((x₂−x₁)² + (y₂−y₁)²) / (t₂−t₁). Se compara cada ID con su última detección. La primera no tiene velocidad. En video, t = frame/FPS; en webcam se usa el instante de captura en el navegador, no la latencia del servidor. Las etiquetas muestran px/s y una de nueve zonas; el CSV guarda ambas. El mapa de calor acumula detecciones, no personas únicas ni segundos de permanencia.</p>
<p>¿Metros por segundo? Sí, con calibración. Para una superficie plana, corregir distorsión y obtener una homografía con al menos cuatro correspondencias no colineales entre imagen y suelo medido en metros. Transformar el punto de contacto con el suelo de cada objeto y dividir su distancia métrica entre Δt. Un único factor metros/píxel solo sirve donde la escala sea aproximadamente uniforme. Sin calibración, los valores son velocidades aparentes en px/s.</p></div>
</section>
<section class="view" id="v4"><h2>d. Cámara móvil: compensación planar</h2>
<p class="lead">Compensa el movimiento aparente con ORB y una homografía sobre un plano. Compara el seguimiento original con el compensado.</p>
<div class="card"><h3>Demostración planar en vivo</h3>
<p>Apunta a un fondo plano con textura (pared con carteles o suelo), deja una persona u objeto detectable cerca de ese plano y mueve la cámara suavemente. Evita un primer plano del rostro: ocupa el fondo y tiene otra profundidad. Amarillo: trayectoria en la imagen; verde: trayectoria compensada.</p>
<div class="ctl"><div><label for="dcam">Cámara del móvil</label><select id="dcam"><option value="environment">Trasera (o webcam disponible)</option><option value="user">Frontal</option></select></div>
<button class="btn" id="dstart">1. Abrir cámara</button><button class="btn sec" id="dreset" disabled>2. Capturar referencia</button><button class="btn sec" id="dstop" disabled>Detener y guardar CSV</button></div>
<ol><li>Abre la cámara y encuadra un fondo plano con textura. Los puntos cian muestran el fondo disponible; las cajas amarillas quedan excluidas.</li><li>Cuando aparezca «Fondo listo», pulsa Capturar referencia manteniendo la cámara quieta. Esta captura establece el origen visual; no calibra la cámara.</li><li>Al aparecer «Compensación activa», mueve suavemente la cámara y compara amarillo con verde. Si pierdes el encuadre, vuelve a mirar la referencia; si cambias de escena, captura otra.</li></ol>
<p id="dstatus" role="status" aria-live="polite">Paso 1: abre la cámara.</p><p id="dquality" role="status"></p>
<div class="two"><div><b>Imagen capturada y trayectoria original</b><canvas id="draw" style="display:block;width:100%;background:#000;border-radius:12px"></canvas></div><div><b>Trayectorias sobre imagen de referencia</b><canvas id="dstable" style="display:block;width:100%;background:#000;border-radius:12px"></canvas></div></div>
<p id="dcounts">Cruces compensados: entran 0 · salen 0</p>
<div style="overflow-x:auto" role="region" aria-label="Comparación de trayectorias" tabindex="0"><table id="dtable"><tr><th>ID</th><th>Original (px/s)</th><th>Compensada (px de referencia/s)</th><th>Zona de referencia</th><th>Recorrido comparable original / compensado (px)</th></tr></table></div><p id="ddownload"></p>
<p>Las coordenadas compensadas pertenecen a la imagen capturada como referencia, de hasta 640 px de ancho; no son 3D ni metros. La línea y la cuadrícula permanecen fijas en esa referencia. Los recorridos comparables suman únicamente intervalos con compensación válida en ambos extremos. No se unen trayectorias a través de una pérdida. El CSV separa cada referencia aceptada con <code>segmento</code>.</p>
<p>Prueba 1: mueve la cámara ante un objeto estacionado próximo al plano; su recorrido compensado debe ser menor que el original. Prueba 2: con una persona caminando, debe persistir movimiento después de compensar. Se necesita suficiente fondo visible; la deriva, el paralaje, el desenfoque y los cambios de caja introducen error.</p>
<video id="dvideo" playsinline muted style="display:none"></video></div>
<div class="card" style="margin-top:16px"><h3>Compensar el movimiento de la cámara</h3>
<p>Al mover la cámara, incluso un objeto quieto cambia de posición en la imagen. Esto contamina velocidades, zonas y cruces de una línea fija en pantalla; también dificulta mantener IDs. El tracking puede seguir funcionando, pero esas medidas dejan de representar movimiento respecto al entorno.</p>
<ol><li>Calibrar la cámara y seguir puntos del fondo, excluyendo las cajas de objetos detectados por YOLO.</li><li>Estimar el movimiento de cámara con correspondencias y RANSAC. Una VO monocular obtiene rotación y dirección de traslación con matriz esencial y recoverPose; para mantener poses coherentes necesita triangulación y seguimiento del mapa, por ejemplo con PnP.</li><li>Con profundidad o un plano del suelo conocido, transformar los puntos de los objetos al mismo sistema del mundo usando la pose inversa de la cámara. Una pose sola no convierte un píxel en un punto 3D.</li><li>Asociar los objetos y medir trayectorias, velocidad y cruces en ese sistema estable. La línea y las zonas deben estar ancladas al entorno. Si se pierde la pose, suspender esas medidas hasta recuperarla.</li></ol>
<p>La VO estima cómo se mueve la cámara; YOLO + tracker mantiene qué objeto es cada uno. Ejemplo: al girar la cámara frente a un auto estacionado, su centro se mueve en píxeles, pero su posición mundial debe permanecer estable. Un peatón que camina conserva movimiento residual después de compensar la cámara.</p>
<p>La opción BoT-SORT de esta aplicación ya ofrece compensación global 2D de cámara (GMC). Ayuda a mantener IDs ante movimientos moderados, pero no es VO 3D ni estabiliza aquí el CSV, la línea o el mapa de calor. Una transformación 2D no resuelve el paralaje de una escena con distintas profundidades. La VO monocular tampoco proporciona escala métrica por sí sola: requiere una referencia conocida, estéreo, RGB-D o fusión visual-inercial con escala observable.</p>
<p>Esta demostración ejecuta compensación visual planar con ORB, correspondencias y homografía RANSAC estimada directamente contra la referencia; no implementa VO 3D. Para escenas generales se necesita el procedimiento 3D descrito arriba. Referencias: <a class="dl" href="https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html">OpenCV: pose y calibración</a><a class="dl" href="https://docs.ultralytics.com/modes/track/">Ultralytics: BoT-SORT/GMC</a></p></div></section></main>
<script>
const $=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)];
$$('.nav button').forEach(b=>b.onclick=()=>{
 $$('.nav button').forEach(x=>x.classList.toggle('on',x==b));
 $$('.view').forEach(v=>v.classList.toggle('on',v.id==b.dataset.v));
 if(b.dataset.v=='v2'||b.dataset.v=='v3'){
  // Reutilizar el análisis conserva el video, los IDs y los resultados al cambiar de sección.
  $(b.dataset.v=='v3'?'#ctools':'#btools').appendChild($('#trackingWorkspace'));
  drawLine();
 }
 resize();
});
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
 why:'La odometría suma movimientos relativos cuadro a cuadro, y los errores pueden acumularse como deriva. La optimización local la reduce; los cierres de bucle de un sistema SLAM permiten corregirla globalmente. En tracking 2D las nuevas detecciones corrigen la posición, pero pueden persistir cambios de ID.',
 run(){const N=200,gt=[],es=[];let ex=0,ey=0;for(let i=0;i<N;i++){const a=i/N*6.28;ex+=Math.sin(i*1.7)*.02+.013;ey+=Math.cos(i*2.3)*.02+.009;gt.push(new THREE.Vector3(Math.cos(a)*4,0,Math.sin(a)*4));es.push(new THREE.Vector3(Math.cos(a)*4+ex,0,Math.sin(a)*4+ey))}
  const g=LN(gt,0x7be0a8),e=LN(es,0xff7a8a);lab('Trayectoria real',[-5,.5,5.2],3);lab('Trayectoria estimada',[5,.5,5.2],3);const h=sph(.2,0xffb86b,[4,0,0]);
  return t=>{const k=Math.max(2,Math.floor(t*N));g.geometry.setDrawRange(0,k);e.geometry.setDrawRange(0,k);h.position.copy(es[k-1]);return 'Error de posición acumulado: '+gt[k-1].distanceTo(es[k-1]).toFixed(2)+' m'}}},
{n:'Correspondencias y outliers',l:[12,62],
 why:'Hay que detectar puntos característicos (ORB, SIFT, SuperPoint), emparejarlos entre cuadros y descartar los falsos con RANSAC sobre la geometría epipolar. Unos pocos emparejamientos erróneos bastan para arruinar la pose. El tracking 2D asocia detecciones mediante movimiento, IoU y, según el tracker, apariencia; también puede fallar por oclusiones.',
 run(){box(3.6,4.2,.05,0x232a55,[-4,0,0]);box(3.6,4.2,.05,0x232a55,[4,0,0]);lab('Cuadro t',[-4,2.7,0],2.4);lab('Cuadro t+1',[4,2.7,0],2.4);
  const N=36,ls=[],bad=[];for(let i=0;i<N;i++){const x=(rnd()-.5)*3,y=(rnd()-.5)*3.6,o=rnd()<.3;bad.push(o);const q=o?[(rnd()-.5)*3,(rnd()-.5)*3.6]:[x+.2,y+.1];ls.push(LN([new THREE.Vector3(-4+x,y,.1),new THREE.Vector3(4+q[0],q[1],.1)],0x7cc4ff));ls[i].visible=false}
  return t=>{const v=Math.floor(Math.min(1,t*3)*N);let ok=0;ls.forEach((l,i)=>{l.visible=i<v;const o=bad[i];l.material.color.set(t>.45?(o?0xff7a8a:0x7be0a8):0x7cc4ff);l.material.opacity=(t>.75&&o)?Math.max(0,1-(t-.75)*8):1;if(!o)ok++});return t<.45?'Emparejando puntos…':t<.75?'RANSAC: iterando hipótesis':'Inliers conservados: '+ok+' de '+N}}},
{n:'Cámara en movimiento',l:[9,55],
 why:'En seguridad la cámara está fija: el fondo suele permanecer estable, aunque puede haber sombras, reflejos o vegetación en movimiento. En odometría la cámara se mueve, así que todo el fondo se desplaza (ego-motion) y hay que separar ese flujo del movimiento propio de personas y vehículos.',
 run(){const n=500,pos=new Float32Array(n*3);for(let i=0;i<n;i++){pos[i*3]=(rnd()-.5)*16;pos[i*3+1]=(rnd()-.5)*6;pos[i*3+2]=(rnd()-.5)*16}const z0=Float32Array.from({length:n},(_,i)=>pos[i*3+2]);
  const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(pos,3));G.add(new THREE.Points(g,new THREE.PointsMaterial({color:0x9aa3d6,size:.09})));
  const cs=[0xffb86b,0xff7a8a,0x7be0a8].map((c,i)=>box(.8,.8,.8,c,[0,0,0]));
  return t=>{const ego=t>.45;for(let i=0;i<n;i++)pos[i*3+2]=((z0[i]+8+(ego?(t-.45)*40:0))%16+16)%16-8;g.attributes.position.needsUpdate=true;
   cs.forEach((c,i)=>c.position.set(Math.sin(t*9+i*2)*5,.5+i*.8-1,Math.cos(t*7+i)*3));return ego?'Cámara móvil: todo el fondo se mueve':'Cámara fija: solo se mueven los objetos'}}},
{n:'Carga computacional',l:[14,95],
 why:'El tracking 2D ejecuta un detector, un filtro de Kalman y una asociación húngara: unas decenas de milisegundos en GPU. La odometría requiere puntos, correspondencias y estimación geométrica. Sistemas más completos añaden optimización local y, en SLAM, mapa y cierre de bucle: optimización no lineal con miles de variables que debe cumplirse en tiempo real.',
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
let FILE=null,RES=null,JID=null,VIDEO_BUSY=false;
const METRIC={file:null,live:null};let MSHOT=null,MPOINTS=[],MSOURCE=null;
const fuenteM=()=>$('#mlive').style.display==='none'?'file':'live';
function conversionM(v,ref){return v===null||v===undefined||!ref?'—':(v*ref.metros_por_pixel).toFixed(3)}
function pintarReferencia(x,w,h,ref){
 if(!ref)return;const [a,b]=ref.puntos;x.save();x.setLineDash([]);x.strokeStyle=x.fillStyle='#3de2ef';x.lineWidth=3;x.font=Math.max(14,w/55)+'px sans-serif';
 x.beginPath();x.moveTo(a[0]*w,a[1]*h);x.lineTo(b[0]*w,b[1]*h);x.stroke();
 for(const p of ref.puntos){x.beginPath();x.arc(p[0]*w,p[1]*h,5,0,Math.PI*2);x.fill()}
 x.fillText('Referencia: '+ref.metros+' m',Math.min(a[0]*w,w-180),Math.max(20,a[1]*h-12));x.restore();
}
function estadoM(){const r=METRIC[fuenteM()];$('#mestado').textContent=r?'Referencia aplicada: '+r.metros+' m = '+r.pixeles.toFixed(1)+' px; '+r.metros_por_pixel.toPrecision(5)+' m/px. Velocidad estimada = px/s × este factor.':'Sin referencia: las velocidades se muestran en px/s.'}
function limpiarM(){MSHOT=null;MPOINTS=[];MSOURCE=null;$('#mcanvas').style.display='none';estadoM()}
$('#mref').onclick=()=>{
 if(VIDEO_BUSY){$('#mestado').textContent='Espera a que termine el análisis antes de modificar la referencia.';return}
 const source=fuenteM(),v=$(source==='file'?'#pv':'#lv');
 if(!v.videoWidth||!v.videoHeight||source==='live'&&!RUN){$('#mestado').textContent='Primero selecciona un video o inicia la cámara en vivo.';return}
 MSHOT=document.createElement('canvas');MSHOT.width=v.videoWidth;MSHOT.height=v.videoHeight;MSHOT.getContext('2d').drawImage(v,0,0);MPOINTS=[];MSOURCE=source;
 const c=$('#mcanvas');c.width=Math.min(960,v.videoWidth);c.height=Math.round(c.width*v.videoHeight/v.videoWidth);c.style.display='block';c.getContext('2d').drawImage(MSHOT,0,0,c.width,c.height);
 $('#mestado').textContent='Imagen congelada para medir. Toca el primer extremo y luego el segundo; la cámara sigue procesando.';
};
$('#mcanvas').onpointerdown=e=>{
 if(!MSHOT||VIDEO_BUSY)return;const c=e.currentTarget,b=c.getBoundingClientRect();if(MPOINTS.length===2)MPOINTS=[];
 MPOINTS.push([Math.max(0,Math.min(1,(e.clientX-b.left)/b.width)),Math.max(0,Math.min(1,(e.clientY-b.top)/b.height))]);
 const x=c.getContext('2d');x.drawImage(MSHOT,0,0,c.width,c.height);x.fillStyle='#3de2ef';for(const p of MPOINTS){x.beginPath();x.arc(p[0]*c.width,p[1]*c.height,6,0,Math.PI*2);x.fill()}
 if(MPOINTS.length===2){pintarReferencia(x,c.width,c.height,{puntos:MPOINTS,metros:$('#metros').value||'?'});$('#mestado').textContent='Dos extremos marcados. Ingresa su longitud real y pulsa Aplicar referencia.'}
};
$('#maplicar').onclick=()=>{
 if(VIDEO_BUSY)return;
 if(!MSHOT||MPOINTS.length!==2||MSOURCE!==fuenteM()){$('#mestado').textContent='Pulsa Marcar referencia y selecciona dos extremos en la fuente actual.';return}
 const metros=+$('#metros').value,[a,b]=MPOINTS,pixeles=Math.hypot((b[0]-a[0])*MSHOT.width,(b[1]-a[1])*MSHOT.height);
 if(!Number.isFinite(metros)||metros<=0||pixeles<5){$('#mestado').textContent='Ingresa una longitud mayor que cero y marca un segmento de al menos 5 píxeles.';return}
 METRIC[MSOURCE]={puntos:MPOINTS.map(p=>p.slice()),metros,pixeles,metros_por_pixel:metros/pixeles,ancho:MSHOT.width,alto:MSHOT.height};
 const c=$('#mcanvas'),x=c.getContext('2d');x.drawImage(MSHOT,0,0,c.width,c.height);pintarReferencia(x,c.width,c.height,METRIC[MSOURCE]);estadoM();drawLine();
 if(MSOURCE==='live'&&LAST){if(RUN)tablaLive(LAST);else{summary();drawL(true)}}
};
$('#mborrar').onclick=()=>{if(VIDEO_BUSY)return;METRIC[fuenteM()]=null;limpiarM();drawLine();if(fuenteM()==='live'&&LAST){if(RUN)tablaLive(LAST);else{summary();drawL(true)}}};
$('#conf').oninput=e=>$('#cv_').textContent=(+e.target.value).toFixed(2);
$('#drop').onclick=()=>$('#file').click();$('#file').onchange=e=>e.target.files[0]&&setFile(e.target.files[0]);
$('#drop').ondragover=e=>{e.preventDefault();$('#drop').classList.add('h')};$('#drop').ondragleave=()=>$('#drop').classList.remove('h');
$('#drop').ondrop=e=>{e.preventDefault();$('#drop').classList.remove('h');e.dataTransfer.files[0]&&setFile(e.dataTransfer.files[0])};
function setFile(f){if(VIDEO_BUSY)return;FILE=f;RES=null;JID=null;METRIC.file=null;limpiarM();$('#out').style.display='none';$('#retryvideo').style.display='none';$('#pvw').style.display='block';const v=$('#pv');v.src=URL.createObjectURL(f);v.onloadeddata=drawLine;$('#go').disabled=false;$('#drop').firstChild.textContent=f.name+' ('+(f.size/1e6).toFixed(1)+' MB)'}
const lineP=()=>{const p=$('#pos').value/100;return $('#ori').value=='h'?[0,p,1,p]:[p,0,p,1]};
function drawLine(){
 const v=$('#pv'),c=$('#pc');if(!v.videoWidth||!v.videoHeight)return;
 const escala=Math.min(v.clientWidth/v.videoWidth,v.clientHeight/v.videoHeight),w=v.videoWidth*escala,h=v.videoHeight*escala;
 c.width=Math.round(w);c.height=Math.round(h);Object.assign(c.style,{width:w+'px',height:h+'px',left:(v.clientWidth-w)/2+'px',top:(v.clientHeight-h)/2+'px',right:'auto',bottom:'auto'});
 const x=c.getContext('2d'),l=lineP();x.setLineDash([10,6]);x.strokeStyle='#ffb86b';x.lineWidth=3;x.beginPath();x.moveTo(l[0]*c.width,l[1]*c.height);x.lineTo(l[2]*c.width,l[3]*c.height);x.stroke();pintarReferencia(x,c.width,c.height,METRIC.file);
}
addEventListener('resize',drawLine);
$('#pos').oninput=$('#ori').onchange=drawLine;
function subirVideo(fd){return new Promise((resolve,reject)=>{
 const xhr=new XMLHttpRequest();xhr.open('POST','/api/track');xhr.responseType='json';xhr.timeout=600000;
 xhr.upload.onprogress=e=>{if(e.lengthComputable){const p=Math.round(100*e.loaded/e.total);$('#pg').style.width=p+'%';$('#st').textContent='Subiendo video: '+p+'% · '+(e.loaded/1e6).toFixed(1)+' / '+(e.total/1e6).toFixed(1)+' MB'}};
 xhr.upload.onload=()=>{$('#st').textContent='Video enviado. Esperando confirmación del servidor…'};
 xhr.onload=()=>{
  if(xhr.status===413)return reject(new Error('El túnel o servidor rechazó el tamaño del archivo. Comprime el video o usa un fragmento más corto (HTTP 413).'));
  if(xhr.status<200||xhr.status>=300)return reject(new Error(typeof xhr.response?.detail==='string'?xhr.response.detail:'La subida falló (HTTP '+xhr.status+'). Revisa la ejecución de Colab.'));
  if(!xhr.response?.id)return reject(new Error('El servidor no devolvió un identificador de trabajo válido.'));
  resolve(xhr.response);
 };
 xhr.onerror=()=>reject(new Error('Se perdió la conexión durante la subida. Revisa el enlace y la conexión.'));
 xhr.ontimeout=()=>reject(new Error('La subida tardó demasiado. Prueba un video más pequeño.'));
 xhr.send(fd);
})}
$('#go').onclick=async()=>{
 if(!FILE||VIDEO_BUSY)return;
 const fd=new FormData();fd.append('file',FILE);fd.append('referencia',JSON.stringify(METRIC.file));fd.append('tracker',$('#trk').value);fd.append('conf',$('#conf').value);fd.append('line',JSON.stringify(lineP()));fd.append('classes',$$('.cls:checked').map(c=>c.value).join(','));
 VIDEO_BUSY=true;RES=null;$('#go').disabled=true;$('#st').textContent='Subiendo video…';$('#pg').style.width='0%';$('#out').style.display='none';$('#retryvideo').style.display='none';
 try{const r=await subirVideo(fd);JID=r.id;$('#pg').style.width='0%';poll(JID)}
 catch(e){$('#st').textContent=e.message;VIDEO_BUSY=false;$('#go').disabled=false}
};
$('#retryvideo').onclick=()=>{if(JID){VIDEO_BUSY=true;$('#go').disabled=true;$('#retryvideo').style.display='none';poll(JID)}};
async function poll(jid,reintentos=0){
 if(jid!==JID)return;
 try{
  const respuesta=await fetch('/api/job/'+jid,{signal:AbortSignal.timeout(15000)});
  if(!respuesta.ok)throw new Error('HTTP '+respuesta.status);
  const j=await respuesta.json();if(jid!==JID)return;
  $('#pg').style.width=(j.p||0)*100+'%';
  if(j.state=='done'){if(!j.res)throw new Error('Faltan resultados');$('#st').textContent='Análisis completo';VIDEO_BUSY=false;$('#go').disabled=false;show(j.res)}
  else if(j.state=='error'){$('#st').textContent='Error: '+j.err;VIDEO_BUSY=false;$('#go').disabled=false}
  else{
   $('#st').textContent=j.state=='queued'?'En cola, esperando un modelo disponible…':j.state=='encoding'?'Preparando video reproducible, mapa de calor y CSV…':'Procesando: '+Math.round((j.p||0)*100)+'%'+(j.frames?' · '+j.frames+' / '+j.total+' fotogramas':'');
   setTimeout(()=>poll(jid),800);
  }
 }catch(e){
  if(jid!==JID)return;
  if(reintentos<2){$('#st').textContent='No se pudo consultar el avance; reintentando…';setTimeout(()=>poll(jid,reintentos+1),2000)}
  else{$('#st').textContent='Se perdió la consulta al servidor. Pulsa Consultar resultado; el trabajo podría seguir procesándose.';VIDEO_BUSY=false;$('#go').disabled=false;$('#retryvideo').style.display='inline-block'}
 }
}
let ch=[];
function show(r){RES=r;$('#out').style.display='block';$('#rv').src='/files/'+JID+'/video.mp4';
 const k=[['Entran',r.cin],['Salen',r.cout],['Objetos únicos (IDs)',r.ids],['Cuadros analizados',r.frames],['FPS de inferencia',r.inf_fps],['FPS del video',r.fps],['Tamaño del video (MB)',r.mb]];
 $('#kp').innerHTML=k.map(x=>'<div class="card kpi"><b>'+x[1]+'</b><span>'+x[0]+'</span></div>').join('');
 ch.forEach(c=>c.destroy());const o={plugins:{legend:{labels:{color:'#9aa3d6'}}},scales:{x:{ticks:{color:'#9aa3d6'},grid:{color:'#283060'}},y:{ticks:{color:'#9aa3d6'},grid:{color:'#283060'}}}};
 ch=[new Chart($('#c1'),{type:'line',data:{labels:r.series.map((_,i)=>((i*r.st)/r.fps).toFixed(1)+' s'),datasets:[{label:'Objetos activos',data:r.series,borderColor:'#7cc4ff',pointRadius:0,tension:.3,fill:true,backgroundColor:'#7cc4ff22'}]},options:o}),
 new Chart($('#c2'),{type:'bar',data:{labels:Object.keys(r.pc),datasets:[{label:'Entran',data:Object.values(r.pc).map(v=>v[0]),backgroundColor:'#7be0a8'},{label:'Salen',data:Object.values(r.pc).map(v=>v[1]),backgroundColor:'#ff7a8a'}]},options:o})];
 $('#tb').innerHTML='<tr><th>ID</th><th>Clase</th><th>Duración (s)</th><th>Recorrido (px)</th><th>Velocidad media (px/s)</th><th>Media estimada (m/s)</th></tr>'+r.rows.slice(0,15).map(x=>'<tr><td>#'+x.id+'</td><td>'+x.cls+'</td><td>'+x.dur+'</td><td>'+x.dist+'</td><td>'+(x.vel===null?'—':x.vel)+'</td><td>'+(x.velocidad_m_s??'—')+'</td></tr>').join('');
 $('#dls').textContent=r.referencia?'Referencia usada: '+r.referencia.metros+' m = '+r.referencia.pixeles.toFixed(1)+' px; '+r.referencia.metros_por_pixel.toPrecision(5)+' m/px. m/s estimados. ':'Sin referencia métrica; m/s no disponibles. ';
 $('#dls').innerHTML+=['video.mp4','mapa_calor.png','trayectorias.csv','resultados.json'].map(n=>'<a class="dl" href="/files/'+JID+'/'+n+'" download>Descargar '+n+'</a>').join('');drawT()}
$('#hm').onchange=()=>drawT();
function drawT(){
 const r=RES;if(!r)return;const calor=$('#hm').checked,im=new Image();
 $('#heatlegend').style.display=calor?'block':'none';
 const total=r.heat.flat().reduce((a,b)=>a+b,0),maximo=Math.max(...r.heat.flat(),0);
 $('#heatinfo').textContent=calor?(total?'Densidad suavizada de '+total+' detecciones. Máximo original por celda: '+maximo+'. El rojo indica mayor presencia; no cuenta objetos únicos.':'Sin detecciones: no se colorea presencia inexistente. Revisa las clases y la confianza.'):'Trayectorias por ID; las etiquetas F1C1–F3C3 identifican las nueve zonas.';
 im.onload=()=>{
  if(RES!==r||$('#hm').checked!==calor)return;
  const c=$('#tc'),escala=Math.min(1,960/Math.max(r.w,r.h));c.width=Math.round(r.w*escala);c.height=Math.round(r.h*escala);
  const x=c.getContext('2d'),w=c.width,h=c.height;x.drawImage(im,0,0,w,h);
  if(!calor){x.fillStyle='#0f122688';x.fillRect(0,0,w,h);r.tr.forEach(t=>{x.strokeStyle='hsl('+(t.id*47%360)+',90%,65%)';x.lineWidth=2;x.beginPath();t.p.forEach((q,i)=>i?x.lineTo(q[0]*w/r.w,q[1]*h/r.h):x.moveTo(q[0]*w/r.w,q[1]*h/r.h));x.stroke()})}
  x.strokeStyle='#ffffff66';x.lineWidth=1;x.setLineDash([]);
  for(let j=0;j<3;j++)for(let i=0;i<3;i++){x.strokeRect(i*w/3,j*h/3,w/3,h/3);x.fillStyle='#0f1226cc';x.fillRect(i*w/3+3,j*h/3+3,42,18);x.fillStyle='#ffffff';x.font='12px sans-serif';x.fillText('F'+(j+1)+'C'+(i+1),i*w/3+6,j*h/3+16)}
  const l=r.line;x.strokeStyle='#ffb86b';x.lineWidth=3;x.setLineDash([12,8]);x.beginPath();x.moveTo(l[0]*w,l[1]*h);x.lineTo(l[2]*w,l[3]*h);x.stroke();pintarReferencia(x,w,h,r.referencia);
 };
 im.onerror=()=>{$('#heatinfo').textContent='No se pudo cargar '+(calor?'el mapa de calor':'la imagen de referencia')+'. Revisa el servidor y vuelve a activar la casilla.'};
 im.src='/files/'+JID+'/'+(calor?'mapa_calor.png':'poster.jpg');
}
/* ---------- Cámara en vivo ---------- */
$$('.tabs button').forEach(b=>b.onclick=()=>{$$('.tabs button').forEach(x=>x.classList.toggle('on',x==b));$('#mvid').style.display=b.dataset.m=='mvid'?'block':'none';$('#mlive').style.display=b.dataset.m=='mlive'?'block':'none';$('#out').style.display=(b.dataset.m=='mvid'&&RES)?'block':'none';limpiarM();if(b.dataset.m=='mlive')listCams()});
let WS=null,STRM=null,LAST=null,TRK={},ROWS=[],LF=0,LCH=null,LT0=0,LP=0,LFPS=0,RUN=false,SENT_TIME=0,LHEAT=[];const OFF=document.createElement('canvas');
const cfg=()=>({tracker:$('#trk').value,conf:+$('#conf').value,line:lineP(),classes:$$('.cls:checked').map(c=>+c.value)});
['conf','pos','ori'].forEach(id=>$('#'+id).addEventListener('input',()=>{if(WS&&WS.readyState==1)WS.send(JSON.stringify(cfg()))}));
async function listCams(){try{const d=(await navigator.mediaDevices.enumerateDevices()).filter(x=>x.kind=='videoinput'),s=$('#cam'),v=s.value;s.innerHTML='<option value="">Cámara predeterminada</option>'+d.map((x,i)=>'<option value="'+x.deviceId+'">'+(x.label||'Cámara '+(i+1))+'</option>').join('');s.value=v}catch(e){}}
$('#lgo').onclick=async()=>{if(DRUN)stopD();$('#lerr').textContent='';
 try{STRM=await navigator.mediaDevices.getUserMedia({video:{deviceId:$('#cam').value?{exact:$('#cam').value}:undefined,width:{ideal:1280},height:{ideal:720}},audio:false})}catch(e){$('#lerr').textContent='No se pudo abrir la cámara. Permite el acceso en el navegador o elige otra cámara.';return}
 const v=$('#lv');v.srcObject=STRM;await v.play();listCams();
 METRIC.live=null;limpiarM();TRK={};ROWS=[];LHEAT=Array.from({length:18},()=>Array(32).fill(0));LF=0;LAST=null;LP=0;LFPS=0;RUN=true;LT0=performance.now();$('#lgo').disabled=true;$('#lst').disabled=false;$('#ltb').innerHTML='';$('#ldl').innerHTML='';
 if(LCH)LCH.destroy();LCH=new Chart($('#lch'),{type:'line',data:{labels:[],datasets:[{label:'Objetos activos',data:[],borderColor:'#7cc4ff',pointRadius:0,tension:.3,fill:true,backgroundColor:'#7cc4ff22'}]},options:{animation:false,plugins:{legend:{display:false}},scales:{x:{ticks:{color:'#9aa3d6',maxTicksLimit:6},grid:{color:'#283060'}},y:{beginAtZero:true,ticks:{color:'#9aa3d6',precision:0},grid:{color:'#283060'}}}}});
 WS=new WebSocket((location.protocol=='https:'?'wss://':'ws://')+location.host+'/ws/live');
 WS.onopen=()=>{WS.send(JSON.stringify(cfg()));sendF()};
 WS.onmessage=e=>{const r=JSON.parse(e.data);if(r.error){$('#lerr').textContent=r.error;stopLive(false);return}onRes(r);sendF()};
 WS.onclose=()=>{if(RUN)stopLive(false)};drawL()};
$('#lst').onclick=()=>stopLive(true);
function sendF(){if(!RUN)return;const v=$('#lv');if(WS.readyState!=1||!v.videoWidth)return setTimeout(sendF,50);const w=Math.min(640,v.videoWidth);OFF.width=w;OFF.height=Math.round(w*v.videoHeight/v.videoWidth);SENT_TIME=(performance.now()-LT0)/1000;OFF.getContext('2d').drawImage(v,0,0,OFF.width,OFF.height);OFF.toBlob(b=>{if(RUN&&WS.readyState==1)WS.send(b)},'image/jpeg',.7)}
function medirLive(prev,t,x,y,w,h){
 const zona='F'+(Math.min(2,Math.max(0,Math.floor(3*y/h)))+1)+'C'+(Math.min(2,Math.max(0,Math.floor(3*x/w)))+1);
 const vel=prev&&t>prev[0]?Math.hypot(x-prev[1],y-prev[2])/(t-prev[0]):null;
 return {vel,zona};
}
function onRes(r){LAST=r;LF++;const v=$('#lv'),s=v.videoWidth/r.w,n=performance.now(),t=SENT_TIME;LFPS=LP?LFPS*.85+.15*(1000/(n-LP)):1;LP=n;
 r.b.forEach(b=>{const cx=(b[0]+b[2])/2*s,cy=(b[1]+b[3])/2*v.videoHeight/r.h;
 const tr=TRK[b[4]]=TRK[b[4]]||{n:b[6],p:[],start:t},m=medirLive(tr.last,t,cx,cy,v.videoWidth,v.videoHeight);
 tr.last=[t,cx,cy];tr.vel=m.vel;tr.zona=m.zona;tr.p.push([cx,cy]);
 LHEAT[Math.min(17,Math.max(0,Math.floor(cy/v.videoHeight*18)))][Math.min(31,Math.max(0,Math.floor(cx/v.videoWidth*32)))]++;
 ROWS.push([b[4],b[6],LF,t.toFixed(3),cx.toFixed(1),cy.toFixed(1),m.vel===null?'':m.vel.toFixed(1),m.zona])});
 tablaLive(r);
 $('#lk').innerHTML=[['Entran',r.cin],['Salen',r.cout],['Activos',r.b.length],['IDs únicos',Object.keys(TRK).length],['FPS de extremo a extremo',LFPS.toFixed(1)],['Latencia de la GPU (ms)',r.ms]].map(x=>'<div class="card kpi"><b>'+x[1]+'</b><span>'+x[0]+'</span></div>').join('');
 if(LF%3==0){const d=LCH.data;d.labels.push(t.toFixed(0)+' s');d.datasets[0].data.push(r.b.length);if(d.labels.length>150){d.labels.shift();d.datasets[0].data.shift()}LCH.update('none')}}
function tablaLive(r){
 $('#ltb').innerHTML='<tr><th>ID</th><th>Clase</th><th>px/s</th><th>m/s estimados</th><th>Zona</th></tr>'+r.b.map(b=>{const t=TRK[b[4]];return '<tr><td>#'+b[4]+'</td><td>'+t.n+'</td><td>'+(t.vel===null?'—':t.vel.toFixed(1))+'</td><td>'+conversionM(t.vel,METRIC.live)+'</td><td>'+t.zona+'</td></tr>'}).join('');
}
function drawL(full){const v=$('#lv'),c=$('#lc');if(!v.videoWidth){if(RUN)requestAnimationFrame(()=>drawL());return}
 c.width=v.videoWidth;c.height=v.videoHeight;const x=c.getContext('2d'),W=c.width,H=c.height,lw=Math.max(2,W/400);x.drawImage(v,0,0,W,H);
 if($('#lhm').checked){const mx=Math.max(1,...LHEAT.flat());LHEAT.forEach((row,j)=>row.forEach((v,i)=>{if(v){x.fillStyle='rgba(255,184,107,'+(.15+.65*v/mx)+')';x.fillRect(i*W/32,j*H/18,W/32,H/18)}}))}
 x.strokeStyle='#ffffff55';x.lineWidth=1;for(let i=1;i<3;i++){x.beginPath();x.moveTo(i*W/3,0);x.lineTo(i*W/3,H);x.moveTo(0,i*H/3);x.lineTo(W,i*H/3);x.stroke()}
 const l=lineP();x.strokeStyle='#ffb86b';x.lineWidth=lw*1.5;x.setLineDash([14,8]);x.beginPath();x.moveTo(l[0]*W,l[1]*H);x.lineTo(l[2]*W,l[3]*H);x.stroke();x.setLineDash([]);
 for(const id in TRK){const p=full?TRK[id].p:TRK[id].p.slice(-60);x.strokeStyle='hsl('+(id*47%360)+',90%,65%)';x.lineWidth=lw;x.beginPath();p.forEach((q,i)=>i?x.lineTo(q[0],q[1]):x.moveTo(q[0],q[1]));x.stroke()}
 if(LAST&&!full){const s=W/LAST.w;x.font='600 '+Math.max(14,W/50)+'px Manrope,sans-serif';LAST.b.forEach(b=>{const col='hsl('+(b[4]*47%360)+',90%,65%)';x.strokeStyle=col;x.fillStyle=col;x.lineWidth=lw*1.2;x.strokeRect(b[0]*s,b[1]*s,(b[2]-b[0])*s,(b[3]-b[1])*s);const tr=TRK[b[4]];x.fillText('#'+b[4]+' '+b[6]+' '+(tr.vel===null?'--':tr.vel.toFixed(1))+' px/s '+(METRIC.live?'≈'+conversionM(tr.vel,METRIC.live)+' m/s ':'')+tr.zona,b[0]*s,Math.max(14,b[1]*s-6))})}
 if(LAST){x.fillStyle='#0f1226cc';x.fillRect(0,0,W/4,H/6);x.font='700 '+Math.max(16,W/40)+'px Manrope,sans-serif';x.fillStyle='#7be0a8';x.fillText('Entran: '+LAST.cin,12,H/20);x.fillStyle='#ff7a8a';x.fillText('Salen: '+LAST.cout,12,H/10)}
 pintarReferencia(x,W,H,METRIC.live);
 if(RUN&&!full)requestAnimationFrame(()=>drawL())}
function stopLive(save){if(!RUN)return;RUN=false;if(save)drawL(true);try{WS.close()}catch(e){}STRM&&STRM.getTracks().forEach(t=>t.stop());$('#lgo').disabled=false;$('#lst').disabled=true;if(save)summary()}
function summary(){const rows=Object.entries(TRK).map(([id,t])=>{let d=0;for(let i=1;i<t.p.length;i++)d+=Math.hypot(t.p[i][0]-t.p[i-1][0],t.p[i][1]-t.p[i-1][1]);return{id,n:t.n,f:t.p.length,d:Math.round(d),vel:t.last[0]>t.start?(d/(t.last[0]-t.start)).toFixed(1):"—",zona:t.zona}}).sort((a,b)=>b.d-a.d).slice(0,15);
 $('#ltb').innerHTML='<tr><th>ID</th><th>Clase</th><th>Detecciones</th><th>Recorrido (px)</th><th>Media (px/s)</th><th>Media estimada (m/s)</th><th>Última zona</th></tr>'+rows.map(x=>'<tr><td>#'+x.id+'</td><td>'+x.n+'</td><td>'+x.f+'</td><td>'+x.d+'</td><td>'+x.vel+'</td><td>'+conversionM(x.vel==='—'?null:+x.vel,METRIC.live)+'</td><td>'+x.zona+'</td></tr>').join('');
 const ref=METRIC.live;const csv='id,clase,frame,tiempo_s,cx,cy,velocidad_px_s,zona,velocidad_m_s_aprox,metros_por_pixel,referencia_m\n'+ROWS.map(r=>[...r,r[6]===''||!ref?'':conversionM(+r[6],ref),ref?.metros_por_pixel??'',ref?.metros??''].join(',')).join('\n');$('#ldl').innerHTML='<a class="dl" download="trayectorias_en_vivo.csv" href="'+URL.createObjectURL(new Blob([csv],{type:'text/csv'}))+'">Descargar trayectorias_en_vivo.csv</a>'}

/* ---------- d: compensación planar independiente de b/c ---------- */
let DRUN=false,DWS=null,DSTREAM=null,DFRAME=document.createElement('canvas'),DREF=document.createElement('canvas'),
    DHASREF=false,DCAPTURE=false,DSEG=0,DF=0,DT0=0,DTS=0,DTRACK={},DROWS=[],DIN=0,DOUT=0,DLP=null;
function proyectarD(h,p){
 const z=h[2][0]*p[0]+h[2][1]*p[1]+h[2][2];
 if(!Number.isFinite(z)||Math.abs(z)<1e-8)return null;
 const q=[(h[0][0]*p[0]+h[0][1]*p[1]+h[0][2])/z,(h[1][0]*p[0]+h[1][1]*p[1]+h[1][2])/z];
 return q.every(Number.isFinite)?q:null;
}
function resetD(){DTRACK={};DIN=0;DOUT=0;$('#dtable').innerHTML='';$('#dcounts').textContent='Cruces compensados: entran 0 · salen 0';}
$('#dstart').onclick=async()=>{
 if(DRUN)return;
 $('#dstart').disabled=true;if(RUN)stopLive(true);$('#dstatus').textContent='Abriendo cámara…';
 try{
  DSTREAM=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:$('#dcam').value},width:{ideal:640},height:{ideal:480}},audio:false});
  const v=$('#dvideo');v.srcObject=DSTREAM;await v.play();
  DRUN=true;DSEG=0;DF=0;DROWS=[];DCAPTURE=false;DHASREF=false;DREF.width=0;DLP=null;resetD();DT0=performance.now();
  $('#ddownload').innerHTML='';$('#dquality').textContent='';$('#dcam').disabled=true;$('#dstop').disabled=false;
  $('#dreset').disabled=true;$('#dreset').textContent='2. Capturar referencia';$('#dstatus').textContent='Conectando con el servidor…';
  DWS=new WebSocket((location.protocol=='https:'?'wss://':'ws://')+location.host+'/ws/live');
  const socket=DWS;
  DWS.onopen=()=>{if(!DRUN||socket!==DWS)return;socket.send(JSON.stringify({tracker:'botsort',conf:.3,planar:true}));sendD()};
  DWS.onmessage=e=>{if(!DRUN||socket!==DWS)return;const r=JSON.parse(e.data);if(r.error){$('#dstatus').textContent=r.error;stopD();return}resultD(r);sendD()};
  DWS.onerror=()=>{if(socket!==DWS)return;$('#dstatus').textContent='Error de conexión. Revisa el servidor y vuelve a iniciar.';stopD()};
  DWS.onclose=()=>{if(DRUN&&socket===DWS){$('#dstatus').textContent='Conexión cerrada; descarga los datos disponibles.';stopD()}};
 }catch(e){$('#dstatus').textContent='No se pudo iniciar: '+e.message;stopD()}
};
$('#dreset').onclick=()=>{if(!DRUN||$('#dreset').disabled)return;DCAPTURE=true;$('#dreset').disabled=true;$('#dstatus').textContent='Validando referencia: mantén la cámara quieta…'};
$('#dstop').onclick=()=>{stopD();$('#dstatus').textContent=DROWS.length?'Sesión detenida. Descarga el CSV.':'Sesión detenida sin mediciones: primero debes capturar una referencia válida.'};
function sendD(){
 if(!DRUN||DWS.readyState!=1)return;
 if(DCAPTURE){DWS.send(JSON.stringify({capturar_referencia:true}));DCAPTURE=false}
 const v=$('#dvideo');DFRAME.width=Math.min(640,v.videoWidth);DFRAME.height=Math.round(DFRAME.width*v.videoHeight/v.videoWidth);
 DTS=(performance.now()-DT0)/1000;DFRAME.getContext('2d').drawImage(v,0,0,DFRAME.width,DFRAME.height);
 const socket=DWS;DFRAME.toBlob(b=>{if(DRUN&&socket===DWS&&socket.readyState==1&&b)socket.send(b)},'image/jpeg',.85);
}
function trazoD(x,p,color){
 x.strokeStyle=color;x.lineWidth=2;x.beginPath();let continuar=false;
 p.forEach(q=>{if(!q){continuar=false;return}if(continuar)x.lineTo(q[0],q[1]);else x.moveTo(q[0],q[1]);continuar=true});x.stroke();
}
function rejillaD(x,w,h){
 x.strokeStyle='#ffffff66';x.lineWidth=1;
 for(let i=1;i<3;i++){x.beginPath();x.moveTo(i*w/3,0);x.lineTo(i*w/3,h);x.moveTo(0,i*h/3);x.lineTo(w,i*h/3);x.stroke()}
 x.strokeStyle='#ffb86b';x.setLineDash([8,6]);x.beginPath();x.moveTo(0,h/2);x.lineTo(w,h/2);x.stroke();x.setLineDash([]);
}
function resultD(r){
 DF++;const q=r.planar,w=r.w,h=r.h,c=$('#draw'),stable=$('#dstable');c.width=w;c.height=h;stable.width=w;stable.height=h;
 const x=c.getContext('2d'),y=stable.getContext('2d');x.drawImage(DFRAME,0,0,w,h);rejillaD(x,w,h);
 if(!q||!q.etapa){$('#dstatus').textContent='Actualiza app.py en el servidor para usar el flujo de captura de referencia.';stopD();return}
 if(q.nueva&&q.ok){DSEG++;resetD();DREF.width=w;DREF.height=h;DREF.getContext('2d').drawImage(DFRAME,0,0,w,h);DHASREF=true;$('#dreset').textContent='Capturar otra referencia'}
 if(DHASREF)y.drawImage(DREF,0,0,w,h);else y.drawImage(DFRAME,0,0,w,h);
 rejillaD(y,w,h);
 x.fillStyle='#56e4ed';(q.puntos||[]).forEach(p=>{x.beginPath();x.arc(p[0],p[1],2,0,Math.PI*2);x.fill()});
 if(!q.ok){y.fillStyle='#0f1226dd';y.fillRect(0,0,w,44);y.fillStyle='#ffffff';y.font='14px sans-serif';y.fillText(DHASREF?'Referencia conservada · compensación suspendida':'Vista previa · primero captura una referencia',10,27)}
 $('#dreset').disabled=!q.lista||DCAPTURE;
 $('#dstatus').textContent=(DHASREF?'Referencia '+DSEG+' · ':'Paso 2 · ')+q.estado;
 const fps=DLP===null||DTS<=DLP?'—':(1/(DTS-DLP)).toFixed(1);DLP=DTS;
 $('#dquality').textContent='Fondo disponible: '+q.fondo_pct+'% · Puntos ORB: '+q.num_puntos+' · Inliers: '+q.inliers+' · Error geométrico: '+(q.error_px===null?'—':q.error_px+' px')+' · Procesamiento: '+r.ms+' ms · Actualizaciones/s: '+fps;
 let filas='';
 // Un ID ausente o una pose perdida corta el intervalo de comparación.
 const activos=new Set(r.b.map(b=>String(b[4])));
 Object.entries(DTRACK).forEach(([id,t])=>{if(!activos.has(id)){t.prevRaw=null;t.prevStable=null;t.side=0;if(t.raw.at(-1)!==null)t.raw.push(null);if(t.stable.at(-1)!==null)t.stable.push(null)}});
 r.b.forEach(b=>{
  const id=b[4],p=[(b[0]+b[2])/2,(b[1]+b[3])/2],t=DTRACK[id]=DTRACK[id]||{raw:[],stable:[],done:new Set(),distRaw:0,distStable:0,side:0};
  const raw=medirLive(t.prevRaw,DTS,...p,w,h);t.raw.push(p);if(t.raw.length>100)t.raw.shift();
  x.strokeStyle='#ffe066';x.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1]);x.fillStyle='#ffe066';x.font='14px sans-serif';x.fillText('#'+id+' '+b[6],b[0],Math.max(15,b[1]-5));trazoD(x,t.raw,'#ffe066');
  const pos=q.ok?proyectarD(q.h,p):null;
  let vel=null,zona='',sx='',sy='';
  if(pos){
   vel=medirLive(t.prevStable,DTS,...pos,w,h).vel;
   zona=pos[0]<0||pos[0]>=w||pos[1]<0||pos[1]>=h?'Fuera':medirLive(null,DTS,...pos,w,h).zona;
   const side=pos[1]-h/2;
   if(t.prevStable&&t.prevRaw&&t.prevStable[0]===t.prevRaw[0]){
    t.distRaw+=Math.hypot(p[0]-t.prevRaw[1],p[1]-t.prevRaw[2]);t.distStable+=Math.hypot(pos[0]-t.prevStable[1],pos[1]-t.prevStable[2]);
    if(t.side*side<0){const dir=side>0?'in':'out';if(!t.done.has(dir)){t.done.add(dir);if(dir=='in')DIN++;else DOUT++}}
   }
   if(side)t.side=side;
   t.prevStable=[DTS,...pos];t.stable.push(pos);sx=pos[0].toFixed(2);sy=pos[1].toFixed(2);
   const corners=[[b[0],b[1]],[b[2],b[1]],[b[2],b[3]],[b[0],b[3]]].map(a=>proyectarD(q.h,a));
   if(corners.every(Boolean))trazoD(y,[...corners,corners[0]],'#7be0a8');
   y.fillStyle='#7be0a8';y.beginPath();y.arc(...pos,4,0,Math.PI*2);y.fill();
  }else{t.prevStable=null;t.side=0;if(t.stable.at(-1)!==null)t.stable.push(null)}
  t.prevRaw=[DTS,...p];if(t.stable.length>100)t.stable.shift();trazoD(y,t.stable,'#7be0a8');
  const formato=v=>v===null?'—':v.toFixed(2);
  filas+='<tr><td>#'+id+'</td><td>'+formato(raw.vel)+'</td><td>'+formato(vel)+'</td><td>'+(zona||'Sin compensación')+'</td><td>'+t.distRaw.toFixed(1)+' / '+t.distStable.toFixed(1)+'</td></tr>';
  if(DHASREF)DROWS.push([DSEG,DF,DTS.toFixed(3),id,b[6],...p.map(v=>v.toFixed(2)),sx,sy,raw.vel===null?'':raw.vel.toFixed(2),vel===null?'':vel.toFixed(2),zona,pos?1:0,q.inliers,DIN,DOUT,t.distRaw.toFixed(2),t.distStable.toFixed(2),q.error_px===null?'':q.error_px]);
 });
 $('#dtable').innerHTML='<tr><th>ID</th><th>Original (px/s)</th><th>Compensada (px de referencia/s)</th><th>Zona de referencia</th><th>Recorrido comparable original / compensado (px)</th></tr>'+filas;
 $('#dcounts').textContent='Cruces compensados: entran '+DIN+' · salen '+DOUT;
}
function stopD(){
 DRUN=false;try{DWS&&DWS.close()}catch(e){}DSTREAM&&DSTREAM.getTracks().forEach(t=>t.stop());
 $('#dstart').disabled=false;$('#dcam').disabled=false;$('#dreset').disabled=true;$('#dstop').disabled=true;
 if(DROWS.length){const csv='segmento,frame,tiempo_s,id,clase,cx,cy,x_referencia,y_referencia,velocidad_original_px_s,velocidad_referencia_px_s,zona_referencia,pose_valida,inliers,entran_referencia,salen_referencia,recorrido_original_comparable_px,recorrido_compensado_comparable_px,error_geometrico_px\n'+DROWS.map(r=>r.join(',')).join('\n');
 const a=document.createElement('a');a.className='dl';a.download='comparacion_planar.csv';a.textContent='Descargar comparación planar (CSV)';a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));$('#ddownload').replaceChildren(a)}
}

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
