# -*- coding: utf-8 -*-
"""Radar de tráfico · YOLO11n + ByteTrack · archivo independiente para Google Colab.

1. Selecciona un entorno con GPU (opcional) y carga el repositorio en Colab.
2. Ejecuta en una celda:
   !pip -q install -r apps/radar_trafico/requirements.txt
   !python apps/radar_trafico/app.py
3. Abre el LINK PÚBLICO impreso, carga un MP4 y calibra P1/P2.

El enlace da acceso a quien lo conozca. Descarga tus resultados antes de cerrar
Colab: /content es temporal. Interrumpir la celda cierra el servidor y el túnel.
Comprobación pequeña, sin modelos ni videos: !python apps/radar_trafico/app.py --check
"""

import asyncio
import csv
import math
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path
from typing import Literal


def pixel_scale(reference, distance, width, height):
    if not math.isfinite(distance) or not 0 < distance <= 10000:
        raise ValueError("La distancia debe ser mayor que 0 y como máximo 10000 m.")
    if len(reference) != 2 or any(
        len(p) != 2 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in p)
        for p in reference
    ):
        raise ValueError("Marca P1 y P2 dentro de la imagen.")
    if math.dist(*reference) < 0.01:
        raise ValueError("Separa los dos puntos de referencia.")
    a, b = reference
    return math.hypot((b[0] - a[0]) * width, (b[1] - a[1]) * height) / distance


class SpeedEstimator:
    """Desplazamiento neto en 0,5 s; misma interpolación que el radar original."""

    def __init__(self, scale):
        self.scale = scale
        self.tracks = {}

    def update(self, key, point, timestamp):
        samples = self.tracks.get(key)
        if samples is None or not 0 < timestamp - samples[-1][0] <= 0.5:
            self.tracks[key] = deque([(timestamp, point)])
            return None
        samples.append((timestamp, point))
        start = timestamp - 0.5
        while len(samples) > 2 and samples[1][0] <= start:
            samples.popleft()
        first_time, first_point = samples[0]
        if timestamp - first_time < 0.5 - 1e-9:
            return None
        next_time, next_point = samples[1]
        fraction = max(0, min(1, (start - first_time) / (next_time - first_time)))
        start_point = tuple(a + (b - a) * fraction for a, b in zip(first_point, next_point))
        # shortcut: una escala local supone profundidad similar; usar homografía para áreas amplias.
        return math.dist(start_point, point) / self.scale / 0.5 * 3.6

    def expire(self, timestamp):
        for key in [k for k, samples in self.tracks.items() if timestamp - samples[-1][0] > 0.5]:
            del self.tracks[key]


def self_check():
    assert pixel_scale(((0, 0), (1, 1)), 5, 300, 400) == 100
    estimator = SpeedEstimator(10)
    assert estimator.update(1, (0, 0), 0) is None
    assert estimator.update(1, (10, 0), 0.25) is None
    assert math.isclose(estimator.update(1, (30, 0), 0.75), 14.4)
    assert estimator.update(1, (40, 0), 2) is None
    assert not (60 > 60) and 60.1 > 60
    for points, distance in [(((0, 0), (0, 0)), 5), (((-1, 0), (1, 1)), 5), (((0, 0), (1, 1)), 0)]:
        try:
            pixel_scale(points, distance, 300, 400)
        except ValueError:
            pass
        else:
            raise AssertionError("Se aceptó una calibración inválida")
    print("OK: escala, interpolación, pérdida de seguimiento y límite estricto.")


if __name__ == "__main__" and "--check" in sys.argv:
    self_check()
    sys.exit(0)

import av
import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field, model_validator

ROOT = Path(os.environ.get("JOBS_DIR", "/content/traffic-monitor"))
MODEL = ROOT / "yolo11n.pt"
MAX_UPLOAD = 1024**3
CLASSES = {"car": 2, "motorcycle": 3, "bus": 5, "truck": 7}
NAMES = {"car": "Automóvil", "motorcycle": "Motocicleta", "bus": "Autobús", "truck": "Camión"}
FILES = {"output.mp4", "vehicles.csv", "heatmap.png", "traffic-by-time.png"}
jobs = {}
busy = threading.Lock()
stopping = threading.Event()
executor = ThreadPoolExecutor(max_workers=1)
app = FastAPI(title="Traffic Monitor · Colab", docs_url=None, redoc_url=None)


class UploadLimit:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] != "/api/videos" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        maximum = MAX_UPLOAD + 65536
        try:
            length = int(dict(scope["headers"]).get(b"content-length", b"0"))
        except ValueError:
            return await JSONResponse({"detail": "Content-Length inválido."}, 400)(scope, receive, send)
        if length > maximum:
            return await JSONResponse({"detail": "El máximo es 1 GiB por video."}, 413)(scope, receive, send)
        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > maximum:
                    raise HTTPException(413, "El máximo es 1 GiB por video.")
            return message

        await self.app(scope, limited_receive, send)


app.add_middleware(UploadLimit)


class Calibration(BaseModel):
    reference: tuple[tuple[float, float], tuple[float, float]]
    distance_m: float = Field(gt=0, le=10000, allow_inf_nan=False)
    reference_time: float = Field(default=0, ge=0, allow_inf_nan=False)
    speed_limit: float = Field(default=60, gt=0, le=300, allow_inf_nan=False)
    confidence: float = Field(default=0.25, ge=0.05, le=0.9, allow_inf_nan=False)
    classes: list[Literal["car", "motorcycle", "bus", "truck"]] = Field(
        default_factory=lambda: list(CLASSES), min_length=1
    )
    recording_started_at: datetime | None = None

    @model_validator(mode="after")
    def valid_points(self):
        pixel_scale(self.reference, self.distance_m, 1, 1)
        return self


def get_job(key):
    if not re.fullmatch(r"[0-9a-f]{32}", key) or key not in jobs:
        raise HTTPException(404, "Video no encontrado.")
    return jobs[key]


def directory(key):
    return ROOT / key


def inspect_video(path):
    with av.open(str(path)) as container:
        if "mp4" not in container.format.name.split(",") or not container.streams.video:
            raise ValueError("Selecciona un MP4 con una pista de video.")
        stream = container.streams.video[0]
        if not stream.width or not stream.height or stream.width * stream.height > 3840 * 2160:
            raise ValueError("Resolución inválida; máximo 3840 × 2160 o equivalente vertical.")
        duration = float(stream.duration * stream.time_base) if stream.duration else (container.duration or 0) / av.time_base
        if duration <= 0 or next(container.decode(stream), None) is None:
            raise ValueError("No se pudo leer el video o su duración.")
        return dict(width=stream.width, height=stream.height, duration=duration, fps=float(stream.average_rate or 0))


def reference_jpeg(path, timestamp):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        origin = float((stream.start_time or 0) * stream.time_base)
        container.seek(int((timestamp + origin) / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.time is not None and frame.time - origin >= timestamp - 1e-6:
                ok, data = cv2.imencode(".jpg", frame.to_ndarray(format="bgr24"))
                if ok:
                    return data.tobytes()
    raise ValueError("No hay un fotograma disponible en ese instante.")


@app.post("/api/videos", status_code=201)
async def upload(file: UploadFile = File(...)):
    key = uuid.uuid4().hex
    target = directory(key)
    try:
        if not file.filename or not file.filename.lower().endswith(".mp4"):
            raise HTTPException(415, "Selecciona un archivo MP4.")
        target.mkdir(parents=True)
        size = 0
        with (target / "input.upload").open("wb") as stream:
            while chunk := await file.read(1024**2):
                size += len(chunk)
                if size > MAX_UPLOAD:
                    raise HTTPException(413, "El máximo es 1 GiB por video.")
                await asyncio.to_thread(stream.write, chunk)
        try:
            info = await asyncio.to_thread(inspect_video, target / "input.upload")
        except Exception as exc:
            raise HTTPException(422, f"Video no válido: {exc}") from exc
        (target / "input.upload").replace(target / "input.mp4")
        job = dict(id=key, name=file.filename.replace("\\", "/").split("/")[-1][:255], state="ready", progress=0, **info)
        jobs[key] = job
        return job
    except Exception:
        (target / "input.upload").unlink(missing_ok=True)
        (target / "input.mp4").unlink(missing_ok=True)
        if target.exists():
            target.rmdir()
        raise
    finally:
        await file.close()


@app.get("/api/jobs/{key}")
def status(key: str):
    return get_job(key).copy()


@app.get("/api/jobs/{key}/frame")
def frame(key: str, time: float = Query(default=0, ge=0, allow_inf_nan=False)):
    job = get_job(key)
    if time >= job["duration"]:
        raise HTTPException(422, "El instante está fuera del video.")
    try:
        return Response(reference_jpeg(directory(key) / "input.mp4", time), media_type="image/jpeg")
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/api/jobs/{key}/preview")
def preview(key: str):
    get_job(key)
    try:
        return Response((directory(key) / "preview.jpg").read_bytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})
    except FileNotFoundError:
        raise HTTPException(404, "Esperando el primer fotograma.") from None


@app.post("/api/jobs/{key}/start")
def start(key: str, config: Calibration):
    job = get_job(key)
    if config.reference_time >= job["duration"]:
        raise HTTPException(422, "El instante está fuera del video.")
    if not busy.acquire(blocking=False):
        raise HTTPException(409, "Ya hay un análisis activo; espera a que termine.")
    try:
        if job["state"] != "ready":
            raise HTTPException(409, "Carga el video otra vez para una nueva ejecución.")
        job.update(state="running", progress=0)
        executor.submit(analyze, key, config)
    except Exception:
        busy.release()
        if job["state"] == "running":
            job["state"] = "ready"
        raise
    return job.copy()


@app.get("/files/{key}/{name}")
def download(key: str, name: str, download: bool = False):
    job = get_job(key)
    if name not in FILES or job["state"] != "done":
        raise HTTPException(404, "El resultado todavía no está disponible.")
    path = directory(key) / name
    if not path.is_file():
        raise HTTPException(404, "Archivo no encontrado.")
    return FileResponse(path, filename=f"{key[:8]}-{name}" if download else None)


def write_reports(target, tracks, grid, background, config, duration, video_name):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    fields = ["id", "clase", "video", "primera_aparicion_s", "ultima_aparicion_s", "fecha_primera_aparicion", "observaciones", "maxima_kmh", "limite_kmh", "exceso"]
    with (target / "vehicles.csv").open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.writer(output)
        writer.writerow(fields)
        safe_name = "'" + video_name if video_name.lstrip().startswith(("=", "+", "-", "@")) else video_name
        for key, track in tracks.items():
            date = (config.recording_started_at + timedelta(seconds=track["first"])).isoformat() if config.recording_started_at else ""
            writer.writerow([key, NAMES[track["class"]], safe_name, track["first"], track["last"], date, track["observations"], track["maximum"], config.speed_limit, track["alert"]])

    height, width = background.shape[:2]
    figure = Figure(figsize=(max(3.6, min(10, 6 * width / height + 1)), 6), layout="constrained", facecolor="#172129")
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    extent = (0, width, height, 0)
    ax.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB), extent=extent)
    density = cv2.GaussianBlur(grid, (0, 0), 2)
    if density.max() > 0:
        density /= density.max()
        overlay = ax.imshow(np.ma.masked_where(density < 0.01, density), extent=extent, cmap="inferno", alpha=0.25 + 0.6 * density, vmin=0, vmax=1)
        colorbar = figure.colorbar(overlay, ax=ax, fraction=0.045, pad=0.04)
        colorbar.set_ticks([0, 0.5, 1], labels=["Baja", "Media", "Alta"])
        colorbar.ax.tick_params(colors="#9aabb8", labelsize=9)
        colorbar.outline.set_edgecolor("#2a3943")
    else:
        ax.text(0.5, 0.5, "Sin trayectorias", transform=ax.transAxes, ha="center", color="white", bbox=dict(facecolor="black"))
    ax.set_title("Intensidad relativa de trayectorias", color="#e5edf3", fontsize=11, pad=12)
    ax.set_axis_off()
    figure.savefig(target / "heatmap.png", dpi=150, bbox_inches="tight", pad_inches=0.12)
    figure.clear()

    cars = [track for track in tracks.values() if track["class"] == "car"]
    if config.recording_started_at:
        origin = config.recording_started_at.replace(minute=0, second=0, microsecond=0)
        offset = (config.recording_started_at - origin).total_seconds()
        interval = 3600
        labels = [(origin + timedelta(hours=i)).isoformat(timespec="minutes") for i in range(int((offset + duration) // interval) + 1)]
        title = "Automóviles por hora de grabación"
    else:
        offset = 0
        interval = 1 if duration < 60 else 60
        labels = [str(timedelta(seconds=i * interval)) for i in range(int(duration // interval) + 1)]
        title = "Automóviles por " + ("segundo" if interval == 1 else "minuto") + " del video"
    counts = Counter(int((offset + track["first"]) // interval) for track in cars)
    figure = Figure(figsize=(10, 5.5), layout="constrained", facecolor="#172129")
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    ax.set_facecolor("#172129")
    bars = ax.bar(range(len(labels)), [counts[i] for i in range(len(labels))], color="#25c2a5", width=0.6, zorder=3)
    if len(labels) <= 30:
        ax.bar_label(bars, labels=[str(counts[i]) if counts[i] else "" for i in range(len(labels))], padding=6, color="#e5edf3", fontsize=11)
    ticks = list(range(0, len(labels), max(1, math.ceil(len(labels) / 10))))
    ax.set_xticks(ticks, [labels[i].replace("T", "\n") for i in ticks], rotation=25 if len(ticks) > 4 else 0, ha="right" if len(ticks) > 4 else "center")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(0, max(1, max(counts.values(), default=0)) * 1.2)
    ax.set_title(title, color="#e5edf3", fontsize=14, pad=18)
    ax.set_ylabel("Automóviles (IDs nuevos)", color="#9aabb8", labelpad=12)
    ax.set_xlabel("Inicio del intervalo · " + ("hora indicada de la cámara" if config.recording_started_at else "tiempo transcurrido"), color="#9aabb8", labelpad=12)
    ax.tick_params(colors="#9aabb8")
    ax.grid(axis="y", color="#2a3943", linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    figure.suptitle(f"{len(cars)} IDs de automóviles · {duration:.1f} s de video", color="#9aabb8", fontsize=11)
    figure.supxlabel("Cada ID se cuenta en su primera aparición; los intervalos extremos pueden ser parciales.", color="#9aabb8", fontsize=9)
    figure.savefig(target / "traffic-by-time.png", dpi=150, bbox_inches="tight")
    figure.clear()


def analyze(key, config):
    job, target = jobs[key], directory(key)
    output = None
    try:
        import torch
        from ultralytics import YOLO

        device = 0 if torch.cuda.is_available() else "cpu"
        job["device"] = torch.cuda.get_device_name(0) if device == 0 else "CPU"
        model = YOLO(str(MODEL))  # Una instancia nueva evita heredar IDs de otra ejecución.
        scale = pixel_scale(config.reference, config.distance_m, job["width"], job["height"])
        estimator = SpeedEstimator(scale)
        tracks, visited = {}, {}
        factor = min(1, 320 / max(job["width"], job["height"]))
        grid = np.zeros((max(1, round(job["height"] * factor)), max(1, round(job["width"] * factor))), dtype=np.float32)
        background = None
        last_emit, origin, previous = 0, None, None
        with av.open(str(target / "input.mp4")) as source:
            stream = source.streams.video[0]
            output = av.open(str(target / "output.partial.mp4"), "w", options={"movflags": "+faststart"})
            encoded = output.add_stream("libx264", rate=stream.average_rate or Fraction(30))
            encoded.width, encoded.height = stream.width + stream.width % 2, stream.height + stream.height % 2
            encoded.pix_fmt = "yuv420p"
            encoded.time_base = encoded.codec_context.time_base = Fraction(1, 90000)
            encoded.options = {"preset": "ultrafast", "crf": "23"}
            for index, frame in enumerate(source.decode(stream)):
                if stopping.is_set():
                    raise InterruptedError("Análisis interrumpido al cerrar el servidor.")
                if frame.pts is None:
                    raise ValueError("El video contiene un fotograma sin marca temporal.")
                absolute = frame.pts * frame.time_base
                if origin is None:
                    origin = absolute
                timestamp = float(absolute - origin)
                if previous is not None and timestamp <= previous:
                    raise ValueError("Las marcas temporales del video no son crecientes.")
                previous = timestamp
                image = frame.to_ndarray(format="bgr24")
                if background is None:
                    background = image.copy()
                result = model.track(image, persist=True, tracker="bytetrack.yaml", device=device, half=device == 0, imgsz=640, conf=config.confidence, classes=[CLASSES[c] for c in config.classes], verbose=False)[0]
                estimator.expire(timestamp)
                boxes = result.boxes
                detected = len(boxes) if boxes is not None else 0
                if boxes is not None:
                    ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)
                    for tid, bbox, cls in zip(ids, boxes.xyxy.tolist(), boxes.cls.int().tolist()):
                        class_name = result.names[cls]
                        x1, y1, x2, y2 = bbox
                        point = ((x1 + x2) / 2, y2)
                        speed = estimator.update(tid, point, timestamp) if tid is not None else None
                        if tid is not None:
                            track = tracks.setdefault(tid, dict(first=timestamp, last=timestamp, observations=0, maximum=None, alert=False, **{"class": class_name}))
                            track["last"] = timestamp
                            track["observations"] += 1
                            if speed is not None:
                                track["maximum"] = max(track["maximum"] or 0, speed)
                                if speed > config.speed_limit and not track["alert"]:
                                    track.update(alert=True, alert_time=timestamp)
                            row = min(grid.shape[0] - 1, max(0, int(point[1] / stream.height * grid.shape[0])))
                            column = min(grid.shape[1] - 1, max(0, int(point[0] / stream.width * grid.shape[1])))
                            cell = (row, column)
                            cells = visited.setdefault(tid, set())
                            if cell not in cells:
                                cells.add(cell)
                                grid[cell] += 1
                        color = (60, 60, 235) if speed is not None and speed > config.speed_limit else (70, 175, 80) if speed is not None else (230, 185, 65)
                        label = f"#{tid if tid is not None else '?'} {class_name} " + (f"{speed:.1f} km/h" if speed is not None else "Midiendo")
                        cv2.rectangle(image, (round(x1), round(y1)), (round(x2), round(y2)), color, 2)
                        cv2.putText(image, label, (round(x1), max(20, round(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                p1, p2 = [tuple(round(v * size) for v, size in zip(point, (stream.width, stream.height))) for point in config.reference]
                cv2.line(image, p1, p2, (240, 180, 40), 2, cv2.LINE_AA)
                cv2.putText(image, f"Referencia: {config.distance_m:g} m", p1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 180, 40), 2)
                padded = cv2.copyMakeBorder(image, 0, stream.height % 2, 0, stream.width % 2, cv2.BORDER_CONSTANT)
                annotated = av.VideoFrame.from_ndarray(padded, format="bgr24")
                annotated.pts, annotated.time_base = round((absolute - origin) * 90000), Fraction(1, 90000)
                for packet in encoded.encode(annotated):
                    output.mux(packet)
                if time.monotonic() - last_emit >= 0.5:
                    ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if not ok:
                        raise RuntimeError("No se pudo generar la vista previa.")
                    (target / "preview.tmp").write_bytes(jpeg.tobytes())
                    (target / "preview.tmp").replace(target / "preview.jpg")
                    alerts = [dict(id=tid, speed=t["maximum"], time=t["alert_time"], label=NAMES[t["class"]]) for tid, t in tracks.items() if t["alert"]]
                    maxima = [t["maximum"] for t in tracks.values() if t["maximum"] is not None]
                    job.update(progress=min(0.99, timestamp / job["duration"]), video_time=timestamp, frames=index + 1, detected=detected, tracked=len(tracks), measured=len(maxima), average=sum(maxima) / len(maxima) if maxima else None, alerts=sorted(alerts, key=lambda item: item["time"], reverse=True)[:20], alert_count=len(alerts))
                    last_emit = time.monotonic()
            if background is None:
                raise ValueError("El video no contiene fotogramas decodificables.")
            for packet in encoded.encode():
                output.mux(packet)
            output.close()
            output = None
        (target / "output.partial.mp4").replace(target / "output.mp4")
        job["state"] = "reports"
        write_reports(target, tracks, grid, background, config, timestamp, job["name"])
        maxima = [t["maximum"] for t in tracks.values() if t["maximum"] is not None]
        alerts = [dict(id=tid, speed=t["maximum"], time=t["alert_time"], label=NAMES[t["class"]]) for tid, t in tracks.items() if t["alert"]]
        job.update(state="done", progress=1, video_time=timestamp, frames=index + 1, tracked=len(tracks), measured=len(maxima), average=sum(maxima) / len(maxima) if maxima else None, alert_count=len(alerts), alerts=sorted(alerts, key=lambda item: item["time"], reverse=True)[:20])
    except Exception as exc:
        job.update(state="error", error=f"{type(exc).__name__}: {exc}")
    finally:
        if output is not None:
            try:
                output.close()
            except Exception:
                pass
        (target / "output.partial.mp4").unlink(missing_ok=True)
        busy.release()


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML


HTML = r'''<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Radar de tráfico · Colab</title>
<style>
:root{color-scheme:dark;--bg:#10171d;--card:#172129;--line:#2a3943;--text:#e5edf3;--muted:#9aabb8;--accent:#25c2a5}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.6 system-ui,sans-serif}
header{border-bottom:1px solid var(--line);padding:18px max(18px,calc((100vw - 1440px)/2));background:var(--card)}header strong{font-size:20px}header span{color:var(--muted);margin-left:16px}
main{max-width:1480px;margin:auto;padding:28px 20px}h1{font-size:26px;margin:0}h2{font-size:18px;margin:0 0 14px}p{margin:10px 0 18px}.muted,.hint{color:var(--muted)}.hint{font-size:13px}
.grid{display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:20px;align-items:start}.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px;min-width:0}.stack{display:grid;gap:16px}
label,legend{display:block;margin:12px 0 6px;font-weight:600;font-size:14px}input:not([type=checkbox]),select{width:100%;padding:10px;border:1px solid var(--line);border-radius:7px;background:var(--bg);color:var(--text);font:inherit}input[type=file]{max-width:560px}
button,.button{display:inline-block;border:1px solid var(--line);border-radius:7px;padding:10px 14px;background:var(--accent);color:#09261f;font:600 14px system-ui;cursor:pointer;text-decoration:none}button.secondary{background:var(--card);color:var(--text)}button:disabled{opacity:.5;cursor:not-allowed}button[aria-pressed=true]{outline:2px solid var(--accent);outline-offset:2px}
:focus-visible{outline:2px solid var(--accent);outline-offset:3px}.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.row input[type=number]{width:130px}.error{color:#ff8a92;overflow-wrap:anywhere}.notice{padding:12px;background:#202e37;border-radius:7px}.stage{position:relative;margin:auto;width:100%;background:#080e12}.stage img,.stage svg{display:block;width:100%;height:100%}.stage svg{position:absolute;inset:0;touch-action:none;cursor:crosshair}.stage circle{cursor:grab;stroke:white;stroke-width:3}.stage circle:focus{stroke-width:8;outline:none}.stage text{fill:white;stroke:#10171d;paint-order:stroke;stroke-width:3}
.viewer{width:100%;max-height:65vh;object-fit:contain;background:#080e12;display:block;border-radius:7px}.kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:18px 0}.kpis b{font-size:28px;display:block}.kpis span{font-size:13px;color:var(--muted)}progress{width:100%;accent-color:var(--accent);height:12px}.alerts{max-height:280px;overflow:auto;padding-left:20px}.alerts li{padding:8px 0;border-bottom:1px solid var(--line)}fieldset{border:0;padding:0;margin:0}.classes{display:flex;flex-wrap:wrap;gap:4px 14px}.classes label{font-weight:400;display:flex;align-items:center;gap:6px}.classes input{accent-color:var(--accent)}.reports{display:grid;grid-template-columns:minmax(0,.85fr) minmax(0,1.15fr);gap:16px;margin-top:18px;align-items:start}.reports img{width:100%;max-height:560px;object-fit:contain;display:block;border-radius:8px}.reports a{display:block}.reports h2{margin-bottom:6px}a{color:var(--accent)}[hidden]{display:none!important}
@media(max-width:900px){.grid{grid-template-columns:1fr}.reports{grid-template-columns:1fr}.kpis{grid-template-columns:repeat(2,minmax(0,1fr))}header span{display:block;margin:0}main{padding:20px 14px}.card{padding:16px}}
</style></head><body>
<header><strong>Traffic Monitor</strong><span>YOLO11n + ByteTrack · Google Colab</span></header>
<main><h1>Radar de tráfico</h1><p class="muted">Carga un video de una cámara fija, indica una distancia conocida y analiza el tránsito.</p>
<p class="hint">Estimaciones académicas: la perspectiva cambia la escala. Calibra cerca de los vehículos; no tiene validez para sanciones.</p>
<section class="card"><label for="file">1. Selecciona un video MP4 (máximo 1 GiB)</label><div class="row"><input id="file" type="file" accept="video/mp4,.mp4"><button id="upload">Cargar video</button></div><p id="uploadStatus" role="status" class="hint"></p></section>
<p id="error" class="error" role="alert" hidden></p>
<section id="setup" hidden style="margin-top:20px"><div class="grid">
<div class="card"><h2>2. Marca P1 y P2 en el suelo</h2><p id="videoName" class="hint"></p>
<div class="stage" id="stage"><img id="frame" alt="Fotograma de la calle para calibrar la distancia"><svg id="overlay" aria-label="Puntos de referencia sobre el video"><line id="referenceLine" stroke="#25c2a5" stroke-width="3" vector-effect="non-scaling-stroke"/><g><circle data-point="0" tabindex="0" role="button" aria-label="Punto 1. Usa las flechas para moverlo."/><text id="p1text">P1</text></g><g><circle data-point="1" tabindex="0" role="button" aria-label="Punto 2. Usa las flechas para moverlo."/><text id="p2text">P2</text></g></svg></div>
<div class="row" style="margin-top:18px"><button type="button" id="select1" class="secondary" aria-pressed="true">Marcar P1</button><button type="button" id="select2" class="secondary" aria-pressed="false">Marcar P2</button></div>
<p class="hint">Haz clic para colocar el punto seleccionado o arrastra un punto. Tab y flechas también permiten ajustarlo; Mayús mueve más rápido.</p>
<label for="frameTime">Elegir otro momento del video (segundos)</label><div class="row"><input id="frameTime" type="number" min="0" step="0.1" value="0" aria-describedby="frameHint"><button id="showFrame" class="secondary">Mostrar fotograma</button></div>
<p id="frameHint" class="hint">0 corresponde al inicio. Este campo solo selecciona la imagen para calibrar; no cambia los FPS ni la rapidez del análisis.</p></div>
<form class="card" id="config"><fieldset id="parameters"><h2>3. Configura el análisis</h2>
<label for="source">¿Cómo obtendrás la distancia?</label><select id="source"><option value="manual">Manual: conozco o medí la distancia</option><option value="maps">Medir en Google Maps</option></select>
<p class="hint" id="maps" hidden>Ubica los mismos puntos en la vista satelital. Haz clic derecho en P1, elige «Medir distancia», marca P2 y copia los metros aquí. <a href="https://www.google.com/maps" target="_blank" rel="noopener noreferrer">Abrir Google Maps</a>. No se importa automáticamente.</p>
<label for="distance">Distancia entre P1 y P2 (metros)</label><input id="distance" type="number" min="0.01" max="10000" step="any" required placeholder="Ejemplo: 10"><p id="scale" class="notice" aria-live="polite">Introduce la distancia real.</p>
<label for="limit">Límite para alertas (km/h)</label><input id="limit" type="number" min="0.01" max="300" step="any" value="60" required><p class="hint">Solo se alerta al superar este valor. No modifica la velocidad calculada.</p>
<label for="confidence">Confianza mínima</label><input id="confidence" type="number" min="0.05" max="0.9" step="0.01" value="0.25" required>
<fieldset><legend>Vehículos a detectar</legend><div class="classes"><label><input type="checkbox" name="class" value="car" checked>Auto</label><label><input type="checkbox" name="class" value="motorcycle" checked>Moto</label><label><input type="checkbox" name="class" value="bus" checked>Bus</label><label><input type="checkbox" name="class" value="truck" checked>Camión</label></div></fieldset>
<label for="recording">Inicio de grabación (opcional)</label><input id="recording" type="datetime-local"><p class="hint">Fecha y hora local de la cámara para contar autos por hora. Sin este dato, la gráfica usa el tiempo transcurrido del video.</p>
<button id="start" type="submit" disabled>Iniciar análisis</button><p class="hint">Se analizan todos los fotogramas. Cada ID obtiene velocidad después de 0,5 segundos de seguimiento continuo.</p></fieldset></form>
</div></section>
<section id="results" hidden style="margin-top:24px"><h2>Monitor y resultados</h2><p id="state" role="status"></p><progress id="progress" max="1" value="0" aria-label="Avance del análisis"></progress>
<div class="kpis"><div class="card"><b id="tracked">—</b><span>IDs seguidos</span></div><div class="card"><b id="measured">—</b><span>IDs con velocidad válida</span></div><div class="card"><b id="average">—</b><span>Promedio de máximas (km/h)</span></div><div class="card"><b id="alertCount">0</b><span>IDs con exceso de velocidad</span></div></div>
<div class="grid"><div class="card"><img id="preview" class="viewer" alt="Vista previa con detecciones y velocidades" hidden><video id="output" class="viewer" controls playsinline hidden aria-label="Video analizado"></video></div><div class="card"><h2>Alertas recientes</h2><p class="hint">Una alerta por ID. Se muestra su máxima velocidad válida.</p><ul id="alerts" class="alerts"></ul></div></div>
<div id="downloads" class="card" hidden style="margin-top:18px"><h2>Descargar resultados</h2><div id="links" class="row"></div><p class="hint">Descarga los archivos antes de cerrar Colab. Los IDs son trayectorias: un vehículo perdido puede recibir otro ID.</p></div>
<div id="reports" class="reports" hidden><div class="card"><h2>Mapa de calor</h2><p class="hint">Las zonas más cálidas concentran más trayectorias. Cada ID aporta una visita por celda.</p><a id="heatmapFull" target="_blank" rel="noopener noreferrer" aria-label="Abrir mapa de calor a tamaño completo"><img id="heatmap" alt="Mapa de calor de trayectorias"></a><p class="hint">Pulsa la imagen para verla a tamaño completo.</p></div><div class="card"><h2>Automóviles por intervalo</h2><p class="hint">Conteo por primera aparición de cada ID, no por cantidad de fotogramas. Un vehículo puede recibir otro ID si se pierde su seguimiento.</p><a id="trafficFull" target="_blank" rel="noopener noreferrer" aria-label="Abrir gráfica temporal a tamaño completo"><img id="traffic" alt="Gráfica de primeras apariciones de automóviles por intervalo"></a><p class="hint">Una barra vacía indica que no aparecieron IDs nuevos en ese intervalo.</p></div></div>
</section></main>
<script>
const $=id=>document.getElementById(id);
let job=null,points=[[.5,.35],[.5,.65]],selected=0,drag=null,imageReady=false,locked=false,frameTime=0,pollTimer=null;
const number=v=>v==null?'—':Number(v).toLocaleString('es-PE',{maximumFractionDigits:2});
function error(message){$('error').textContent=message||'';$('error').hidden=!message;}
async function api(url,options){const response=await fetch(url,options);if(!response.ok){const body=await response.json().catch(()=>null);throw new Error(typeof body?.detail==='string'?body.detail:body?.detail?.[0]?.msg||`Error HTTP ${response.status}`);}return response.json();}
function select(index){selected=index;[0,1].forEach(i=>$('select'+(i+1)).setAttribute('aria-pressed',String(i===index)));}
function draw(){if(!job)return;const w=job.width,h=job.height,coords=points.map(p=>[p[0]*w,p[1]*h]);const line=$('referenceLine');['x1','y1','x2','y2'].forEach((key,i)=>line.setAttribute(key,coords.flat()[i]));document.querySelectorAll('[data-point]').forEach((circle,i)=>{circle.setAttribute('cx',coords[i][0]);circle.setAttribute('cy',coords[i][1]);circle.setAttribute('r',Math.max(w*.012,Math.min(w,h)*.018));const text=$('p'+(i+1)+'text');text.setAttribute('x',coords[i][0]+w*.02);text.setAttribute('y',Math.max(h*.035,coords[i][1]-h*.02));text.style.fontSize=Math.max(w*.025,h*.02)+'px';});const pixels=Math.hypot((points[1][0]-points[0][0])*w,(points[1][1]-points[0][1])*h),distance=Number($('distance').value),valid=distance>0&&distance<=10000&&Math.hypot(points[1][0]-points[0][0],points[1][1]-points[0][1])>=.01;$('scale').textContent=valid?`${number(pixels)} píxeles ÷ ${number(distance)} m = ${number(pixels/distance)} píxeles por metro`:'Introduce una distancia positiva y separa P1 de P2.';$('start').disabled=locked||!imageReady||!valid;}
function position(event){const r=$('overlay').getBoundingClientRect();return [Math.max(0,Math.min(1,(event.clientX-r.left)/r.width)),Math.max(0,Math.min(1,(event.clientY-r.top)/r.height))];}
$('overlay').onpointerdown=event=>{if(locked||!imageReady)return;const handle=event.target.getAttribute('data-point');drag=handle===null?selected:Number(handle);event.currentTarget.setPointerCapture(event.pointerId);points[drag]=position(event);select(1-drag);draw();};
$('overlay').onpointermove=event=>{if(drag!==null&&!locked){points[drag]=position(event);draw();}};
$('overlay').onpointerup=$('overlay').onpointercancel=()=>{drag=null;};
document.querySelectorAll('[data-point]').forEach((circle,i)=>circle.onkeydown=event=>{if(locked||!imageReady||!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(event.key))return;event.preventDefault();const step=event.shiftKey?.02:.005;points[i]=[Math.max(0,Math.min(1,points[i][0]+(event.key==='ArrowRight'?step:event.key==='ArrowLeft'?-step:0))),Math.max(0,Math.min(1,points[i][1]+(event.key==='ArrowDown'?step:event.key==='ArrowUp'?-step:0)))];draw();});
$('select1').onclick=()=>select(0);$('select2').onclick=()=>select(1);$('distance').oninput=draw;
$('source').onchange=()=>{$('maps').hidden=$('source').value!=='maps';$('distance').value='';draw();};
function loadFrame(timestamp){imageReady=false;frameTime=timestamp;draw();$('frame').src=`/api/jobs/${job.id}/frame?time=${timestamp}`;}
$('frame').onload=()=>{imageReady=true;draw();};$('frame').onerror=()=>{imageReady=false;draw();error('No se pudo cargar el fotograma. Selecciona otro instante.');};
$('showFrame').onclick=()=>{const timestamp=Number($('frameTime').value);if(!Number.isFinite(timestamp)||timestamp<0||timestamp>=job.duration){error('El instante está fuera del video.');return;}error('');loadFrame(timestamp);};
$('upload').onclick=async()=>{const file=$('file').files[0];if(!file){error('Selecciona un MP4.');return;}if(file.size>1024**3){error('El máximo es 1 GiB por video.');return;}error('');$('upload').disabled=true;$('setup').hidden=true;$('results').hidden=true;$('uploadStatus').textContent='Subiendo y validando el video…';try{const body=new FormData();body.append('file',file);job=await api('/api/videos',{method:'POST',body});clearTimeout(pollTimer);locked=false;points=[[.5,.35],[.5,.65]];select(0);$('parameters').disabled=false;$('showFrame').disabled=false;$('frameTime').disabled=false;document.querySelectorAll('[data-point]').forEach(c=>c.setAttribute('tabindex','0'));$('setup').hidden=false;$('videoName').textContent=`${job.name} · ${job.width} × ${job.height} · ${number(job.duration)} s`;$('stage').style.aspectRatio=`${job.width}/${job.height}`;$('stage').style.maxWidth=`${job.width/job.height*65}vh`;$('overlay').setAttribute('viewBox',`0 0 ${job.width} ${job.height}`);$('frameTime').value=0;$('frameTime').max=Math.max(0,job.duration-.001);$('uploadStatus').textContent='Video listo. Marca los puntos y configura el análisis.';loadFrame(0);}catch(e){error(e.message);$('uploadStatus').textContent='No se pudo cargar el video.';}finally{$('upload').disabled=false;}};
$('config').onsubmit=async event=>{event.preventDefault();if(!job||!imageReady||locked)return;const classes=[...document.querySelectorAll('[name=class]:checked')].map(c=>c.value);if(!classes.length){error('Selecciona al menos una clase de vehículo.');return;}error('');$('start').disabled=true;try{const config={reference:points,distance_m:Number($('distance').value),reference_time:frameTime,speed_limit:Number($('limit').value),confidence:Number($('confidence').value),classes,recording_started_at:$('recording').value||null};await api(`/api/jobs/${job.id}/start`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(config)});locked=true;$('parameters').disabled=true;$('showFrame').disabled=true;$('frameTime').disabled=true;document.querySelectorAll('[data-point]').forEach(c=>c.setAttribute('tabindex','-1'));$('upload').disabled=true;$('file').disabled=true;$('results').hidden=false;$('downloads').hidden=true;$('reports').hidden=true;$('output').hidden=true;$('output').removeAttribute('src');$('preview').hidden=true;$('alerts').replaceChildren();['tracked','measured','average'].forEach(id=>$(id).textContent='—');$('alertCount').textContent='0';poll();$('results').scrollIntoView({behavior:'smooth',block:'start'});}catch(e){error(e.message);draw();}};
async function poll(){try{const state=await api(`/api/jobs/${job.id}`);$('state').textContent=state.state==='reports'?'Generando informes…':state.state==='done'?'Análisis completado':state.state==='error'?'El análisis terminó con un error':`Procesando en ${state.device||'el dispositivo disponible'} · ${number(state.video_time||0)} / ${number(state.duration)} s`;$('progress').value=state.progress||0;$('tracked').textContent=number(state.tracked);$('measured').textContent=number(state.measured);$('average').textContent=number(state.average);$('alertCount').textContent=number(state.alert_count||0);$('alerts').replaceChildren(...(state.alerts||[]).map(a=>{const li=document.createElement('li');li.textContent=`${a.label} #${a.id}: ${number(a.speed)} km/h · alerta en ${number(a.time)} s`;return li;}));if(state.frames){$('preview').hidden=false;$('preview').src=`/api/jobs/${job.id}/preview?t=${Date.now()}`;}if(state.state==='done'){const base=`/files/${job.id}/`;$('preview').hidden=true;$('output').src=base+'output.mp4';$('output').hidden=false;const files=[['output.mp4','Video anotado'],['vehicles.csv','CSV por vehículo'],['heatmap.png','Mapa de calor'],['traffic-by-time.png','Gráfica temporal']];$('links').replaceChildren(...files.map(([file,label])=>{const a=document.createElement('a');a.className='button';a.textContent=label;a.href=base+file+'?download=true';return a;}));$('heatmap').src=$('heatmapFull').href=base+'heatmap.png';$('traffic').src=$('trafficFull').href=base+'traffic-by-time.png';$('downloads').hidden=false;$('reports').hidden=false;$('upload').disabled=false;$('file').disabled=false;return;}if(state.state==='error'){error(state.error);$('upload').disabled=false;$('file').disabled=false;return;}}catch(e){error('No se pudo consultar el progreso; reintentando. '+e.message);}pollTimer=setTimeout(poll,1000);}
</script></body></html>'''


def main():
    import uvicorn
    import torch
    ROOT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "matplotlib"))
    from ultralytics import YOLO

    print("Preparando YOLO11n. Dispositivo:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", flush=True)
    YOLO(str(MODEL))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    tunnel = None
    try:
        thread.start()
        deadline = time.monotonic() + 30
        while not server.started:
            if not thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("No se pudo iniciar el servidor en el puerto 8000.")
            time.sleep(0.1)
        binary = ROOT / "cloudflared"
        if not binary.is_file():
            print("Descargando el túnel oficial de Cloudflare…", flush=True)
            request = urllib.request.Request("https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", headers={"User-Agent": "TrafficMonitor"})
            temporary = binary.with_suffix(".download")
            try:
                with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as target:
                    while chunk := response.read(1024**2):
                        target.write(chunk)
                temporary.chmod(0o700)
                temporary.replace(binary)
            finally:
                temporary.unlink(missing_ok=True)
        tunnel = subprocess.Popen([str(binary.resolve()), "tunnel", "--url", "http://127.0.0.1:8000", "--no-autoupdate"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        found = False
        for line in tunnel.stderr:
            match = re.search(r"https://[-a-z0-9]+\.trycloudflare\.com", line)
            if match:
                found = True
                print("\nLINK PÚBLICO:", match.group(0), "\nMantén esta celda ejecutándose. Interrúmpela para cerrar.\n", flush=True)
            elif not found and ("error" in line.lower() or "failed" in line.lower()):
                print(line.strip(), flush=True)
        raise RuntimeError("El túnel terminó. Ejecuta de nuevo la celda para obtener otro enlace.")
    except KeyboardInterrupt:
        print("\nCerrando el radar y el túnel…", flush=True)
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
        thread.join(timeout=5)
        executor.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    main()
