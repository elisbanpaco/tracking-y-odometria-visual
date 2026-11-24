# Radar de tráfico

[Volver al proyecto](../../README.md)

Convierte un video de una cámara fija en una experiencia de análisis de tránsito: marca una distancia conocida, estima la velocidad de los vehículos y consulta dónde y cuándo se concentra su paso. La aplicación utiliza YOLO11n y ByteTrack y reúne el radar y los reportes en un mismo flujo.

## Ponlo en marcha

Puedes usar el [laboratorio de Colab](../../demo_colab.ipynb) y seleccionar **Etapa 03 - Niveles 1 y 2: Radar y analítica**, o ejecutar los comandos siguientes.

Desde la raíz del repositorio, en Google Colab:

```python
!pip -q install uv
!uv pip install --system -r apps/radar_trafico/requirements.txt
!python apps/radar_trafico/app.py
```

Estos comandos usan `uv` con el Python de Colab. Si prefieres `pip`, puedes instalar los mismos requisitos con `!pip -q install -r apps/radar_trafico/requirements.txt`. Ambos flujos se explican en la [guía de inicio](../../README.md#primeros-pasos-en-google-colab).

Abre el `LINK PÚBLICO` de la salida. El modelo y el túnel se descargan cuando hacen falta. Consulta los [primeros pasos](../../README.md#primeros-pasos-en-google-colab) para preparar el entorno y ejecutar una aplicación a la vez.

## De un video a una estimación de velocidad

1. Carga un **MP4 de cámara fija, de hasta 1 GiB**.
2. Selecciona un fotograma donde puedas reconocer dos puntos sobre el suelo, cerca del paso de los vehículos.
3. Marca P1 y P2 e introduce su distancia real en metros. Puedes medirla previamente o consultar Google Maps siguiendo las indicaciones de la interfaz; el valor se introduce manualmente.
4. Elige el límite de velocidad, la confianza y las clases: autos, motos, buses o camiones.
5. Si conoces la fecha y hora de inicio de la grabación, indícala para obtener una gráfica por horas.
6. Inicia el análisis, observa las velocidades y alertas y descarga los archivos al finalizar.

La calibración convierte una distancia de la imagen en una referencia física:

```text
píxeles por metro = distancia entre P1 y P2 en píxeles / distancia real en metros
km/h = velocidad en píxeles por segundo / píxeles por metro × 3,6
```

La aplicación sigue el punto inferior central de cada caja y estima la velocidad con una ventana de 0,5 segundos de seguimiento continuo. Una alerta se activa cuando la velocidad **supera** el límite elegido.

## Qué mirar en los resultados

| Archivo | Qué puedes explorar |
| --- | --- |
| `output.mp4` | Seguimiento, IDs, velocidades estimadas y alertas sobre el video |
| `vehicles.csv` | Primera y última aparición de cada ID, observaciones, velocidad máxima y exceso |
| `heatmap.png` | Zonas con mayor presencia de trayectorias; cada ID aporta una visita por celda |
| `traffic-by-time.png` | Intervalos en los que aparecen más IDs nuevos de automóviles |

Con la hora inicial indicada, las barras agrupan automóviles por hora. Sin ese dato, utilizan segundos o minutos según la duración del video. El mapa ayuda a explorar el uso de la calle, pero no reconoce ni etiqueta carriles automáticamente.

Para comparar pruebas, conserva el video, la distancia de referencia y la configuración utilizada. Cambiar la calibración cambia la interpretación de las velocidades.

## Relación con el monitor de tráfico del enunciado

Aquí se reúnen los niveles 1 y 2 de la etapa 03: radar con cámara fija, escala píxeles/metro, alertas y analítica de tránsito. La entrada implementada es un archivo MP4; para usar una webcam en esta aplicación, primero debes grabar el video.

Los reportes se generan al terminar el análisis, antes de cerrar el programa. El CSV guarda un resumen por ID, mientras que el [tracking 2D](../tracking_2d/README.md) exporta coordenadas por fotograma.

Como siguiente experimento, utiliza un video grabado en movimiento y observa por qué deja de ser suficiente una calibración fija. La [aplicación de odometría](../odometria_visual/README.md) aborda el nivel 3.

## Cómo interpretar las velocidades

La escala es local: un metro no ocupa los mismos píxeles cerca y lejos de la cámara. Calibra cerca de los vehículos y recuerda que la perspectiva puede distorsionar las estimaciones. Esta aplicación no compensa el movimiento de la cámara. Una pérdida de seguimiento también puede asignar un nuevo ID al mismo vehículo y afectar los conteos.

Las velocidades tienen finalidad académica y no son mediciones para sanciones de tránsito.

## Archivos y comprobación rápida

Los trabajos, el modelo `yolo11n.pt` y `cloudflared` se guardan en `/content/traffic-monitor`. Puedes cambiar esa carpeta con la variable de entorno `JOBS_DIR`. Esta aplicación no utiliza `WEIGHTS` ni `NPOOL`; procesa y genera video con PyAV.

Para comprobar escala, interpolación, pérdida de seguimiento y comparación del límite sin instalar dependencias externas:

```bash
python apps/radar_trafico/app.py --check
```

Al interrumpir la ejecución principal, el programa solicita el cierre del análisis, del servidor y del túnel. Descarga los resultados antes de cerrar Colab.
