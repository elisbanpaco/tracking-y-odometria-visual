# Tracking 2D y trayectorias

[Volver al proyecto](../../README.md)

Sigue personas y vehículos a lo largo de un video, cuenta cuántos cruzan una línea y descubre por dónde pasan. Esta aplicación combina detección con YOLO y seguimiento con ByteTrack o BoT-SORT; también incluye animaciones para explorar por qué estimar el movimiento 3D de una cámara es más difícil que seguir objetos en una imagen.

## Ponlo en marcha

Puedes usar el [laboratorio de Colab](../../demo_colab.ipynb) y seleccionar **Etapa 02 - Tracking 2D**, o ejecutar los comandos siguientes.

Desde la raíz del repositorio, en Google Colab:

```python
!pip -q install uv
!uv pip install --system -r apps/tracking_2d/requirements.txt
!python apps/tracking_2d/app.py
```

Estos comandos usan `uv` con el Python de Colab. Si prefieres `pip`, puedes instalar los mismos requisitos con `!pip -q install -r apps/tracking_2d/requirements.txt`. Ambos flujos se explican en la [guía de inicio](../../README.md#primeros-pasos-en-google-colab).

Abre el `LINK PÚBLICO` de la salida. Si necesitas ayuda para cargar los archivos o cambiar de aplicación, consulta los [primeros pasos](../../README.md#primeros-pasos-en-google-colab).

El procesamiento de videos necesita **FFmpeg**. Comprueba si está disponible con `!ffmpeg -version`. Si falta en Colab, ejecuta `!apt-get -qq update` y después `!apt-get -qq install -y ffmpeg`. El arranque también utiliza `wget` y `chmod` para preparar el túnel.

## Recorrido de la etapa 02

| Apartado | Qué probar |
| --- | --- |
| a · Comparación conceptual | Explora las animaciones de tracking 2D y estimación 3D. Las cargas y tiempos mostrados son ilustrativos. |
| b · Tracking | Elige video o webcam fija, ByteTrack o BoT-SORT y una línea de conteo. Observa IDs, cruces y trayectorias. |
| c · Velocidad y zonas | Consulta px/s, la cuadrícula 3 × 3 y el mapa de calor. Puedes cambiar entre b y c conservando el análisis. |
| d · Cámara móvil | Abre la cámara, captura una referencia de un fondo plano con textura y compara trayectorias originales y compensadas. |

El conteo registra como máximo una entrada y una salida por ID. Una pérdida de seguimiento puede asignar otro ID al mismo objeto.

### Calibrar una distancia en el apartado c

1. Selecciona un video o inicia la webcam.
2. Pulsa «Marcar referencia» y marca los extremos de una distancia conocida.
3. Introduce los metros y pulsa «Aplicar referencia». En video, hazlo antes del análisis.

La conversión a m/s usa una escala local: la referencia y los objetos deben estar en un plano y a una profundidad similares. Reiniciar la cámara o cambiar de archivo requiere marcar otra referencia. La perspectiva puede invalidar una única escala para toda la escena.

### Compensar movimiento en el apartado d

1. Abre la cámara y encuadra un fondo plano con textura.
2. Cuando aparezca «Fondo listo», mantén la cámara quieta y pulsa «Capturar referencia».
3. Mueve suavemente la cámara y compara amarillo (original) con verde (compensado).
4. Pulsa «Detener y guardar CSV» antes de cerrar.

ORB y una homografía estiman la transformación respecto a la referencia. Si se pierde, se suspenden las mediciones compensadas. Los cruces se cuentan sobre la línea fija de esa referencia; no se unen recorridos a través de pérdidas de seguimiento.

Esta demostración trabaja en píxeles de referencia sobre un plano: no estima pose 3D ni distancias métricas. La [aplicación de odometría visual](../odometria_visual/README.md) corresponde al nivel 3 de la etapa 03. El [radar](../radar_trafico/README.md) reúne velocidades en km/h, alertas y reportes de tránsito.

## Resultados

| Modo | Qué obtienes |
| --- | --- |
| Video | Video anotado con px/s y zonas, resumen de velocidades medias y mapa de calor; descargas `video.mp4`, `mapa_calor.png`, `trayectorias.csv` y `resultados.json` |
| Webcam fija | Velocidad aparente, zona, trayectorias, conteos y mapa de calor en vivo; descarga `trayectorias_en_vivo.csv` al detener |
| Cámara móvil planar | Comparación de recorridos, velocidades y cruces en referencia; descarga `comparacion_planar.csv` al detener |

Los CSV de tracking incluyen ID, clase, frame, tiempo, centro, velocidad en px/s y zona. Si aplicaste una referencia métrica, incluyen la escala y la velocidad aproximada en m/s. El CSV planar separa las referencias por segmento y registra la validez de la compensación.

El mapa de calor acumula detecciones: no representa objetos únicos ni tiempo exacto de permanencia. La velocidad compara la última posición de cada ID con la actual; el ruido de las cajas puede afectar el resultado. Los datos de webcam y cámara móvil permanecen en el navegador hasta descargarlos.

## Ajustes opcionales

Puedes iniciar con los valores predeterminados. Si falta memoria, reduce el número de modelos cargados:

```python
!NPOOL=1 python apps/tracking_2d/app.py
```

| Variable de entorno | Valor predeterminado | Para qué sirve |
| --- | --- | --- |
| `JOBS_DIR` | `/content/jobs` | Carpeta de trabajos de video |
| `WEIGHTS` | `yolo11l.pt` | Modelo YOLO |
| `NPOOL` | `6` | Instancias del modelo y trabajadores |

Seis instancias pueden requerir más memoria de la disponible. El modelo y `cloudflared` se descargan en el directorio desde el que ejecutas el programa. Al subir un video, la aplicación elimina carpetas de trabajos de más de dos horas dentro de `JOBS_DIR`: dedica esa ruta solo a sus resultados y descarga lo que quieras conservar.

Descarga los resultados antes de interrumpir la ejecución. El cierre solicita detener los análisis, libera el servidor y termina el túnel.

Comprobaciones incorporadas, sin crear archivos de pruebas en el repositorio:

```bash
python apps/tracking_2d/app.py --autoprueba
python apps/tracking_2d/app.py --autoprueba-planar
JOBS_DIR=/tmp/prueba-video python apps/tracking_2d/app.py --autoprueba-video
```

La primera valida cálculos básicos; la segunda usa OpenCV y escenas sintéticas. La tercera requiere las dependencias y FFmpeg y comprueba exportaciones con detecciones simuladas; no valida YOLO real.
