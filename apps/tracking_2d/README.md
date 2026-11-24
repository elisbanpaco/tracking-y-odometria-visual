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

## Tu primer experimento

1. Elige una escena de cámara fija donde personas o vehículos crucen una zona visible.
2. Abre la sección de tracking y selecciona video o cámara en vivo.
3. Elige ByteTrack o BoT-SORT, las clases que quieres seguir y la confianza mínima de detección.
4. Sitúa la línea virtual en el paso de los objetos y comienza el análisis.
5. Observa los IDs y los cruces. En webcam, pulsa «Detener y guardar» para obtener el resumen y descargar el CSV.

Prueba ambos trackers con el mismo video y la misma línea. Observa qué ocurre con los IDs cuando dos objetos se cruzan o uno queda oculto. El conteo distingue dos sentidos y registra como máximo una entrada y una salida por ID.

## Qué puedes obtener

| Con un video | Con la webcam |
| --- | --- |
| Video anotado, trayectorias y conteo de cruces | Cajas, IDs, trayectorias y conteo en vivo |
| Mapa de calor y gráficas de actividad y cruces | Gráfica de objetos activos |
| Resumen por ID con velocidad media en px/s | Resumen de recorridos al detener |
| Descargas `trayectorias.csv` y `resultados.json` | Descarga `trayectorias_en_vivo.csv` |

Los CSV registran ID, clase, frame, tiempo y centro del objeto `(cx, cy)`: puedes usarlos para reconstruir las trayectorias. Los trabajos de video se guardan en el servidor; las filas de la webcam permanecen en el navegador hasta descargarlas.

## Qué enseña y hasta dónde llega

Esta experiencia corresponde a la etapa 02: comparar tracking 2D y odometría, seguir objetos, contar cruces y analizar trayectorias. La velocidad media en px/s se calcula al terminar el video. El modo webcam no ofrece velocidad instantánea ni mapa de calor, por lo que ese apartado del enunciado queda cubierto solo parcialmente.

Los píxeles describen desplazamientos en la imagen. Para estimar metros por segundo necesitas una referencia física y considerar la perspectiva; puedes explorar esa calibración en el [radar de tráfico](../radar_trafico/README.md).

Después, prueba una cámara móvil: el fondo y los objetos estacionados también se desplazan en la imagen. La [odometría visual](../odometria_visual/README.md) permite estudiar cómo compensar ese movimiento. Un cambio de ID tras una oclusión puede afectar tanto el conteo como la trayectoria.

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
