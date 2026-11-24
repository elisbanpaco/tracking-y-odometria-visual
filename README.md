# Tracking y odometría visual

¿Qué cambia cuando se mueve un objeto? ¿Y cuando se mueve la cámara? Este proyecto explora ambas preguntas con tres aplicaciones web: seguimiento de objetos, un radar de tráfico y una demostración de odometría visual.

Puedes usar tus propios videos o una webcam, observar las trayectorias y descargar los resultados para analizarlos. El recorrido parte del tracking 2D y llega a la estimación del movimiento de la cámara para mostrar cómo se complementan.

## Dos formas de entender el movimiento

| | Tracking de objetos | Odometría visual |
| --- | --- | --- |
| ¿Qué sigue? | Objetos identificados entre fotogramas | El movimiento de la cámara respecto al entorno |
| ¿Qué obtiene aquí? | IDs, posiciones en imagen y trayectorias | Una pose y una trayectoria estimadas de la cámara |
| ¿Para qué se combinan? | Seguir un vehículo mientras cambia su posición en la imagen | Ayudar a separar el movimiento de la cámara del movimiento del vehículo |

Con una cámara fija, el desplazamiento de un auto en la imagen puede servir para estimar su velocidad si conocemos la escala. Al mover la cámara, incluso un auto estacionado parece desplazarse: la odometría aporta una estimación del movimiento de la cámara para compensar ese efecto.

## Elige una experiencia

| Aplicación | Qué puedes explorar | Entrada |
| --- | --- | --- |
| [Tracking 2D y trayectorias](apps/tracking_2d/README.md) | Detectar objetos con YOLO, comparar ByteTrack y BoT-SORT, contar cruces y consultar trayectorias | Video o webcam |
| [Radar de tráfico](apps/radar_trafico/README.md) | Calibrar una distancia, estimar km/h, mostrar alertas y generar analítica | Video MP4 de cámara fija |
| [Odometría visual](apps/odometria_visual/README.md) | Estimar la trayectoria de una cámara móvil con ORB y comparar velocidades aparentes y compensadas | Video o webcam en movimiento |

Para una primera prueba, sigue el orden de la tabla. Cada guía explica cómo iniciar la aplicación, qué observar y qué resultados puedes descargar.

## Primeros pasos en Google Colab

Abre [demo_colab.ipynb](demo_colab.ipynb), elige una experiencia y ejecuta las celdas en orden. El notebook descarga automáticamente el código y los requisitos desde `apps/` en la rama `main` de GitHub. No necesitas copiar archivos manualmente.

[Abrir en Colab](https://colab.research.google.com/github/elisbanpaco/tracking-y-odometria-visual/blob/main/demo_colab.ipynb)

El enlace requiere publicar el notebook en `main`. También puedes descargarlo y subirlo directamente a Colab.

Si prefieres ejecutar los comandos directamente, sigue estos pasos:

Necesitas el código, conexión a Internet y un video o una webcam según la experiencia elegida. Puedes usar GPU para acelerar el procesamiento; también se contempla CPU, con menor rendimiento.

1. Carga el repositorio en el entorno de Colab y usa `%cd` para situarte en su carpeta raíz.
2. Instala `uv` en el entorno de Colab y úsalo para instalar los requisitos de una aplicación. Por ejemplo, el radar:

   ```python
   !pip -q install uv
   !uv pip install --system -r apps/radar_trafico/requirements.txt
   !python apps/radar_trafico/app.py
   ```

3. Espera la preparación del modelo y abre el `LINK PÚBLICO` que aparece en la salida.
4. Sigue los pasos de la interfaz y descarga tus resultados antes de cerrar Colab.

`uv pip` utiliza los mismos archivos `requirements.txt` de cada aplicación. En Colab, `--system` instala en el Python del entorno en lugar de exigir un entorno virtual. Puedes consultar este comportamiento en la [documentación de uv](https://docs.astral.sh/uv/pip/environments/).

Si prefieres `pip`, sustituye la instalación de requisitos por `!pip -q install -r apps/radar_trafico/requirements.txt`; el comando para iniciar la aplicación es el mismo.

También puedes subir solamente `app.py` y `requirements.txt` de una aplicación a una misma carpeta. Desde esa carpeta, instala `uv` con `!pip -q install uv`, ejecuta `!uv pip install --system -r requirements.txt` y luego `!python app.py`.

Las tres aplicaciones usan el puerto 8000: ejecuta una a la vez. Descarga los resultados antes de interrumpir la ejecución; las tres incorporan cierre del servidor y del túnel. El notebook además cierra el grupo de procesos de su propia ejecución y comprueba que el puerto esté libre antes de iniciar otra experiencia. Si un proceso externo mantiene el puerto ocupado, detén esa ejecución antes de continuar.

El arranque descarga los modelos YOLO y, cuando hace falta, `cloudflared` para Linux x86_64. Las interfaces también utilizan recursos externos. El enlace público permite acceder a quien lo conozca. Para usar la webcam, concede el permiso de cámara en el navegador. Los archivos de `/content` son temporales.

## Qué encontrarás en el repositorio

```text
apps/
├── tracking_2d/
├── radar_trafico/
└── odometria_visual/
```

Cada aplicación incluye su guía, sus requisitos y un `app.py` autónomo con el servidor y la interfaz web. El radar y los reportes de tránsito comparten el procesamiento en una sola aplicación.

El notebook de la raíz presenta el recorrido por etapas y ejecuta las aplicaciones descargadas de GitHub. Al publicar cambios en `apps/` en la rama `main`, estarán disponibles al repetir la descarga. La celda muestra el commit descargado para identificar la versión utilizada.

Los modelos, las cachés y las carpetas habituales de resultados se excluyen de Git. Puedes incorporar videos, imágenes y CSV seleccionados como evidencias: esos formatos no se ignoran globalmente.

## Del experimento a la actividad académica

El objetivo es comprender la diferencia entre tracking y odometría visual mediante ejemplos de su complementariedad.

| Parte del enunciado | Cómo se relaciona con el proyecto |
| --- | --- |
| Etapa 01 · Conceptos | Base teórica: pasos y tipos de odometría, comparación con odometría de ruedas, relación con VIO y SLAM, aplicaciones y diferencias frente al tracking, incluyendo RA, experiencias inmersivas y VFX. Las respuestas teóricas no están incluidas aquí. |
| Etapa 02 · Aplicaciones y conceptos | Tracking incluye una comparación visual entre seguimiento 2D y estimación 3D, YOLO con ByteTrack o BoT-SORT, conteo por línea, CSV de trayectorias y analítica de video. Odometría permite explorar el caso de cámara móvil. |
| Etapa 03 · Nivel 1: radar de velocidad | Radar incorpora calibración con una distancia conocida, estimación en km/h y alertas por exceso. La distancia puede obtenerse manualmente con Google Maps. |
| Etapa 03 · Nivel 2: analítica de tránsito | Radar exporta CSV, mapa de calor y gráfica temporal al terminar de procesar el video. |
| Etapa 03 · Nivel 3: cámara en movimiento | Odometría combina YOLO y ByteTrack con ORB, emparejamiento y estimación de pose para comparar movimiento aparente y compensado. |

Hay diferencias entre lo solicitado y el alcance actual que conviene tener presentes al preparar una demostración: tracking resume la velocidad media en px/s al finalizar el video; su webcam no muestra velocidad instantánea ni mapa de calor. Radar recibe archivos MP4 y genera reportes al terminar el análisis, antes de cerrar el programa. Su CSV resume cada ID y el mapa muestra ocupación de zonas sin clasificar carriles automáticamente. El informe final tampoco está incluido.

## Cómo interpretar los resultados

Las velocidades son estimaciones académicas. La perspectiva afecta la escala del radar; la odometría depende de supuestos sobre el suelo y la cámara y puede acumular error. Además, un objeto que pierde su seguimiento puede recibir otro ID y alterar los conteos. Cada guía desarrolla estos límites para ayudarte a interpretar lo que ves.

El radar incluye una comprobación pequeña que funciona sin modelos ni dependencias externas:

```bash
python apps/radar_trafico/app.py --check
```

Comprueba escala, interpolación y pérdida de seguimiento; la validación de una escena requiere probarla con sus propios videos y calibración. Los requisitos por aplicación no fijan todas las versiones del entorno.

## Compartir mejoras

Para reportar un problema, incluye la aplicación, el entorno, los pasos para reproducirlo y el mensaje de error. Si propones una mejora, explica qué ejemplo o resultado permite entender mejor. Comparte únicamente material que tengas permiso para publicar y evita incluir enlaces temporales o datos privados.
