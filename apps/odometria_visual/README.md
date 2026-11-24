# Odometría visual con cámara en movimiento

[Volver al proyecto](../../README.md)

Un auto estacionado puede parecer que se mueve si caminas con la cámara. Esta aplicación permite explorar ese efecto: estima la trayectoria de la cámara y compara la velocidad aparente de los objetos con una velocidad compensada por ese movimiento.

Es el paso que conecta el seguimiento de objetos con la navegación visual del proyecto.

## Ponlo en marcha

Puedes usar el [laboratorio de Colab](../../demo_colab.ipynb) y seleccionar **Etapa 03 - Nivel 3: Odometría visual**, o ejecutar los comandos siguientes.

Desde la raíz del repositorio, en Google Colab:

```python
!pip -q install uv
!uv pip install --system -r apps/odometria_visual/requirements.txt
!python apps/odometria_visual/app.py
```

Estos comandos usan `uv` con el Python de Colab. Si prefieres `pip`, puedes instalar los mismos requisitos con `!pip -q install -r apps/odometria_visual/requirements.txt`. Ambos flujos se explican en la [guía de inicio](../../README.md#primeros-pasos-en-google-colab).

Abre el `LINK PÚBLICO` de la salida. El arranque utiliza `wget` y `chmod` para preparar el túnel Linux de Cloudflare cuando falta. Encontrarás la preparación del entorno en los [primeros pasos](../../README.md#primeros-pasos-en-google-colab).

## Tu primer recorrido

1. Elige una webcam o un video grabado caminando, con suelo y detalles del entorno visibles.
2. Ajusta la altura de la cámara y su campo de visión. Revisa también el número de características y el límite de velocidad.
3. Inicia la odometría y observa la trayectoria 3D, las gráficas y los indicadores de seguimiento.
4. Compara las velocidades aparente y compensada de un objeto estacionado que el detector reconozca, como un auto.
5. Pulsa «Detener y guardar» para descargar los registros.

Comienza con un movimiento suave y una escena con textura. Prueba después una pared lisa o un giro de cámara para observar cuándo se debilita la estimación. Si usas webcam, concede permiso de cámara en el navegador; los videos locales se reproducen y envían desde el navegador.

## Qué ocurre entre dos fotogramas

1. **YOLO y ByteTrack** detectan personas y vehículos y mantienen sus IDs. Las cajas con ID se enmascaran para buscar puntos del fondo.
2. **ORB**, que combina FAST orientado y BRIEF rotado, extrae características y las empareja mediante distancia Hamming.
3. **RANSAC y la matriz esencial** ayudan a descartar correspondencias falsas y recuperar rotación y dirección de traslación.
4. **Una homografía y la altura de cámara** aportan una estimación de escala, suponiendo que los puntos seleccionados pertenecen al suelo.
5. Las poses se acumulan para dibujar la trayectoria y estimar las posiciones y velocidades compensadas de los objetos.

La idea es observar cuánto del desplazamiento de la imagen puede explicarse por el movimiento de la cámara.

## Qué puedes descargar

| Archivo | Contenido |
| --- | --- |
| `trayectoria_camara.csv` | Tiempo, posición estimada en metros, rumbo, velocidad de cámara, inliers y estado del seguimiento |
| `vehiculos.csv` | Tiempo, ID, clase, posición estimada, velocidades aparente y compensada y estado; también puede incluir personas |

Los *inliers* son correspondencias aceptadas por la estimación geométrica. Los registros se mantienen en el navegador hasta descargarlos al detener. Esta aplicación exporta CSV; no guarda un video anotado ni una carpeta de trabajos en el servidor.

La interfaz muestra **velocidad compensada estimada**. El CSV conserva el nombre `v_real_kmh` por compatibilidad; representa esa estimación, no una medición de referencia.

## Relación con el enunciado

Esta experiencia aborda el nivel 3 de la etapa 03 y complementa el caso de cámara móvil de la etapa 02. Permite comparar la hipótesis de cámara fija con una estimación que incorpora odometría mediante ORB y geometría entre fotogramas.

Para completar la comparación, prueba primero el [radar de tráfico](../radar_trafico/README.md) con una cámara fija y después observa el efecto de moverla. El [tracking 2D](../tracking_2d/README.md) ayuda a entender qué información aportan por sí solos los IDs y las trayectorias en imagen.

## Qué puede afectar la estimación

Una cámara monocular necesita una referencia adicional para obtener escala métrica. Aquí se utiliza la altura indicada y un supuesto de suelo plano. La mitad inferior de la imagen también puede contener paredes u otros planos y producir una escala incorrecta; el campo de visión configurado influye en el cálculo.

Poca textura, desenfoque y oclusiones pueden debilitar o perder el seguimiento. Los errores se acumulan en la trayectoria y el envío de fotogramas desde el navegador condiciona el muestreo. Por eso, la compensación no garantiza velocidad cero en objetos estacionados ni velocidades métricas exactas.

## Ajustes opcionales

Si necesitas reducir el consumo de memoria, carga una sola instancia del modelo:

```python
!NPOOL=1 python apps/odometria_visual/app.py
```

| Variable de entorno | Valor predeterminado | Para qué sirve |
| --- | --- | --- |
| `WEIGHTS` | `yolo11m.pt` | Modelo YOLO |
| `NPOOL` | `3` | Instancias del modelo para sesiones |

Las tres instancias predeterminadas pueden requerir más memoria de la disponible. El modelo y `cloudflared` se descargan en el directorio desde el que ejecutas el programa. Esta aplicación no utiliza `JOBS_DIR`. Ejecuta una sola aplicación a la vez y descarga los CSV antes de cerrar la página.

Después de guardar los CSV, interrumpe la ejecución para cerrar el servidor y el túnel.
