## CARACTERIZACION

El programa de caracterización está hecho y funcionando: procesó los 182 audios normalizados y generó las etiquetas y los números listos para el KNN. Probé que da los resultados esperados y que no toca `audios/` ni `audios_normalizados/`.

**Una sola carpeta nueva: `caracteristicas/`**
```
caracteristicas/
  etiquetas.csv    una fila por audio: archivo, comando, etiqueta (0/1/2), sesión
  clases.json      reconocer=0, identificar=1, salir=2, otros=3
  dataset.npz      lo que usás en el KNN: X (91 números por audio) e y (la etiqueta de cada fila)
  secuencias/      un archivo por audio con su MFCC a lo largo del tiempo (para KNN con DTW)
  _control.json    uso interno, no hace falta abrirlo
```
- **Etiquetas:** salen del nombre de la carpeta del audio. Si agregás un comando nuevo, recibe el número siguiente y los anteriores no cambian.
- **Sin avisos:** no maneja `sin_voz` ni ningún otro aviso. Caracteriza todo lo que haya en `audios_normalizados/`.
- **Sin estandarizar:** no hace z-score ni PCA. Eso se ajusta dentro del KNN usando solo los datos de entrenamiento, para no contaminar la prueba.

**Cómo usarlo después**
```python
from caracterizar_audios import cargar_dataset, cargar_secuencias
X, y, archivos, sesiones = cargar_dataset()        # KNN euclídeo rápido
secuencias, y, archivos = cargar_secuencias()      # KNN con DTW
```

**Qué se calcula (del informe).** Pre-énfasis, tramos de 25 ms con ventana Hamming, FFT, 40 filtros Mel, log y DCT, quedándose con 13 MFCC. Se suman Δ y ΔΔ, y quedan 39 números por tramo. Usé tramos de 25 ms y no la configuración n_fft 2048 del informe, porque a 16 kHz esta daría solo unos 10 tramos por palabra y es demasiado gruesa.

**Resultados medidos.** Los medí dejando afuera una sesión (un hablante o condición distinta) en cada vuelta, con K=3:
- **KNN euclídeo con `X`:** 87,4 % de aciertos.
- **KNN con DTW sobre las secuencias:** 97,8 % de aciertos.

El DTW es claramente mejor, como recomienda el informe, por eso guardé ambas representaciones.

**Al ampliar la base.** Volvé a correr los dos programas, primero `normalizar_audios.py` y después `caracterizar_audios.py`. Cada uno procesa solo lo nuevo. Si cambiás parámetros del pipeline, se niega a mezclar y pide `--forzar`.

**Al filtrar audios malos.** Si borrás un audio solo de `audios/`, su copia normalizada sigue existiendo, porque el normalizador nunca borra nada. Para que deje de entrar al dataset, borralo también de `audios_normalizados/`. En la próxima corrida queda excluido de `dataset.npz` y de `etiquetas.csv`, aunque su archivo en `secuencias/` se conserva.

El código está en [src/caracterizar_audios.py](src/caracterizar_audios.py). El siguiente paso sería el KNN: partición entrenamiento, validación y prueba, la distancia DTW, la elección de K y el umbral de rechazo.

### Qué es cada archivo de `caracteristicas/` (explicado simple)

**`.npy`**: es el formato de NumPy para guardar **una tabla de números (una matriz)** tal cual, en binario. No se lee con el bloc de notas: se abre desde Python con `np.load("archivo.npy")`. Es rápido y no pierde precisión. En esta carpeta hay un `.npy` por cada audio.

**`etiquetas.csv`**: es una **tabla de texto** con las columnas separadas por comas (se puede abrir con Excel o el bloc de notas). Tiene una fila por audio y dice cuál es la "respuesta correcta" de cada uno:
`archivo, comando, etiqueta, sesion`. Por ejemplo `salir/salir_s1_normal_001.wav, salir, 2, 1_normal` significa que ese audio es del comando *salir*, cuyo número es el 2, grabado en la sesión *1_normal*. El KNN usa la etiqueta para aprender y para saber si acertó.

**`clases.json`**: es la **lista de equivalencias entre el nombre del comando y su número**, porque el algoritmo trabaja con números y no con palabras: `reconocer = 0`, `identificar = 1`, `salir = 2`, `otros = 3`. (`.json` es un formato de texto con pares "nombre: valor", legible por personas y por programas.) Si algún día se agrega un comando nuevo, se le asigna el siguiente número y los que ya existen **no cambian nunca**; así las etiquetas guardadas antes siguen valiendo.

**`_control.json`**: es el **cuaderno interno del programa**; no hace falta abrirlo. Anota con qué parámetros se calcularon los números (por ejemplo, el tamaño de ventana) y cuáles audios ya se procesaron (con su tamaño y fecha). Gracias a eso, cuando se vuelve a ejecutar, el programa sabe cuáles audios son nuevos y **no repite los que ya hizo**, y avisa si cambiaron los parámetros para no mezclar resultados hechos con criterios distintos.

**`dataset.npz`**: es un paquete de varias tablas `.npy` juntas. Contiene `X` (91 números que resumen cada audio), `y` (su etiqueta) y los nombres de archivo y sesión. Es la forma rápida de cargar todo de una vez.

### Por qué la carpeta se llama `secuencias/`

Cada audio se corta en pedacitos muy cortos y de cada pedacito se calcula una "foto" de cómo suena. Esas fotos, puestas **una detrás de otra en el orden en que ocurren**, forman una **secuencia**: la palabra vista como algo que cambia a lo largo del tiempo (primero suena una "r", después una "e", etc.). Se llama así para distinguirla de `X` (el vector de 91 números del `dataset.npz`), que **resume** todo el audio en una sola fila y pierde ese orden. Las secuencias conservan el orden, y eso es justo lo que necesita el DTW, que compara dos palabras estirando o encogiendo el tiempo (si una persona dice "reconocer" más lento que otra, igual coinciden).

### Qué información tiene ahora cada `.npy` de `secuencias/`

Cada archivo es **una matriz con 39 columnas y tantas filas como pedacitos tenga el audio**:

- **Cada fila = un pedacito de audio.** Se miran ventanas de 25 ms que avanzan de a 10 ms. Por ejemplo, un audio normalizado de unos 1,15 s da una matriz de 114 filas × 39 columnas (los audios del dataset van de 27 a 148 filas).
- **Columnas 1 a 13: MFCC.** Describen la **forma del sonido** en ese instante, es decir, el timbre: qué sonido se está pronunciando (una "s" suena muy distinta de una "a"), sin depender tanto de si se habla fuerte o con una voz más grave o aguda.
- **Columnas 14 a 26: Δ (delta).** Dicen **qué tan rápido cambia** cada MFCC de un pedacito al siguiente: la velocidad con que la boca pasa de un sonido al otro.
- **Columnas 27 a 39: ΔΔ (delta-delta).** Dicen **cuánto se acelera ese cambio**.

Son números decimales (`float32`) **sin normalizar**: se guardan tal cual salen del cálculo. La normalización (restar la media por audio, estandarizar) se hace después, dentro del KNN, ajustándola solo con los datos de entrenamiento.