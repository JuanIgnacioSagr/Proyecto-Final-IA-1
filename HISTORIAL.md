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

Desde la carpeta raíz del proyecto, corré estos dos comandos en este orden:

```
python src/normalizar_audios.py
python src/caracterizar_audios.py
```

Cada uno procesa **solo los audios nuevos**, así que no repiten lo que ya está hecho. El primero lee `audios/` y escribe en `audios_normalizados/`. El segundo lee `audios_normalizados/` y actualiza `caracteristicas/`, incluidos `etiquetas.csv`, `dataset.npz` y `secuencias/`.

Si antes querés ver qué se va a procesar sin que escriba nada, agregá `--simular` a cualquiera de los dos:
```
python src/normalizar_audios.py --simular
python src/caracterizar_audios.py --simular
```

**Si grabaste un comando nuevo** (por ejemplo la carpeta `otros/`), el segundo programa le asigna automáticamente el siguiente número en `clases.json`. `otros` ya tiene reservado el 3, y los números que ya existían no cambian.

**Si algo falla:**
- Si algún programa dice que los parámetros cambiaron, es porque se modificó el código del pipeline. Agregá `--forzar` para reprocesar todo, y no mezclar audios hechos con criterios distintos.
- Si alguno de los dos dice que no existe la carpeta de entrada, revisá que estés parado en la raíz del proyecto.
Con lo que ya tenés, la secuencia para llegar al KNN es esta.

**1. Completar y revisar el dataset**
- Revisar `etiquetas.csv`: cuántos audios hay por comando y por persona.
- Grabar `otros/` (otras palabras, silencio y ruido), si todavía no está.
- Si podés, sumar más personas. Después de agregar audios, correr `normalizar_audios.py` y `caracterizar_audios.py`.

**2. Separar entrenamiento, validación y prueba**
- Separalos **por persona**, no al azar: por ejemplo, algunas personas para entrenar, otras para elegir parámetros y una o dos que no se toquen hasta el final.
- Si no, el KNN vería a la misma persona en entrenamiento y en prueba, y la exactitud saldría inflada.

**3. Armar el KNN con DTW**
- Se usa la distancia DTW sobre las `secuencias/`, con la resta de la media por audio (`cmn`). En mi prueba fue el método que mejor funcionó (97,8 %).
- Para clasificar un audio nuevo se calcula su DTW contra todos los de entrenamiento, se toman los K más cercanos y gana el comando que más se repite.
- Como alternativa rápida y de referencia, se puede hacer un KNN euclídeo con `X` (z-score ajustado solo con entrenamiento).

**4. Elegir K y ajustes con la validación**
- Probar K impares (1, 3, 5, 7) y, si hace falta, el ancho de la banda del DTW y cuántos coeficientes usar (13 o los 39).

**5. Definir el umbral de rechazo**
- Es lo que el enunciado pide: si la distancia al vecino más cercano supera un umbral, el acceso se deniega.
- Se calibra con la validación, usando `otros/` y personas desconocidas: se busca un umbral que rechace lo ajeno sin rechazar demasiado los comandos válidos.

**6. Evaluar una sola vez con el conjunto de prueba**
- Medir exactitud, matriz de confusión, aciertos por persona y porcentaje de rechazos.
- Estos son los números para el informe.

**7. Dejar el modelo final listo**
- Entrenar con todos los datos de entrenamiento más validación y guardar las secuencias de referencia, K y el umbral en un archivo.

**8. Conectarlo a la demo con el micrófono**
- Grabar audio, pasarlo por `normalizar_audio`, luego por `caracterizar_audio`, compararlo con el modelo y devolver *Reconocer*, *Identificar* o *Salir*, o "no es un comando".
- Todo reutiliza el mismo código de las etapas anteriores.

**Archivos que se crearían:**
- `src/knn_voz.py`: DTW, clasificación y rechazo.
- `src/evaluar_knn.py`: separación, elección de K y umbral, y resultados.
- `src/reconocedor_voz.py`: la parte en vivo con el micrófono.

El DTW está escrito en Python puro, así que comparar un audio contra unos 200 de referencia tarda una fracción de segundo. Es suficiente para la demo.

¿Querés que arme el plan para empezar con `knn_voz.py` y `evaluar_knn.py`?

Terminé de probar y revisar los gráficos; cierro con el resumen.

Hice la caracterización de huellas y el programa PCA. Los probé con tus huellas y con los audios, y funcionan. Con las huellas dio un acierto de 0,99 sin `matada` y de 0,92 con `matada`, los mismos números del prototipo.

## Huellas: nueva carpeta `caracteristicas_huellas/`

Es parecida a la de audios:
```
caracteristicas_huellas/
  etiquetas.csv   archivo, efecto, identidad (1/2/3), etiqueta (0/1/2)
  clases.json     identidad_1=0, identidad_2=1, identidad_3=2, intruso=3 (reservado)
  dataset.npz     X (72 números por huella) + X_fingercode (256) + X_minucias (18), con y, archivo, efecto
  vectores/       un archivo por huella con sus rasgos (<efecto>/id<N>/<nombre>.npz)
  graficos/       aquí quedan los PCA
  _control.json   uso interno
```
- **Rasgo principal `X` (72 números).** Para cada huella se mide la orientación de sus crestas. Se arma un histograma con 24 rangos para toda la huella y otros 12 rangos para cada cuadrante (2×2). Resume hacia dónde van las crestas, sin importar el ruido ni los rayones.
- **Rasgos auxiliares.** `X_fingercode` mide la energía de filtros Gabor por anillos y sectores. `X_minucias` mide la densidad de terminaciones y bifurcaciones por zona.
- **Incremental como en audios.** Procesa solo las huellas nuevas, tiene `--simular` y `--forzar`, y se niega a mezclar si cambian los parámetros. Nunca borra nada, y `huellasDactilares/` y `huellasProcesadas/` quedaron intactas.
- **Comandos:**
```
python src/preprocesar.py            # si hay huellas nuevas en huellasDactilares/
python src/caracterizar_huellas.py   # vectoriza lo pendiente
```

## Programa PCA: `src/graficar_pca.py`
```
python src/graficar_pca.py --datos huellas
python src/graficar_pca.py --datos audios
```
Genera `pca_2d.png` y `pca_3d.png` dentro de `graficos/` de cada carpeta. En huellas el color es la identidad y el marcador el efecto. En audios el color es el comando y el marcador la sesión. En audios aplica z-score y en huellas no; ambas cosas se pueden cambiar con `--escalar` o `--no-escalar`.

## Qué muestran los gráficos
- **Huellas.** Se ven 3 grupos bien separados por identidad: PC1 46 % y PC2 17,7 % de la varianza. Los efectos casi no los mezclan.
- **Audios.** Se separa más por sesión (quién habla, y la sesión de ruido) que por comando. Los comandos se solapan bastante en 2D (PC1 12,9 % y PC2 10,6 %). No es contradictorio con el 97 % de aciertos del DTW, que usa la secuencia completa y no un resumen en 2D. Pero conviene tenerlo presente para el informe, y es otro argumento para sumar más personas.

## Dos cosas que debés saber
1. **Falta `impresionMatada` en `huellasProcesadas/`.** Hoy el dataset tiene 75 huellas (25 por identidad). En una prueba temporal con las 90 huellas, el efecto `matada` es el único que rompe el agrupamiento: acierto de 0,47 solo en esas, y 0,92 en total contra 0,99 sin ellas. Si lo regenerás con `python src/preprocesar.py`, la próxima corrida lo suma sola. Esa corrida sobrescribe los PNG ya generados, pero con resultado idéntico.
2. **Dejé sin hacer lo que acordamos para después.** Clusters con centroides y radio de umbral (K-means), y fronteras de decisión y curva de exactitud (KNN). Cada uno necesita el algoritmo ya corriendo.

Código: [src/caracterizar_huellas.py](src/caracterizar_huellas.py) y [src/graficar_pca.py](src/graficar_pca.py). Los gráficos están en [caracteristicas_huellas/graficos/](caracteristicas_huellas/graficos/) y [caracteristicas/graficos/](caracteristicas/graficos/).

Midió el comportamiento real de los dos modelos con una huella y un audio nuevos. No modifiqué nada.

## Huella nueva (`kmeans_huellas.py`)

**Recorrido:**
1. **Preprocesar** (`preprocesar_huella`, ~0,26 s): pasa a gris, alinea y rota, filtra con Gabor y binariza. Es el mismo pipeline para cualquier imagen.
2. **Vectorizar** (`rasgo_orientacion`, ~0,02 s): la convierte en 72 números, el histograma de orientación de las crestas.
3. **Decidir** (`identificar_huella`): calcula la distancia a los 3 centroides y se queda con el más cercano.
   - Si esa distancia es **menor o igual al radio** de ese cluster, responde `identidad_N`.
   - Si es **mayor**, responde **`NO AUTORIZADO`**.

Devuelve la identidad (o el rechazo) y la distancia.

**Lo que probé:**
- Una huella de la identidad 3 que no estaba en el entrenamiento → `identidad_3`, distancia 0,056, dentro del radio.
- Una impresión rotada de la identidad 2 → **`NO AUTORIZADO`**, distancia 0,215 contra un radio de 0,1726. Es una persona autorizada rechazada. Es el falso rechazo que mostraba la métrica: 5 de las 30 huellas de prueba.
- Una imagen en blanco y una de ruido aleatorio → `NO AUTORIZADO`. Bien.

**Cómo se comporta:**
- **Rechazo solo por distancia.** Nada verifica que la imagen sea una huella. Un dibujo parecido podría caer dentro del radio.
- **Umbral sin calibrar.** El radio está definido (percentil 95) pero nunca se probó con huellas de intrusos reales. Con el umbral actual se rechazan algunas personas autorizadas (6,7 % en entrenamiento y 16,7 % en prueba), y no sé cuántos intrusos entrarían.
- **Huellas cortadas.** Las muy recortadas, como el efecto `matada` que falta, son el punto débil.

## Audio nuevo (`knn_audios.py`)

**Recorrido (`clasificar_audio`, ~0,3 s en total):**
1. **Normalizar:** mismo acondicionamiento que el dataset (quita continua, pasa-banda, recorta silencios, lleva a −20 dBFS).
2. **Caracterizar:** calcula los MFCC con Δ y ΔΔ, la secuencia completa.
3. **DTW:** compara contra los 197 audios de referencia.
4. **Votar:** toma los 3 más cercanos (K = 3) y gana el comando más repetido.

Devuelve el comando, la distancia al vecino más cercano y los votos, por ejemplo `salir, 0.01, {salir: 3}`.

**Lo que probé:**
- Un audio real de la base dio `salir` con distancia 0,01. Es casi cero porque ese audio ya está en la referencia, así que **no vale como prueba de exactitud**.
- Silencio, ruido blanco y un tono puro → todos respondieron **`identificar`** con distancias 2,8 a 3,6 y votos dispersos (1-2-0). Es decir, siempre hay una respuesta, aunque no se haya dicho ningún comando.

**Cómo se comporta:**
- **No rechaza.** Siempre elige uno de los tres comandos, aunque diga "hola" o no diga nada.
- **Hay señales útiles.** Cuando el audio no es un comando, las distancias son grandes y los votos están divididos. Con un umbral de distancia o exigiendo votos unánimes se podría rechazar. Hoy eso no está.
- **Precisión.** En la prueba falló en 2 de 22 audios, ambos de *reconocer*.

## Resumen de la diferencia

| | Huella | Voz |
|---|---|---|
| Respuesta posible | 3 identidades o `NO AUTORIZADO` | siempre `reconocer`, `identificar` o `salir` |
| Rechaza | sí, por radio de umbral (sin calibrar) | no |
| Qué dice la distancia | qué tan lejos está del centro de su grupo | qué tan parecido es a su vecino más cercano |
| Tiempo | ~0,3 s | ~0,3 s |

## Para la interfaz tendremos que decidir
1. **Umbral de rechazo para la voz:** distancia máxima o votos unánimes.
2. **Huellas de intrusos:** para calibrar el umbral y medir cuánto entra de más.
3. **Qué hace el programa en cada caso:** por ejemplo, si el comando es *identificar*, pedir la huella, y si da `NO AUTORIZADO`, negar el acceso. Esto lo definimos con vos antes de armar la interfaz.

¿Arrancamos con el plan de la interfaz?