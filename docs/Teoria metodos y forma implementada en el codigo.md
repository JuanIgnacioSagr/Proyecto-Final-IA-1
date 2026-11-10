# Teoría de los métodos y forma en que están implementados en el código

Documento de estudio para **explicar K-means (huellas) y KNN (voz)**: qué es cada método, cómo funciona paso a paso, qué gráficos se usan para mostrarlo y en qué parte del código está cada cosa.
Todos los números de este documento salen de las corridas reales del proyecto.

**Archivos del código**

| Archivo | Para qué sirve |
|---|---|
| `src/kmeans_huellas.py` | K-means de las huellas (no supervisado), con umbral de rechazo |
| `src/knn_audios.py` | KNN de los comandos de voz (supervisado), con distancia euclídea y DTW |
| `src/metricas.py` | Matriz de confusión y métricas de error (lo usan los dos) |
| `src/graficar_pca.py` | Gráficos PCA 2D y 3D (ver `GRAFICOS.md`) |

Cómo correrlos (desde la raíz del proyecto):
```
python src/kmeans_huellas.py
python src/knn_audios.py
```

---

## 0. Del dato crudo al vector: filtrado y reacondicionamiento en huellas y en voz

Antes de aplicar K-means o KNN, **las dos señales** (la imagen de la huella y el audio) pasan por el mismo tipo de proceso en tres etapas:

1. **Reacondicionamiento:** dejar la señal "en condiciones" (nivel, recorte, alineación) para que dos muestras de la misma persona o comando se parezcan aunque se hayan tomado distinto.
2. **Filtrado:** sacar lo que no es información (ruido, rayones, frecuencias inútiles) y resaltar lo que sí.
3. **Vectorización:** resumir la señal limpia en una **lista de números de largo fijo** (el vector) que los algoritmos pueden comparar.

### 0.1 Los pasos, lado a lado

| Para qué | **Huellas** (`src/preprocesar.py`) | **Voz** (`src/normalizar_audios.py`) |
|---|---|---|
| Formato común | Escala de grises | Mono, 16 kHz |
| Nivel / contraste | **Normalización de contraste** (se estira el rango de grises entre los percentiles 1 y 99) | **Normalización de volumen** (RMS a −20 dBFS, con limitador de pico) y se quita la componente continua |
| Quedarse con lo útil | **Máscara** de la zona con crestas (descarta el fondo blanco) | **Recorte de silencios** por energía (descarta lo que no es voz, con 100 ms de margen) |
| **Reajuste** | **Alineación:** se centra la huella en el lienzo y se gira para ponerla vertical (cubre rotación y traslación; con una guarda para centros imposibles) | No hace falta girar; el desajuste en el tiempo (decir la palabra más rápido o más lento) se compensa después con DTW y restando la media de cada coeficiente |
| **Filtrado** | **Filtro de Gabor orientado** (previa mediana 3×3): sigue la dirección de cada cresta y rellena cortes y rayones; elimina el ruido | **Filtro pasa-banda Butterworth** de 80 a 7000 Hz, de fase cero: elimina zumbidos graves y ruido muy agudo |
| **Binarización** | **Otsu** sobre la salida del Gabor + limpieza de motas y agujeros: crestas negras, fondo blanco | **No existe** en voz (el audio no es una imagen en blanco y negro) |
| Resumir (caracterización, `src/caracterizar_*.py`) | Orientación de las crestas → **histograma** | Pre-énfasis, ventanas de 25 ms, FFT, **40 filtros Mel**, log y DCT → **MFCC** (+ Δ y ΔΔ) |
| Vector final | **72 números** por huella | **91 números** por audio (y la secuencia de 39 por tramo para DTW) |

### 0.2 Por qué cada vector tiene esa cantidad de componentes

**La binarización no fija el tamaño del vector.** Solo deja la huella en blanco y negro para poder medir limpiamente la orientación de las crestas. El tamaño sale de **qué se mide después y cuántas partes se miden**:

**Huella: 72 = 24 + 4 × 12**
- **24:** histograma de orientación de las crestas de **toda** la huella. Los ángulos de 0° a 180° se dividen en 24 rangos de 7,5°; cada número dice qué fracción de la huella tiene crestas con esa orientación.
- **4 × 12 = 48:** la huella se divide en **4 cuadrantes** (arriba-izquierda, arriba-derecha, abajo-izquierda, abajo-derecha) y en cada uno se arma un histograma de **12 rangos** de 15°. Agrega *dónde* están las orientaciones, no solo cuáles hay.
- **Por qué esos valores** (medido con K-means entrenado con las impresiones 1 a 3): el histograma global solo (24 números) acierta **61 %**; con los 4 cuadrantes (72 números) acierta **93 %**. Otras formas de dividir la huella o rasgos más grandes separaron peor: 3 anillos concéntricos 58 %, FingerCode (256) 59 %, HOG (2520) 61 %, comparadas con las 90 huellas. Dividir más también hace que las huellas trasladadas o recortadas se parezcan menos entre sí.
- Hay dos bloques opcionales que se calculan pero no se usan para K-means: **FingerCode = 256** (4 anillos × 8 sectores × 8 filtros de Gabor) y **minucias = 18** (3 × 3 zonas × 2 tipos: terminaciones y bifurcaciones).

**Voz: 39 por tramo y 91 por audio**
- **13 MFCC:** los primeros 13 coeficientes (sin el C0, que es el volumen) de la transformada de las 40 energías Mel. Es el tamaño estándar para reconocimiento de voz: alcanza para describir la forma del tracto vocal y más coeficientes agregan detalle fino y ruido.
- **39 = 13 + 13 + 13:** los 13 MFCC, su **Δ** (velocidad de cambio entre tramos) y su **ΔΔ** (aceleración). Los Δ capturan cómo se mueve la boca de un sonido al siguiente.
- Cada audio dura distinto y queda como una **matriz de tramos × 39** (un tramo cada 10 ms). Para el KNN euclídeo hace falta un vector de largo fijo:
  **91 = 4 × 13 + 39** → la **media de los 13 MFCC en cada cuarto** del audio (52 números, conserva el orden en el tiempo) más el **desvío de cada uno de los 39 coeficientes** (cuánto varía a lo largo del audio).
- **Por qué esos valores:** con K = 3, dejando afuera una persona entera en cada vuelta: media global de los 13 MFCC → 50 % de aciertos; media + desvío → 69 %; **4 cuartos + desvío (los 91)** → 87 %. Con la secuencia completa y DTW (sin vector fijo) sube a 98 %.

> Resumen: en las huellas el vector cuenta **ángulos de crestas por región**; en la voz, **coeficientes de timbre por tramo y por cuarto de palabra**. El largo de cada vector es una decisión de diseño (ángulos, regiones, cuartos, coeficientes) medida con los datos, no algo que imponga el filtrado.

---

## 1. Panorama: aprendizaje supervisado y no supervisado

| | **No supervisado** (K-means, huellas) | **Supervisado** (KNN, voz) |
|---|---|---|
| ¿Usa las etiquetas para aprender? | **No.** Agrupa solo mirando qué tan parecidos son los vectores | **Sí.** Aprende de ejemplos que ya traen su clase (reconocer / identificar / salir) |
| Qué produce | Grupos (clusters) con un centro cada uno | Una regla para clasificar un ejemplo nuevo |
| Cuántos grupos | Se **elige K** de antemano (acá K = 3, una por persona autorizada) | Se **elige K** = cuántos vecinos se consultan (no tiene que ver con la cantidad de clases) |
| Cómo se mide si anda bien | Se compara después con las identidades reales | Se mide con datos que el modelo no vio (validación y prueba) |

Cuidado con el nombre: **la "K" de K-means y la "K" de KNN son cosas distintas.** En K-means es la *cantidad de grupos*; en KNN es la *cantidad de vecinos* que votan.

En este proyecto se usó K-means con huellas (el enunciado pide agrupar y rechazar a quien no se parece a ninguno de los 3 grupos) y KNN con la voz (hay que decidir cuál de los 3 comandos se dijo).

---

## 2. Conceptos base

### 2.1 Vector y espacio de rasgos
Un **vector** es una lista de números que describe un objeto. Una huella se resume en **72 números** (cómo se orientan sus crestas) y un audio en **91 números** (cómo suena). Un vector de 72 números es un **punto en un espacio de 72 dimensiones**. No se puede dibujar, pero las distancias se calculan igual que en 2D.

### 2.2 Distancia euclídea
Es la distancia "en línea recta" entre dos vectores. En 2D, entre A = (1, 1) y B = (4, 5):

`d(A, B) = √[(4 − 1)² + (5 − 1)²] = √(9 + 16) = √25 = 5`

Con 72 números es lo mismo: se restan los números uno a uno, se elevan al cuadrado, se suman y se saca la raíz.
`d(a, b) = √[(a₁ − b₁)² + (a₂ − b₂)² + … + (a₇₂ − b₇₂)²]`
*Código:* `distancias_euclideas()` en `kmeans_huellas.py` y en `knn_audios.py`.

### 2.3 Cluster
Un **cluster** (grupo, conglomerado) es un conjunto de vectores que quedan **cerca unos de otros y lejos de los demás**. Acá, cada cluster agrupa a las huellas que se parecen entre sí; se espera que cada uno corresponda a una persona.

### 2.4 Centroide
El **centroide** es el **centro de un cluster**: el **promedio** de todos sus vectores, número por número. Se escribe `μ`.
Ejemplo: el cluster tiene los puntos (1, 1), (2, 1) y (2, 3). Su centroide es
`((1 + 2 + 2) / 3 , (1 + 1 + 3) / 3) = (1,67 ; 1,67)`.
El centroide **no tiene por qué coincidir con ninguna huella real**: es un punto "promedio". En el gráfico se dibuja con una **✕ grande**.

### 2.5 Inercia
La **inercia** mide qué tan compactos son los clusters: es la **suma de las distancias al cuadrado** de cada vector a su centroide.
`inercia = Σ (distancia de cada vector a su centroide)²`
Cuanto menor, más juntos están los vectores alrededor de sus centros. Sirve para comparar corridas de K-means.

### 2.6 Radio (umbral) de un cluster
Para decidir si una huella "pertenece de verdad" a un cluster se define un **radio R**: la distancia máxima al centroide para aceptarla. Una huella que cae **más lejos** que R de su centroide más cercano se rechaza ("NO AUTORIZADO"). Dibujado en 2D es un círculo alrededor del centroide; en el espacio real de 72 dimensiones es una esfera. El valor de R **se calibra** (ver 3.6): no se elige a ojo.

---

## 3. K-means (huellas)

### 3.1 Idea
Dado un conjunto de vectores y un número K, K-means busca K centros de manera que **cada vector quede asignado al centro más cercano** y que los centros sean el promedio de sus vectores. No necesita saber a qué persona pertenece cada huella.

### 3.2 Los pasos (así están numerados en el código)

| Paso | Qué se hace | Dónde está en el código |
|---|---|---|
| **PASO 1** | Elegir **K huellas al azar** como centroides iniciales | `kmeans_una_vez()` (`rng.choice`) |
| **PASO 2** | **Asignar** cada huella al centroide más cercano (distancia euclídea) | `kmeans_una_vez()` (`D.argmin`) |
| **PASO 3** | **Recalcular** cada centroide como el promedio de las huellas de su cluster | `kmeans_una_vez()` (`miembros.mean`) |
| **PASO 4** | **Reasignar**: volver al paso 2 con los centroides nuevos. Parar cuando **nadie cambia de cluster** (convergió) | `kmeans_una_vez()` (condición `np.array_equal`) |
| **PASO 5** | **Repetir todo con otros centroides iniciales** (10 reinicios) y quedarse con la corrida de **menor inercia** | `kmeans()` |

Pseudocódigo:
```
elegir K centroides al azar
repetir:
    asignar cada punto al centroide más cercano
    si ningún punto cambió de grupo: parar
    mover cada centroide al promedio de sus puntos
```

### 3.3 Ejemplo a mano (6 puntos en 2D, K = 2)
Puntos: A(1,1)  B(2,1)  C(2,3)  D(8,8)  E(9,8)  F(8,10). Centroides iniciales **elegidos al azar**: c1 = A(1,1) y c2 = B(2,1) (a propósito una mala elección).

| Vuelta | Centroides | Asignación (cada punto va al más cercano) | Nuevos centroides |
|---|---|---|---|
| 1 | c1 = (1, 1), c2 = (2, 1) | A→c1; B, C, D, E, F→c2 | c1 = (1, 1); c2 = promedio de B,C,D,E,F = (5,8 ; 6,0) |
| 2 | c1 = (1, 1), c2 = (5,8 ; 6,0) | **B y C cambian a c1** (B: 1,0 a c1 vs 6,28 a c2); D, E, F siguen en c2 | c1 = (1,67 ; 1,67); c2 = (8,33 ; 8,67) |
| 3 | c1 = (1,67 ; 1,67), c2 = (8,33 ; 8,67) | A, B, C→c1; D, E, F→c2. **Nadie cambió → convergió** | — |

Se ve la **reasignación**: en la vuelta 2 los puntos B y C cambian de grupo porque el centroide c2 se corrió hacia el centro de los puntos lejanos. Inercia final = 6,67.
Este mismo paso a paso, con las huellas reales, es el gráfico `kmeans_iteraciones.png`.

### 3.4 Por qué se hacen 10 reinicios
Los centroides iniciales se eligen **al azar**, y según cuáles toque, K-means puede terminar en una solución peor (**óptimo local**). Con las huellas reales (90, ya con el efecto `matada`), los 10 reinicios dieron:

| Reinicio | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Inercia | **3,317** | 3,382 | 3,382 | 4,532 | **3,317** | 3,718 | 4,065 | 3,966 | 5,003 | 3,621 |

Solo **2 de los 10** (el 1 y el 5) llegaron a la mejor solución (3,317); los otros 8 quedaron en soluciones peores (entre 3,382 y 5,003). Con una **sola** corrida, 8 de cada 10 veces se obtendría una solución peor. Por eso se hacen varios reinicios y se queda con la de menor inercia.
(Antes de agregar `matada` los reinicios eran más parejos: 6 de 10 llegaban a la mejor. Las huellas `matada` están deformadas y hacen más difícil el problema.)

### 3.5 De cluster a identidad
K-means devuelve clusters numerados 0, 1, 2 **sin saber a quién corresponde cada uno**. Para ponerles nombre se usa el entrenamiento: se cuenta cuántas huellas de cada identidad cayeron en cada cluster y se hace la **asignación uno a uno que maximiza los aciertos** (algoritmo húngaro).
*Código:* `asignar_cluster_a_identidad()`.

### 3.6 Umbral de rechazo y cómo se calibró (PASO 6)
Regla para una huella nueva (`clasificar()` / `identificar_huella()`):
1. Calcular la distancia a los 3 centroides y quedarse con el más cercano.
2. Si esa distancia es **menor o igual al radio R** → se acepta con la identidad del cluster.
3. Si es **mayor** → **"NO AUTORIZADO"**.

**Los dos errores posibles** al elegir R:
- **Falso rechazo (FRR):** una persona autorizada es rechazada. R chico → más falsos rechazos.
- **Falsa aceptación (FAR):** un intruso es aceptado. R grande → más falsas aceptaciones.
Son un compromiso: no se pueden minimizar los dos a la vez.

**Problema:** en el dataset **no hay huellas de intrusos** para medir la FAR. **Solución usada: intrusos simulados** (*leave-one-identity-out*). `calibrar_radio()` hace, para cada una de las 3 identidades:
1. Esa identidad **hace de intruso** y se la saca del entrenamiento.
2. Con las otras dos se entrena K-means con K = 2 (igual que el modelo real).
3. Para cada R de una grilla (0,10 a 0,60) se mide la **FRR** (huellas de prueba de las 2 identidades conocidas que superan R) y la **FAR** (huellas del intruso que quedan dentro de R).
Se promedian las 3 vueltas y se elige el **R donde las dos curvas se cruzan** (error de igual tasa, EER). Tabla real:

| R | 0,18 | 0,22 | 0,26 | 0,30 | **0,31** | 0,34 | 0,38 | 0,42 | 0,50 |
|---|---|---|---|---|---|---|---|---|---|
| FRR (autorizadas rechazadas) | 48,6 % | 41,7 % | 26,4 % | 18,1 % | **13,9 %** | 9,7 % | 8,3 % | 8,3 % | 6,9 % |
| FAR (intrusos aceptados) | 0 % | 0 % | 0 % | 6,7 % | **14,4 %** | 37,8 % | 53,3 % | 70,0 % | 81,1 % |

**Radio elegido: R = 0,31** (el mismo para los 3 clusters). Con ese radio, ~14 % de las autorizadas se rechazan y ~14 % de los intrusos simulados entran.
Antes se usaba el percentil 95 de las distancias de entrenamiento de cada cluster; con `matada` en el entrenamiento ese método da radios muy grandes (las huellas deformadas lo inflan) y **dejaría entrar a la mayoría de los intrusos simulados (FAR ≈ 77 %)**. El criterio calibrado es mejor.

**Radio aplicado actualmente: R = 0,42.** El 0,31 es el que da la calibración con intrusos simulados; con las huellas reales (las subidas a mano con feedback y las generadas con Anguli) el mejor valor resultó mayor, y se fijó 0,42, el más grande con el que ningún intruso de las huellas subidas a mano fue aceptado (autorizadas rechazadas: 14,8 % en entrenamiento y 8,3 % en prueba; con los datos reales combinados 4,3 % de autorizadas rechazadas y 8,3 % de intrusos aceptados). Ojo: con intrusos simulados dentro del dataset ese radio dejaría entrar el 70 %, por eso se recomienda seguir sumando intrusos reales. `python src/probar_huellas.py --calibrar` **solo calcula y sugiere**; para cambiar el radio del modelo (y el de la interfaz) se usa `python src/probar_huellas.py --aplicar 0.42`.

**Cómo usarlo según lo que importe más** (el programa deja `--radio` para fijarlo a mano): para un sistema de acceso donde pesa más no dejar entrar a un intruso, conviene un R menor (R = 0,28 da FAR ≈ 2 % a costa de ≈ 24 % de falsos rechazos; con 0,26, FAR 0 % y ≈ 26 % de rechazos).

**Límite de la calibración:** los "intrusos" son las otras 2 personas del propio dataset. Con intrusos reales (otras personas, otros dedos) la FAR puede ser distinta; con ellos habría que volver a calibrar. Y el R elegido se eligió con los mismos datos con que se estima el error, así que el 14 % es algo optimista.

**Comprobación con huellas generadas por Anguli (intrusos reales).** Se generaron 72 huellas con los tres niveles de filtro de la interfaz (ver 7.3): 6 por identidad y por nivel (54 autorizadas) y 6 intrusos por nivel (18 huellas maestras nuevas de otras personas). Resultado con R = 0,31:

| Nivel | Autorizadas bien identificadas | Autorizadas rechazadas | Intrusos aceptados | Distancia autorizadas (mediana / máx.) | Distancia intrusos (mediana / mín.) |
|---|---|---|---|---|---|
| Débil | 18 de 18 | 0 | 0 de 6 | 0,109 / 0,132 | 0,703 / 0,526 |
| Medio | 18 de 18 | 0 | 1 de 6 | 0,161 / 0,229 | 0,758 / 0,284 |
| Fuerte | 14 de 18 | 4 | 0 de 6 | 0,206 / 0,410 | 0,820 / 0,748 |
| **Total** | **50 de 54 (93 %)** | **4 (7 %)** | **1 de 18 (6 %)** | | |

Ninguna autorizada fue aceptada con otra identidad. Con estos datos reales el cruce entre falso rechazo y falsa aceptación queda en R ≈ 0,29 (7,4 % / 5,6 %), cerca del 0,31 calibrado, así que **el radio se mantiene**. Es mejor que la estimación con intrusos simulados (~14 %) porque los intrusos de Anguli son huellas de otras personas con dibujos muy distintos; aun así son pocos datos (18 intrusos): un solo intruso aceptado ya pesa 5,6 puntos. Los rechazos de autorizadas se concentran en el nivel **fuerte** (giro hasta 30° y traslación hasta 60 px a la vez), que combina más distorsión que cualquiera de los efectos del dataset por separado.

### 3.7 Datos y partición
- **90 huellas** = 6 efectos (primera, rayada, rotada, ruido, trasladada y **matada**) × 3 identidades × 5 impresiones.
- **Entrenamiento: 54** (impresiones 1, 2 y 3 de cada efecto e identidad). **Prueba: 36** (impresiones 4 y 5). K-means se ajusta **solo con el entrenamiento**; la prueba se usa solo para medir.
- Se usa el vector de 72 números **sin estandarizar** (con z-score separaba peor).
- Opción `--excluir-entrenamiento matada`: entrena sin ese efecto (que se sigue evaluando en la prueba).

### 3.8 Resultados
| | Entrenamiento (54) | Prueba (36) |
|---|---|---|
| Exactitud | 92,6 % | **88,9 %** |
| Tasa de error | 7,4 % | 11,1 % |
| F1 promedio | 0,925 | 0,887 |
| Falso rechazo con R = 0,31 | 10 de 54 = 18,5 % | 5 de 36 = 13,9 % |

Matriz de confusión de la prueba (fila = real, columna = predicho):

| real \ predicho | identidad_1 | identidad_2 | identidad_3 |
|---|---|---|---|
| identidad_1 | 12 | 0 | 0 |
| identidad_2 | 1 | 10 | 1 |
| identidad_3 | 1 | 1 | 10 |

Por efecto (entrenamiento / prueba):

| Efecto | primera | rayada | rotada | ruido | trasladada | **matada** |
|---|---|---|---|---|---|---|
| Exactitud | 100 % / 83,3 % | 100 % / 100 % | 100 % / 100 % | 100 % / 100 % | 100 % / 100 % | **55,6 % / 50,0 %** |

**Qué pasó al agregar `matada`:** las 15 huellas `matada` (recorte muy fuerte, traslación de 100 px, rayones y ruido) tienen un histograma de orientación muy deformado y son las que más fallan: **solo la mitad se asigna bien**. El resto de los efectos queda casi perfecto. Antes de agregarlas, con 75 huellas, la exactitud en prueba era 96,7 %; con las 90 es 88,9 %. Si se **entrena sin `matada`** (`--excluir-entrenamiento matada`) los centroides quedan más limpios: la prueba sube a **91,7 %** (y `matada` pasa de 50 % a 66,7 %).

**Corrección para las huellas trasladadas** (`alinear()` en `src/preprocesar.py`): al trasladar una huella, parte queda fuera del cuadro y **el centro de la huella se corre**. Para volver a centrarla, el programa estima su centro y su giro ajustando una elipse a la parte visible del contorno, pero cuando el recorte era muy grande esa estimación daba un **centro imposible** (por ejemplo x = 287 en una imagen de 275 px) y un giro falso de casi 30°: la huella quedaba girada, cortada y pegada a una esquina del lienzo (`trasladada_id2_imp4`: distancia 0,929, mal asignada). Se agregó una **guarda de plausibilidad**: si el centro estimado cae fuera del 10 %–90 % del cuadro, la estimación no es confiable, **no se gira** y se centra con el centroide de la parte visible. Resultado medido: acierto de las trasladadas 93 % → **100 %**, distancia máxima 0,929 → **0,390**, exactitud en prueba 86,1 % → **88,9 %**, sin empeorar ningún otro efecto ni el radio calibrado (R = 0,31). **Lo que queda:** con R = 0,31, 3 de las 15 huellas trasladadas siguen siendo rechazadas (distancias entre 0,33 y 0,39) porque el recorte les quita parte de la huella; subir R las aceptaría pero también dejaría entrar más intrusos. Se probaron y descartaron (peores o iguales): no girar nunca las huellas cortadas, centrar en el núcleo detectado con Poincaré o por coherencia mínima, tolerar desplazamientos del centro al medir la distancia y bajar el peso de los cuadrantes.

**Qué se puede y qué no se puede decir:** la FAR de ~14 % es una **estimación con intrusos simulados**, no una medición con intrusos reales. Con el efecto `matada`, la separación entre personas es menos nítida (los círculos de umbral del gráfico se solapan en 2D).

### 3.9 Limitaciones de K-means (para el coloquio)
- Hay que **fijar K de antemano** (acá es natural: 3 personas).
- Supone clusters **redondos y de tamaño parecido**; si un grupo es alargado o muy distinto en tamaño, falla.
- Depende del **azar inicial** (por eso los reinicios).
- Es sensible a los **valores atípicos** (una huella muy rara mueve el centroide).
- Usa solo distancia **euclídea**.

### 3.10 Gráficos de K-means (en `caracteristicas_huellas/graficos/`)

**`kmeans_clusters_2d.png` — clusters, centroides y umbral.**
- Cada punto es una huella; el **color** indica la identidad del cluster al que fue asignada, el **círculo** es entrenamiento y el **triángulo** es prueba (o huella no usada para entrenar).
- La **✕ grande** es el **centroide** de cada cluster.
- El **círculo punteado** es el **radio de umbral R = 0,31**: lo que queda afuera se rechaza.
- Un punto con **borde rojo** es una huella asignada a la identidad equivocada (hay varios, casi todos entre los clusters).
- Eje X e Y = PC1 y PC2 del PCA, que solo sirve para **dibujar**: K-means se calcula con los 72 números, no con el plano. El círculo es la **sombra** en el plano de la esfera real de 72 dimensiones: una huella aceptada siempre cae dentro del círculo, pero una huella dentro del círculo puede estar fuera de la esfera (lejos en las dimensiones que no se ven). Por eso los círculos se solapan en el dibujo aunque en 72 dimensiones estén más separados.

**`kmeans_clusters_3d.png` — esferas de aceptación en 3D.** El mismo PCA pero con 3 componentes: alrededor de cada centroide (✕) hay una **esfera translucida de radio R**; una huella **dentro** de la esfera se acepta y **fuera** se rechaza (puntos llenos = aceptadas, huecos = rechazadas; círculo = entrenamiento, triángulo = prueba). Cada punto se dibuja a su **distancia real** (72 dimensiones) del centroide más cercano, así "dentro de la esfera" coincide con la decisión del algoritmo. Con `python src/graficar_kmeans_3d.py --mostrar` se abre interactivo para girarlo, y con `--radio 0.31` se dibuja con otro radio.

**`kmeans_iteraciones.png` — paso a paso.** Cada panel es una vuelta del algoritmo: arranca con las ✕ en lugares al azar (PASO 1 y 2), en cada vuelta se **recalculan** los centroides y las huellas se **reasignan** (el título dice cuántas cambiaron de cluster), hasta que "nadie cambia" (convergió, en 6 vueltas). Acá los colores son clusters, no identidades. Es el gráfico ideal para **explicar la reasignación**.

---

## 4. KNN (voz)

### 4.1 Idea
**KNN (K-Nearest Neighbors, K vecinos más cercanos)** no "aprende" una fórmula: **guarda los ejemplos de entrenamiento** y, para clasificar uno nuevo, **mira a quiénes se parece más** y copia lo que ellos son. Es como preguntarle a los K vecinos más parecidos: gana la clase que más se repite.

### 4.2 Los tres conjuntos de datos
| Conjunto | Para qué sirve | En este proyecto |
|---|---|---|
| **Entrenamiento** | Son los ejemplos de referencia: vectores **con su clase conocida**. Contra ellos se compara | 137 audios (sesiones normal, giuli, mo, ruido) |
| **Validación** | Se clasifican como si fueran nuevos para **elegir K y la distancia**. El modelo no los usó para "aprender" | 38 audios (mama, joaco) |
| **Prueba** | Se usa **una sola vez, al final**, para medir qué tan bien generaliza a datos totalmente nuevos | 22 audios (maxi, mili) |

**Por qué se separan por persona y no al azar:** si el mismo hablante estuviera en entrenamiento y en prueba, el KNN encontraría vecinos casi idénticos (la misma voz diciendo lo mismo) y la exactitud saldría inflada. Dejando personas enteras afuera se mide lo que importa: reconocer el comando dicho **por alguien que el sistema nunca escuchó**.

### 4.3 Los pasos (así están numerados en el código)

| Paso | Qué se hace | Dónde está en el código |
|---|---|---|
| **PASO 1** | Separar los 3 conjuntos (por persona). El entrenamiento es el conjunto de vectores con su clase | `main()` (bloque "PASO 1") |
| **PASO 2** | Para un audio, calcular la **distancia a todos los de entrenamiento**, quedarse con los **K más cercanos** y elegir la **clase más frecuente** | `distancias_euclideas()` / `dtw()` y `knn_predecir()` |
| **PASO 3** | Repetir con los datos de **validación**, medir el porcentaje de aciertos y **elegir K y la distancia** | `main()` (bloque "PASO 2 y 3") |
| **PASO 4** | Con K y la distancia elegidos, evaluar **una sola vez** la **prueba**, usando entrenamiento + validación como referencia | `main()` (bloque "PASO 4") |

**Votación (`knn_predecir`):** cada uno de los K vecinos vota por su clase; gana la que tiene más votos. **Empate** (por ejemplo, K = 3 con un voto para cada clase): gana la clase del vecino más cercano entre las empatadas.

### 4.4 Cómo elegir K
- **K muy chico (1):** el resultado depende de un solo vecino; un audio raro o mal grabado engaña al clasificador (**sensible al ruido**, sobreajuste).
- **K muy grande:** se consulta a vecinos lejanos y se mezclan las clases (se "promedia de más").
- Se usan **K impares** (1, 3, 5, …, 15) para evitar empates entre dos clases.
- Se elige con la **validación** (nunca con la prueba). Resultado:

| Distancia | K=1 | K=3 | K=5 | K=7 | K=9 | K=11 | K=13 | K=15 |
|---|---|---|---|---|---|---|---|---|
| Euclídea | 81,6 % | 94,7 % | 92,1 % | 92,1 % | 92,1 % | 92,1 % | 92,1 % | 94,7 % |
| **DTW** | 94,7 % | **100 %** | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % |

Criterio de desempate: mayor exactitud; si hay empate, **K más chico**; si siguen empatados, la distancia euclídea (más simple). Elegido: **DTW con K = 3**.

### 4.5 Distancia euclídea vs DTW

**Euclídea** (sobre el vector de 91 números): compara los dos audios números contra números. Resume todo el audio en un vector y **pierde el orden en el tiempo**.

**DTW (Dynamic Time Warping, alineamiento temporal dinámico)** compara las **secuencias completas** (un vector de 13 MFCC por cada 10 ms) **estirando y encogiendo el tiempo** para alinear sus partes. Si una persona dice "re-co-no-cer" más lento que otra, el DTW igual hace coincidir cada sílaba con su par. Se llena una tabla donde `acum[i][j]` es el costo mínimo de alinear los primeros `i` tramos de un audio con los primeros `j` del otro:

`acum[i][j] = d(i, j) + mín( acum[i−1][j−1] , acum[i−1][j] , acum[i][j−1] )`

donde `d(i, j)` es la distancia euclídea entre el tramo `i` de un audio y el tramo `j` del otro. Las tres opciones del `mín` son: avanzar en los dos audios, avanzar solo en uno, o avanzar solo en el otro. El resultado final se divide por (n + m) para que no dependa del largo de los audios.
Dos mejoras que usa el código:
- **Banda de Sakoe-Chiba (25 %)**: solo se permite alinear tramos a menos del 25 % de distancia temporal. Evita alineaciones absurdas y ahorra cálculo.
- **Resta de la media (CMN)** de cada coeficiente en cada audio antes de comparar, para quitar el efecto del micrófono y el volumen. Y 1 de cada 2 tramos para acelerar.

*Código:* `dtw()`, `matriz_dtw()`, `preparar_secuencia()`.
Costo: DTW es más lento que la euclídea porque compara todos los pares de tramos; calcular todas las distancias del dataset (197 audios) tarda unos 15 segundos.

### 4.6 Estandarización (z-score) sin fuga de datos
Para la distancia euclídea se estandariza cada número del vector: `z = (x − media) / desvío`, para que ningún número domine por tener una escala mayor. **La media y el desvío se calculan solo con el conjunto de referencia** (`ajustar_zscore()`); si se calcularan con todos los datos, la prueba "contaminaría" el entrenamiento (**fuga de datos**) y la exactitud saldría inflada.

### 4.7 Resultados
| | Validación (38) | Prueba (22) |
|---|---|---|
| Exactitud | **100 %** | **90,9 %** (20 de 22) |
| Tasa de error | 0 % | 9,1 % |
| F1 promedio | 1,000 | 0,907 |

Matriz de confusión de la prueba:

| real \ predicho | reconocer | identificar | salir |
|---|---|---|---|
| reconocer | 6 | 1 | 1 |
| identificar | 0 | 6 | 0 |
| salir | 0 | 0 | 8 |

Por comando: *reconocer* precisión 1,00 / recall 0,75; *identificar* precisión 0,86 / recall 1,00; *salir* precisión 0,89 / recall 1,00. Por persona: joaco 100 %, mama 100 %, maxi 93,3 %, mili 85,7 %.

**Cautela:** la prueba tiene **solo 22 audios**; cada error vale 4,5 puntos porcentuales. Con tan pocos datos, 90,9 % es una estimación poco precisa (no es lo mismo que "el sistema acierta el 90,9 % de las veces"). La validación llegó al 100 % porque se usó para elegir K y distancia: es la prueba la que mide la generalización.

### 4.8 Gráficos de KNN (en `caracteristicas/graficos/`)

**`knn_confusion.png` — matriz de confusión** de validación y de prueba (ver sección 5).

---

## 5. Matrices de confusión y métricas de error

### 5.1 Cómo se lee una matriz de confusión
Es una tabla donde:
- **Cada fila** es la **clase real** del ejemplo.
- **Cada columna** es la **clase que predijo** el sistema.
- La **diagonal** (arriba-izquierda a abajo-derecha) son los **aciertos**.
- Todo lo que está **fuera de la diagonal** son **errores**: la celda [fila i, columna j] cuenta cuántos ejemplos de la clase i fueron clasificados como clase j.

Con la prueba de audios: la celda [reconocer, identificar] = 1 significa "un audio de *reconocer* fue clasificado como *identificar*". Si la matriz fuera perfecta, solo habría números en la diagonal.

En el gráfico de KNN (`knn_confusion.png`), cada celda muestra la **cantidad** y el **porcentaje de su fila** (qué parte de los reales de esa clase cayó ahí), y el color es más oscuro cuanto mayor el porcentaje. La matriz de confusión de K-means se muestra en la consola y se guarda en `metricas_kmeans.json`.
*Código:* `matriz_confusion()` y `dibujar_confusion()` en `src/metricas.py`.

### 5.2 Verdaderos y falsos positivos/negativos
Se miran **por clase**. Tomando la clase *reconocer* en la matriz de prueba:
- **VP (verdadero positivo):** era *reconocer* y se predijo *reconocer* → **6**.
- **FN (falso negativo):** era *reconocer* y se predijo otra cosa → 1 + 1 = **2** (el resto de su fila).
- **FP (falso positivo):** *no* era *reconocer* pero se predijo *reconocer* → **0** (el resto de su columna).

### 5.3 Las métricas
| Métrica | Fórmula | Qué responde | Ejemplo (clase *reconocer*, prueba) |
|---|---|---|---|
| **Exactitud** (accuracy) | aciertos / total | ¿Qué porcentaje acierta en total? | (6 + 6 + 8) / 22 = **90,9 %** |
| **Tasa de error** | 1 − exactitud | ¿Qué porcentaje se equivoca? | **9,1 %** |
| **Precisión** | VP / (VP + FP) | De lo que el sistema dijo "es *reconocer*", ¿cuánto lo era? | 6 / (6 + 0) = **1,00** |
| **Exhaustividad** (recall) | VP / (VP + FN) | De todos los *reconocer* reales, ¿cuántos encontró? | 6 / (6 + 2) = **0,75** |
| **F1** | 2 · precisión · recall / (precisión + recall) | Un solo número que equilibra las dos | 2 · 1 · 0,75 / 1,75 = **0,857** |
| **F1 macro** | promedio simple del F1 de cada clase | Resumen que trata a todas las clases por igual | (0,857 + 0,923 + 0,941) / 3 = **0,907** |

**Cómo interpretarlas:** precisión alta + recall bajo (como *reconocer*: 1,00 y 0,75) significa que cuando dice "reconocer" casi nunca se equivoca, pero **deja pasar** algunos (los confunde con otros comandos). Precisión baja + recall alto significa lo contrario: encuentra todos pero "acusa de más".
La exactitud sola puede engañar si las clases están desbalanceadas; por eso se mira también el F1 por clase.
*Código:* `metricas_clasificacion()` e `imprimir_reporte()` en `src/metricas.py`.

### 5.4 Métricas del umbral en huellas
Con un umbral de rechazo hay dos errores posibles:
- **Falso rechazo (FRR):** una persona autorizada es rechazada. Con R = 0,31: **18,5 %** en entrenamiento y **13,9 %** en prueba.
- **Falsa aceptación (FAR):** un **intruso** es aceptado. Estimada en **~14 %** con intrusos simulados (una identidad dejada afuera cada vez); **no hay medición con intrusos reales**.
Son un compromiso: **agrandar el radio** baja los falsos rechazos pero sube las falsas aceptaciones, y **achicarlo** hace lo contrario. El radio se calibró en el punto donde ambos se igualan (EER); si se prioriza la seguridad, se puede achicar R.

### 5.5 Qué mirar en el coloquio
Dar siempre **la matriz de confusión** (muestra *qué* se confunde con *qué*), **exactitud** y tasa de error de **prueba** (no de entrenamiento) y mencionar el **tamaño de la prueba**; precisión/recall/F1 cuando se pregunte por un comando en particular.

---

## 6. Comparación K-means vs KNN

| | K-means (huellas) | KNN (voz) |
|---|---|---|
| Tipo | No supervisado | Supervisado |
| Etiquetas para aprender | No las usa | Las necesita |
| Qué "aprende" | **3 centroides** (resume los datos) | **Nada**: guarda todos los ejemplos |
| Cómo clasifica | Centroide más cercano (+ umbral de rechazo calibrado, R = 0,31) | K vecinos más cercanos votan |
| Qué es K | Cantidad de grupos (3) | Cantidad de vecinos (3, elegido con la validación) |
| Distancia | Euclídea sobre 72 números | Euclídea sobre 91 números o **DTW** sobre la secuencia |
| Costo al clasificar | Muy bajo (3 distancias) | Mayor (una distancia por cada audio guardado) |
| Datos | 54 entrenamiento / 36 prueba | 137 / 38 / 22 (entrenamiento / validación / prueba) |
| Resultado en prueba | 88,9 % (91,7 % sin `matada` en el entrenamiento) | 90,9 % |
| Gráficos | Clusters + centroides + umbral, iteraciones | Matriz de confusión |

---

## 7. Qué guarda el programa y cómo se usa en la demo

**Huellas** (`python src/kmeans_huellas.py`) genera en `caracteristicas_huellas/`: `modelo_kmeans.npz` (centroides, radios, qué cluster es cada identidad), `metricas_kmeans.json` y los gráficos en `graficos/`.
```python
from kmeans_huellas import identificar_huella
from preprocesar import preprocesar_huella
from caracterizar_huellas import caracterizar_huella
vector = caracterizar_huella(preprocesar_huella("huella.jpg"))["orient"]
identificar_huella(vector)      # -> ("identidad_2", 0.09)  o  ("NO AUTORIZADO", 0.41)
```
**Voz** (`python src/knn_audios.py`) genera en `caracteristicas/`: `modelo_knn.npz` (los ejemplos de referencia, K y la distancia elegida), `metricas_knn.json` y los gráficos. Para la demo:
```python
from knn_audios import clasificar_audio
comando, distancia, votos = clasificar_audio(audio, fs)   # normaliza y clasifica
```
El modelo guardado usa **todos** los audios como referencia (entrenamiento + validación + prueba), una vez que ya se midió con ellos.

### 7.1 Probar la voz en vivo (`src/probar_voz.py`)
Graba un audio igual que para el dataset (16 kHz, mono, 1,5 s), lo pasa por **todo el pipeline** (normalizar → MFCC → KNN con DTW) y dice **qué entendió**:
```
python src/probar_voz.py                      # graba y clasifica, repitiendo
python src/probar_voz.py --comando salir      # decís "salir" cada vez: cuenta los aciertos
python src/probar_voz.py --archivo audio.wav  # clasifica un WAV existente (sin micrófono)
```
La respuesta muestra: el **comando entendido**, la **confianza**, los **votos** de los 3 vecinos y **cuáles son esos vecinos** (audio del dataset, su comando y la distancia). Eso sirve para **explicar la decisión**: "se parece a estos tres audios del dataset".

**Cómo leer la confianza (se basa en los votos):**
- **Alta:** los 3 vecinos votaron el mismo comando. Con audios de personas que el modelo no conocía, los casos de confianza alta acertaron 56 de 57 (98 %).
- **Media:** ganó por 2 contra 1. Con audios nuevos, 2 de 3 acertaron.
- **Baja:** los 3 vecinos votaron cosas distintas (no hay mayoría).

**Qué NO hace todavía:**
- Si el volumen es muy bajo ("No escuché nada") o no se detecta voz, avisa y **no clasifica**.
- Pero **no puede decir "esto no es un comando"**: si se dice otra palabra, el KNN igual elige el comando más parecido. Tampoco sirve un umbral por distancia: medimos que el silencio y el ruido quedan **más cerca** de los comandos (distancia 2,8 a 3,6) que audios reales de personas nuevas (3,4 a 5,7). Para poder rechazar hay que **agregar una clase `otros`** (silencio, ruido y otras palabras) al dataset.

**Prueba de punta a punta** (referencia temporal sin las sesiones mama, joaco, maxi y mili; se clasificaron 60 audios **crudos**, como los entregaría el micrófono): **58 de 60 = 96,7 %**.

### 7.2 Probar huellas nuevas y calibrar con feedback (`src/probar_huellas.py`)
Se suben huellas (autorizadas o intrusas) a la carpeta `huellas_nuevas/` y el programa les aplica todo el pipeline (preprocesar → 72 números → centroide más cercano → umbral). Informa si la huella fue **aceptada** (con qué identidad) o **rechazada**, la distancia, el radio y la distancia a cada persona; guarda una imagen de revisión (original | preprocesada) en `pruebas_huellas/previas/`.
```
python src/probar_huellas.py              # procesa las nuevas y pide feedback (s / n / 1,2,3 / i)
python src/probar_huellas.py --calibrar   # con el feedback acumulado, calcula el mejor radio
```
- **Feedback:** el programa pregunta si el resultado fue correcto. Si el nombre del archivo empieza con `id1_`, `id2_`, `id3_` o `intruso_`, ya sabe la respuesta y no pregunta.
- **Calibración con datos reales:** con huellas autorizadas e intrusas **reales** y su respuesta, `--calibrar` calcula para cada radio R el porcentaje de autorizadas rechazadas (falso rechazo), de autorizadas aceptadas con otra identidad y de intrusos aceptados (**falsa aceptación**). Sugiere el R donde los errores se equilibran y cómo aplicarlo (`python src/kmeans_huellas.py --radio R`). Esto reemplaza la estimación con intrusos simulados de la sección 3.6 por una medición real; hacen falta al menos ~5 huellas de cada tipo para fiarse.

### 7.3 La interfaz (`interfaz/app.py`)
```
python interfaz/app.py
```
Todos los scripts de la interfaz están en la carpeta `interfaz/` y **reutilizan** los módulos de `src/` (no reimplementan ningún algoritmo): `app.py` (ventana, Tkinter), `anguli.py` (generación de huellas), `servicios.py` (lógica sin ventana), `grafico.py` (gráfico de clusters), `estilo.py` (aspecto visual: tema oscuro y widgets planos propios) y `config.py` (rutas y opciones).

**Flujo:**
1. Se elige **Identidad 1, 2, 3 o Intruso** y un **filtro débil, medio o fuerte** (cada nivel combina giro, desplazamiento, rayones y ruido a la vez; ver la tabla de abajo) y se aprieta "Generar huella (Anguli)". Se muestran la **huella generada** y la **huella con filtros** (preprocesada). También se puede "Elegir existente".
2. **Hablar** (o los botones de respaldo): **RECONOCER** → el K-means dice si la huella queda dentro del radio R (*reconocida*) o fuera (*no reconocida, no autorizada*). **IDENTIFICAR** (solo si fue reconocida) → a qué identidad pertenece. **SALIR** → cierra.
3. Opcional: **gráfico de clusters**: plano PCA con las huellas del dataset, los 3 centroides y el círculo del umbral, la **huella nueva** (estrella) y una **línea punteada hasta el cluster más cercano** con la distancia real (72 dimensiones) y R. **Cómo se dibuja:** el PCA aplasta 72 dimensiones en 2 y hace que los puntos parezcan más cerca de lo que están; por eso cada punto se dibuja a su **distancia real** del centroide más cercano (en la dirección que muestra el plano). Así se cumple la regla visualmente: **dentro del círculo = aceptada, fuera = rechazada**, y el largo de la línea es exactamente la distancia escrita.

**Filtros de la interfaz.** Cada nivel combina las cuatro distorsiones a la vez. Para cada una se da un **rango** y en cada generación se sortea el valor máximo que se le pasa a Anguli (Anguli luego elige al azar dentro de ese máximo):

| Nivel | Giro máx. (°) | Desplazamiento máx. (px) | Rayones máx. | Ruido máx. |
|---|---|---|---|---|
| Débil | 3–7 | 6–14 | 1–3 | 2–4 |
| **Medio** (por defecto) | 12–18 | 25–35 | 6–10 | 6–10 |
| Fuerte | 26–30 | 52–60 | 12–15 | 12–15 |

**Por qué rangos y no un valor fijo:** se comprobó que, con los mismos parámetros, **Anguli genera siempre la misma imagen** (la semilla `-seed` solo vale al crear huellas maestras, no impresiones). Sortear el máximo dentro de un rango hace que cada huella salga distinta y siga siendo del nivel; además se pide a Anguli entre 1 y 8 impresiones (`-ni`) y se toma una al azar, porque la impresión número 1 es siempre la misma con los mismos parámetros y las siguientes no (pedir hasta 8 no tarda más: unos 2 s). Con esto, 8 huellas seguidas de la misma identidad y nivel salieron todas distintas. El nivel fuerte llega a lo que tiene el dataset (giro 30, traslación 60). Los valores se cambian en `FILTROS` de `interfaz/anguli.py`.

**`subprocess` (uso en `anguli.py`):** es la librería de Python para **ejecutar y controlar otro programa** como si se lo escribiera en la consola. Se llama a `Anguli.exe` con una lista de argumentos (`-indir … -impr` para una impresión nueva de una identidad; `-num 1 … -seed <al azar>` para un intruso, es decir una huella maestra nueva), con `cwd` en la carpeta de Anguli (necesita sus carpetas de apoyo), `timeout` para cortarlo si tarda de más y `returncode` para saber si terminó bien.

**Orden de archivos:** Anguli deja muchas subcarpetas; la interfaz trabaja en una carpeta **temporal**, toma solo la impresión, la mueve a `huellasConID/` con el nombre `id2_<fecha>.jpg` o `intruso_<fecha>.jpg` (sin subcarpetas) y borra el resto. Esos nombres son los que entiende `probar_huellas.py`.

**Seguridad de la voz:** un comando por voz solo se ejecuta si los 3 vecinos del KNN coinciden (confianza alta); si no, pide repetirlo (`EXIGIR_CONFIANZA_ALTA` en `config.py`). Como no existe la clase "otros", cualquier otra palabra se interpreta como el comando más parecido.

**Registro:** cada huella reconocida queda en `interfaz/registro_interfaz.csv` con su **origen real** (id o intruso) y la decisión del sistema; sirve para medir aciertos y para recalibrar el radio con intrusos generados por Anguli. Si Anguli está en otra ruta, se cambia en `interfaz/config.py` (o la ventana pide elegir el `.exe`).

**Importante:** al grabar más audios o huellas, hay que volver a correr la cadena (normalizar → caracterizar → K-means o KNN) para que los modelos y las métricas se actualicen.

---

## 8. Glosario

- **Vector:** lista de números que describe un objeto.
- **Cluster:** grupo de vectores cercanos entre sí.
- **Centroide:** centro de un cluster (promedio de sus vectores).
- **Inercia:** suma de distancias al cuadrado de los vectores a su centroide (menor = clusters más compactos).
- **Radio / umbral (R):** distancia máxima al centroide para aceptar una huella; se calibra con intrusos simulados.
- **EER (error de igual tasa):** el punto donde el falso rechazo y la falsa aceptación valen lo mismo.
- **Intruso simulado:** una identidad del dataset que se saca del entrenamiento para hacer de persona no autorizada.
- **Distancia euclídea:** distancia en línea recta entre dos vectores.
- **DTW:** distancia que alinea secuencias estirando/encogiendo el tiempo.
- **Entrenamiento / validación / prueba:** ejemplos de referencia / para elegir parámetros / para la medición final.
- **Fuga de datos:** que información de la prueba se "cuele" en el entrenamiento e infle los resultados.
- **Óptimo local:** solución que parece la mejor pero no lo es; K-means puede caer en una según el azar inicial.
- **Matriz de confusión:** tabla real vs predicho.
- **Precisión / recall / F1:** de lo que dijo, cuánto acertó / de lo real, cuánto encontró / equilibrio entre ambos.
- **FRR / FAR:** falso rechazo / falsa aceptación.
- **PCA:** técnica que "aplana" los vectores a 2 o 3 dimensiones solo para dibujarlos (ver `GRAFICOS.md`).

---

## 9. Preguntas probables en el coloquio

**¿Qué es un centroide?** El centro de un cluster, el promedio de todos sus vectores. No tiene que ser un ejemplo real.

**¿Por qué K-means para huellas y KNN para voz?** Las huellas deben agruparse por parecido y rechazar lo que no se parece a ningún grupo (no supervisado, con umbral). Para la voz hay ejemplos etiquetados de cada comando y hay que decidir cuál se dijo (supervisado).

**¿Cómo se eligió K en K-means?** No se eligió: K = 3 porque hay 3 personas autorizadas.

**¿Y en KNN?** Con el conjunto de validación, probando K impares de 1 a 15. K = 1 es muy sensible al ruido; se eligió K = 3 con DTW (100 % en validación).

**¿Por qué varios reinicios en K-means?** Porque los centroides iniciales son al azar y puede quedar una mala solución. Se repite 10 veces y se queda con la de menor inercia.

**¿Cómo sabe K-means cuándo parar?** Cuando en una vuelta ningún elemento cambia de cluster.

**¿Qué es la "reasignación"?** Después de mover los centroides, cada huella vuelve a elegir su centroide más cercano; puede cambiar de cluster.

**¿Por qué separar en tres conjuntos?** Entrenamiento: referencia. Validación: elegir parámetros sin "mirar" la prueba. Prueba: medida honesta final.

**¿Por qué se separó por persona?** Para no inflar la exactitud con el mismo hablante en entrenamiento y prueba.

**¿Por qué DTW?** Porque la misma palabra dicha por dos personas dura distinto; DTW alinea las secuencias en el tiempo. Mejoró la validación de 94,7 % a 100 % con K = 3.

**¿Qué mide la exactitud, y por qué no alcanza?** El porcentaje de aciertos totales. Puede ocultar qué clase falla, por eso se mira la matriz de confusión y el F1 por clase.

**¿Cuánto se puede confiar en el 90,9 %?** Poco: son solo 22 audios de prueba, y un error cambia 4,5 puntos.

**¿Cómo se eligió el umbral de rechazo?** Se probaron radios de 0,10 a 0,60 simulando intrusos (se deja una de las 3 identidades afuera, se entrena con las otras dos y se mide cuántas autorizadas se rechazan y cuántos intrusos entran). Se eligió el radio donde ambos errores se igualan: R = 0,31, con ~14 % de cada uno.

**¿Por qué bajó la exactitud de K-means al agregar `matada`?** Porque esas huellas (recorte fuerte, 100 px de traslación, rayones y ruido) tienen la orientación muy deformada: solo la mitad se asigna bien. Sin ellas en el entrenamiento la prueba sube de 88,9 % a 91,7 %.

**¿Qué falta?** Huellas de intrusos reales para confirmar el umbral y medir la falsa aceptación de verdad, más personas y más audios (y una clase "otros" para que el KNN pueda rechazar lo que no es un comando).
