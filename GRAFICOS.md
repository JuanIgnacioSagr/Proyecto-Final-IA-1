# GRAFICOS — qué se grafica hasta ahora y cómo leerlo

Hasta ahora hay **tres tipos de gráficos**:

| # | Gráfico | Qué muestra | Archivo |
|---|---|---|---|
| 1 | PCA 2D y 3D de **huellas** | Si las 3 identidades se separan según sus vectores | `caracteristicas_huellas/graficos/pca_2d.png` y `pca_3d.png` |
| 2 | PCA 2D y 3D de **audios** | Si los 3 comandos se separan según sus vectores | `caracteristicas/graficos/pca_2d.png` y `pca_3d.png` |
| 3 | Comparación de métodos de filtrado de huellas | Cómo queda la misma huella con cada filtro (A–E) | `huellasProcesadas/_figuras/comparacion_metodos.png` |

Los PCA los genera `src/graficar_pca.py`:
```
python src/graficar_pca.py --datos huellas
python src/graficar_pca.py --datos audios
```
(El gráfico 3 lo genera `src/comparar_metodos.py`.)

---

## 1. ¿Qué son los vectores?

Un **vector** es simplemente **una lista de números** que describe a un objeto. La computadora no puede comparar una foto o un audio "a ojo"; sí puede comparar listas de números, calculando qué tan cerca están.

Cada huella y cada audio se convierte en un vector:

| | Qué objeto es | Cuántos números tiene | Qué describen esos números |
|---|---|---|---|
| **Huella** | una imagen preprocesada | **72** | Hacia dónde apuntan las crestas. Se cuenta qué proporción de la huella tiene cada ángulo: 24 rangos para toda la huella + 12 rangos en cada cuadrante (2×2). Es el "dibujo" de la huella resumido en números |
| **Audio** | una palabra grabada | **91** | Cómo suena la palabra: el MFCC (la "forma" del sonido) promediado en cada cuarto del audio (4 × 13 = 52 números) y cuánto varía cada coeficiente a lo largo del audio (39 números) |

Un punto del gráfico = **un vector** = **una huella** o **un audio**.

- Los vectores de huellas los produce `src/caracterizar_huellas.py` y quedan en `caracteristicas_huellas/dataset.npz` (matriz `X`).
- Los vectores de audios los produce `src/caracterizar_audios.py` y quedan en `caracteristicas/dataset.npz` (matriz `X`).

## 2. ¿Qué componente del programa usa esos vectores?

El **PCA** (Análisis de Componentes Principales, de la librería scikit-learn) dentro de `src/graficar_pca.py`.

Problema que resuelve: no se puede dibujar un punto de 72 u 91 dimensiones, solo de 2 o 3. El PCA busca las 2 o 3 **direcciones en las que los datos más se diferencian entre sí** y "aplana" todos los vectores sobre esas direcciones. Es como sacar una foto a un objeto desde el ángulo en que más se ve.

- En **audios** se aplica primero z-score (se estandarizan los números, porque los MFCC tienen escalas muy distintas).
- En **huellas** no se estandariza, porque son proporciones ya comparables (estandarizar empeoraba la separación).

El PCA **solo se usa para dibujar**. No clasifica nada: ni K-means ni KNN se aplicaron todavía.

---

## 3. Qué significa cada eje

- **Eje X = PC1** (primera componente principal): la dirección en la que los datos más se diferencian.
- **Eje Y = PC2** (segunda): la siguiente dirección con más diferencias, **perpendicular** a la primera.
- **Eje Z = PC3** (solo en 3D): la tercera.

**Los ejes no tienen unidades ni un significado físico** ("no es Hz, ni ángulo, ni segundos"). Son combinaciones de los 72 o 91 números originales. Lo único que importa es la **distancia entre puntos**: dos puntos cercanos se parecen; dos puntos lejanos, no. El **signo** de un eje puede estar invertido de una corrida a otra (izquierda y derecha intercambiadas): no cambia nada.

## 4. Qué grupo de datos se grafica y cómo se identifican los puntos

| | Huellas | Audios |
|---|---|---|
| Cantidad | **90** huellas (6 efectos × 3 identidades × 5 impresiones, incluido el efecto `matada`) | **197** audios: reconocer 60, identificar 76, salir 61 |
| **Color** | **Identidad** (identidad_1, 2, 3): lo que queremos separar | **Comando** (reconocer, identificar, salir): lo que queremos separar |
| **Forma del marcador** | **Efecto** aplicado: primera, rayada, rotada, ruido, trasladada y matada | **Sesión**: quién habla o qué condición (giuli, joaco, mama, maxi, mili, mo, normal, ruido) |

La idea de cada gráfico: **si los puntos del mismo color quedan juntos y los de distinto color separados, los vectores sirven para distinguir las clases**. La forma sirve para ver si el efecto (huellas) o la persona (audios) desordena los grupos.

## 5. Qué son los porcentajes de los ejes

El número entre paréntesis (por ejemplo `PC1 (34.4%)`) es el **porcentaje de la varianza explicada**: qué parte de todas las diferencias entre los vectores captura ese eje.

- Los 72 o 91 números originales contienen el 100 % de las diferencias.
- PC1 se queda con una parte, PC2 con otra, y así sucesivamente.
- **Lo que no capturan los ejes dibujados queda escondido**: en el gráfico no se ve.

| Gráfico | PC1 | PC2 | PC3 | Suma |
|---|---|---|---|---|
| Huellas | 34,4 % | 18,2 % | 11,4 % | 64,0 % |
| Audios | 12,9 % | 10,6 % | 7,3 % | 30,8 % |

Cuanto más alta la suma, más fiel es el dibujo a la realidad. En huellas, el gráfico 2D (52,6 %) muestra algo más de la mitad de la información; en audios, el 2D (23,5 %) muestra menos de un cuarto.

### ¿46 % de qué, exactamente? ¿Cómo se calcula?

**Es un porcentaje de la dispersión (varianza) de los vectores.** No es un porcentaje de aciertos, ni de huellas, ni de audios.

1. Se arma una tabla con todos los vectores: una fila por huella o audio y una columna por cada número del vector (huellas: 90 filas × 72 columnas; audios: 197 filas × 91 columnas).
2. La **varianza** de una columna mide cuánto cambia ese número de una huella a otra (si todas valen casi lo mismo, la varianza es chica).
3. La **varianza total** es la suma de las varianzas de todas las columnas: es la "cantidad total de diferencias" que hay entre los vectores. Corresponde al 100 %.
4. El PCA encuentra los ejes nuevos (PC1, PC2, ...) y mide cuánta de esa varianza total queda a lo largo de cada uno. En fórmula:

   `porcentaje del eje k = (varianza a lo largo del eje k) / (varianza total) × 100`

   (en scikit-learn: `explained_variance_ratio_`). Los porcentajes de **todos** los ejes suman 100 %: hay tantos ejes como números tiene el vector (72 en huellas, 91 en audios).

Ejemplo con los números reales de las huellas:

| | Varianza a lo largo del eje | Cálculo | Porcentaje |
|---|---|---|---|
| Varianza total (las 72 columnas juntas) | 0,1123 | — | 100 % |
| PC1 | 0,0386 | 0,0386 / 0,1123 | **34,4 %** |
| PC2 | 0,0205 | 0,0205 / 0,1123 | **18,2 %** |
| PC3 | 0,0128 | 0,0128 / 0,1123 | **11,4 %** |
| Los otros 69 ejes | el resto | — | 36,0 % |

Con audios es igual. Como se estandariza (z-score), cada una de las 91 columnas pasa a tener varianza 1 y la varianza total es 91,5. PC1 tiene varianza 11,76, o sea 11,76 / 91,5 = **12,9 %**. Si la diferencia estuviera repartida parejo entre los 91 ejes, cada uno tendría solo 1,1 %; PC1 tiene unas 12 veces más, pero sigue siendo poco.

**Cómo imaginarlo:** los 90 puntos forman una nube en un espacio de 72 dimensiones, como una nube alargada en forma de cigarro. PC1 es la dirección en la que la nube es más larga. Decir "34,4 %" significa que, si proyectás todos los puntos sobre esa única línea, conservás el 34,4 % de lo que se extiende la nube; el 65,6 % restante queda en otras direcciones.

**Dos aclaraciones importantes:**
- El PCA **no mira las etiquetas** (identidad o comando) al calcular los porcentajes. Un eje con porcentaje alto es el que más dispersa los datos, no necesariamente el que mejor separa las clases. En huellas ocurre que el eje más disperso también separa las identidades; en audios, el eje más disperso (PC1) separa más por quién habla que por qué comando se dice.
- El porcentaje de varianza **no es lo mismo que el porcentaje de aciertos** de un clasificador.

---

## 6. Conclusiones

### Huellas
- Se ven **3 grupos, uno por identidad**, aunque las huellas tengan rotación, ruido, rayones o traslación: esos efectos casi no los mezclan. Las huellas `matada` (círculos en el gráfico) son las que más se dispersan y se mezclan entre grupos.
- Conclusión: el vector de 72 números (histograma de orientación de crestas) **separa bien las identidades**. Es una buena base para K-means con k=3.
- Medido con K-means (ver la teoría, sección 3.8): acierto de **88,9 %** en la prueba. Los efectos primera, rayada, rotada, ruido y trasladada quedan casi perfectos; `matada` (recorte muy fuerte, 100 px de traslación, rayones y ruido) es el único que rompe el agrupamiento (50 %). Es una limitación real.

### Audios
- Los comandos **no se ven claramente separados en 2D**; hay solapamiento. Pero dentro de cada sesión sí se ve un orden: por ejemplo, en la sesión `normal`, `reconocer` queda arriba, `identificar` en el medio y `salir` abajo (PC2 está relacionado con el comando).
- El eje que más pesa (PC1) separa más por **sesión** (quién habla, volumen, ruido) que por comando: la sesión `ruido` queda a la derecha y `mama` a la izquierda.
- Como el dibujo solo muestra el 23,5 % de la información, **solapar en el gráfico no significa que el KNN vaya a confundirlos**: el KNN usa los 91 números, o la secuencia completa con DTW, no el dibujo.
- Referencia: con KNN usando DTW sobre las secuencias, dejando afuera una sesión entera en cada vuelta, se midió un acierto de **97,8 %** (con 182 audios; hay que volver a medirlo con los 197 actuales cuando se arme el KNN).
- Conclusión: el PCA 2D de audios sirve para **ver la dificultad** (el hablante pesa mucho) y para justificar que conviene sumar más personas y usar DTW. No alcanza para decir cuánto va a acertar el clasificador.

## 7. Gráfico 3: comparación de métodos de filtrado de huellas

Filas: 6 huellas de ejemplo (primera, rotada, trasladada, matada, rayada, ruido). Columnas: la imagen original y el resultado de cada método (A a E):

- **A:** filtro de mediana + umbral. **B:** gaussiano + umbral adaptativo. **C:** filtro de Gabor orientado. **D:** C + limpieza de motas (elegido). **E:** A + limpieza.
- Conclusión: A, B y E dejan puntos sueltos y pierden crestas finas; C deja pocas motas; **D deja 0 motas en todos los efectos** y conserva el núcleo de la huella, por eso es el que usa `preprocesar.py`.

## 8. Cómo se actualizan

- Si agregás audios: `python src/normalizar_audios.py`, luego `python src/caracterizar_audios.py`, y vuelve a correr `python src/graficar_pca.py --datos audios`.
- Si agregás huellas: `python src/preprocesar.py`, luego `python src/caracterizar_huellas.py`, y vuelve a correr `python src/graficar_pca.py --datos huellas`.

## 9. Gráficos de K-means y KNN

Ya están hechos; se explican en `Teoria metodos y forma implementada en el codigo.md` (secciones 3.6, 3.10, 4.8 y 5):

- **K-means (huellas)**, en `caracteristicas_huellas/graficos/`: `kmeans_clusters_2d.png` (clusters con centroides y radio de umbral), `kmeans_iteraciones.png` (paso a paso) y `kmeans_clusters_3d.png` (PCA en 3D con una **esfera de aceptación** de radio R alrededor de cada centroide; con `python src/graficar_kmeans_3d.py --mostrar` se abre interactivo para girarlo). Se generan con `python src/kmeans_huellas.py`.
- **KNN (audios)**, en `caracteristicas/graficos/`: `knn_confusion.png` (matriz de confusión de validación y de prueba). Se genera con `python src/knn_audios.py`.
