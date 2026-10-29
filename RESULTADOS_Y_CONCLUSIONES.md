# Resultados y conclusiones de todas las etapas del proyecto

Proyecto final de Inteligencia Artificial I: **reconocimiento biométrico por huella dactilar (K-means) y comandos de voz (KNN)**, con una interfaz que une los dos.
Estado al 27/10/2026. Todos los números salen de corridas reales del código del repositorio; donde un número es una estimación o tiene pocos datos, está dicho.

> Documentos relacionados: `Teoria metodos y forma implementada en el codigo.md` (explicación de los métodos), `GRAFICOS.md` (qué muestra cada gráfico) y `HISTORIAL.md`.

---

## 1. Resumen

| Etapa | Qué se hizo | Resultado principal |
|---|---|---|
| Huellas · datos | 3 huellas maestras y 6 efectos con Anguli | 90 imágenes (6 efectos × 3 identidades × 5 impresiones) |
| Huellas · preprocesamiento | Un solo pipeline para cualquier imagen | 0 motas sueltas por imagen en todos los efectos |
| Huellas · caracterización | Histograma de orientación de crestas | Vector de **72 números** |
| Huellas · K-means (k = 3) | Clasificación con umbral de rechazo | **88,9 %** de aciertos en prueba (91,7 % entrenando sin el efecto `matada`) |
| Huellas · umbral | Radio R = 0,42 (calibrado con simulación: 0,31; ajustado con huellas reales) | 7 % de autorizadas rechazadas y 6 % de intrusos aceptados con huellas generadas por Anguli |
| Voz · datos | 197 audios, 3 comandos, 8 sesiones | Dataset grabado con el micrófono de la demo |
| Voz · acondicionamiento | Filtro pasa-banda, recorte de silencios, volumen | Todos los audios a −20 dBFS de voz, 16 kHz |
| Voz · caracterización | MFCC + Δ + ΔΔ | Secuencia de 39 por tramo y vector de **91 números** |
| Voz · KNN con DTW (K = 3) | Separación por persona en 3 conjuntos | **100 %** en validación y **90,9 %** en prueba (22 audios) |
| Interfaz | Anguli por `subprocess` + voz + gráfico de clusters | Flujo reconocer → identificar → gráfico funcionando de punta a punta |

**Conclusión general:** el sistema reconoce bien huellas con rotación, ruido, rayones y traslaciones moderadas, y comandos de voz de personas que no estaban en el entrenamiento. Sus puntos débiles medidos son las huellas muy recortadas, la falta de una clase "otros" en la voz y la poca cantidad de datos de prueba (ver sección 6).

---

## 2. Etapa de huellas

### 2.1 Datos (Anguli)
- **3 identidades** (huellas maestras `fp_1/1.jpg`, `2.jpg`, `3.jpg`, de 275 × 400 px).
- **6 efectos**, cada uno con 5 impresiones por identidad (15 por efecto, **90 en total**): `primera` (impresiones de la generación original: giro 30, traslación 16, rayones y ruido hasta 20), `rotada` (giro 30), `trasladada` (traslación 60), `ruido` (nivel 20), `rayada` (traslación 24, giro 7, hasta 20 rayones) y `matada` (traslación 100, giro 7, 8 rayones, ruido 11: el más agresivo).
- Se reorganizaron en `huellasDactilares/<efecto>/id<N>/<efecto>_id<N>_imp<M>.jpg`.

### 2.2 Preprocesamiento (`src/preprocesar.py`)
**Pipeline único**, sin ramas por efecto (en la demo puede entrar cualquier huella): gris → normalización de contraste → máscara de la zona con crestas → **alineación** (centrar y poner vertical) → **filtro de Gabor orientado** → binarización de Otsu → limpieza de motas. Salida: crestas negras sobre lienzo blanco de 320 × 448 px.

**Comparación de cinco métodos de filtrado/binarización** sobre las 90 imágenes (motas sueltas que quedan por imagen, menor es mejor):

| Método | matada | primera | rayada | rotada | ruido | trasladada |
|---|---|---|---|---|---|---|
| A: mediana + Otsu | 146,3 | 88,3 | 97,1 | 90,1 | 57,2 | 146,1 |
| B: gaussiano + umbral adaptativo | 34,1 | 60,1 | 60,7 | 49,0 | 50,0 | 50,3 |
| C: Gabor orientado + Otsu | 9,2 | 7,7 | 5,9 | 6,3 | 3,3 | 6,6 |
| **D: C + limpieza de motas y agujeros (elegido)** | **0** | **0** | **0** | **0** | **0** | **0** |
| E: A + limpieza | 0 | 0 | 0 | 0 | 0 | 0 |

- **D es el único bueno en todos los efectos a la vez**: el Gabor sigue la dirección de cada cresta y rellena rayones y cortes; A, B y E dejan puntos sueltos o pierden crestas finas (en el panel visual `comparacion_metodos.png` se ve claramente).
- Costo: 0,11 s por imagen (A: 0,02 s).

**Corrección de la traslación (cuando la huella queda cortada por el borde).** El centro y el giro de una huella cortada se estimaban ajustando una elipse a la parte visible; en un caso daba un centro imposible (x = 287 en una imagen de 275 px) y un giro falso de casi 30°. Se agregó una **guarda de plausibilidad** en `alinear()`: si el centro cae fuera del 10–90 % del cuadro, no se gira y se centra con la parte visible.

| | Antes | Después |
|---|---|---|
| Acierto en `trasladada` | 93 % | **100 %** |
| Distancia máxima de una `trasladada` | 0,929 | **0,390** |
| Exactitud de K-means en prueba | 86,1 % | **88,9 %** |

No empeoró ningún otro efecto. Se probaron y descartaron (peores o iguales): no girar nunca las cortadas, centrar en el núcleo (Poincaré o coherencia mínima), tolerar desplazamientos del centro al medir distancias y bajar el peso de los cuadrantes. **Queda:** con R = 0,31, 3 de las 15 trasladadas siguen rechazadas (distancias 0,33–0,39) porque el recorte les quita parte de la huella.

### 2.3 Caracterización (`src/caracterizar_huellas.py`)
Se compararon familias de rasgos agrupando con K-means (k = 3) sobre PCA y midiendo el acierto por identidad:

| Rasgo | 90 huellas | Sin `matada` |
|---|---|---|
| **Histograma de orientación de crestas** (24 globales + 4 cuadrantes × 12 = 72) | **0,92** | **0,99** |
| FingerCode (Gabor, 256) | 0,59 | 1,00 |
| HOG (2520) | 0,61 | 1,00 |
| Densidad de minucias (18) | 0,53 | 0,94 |
| LBP, frecuencia de cresta | ≤ 0,37 | — |

- **Elegido: histograma de orientación, 72 números.** Sin estandarizar funciona mucho mejor que estandarizado (0,63 vs 0,29 en la comparación): son proporciones ya comparables.
- El histograma global solo (24 números) acierta 61 %; con los 4 cuadrantes, 93 % (entrenamiento con las impresiones 1–3).
- Los dos efectos sin deformación fuerte dan 100 %; solo `matada` rompe la separación en todos los rasgos.

### 2.4 PCA de las huellas
Varianza explicada: PC1 **34,4 %**, PC2 18,2 %, PC3 11,4 % (suma 64,0 %; el 2D muestra el 52,6 %). El gráfico muestra **tres grupos, uno por identidad**; los efectos casi no los mezclan, salvo `matada`, que se dispersa. El PCA solo sirve para dibujar: K-means usa los 72 números.

### 2.5 K-means (`src/kmeans_huellas.py`, desde cero)
Entrenamiento: impresiones 1–3 (54 huellas); prueba: impresiones 4–5 (36).

| | Entrenamiento | Prueba |
|---|---|---|
| Exactitud | 92,6 % | **88,9 %** |
| F1 promedio | 0,925 | 0,887 |

Matriz de confusión de la prueba (fila = real, columna = predicho):

| real \ predicho | identidad 1 | identidad 2 | identidad 3 |
|---|---|---|---|
| identidad 1 | 12 | 0 | 0 |
| identidad 2 | 1 | 10 | 1 |
| identidad 3 | 1 | 1 | 10 |

| Efecto | primera | rayada | rotada | ruido | trasladada | **matada** |
|---|---|---|---|---|---|---|
| Entrenamiento / prueba | 100 % / 83,3 % | 100 % / 100 % | 100 % / 100 % | 100 % / 100 % | 100 % / 100 % | **55,6 % / 50,0 %** |

- **10 reinicios:** solo 2 llegaron a la mejor inercia (3,317); 8 quedaron en óptimos locales peores (3,38–5,00). Por eso se hacen varios reinicios y se elige el de menor inercia.
- **`matada` (recorte fuerte + 100 px de traslación + rayones + ruido) es la limitación real**: solo la mitad se asigna bien. Entrenando sin ese efecto (`--excluir-entrenamiento matada`), la prueba sube a 91,7 % (y `matada` a 66,7 %).

### 2.6 Umbral de rechazo (radio R)
Un huella se acepta si su distancia al centroide más cercano es ≤ R; si no, es "NO AUTORIZADO". Hay dos errores que se compensan: **falso rechazo** (una persona autorizada es rechazada) y **falsa aceptación** (un intruso entra).

**Calibración 1: intrusos simulados** (se deja una identidad afuera y se entrena con las otras dos). El cruce de los dos errores da **R = 0,31 con 13,9 % de falso rechazo y 14,4 % de falsa aceptación**. El criterio anterior (percentil 95 del cluster) habría dejado entrar a casi 8 de cada 10 intrusos simulados.

**Calibración 2: huellas reales generadas con Anguli** (72 huellas con los tres niveles de filtro de la interfaz; 54 autorizadas y 18 intrusos, que son huellas maestras nuevas de otras personas), con R = 0,31:

| Nivel de filtro | Autorizadas bien identificadas | Intrusos aceptados |
|---|---|---|
| Débil | 18 de 18 | 0 de 6 |
| Medio | 18 de 18 | 1 de 6 |
| Fuerte | 14 de 18 | 0 de 6 |
| **Total** | **50 de 54 (93 %)** | **1 de 18 (6 %)** |

Ninguna autorizada fue aceptada con otra identidad. Los rechazos se concentran en el nivel fuerte (giro 30° y traslación 60 px a la vez).

**Dato con huellas subidas a mano** (`pruebas_huellas/registro.csv`, 21 huellas con tu feedback: 15 autorizadas y 6 intrusas): con R = 0,31 se reconocieron **8 de 15 autorizadas** (7 rechazadas por error, ninguna con otra identidad) y **0 de 6 intrusas** fueron aceptadas. Las autorizadas rechazadas tienen distancias de 0,31 a 0,78; las intrusas, de 0,42 en adelante.

**Datos reales combinados** (69 autorizadas y 24 intrusas):

| R | Autorizadas rechazadas | Intrusos aceptados |
|---|---|---|
| 0,28 | 17,4 % | 0 % |
| **0,31 (actual)** | **15,9 %** | **4,2 %** |
| 0,34 | 13,0 % | 4,2 % |
| 0,36 | 10,1 % | 4,2 % |
| 0,38 | 5,8 % | 4,2 % |
| 0,40 | 5,8 % | 8,3 % |
| 0,42 | 4,3 % | 8,3 % |

El cruce de los dos errores con datos reales queda en **R ≈ 0,38 (5,8 % / 4,2 %)**. Es una **sugerencia a validar** con más intrusos (24 es poco: un intruso más cambia 4 puntos); **decisión tomada: el radio del modelo se fijó en R = 0,42** (el mayor radio con el que ningún intruso de las huellas subidas a mano fue aceptado). Con 0,42: autorizadas rechazadas 14,8 % en entrenamiento y 8,3 % en prueba; con los datos reales combinados, 4,3 % de autorizadas rechazadas y 8,3 % de intrusos aceptados (2 de 24). **Aviso:** con intrusos simulados dentro del dataset (otras identidades de la misma base) a 0,42 entraría el 70 %, por eso conviene seguir sumando intrusos reales para confirmar este valor. Se aplica con `python src/probar_huellas.py --aplicar 0.42` (`--calibrar` solo sugiere).

**Conclusión de la etapa de huellas:** el histograma de orientación con K-means separa bien las tres identidades aunque la huella esté rotada, ruidosa, rayada o trasladada, y rechaza a casi todos los intrusos. Falla con huellas muy recortadas y el umbral conviene revisarlo con más datos reales.

---

## 3. Etapa de voz

### 3.1 Datos (`src/grabar_audios.py`)
**197 audios** de 1,5 s, 16 kHz, mono, grabados con el micrófono de la demo: *reconocer* 60, *identificar* 76, *salir* 61. Hay 8 sesiones (personas o condiciones): normal 58, giuli 34, mama 29, mo 28, ruido 17, maxi 15, joaco 9, mili 7. La sesión `ruido` solo tiene "identificar". Se grabó en tandas con nombre de sesión para poder separar entrenamiento, validación y prueba **por persona**.

**Decisión de diseño:** grabar directo desde Python con el mismo micrófono de la demo, para que la distancia entre audios mida la palabra y no el equipo.

### 3.2 Normalización y filtrado (`src/normalizar_audios.py`)
Pasos: quitar continua → **pasa-banda Butterworth de 80 a 7000 Hz** (fase cero) → **recorte de silencios** por energía (100 ms de margen) → **normalización de volumen** a −20 dBFS con limitador de pico. No se usó sustracción espectral (el micrófono es limpio y el informe advierte de "ruido musical").

- **Problema que resolvió:** los audios originales eran muy bajos y dispares (pico mediano 0,053; voz a −42 dBFS en promedio). Quedaron con voz entre −29,2 y −20,0 dBFS (ganancia mediana +21,9 dB).
- **Duración:** de 1,50 s a una mediana de **0,94 s** (0,29–1,50 s).
- **Avisos:** 6 audios con saturación en el original y 21 con pico limitado; ningún audio sin voz. Los audios con clics fuertes y voz baja quedan algo más bajos que el resto; conviene regrabarlos.
- El proceso es **incremental** y nunca modifica la carpeta original.

### 3.3 Caracterización (`src/caracterizar_audios.py`)
Pre-énfasis 0,97 → tramos de 25 ms cada 10 ms → FFT → **40 filtros Mel** → log → DCT → **13 MFCC** → Δ y ΔΔ = **39 por tramo**. Se guarda la **secuencia completa** (para DTW) y un **vector fijo de 91 números** (media de los 13 MFCC en cada cuarto del audio + desvío de los 39 coeficientes).

Evidencia de por qué 91 (K = 3, dejando una persona entera afuera en cada vuelta):

| Representación | Acierto |
|---|---|
| Media de los 13 MFCC | 50 % |
| Media + desvío | 69 % |
| 4 cuartos de MFCC + desvío (**91 números**) | **87 %** |
| Secuencia completa con DTW | **98 %** |

(Medido con los 182 audios que había entonces.) La secuencia temporal es lo que más aporta.

### 3.4 PCA de los audios
PC1 12,9 %, PC2 10,6 %, PC3 7,3 % (suma 30,8 %; el 2D muestra solo el 23,5 %). Los comandos **se solapan en 2D** y el eje principal separa más por **quién habla** (la sesión `ruido` a la derecha, `mama` a la izquierda) que por qué comando se dice. Solapar en el dibujo no implica confusión del KNN: usa los 91 números o la secuencia completa.

### 3.5 KNN (`src/knn_audios.py`, desde cero)
Tres conjuntos separados **por persona**, nunca al azar: entrenamiento 137 (normal, giuli, mo, ruido), validación 38 (mama, joaco) y prueba 22 (maxi, mili).

Exactitud en validación según K:

| Distancia | K=1 | K=3 | K=5 | K=7 | K=9 | K=11 | K=13 | K=15 |
|---|---|---|---|---|---|---|---|---|
| Euclídea (91 números, z-score) | 81,6 % | 94,7 % | 92,1 % | 92,1 % | 92,1 % | 92,1 % | 92,1 % | 94,7 % |
| **DTW** (secuencias) | 94,7 % | **100 %** | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % |

**Elegido: DTW con K = 3.** K = 1 es el peor (sensible a un audio raro). Resultado final:

| | Validación (38) | Prueba (22) |
|---|---|---|
| Exactitud | 100 % | **90,9 %** (20 de 22) |
| F1 promedio | 1,000 | 0,907 |

Matriz de confusión de la prueba:

| real \ predicho | reconocer | identificar | salir |
|---|---|---|---|
| reconocer | 6 | 1 | 1 |
| identificar | 0 | 6 | 0 |
| salir | 0 | 0 | 8 |

Por persona: joaco 100 %, mama 100 %, maxi 93,3 %, mili 85,7 %. Los dos errores son audios de *reconocer*. **La prueba tiene solo 22 audios:** cada error pesa 4,5 puntos, así que el 90,9 % es una estimación poco precisa.

### 3.6 Prueba de punta a punta (`src/probar_voz.py`)
Graba, normaliza, caracteriza y clasifica, mostrando qué entendió, la confianza (votos de los 3 vecinos) y cuáles son los vecinos. Con una referencia temporal **sin** las sesiones mama, joaco, maxi y mili, se clasificaron **60 audios crudos**: **58 de 60 (96,7 %)**. Por confianza: con los 3 vecinos coincidiendo acertó 56 de 57 (98 %); con 2 contra 1, 2 de 3.

**Limitación medida:** el KNN **siempre** elige uno de los tres comandos. La distancia no sirve para rechazar lo que no es un comando: audios reales de personas nuevas quedan a 3,4–5,7 del vecino más cercano y el silencio o el ruido a 2,8–3,6, es decir, **más cerca**. Lo que sí se puede hacer es avisar "no escuché nada" por volumen y no ejecutar el comando si los votos no coinciden. Para rechazar de verdad hace falta una **clase `otros`** (silencio, ruido y otras palabras).

**Conclusión de la etapa de voz:** el DTW sobre MFCC con KNN reconoce los tres comandos con 91–97 % de aciertos con personas que no estaban en el entrenamiento; la secuencia temporal completa es lo que hace la diferencia. Los datos son pocos (8 sesiones, 22 audios de prueba).

---

## 4. Interfaz (`interfaz/`)

Aplicación de escritorio (Tkinter). Flujo: **1)** elegir Identidad 1, 2, 3 o Intruso y un filtro (débil, medio o fuerte) y generar la huella con **Anguli ejecutado con `subprocess`**; **2)** se muestran la huella generada y la huella con filtros; **3)** por voz (o con botones de respaldo): **reconocer** → "huella reconocida" o "no reconocida"; **identificar** → a qué identidad pertenece (solo si fue reconocida); **salir**; **4)** gráfico de clusters opcional con la huella nueva, el cluster más cercano y la distancia.

Resultados de las pruebas:
- **Anguli por `subprocess`:** una impresión de una identidad tarda ~2 s; un intruso (persona nueva) de 8 a 18 s. Cada generación se hace en una carpeta temporal y deja **un solo archivo** en `huellasConID/` (`id2_<fecha>.jpg`, `intruso_<fecha>.jpg`), sin subcarpetas.
- **Hallazgo:** con los mismos parámetros Anguli genera **siempre la misma imagen** (la semilla solo vale para huellas maestras). Se resolvió sorteando los máximos de cada filtro dentro de un rango por nivel y eligiendo al azar entre 8 impresiones: 8 de 8 huellas seguidas salieron distintas.
- **Recorridos automáticos de la ventana** (generar → reconocer → identificar → gráfico) con una identidad y con un intruso: correctos (identidad reconocida e identificada; intruso rechazado, y "identificar" responde "no hay identidad").
- **Gráfico de clusters:** cada punto se dibuja a su **distancia real** (72 dimensiones) del centroide más cercano; así "dentro del círculo de radio R = huella aceptada" es coherente con la decisión (con el PCA común, una huella rechazada podía verse dentro del círculo).
- **Seguridad de la voz:** un comando solo se ejecuta si los 3 vecinos coinciden.
- **Registro:** cada huella reconocida queda en `interfaz/registro_interfaz.csv` con su origen real, lo que permite medir aciertos y recalibrar el radio con intrusos generados por Anguli.

Pendiente de probar por el usuario: la voz en vivo con el micrófono real y la demostración completa.

---

## 5. Conclusiones generales

1. **Las dos etapas de aprendizaje funcionan** con modelos simples y explicables: K-means con el histograma de orientación para huellas (88,9 % en prueba) y KNN con DTW para voz (90,9 % en prueba; 96,7 % de punta a punta con voces nuevas).
2. **El acondicionamiento pesó más que el algoritmo.** En huellas, el filtro de Gabor orientado pasó de cientos de motas a 0, y corregir la alineación subió la exactitud en traslaciones del 93 al 100 %; en voz, normalizar el volumen y usar la secuencia completa con DTW pasó de 50 % (media) a 98 %.
3. **Elegir los rasgos con datos, no por intuición:** el histograma de orientación ganó a FingerCode, HOG y minucias; el DTW ganó a todo vector fijo.
4. **El umbral de rechazo es la parte más delicada.** Con intrusos simulados da R = 0,31 (~14 % de error en cada sentido); con huellas reales de Anguli es mejor (7 % y 6 %), pero con las huellas subidas a mano el falso rechazo fue alto (7 de 15). Los datos reales combinados sugieren R ≈ 0,38.
5. **Probar separando por persona** (voz) y **por impresión** (huellas) evitó resultados inflados.

---

## 6. Limitaciones y trabajo pendiente

| Limitación | Detalle | Qué hacer |
|---|---|---|
| Huellas muy recortadas | `matada`: 50 % de acierto; 3 de 15 `trasladada` rechazadas | Más impresiones de ese tipo en el entrenamiento o un registro contra plantillas |
| Umbral | Pocos intrusos reales (24); el falso rechazo con huellas subidas a mano fue 47 % con R = 0,31 | Reunir más huellas con feedback y aplicar `python src/kmeans_huellas.py --radio <R>` (el cruce real da ≈ 0,38) |
| Voz sin clase `otros` | Cualquier palabra se interpreta como el comando más parecido | Grabar `otros/` (otras palabras, silencio, ruido) y volver a entrenar |
| Pocos datos de voz | 8 sesiones; 22 audios de prueba (cada error vale 4,5 puntos); la sesión `ruido` solo tiene "identificar" | Grabar más personas y más repeticiones |
| Dataset sintético | Anguli genera huellas con distorsiones controladas; no hay piel seca, sensores distintos ni dedos reales | Informar la limitación al presentar |
| Voz en vivo | Probada con audios y con el micrófono sin hablar; falta la prueba hablada completa en la interfaz | Probar con el micrófono USB |
| Dependencia de Anguli | `Anguli.exe` está fuera del repositorio (ruta en `interfaz/config.py`) | Ajustar la ruta en la computadora de la presentación |

---

## 7. Cómo reproducir

```
# Huellas
python src/preprocesar.py            # huellasDactilares/ -> huellasProcesadas/
python src/caracterizar_huellas.py   # -> caracteristicas_huellas/ (vectores de 72 números)
python src/kmeans_huellas.py         # entrena K-means y calibra el radio
python src/graficar_pca.py --datos huellas

# Voz
python src/grabar_audios.py          # graba el dataset en audios/
python src/normalizar_audios.py      # -> audios_normalizados/ (solo lo nuevo)
python src/caracterizar_audios.py    # -> caracteristicas/ (MFCC y vectores de 91 números)
python src/knn_audios.py             # entrena KNN con DTW y evalúa
python src/graficar_pca.py --datos audios

# Pruebas
python src/probar_voz.py             # grabar un comando y ver qué entendió
python src/probar_huellas.py         # huellas nuevas en huellas_nuevas/ con feedback
python interfaz/app.py               # aplicación completa
```

Los modelos entrenados quedan en `caracteristicas_huellas/modelo_kmeans.npz` y `caracteristicas/modelo_knn.npz`.
