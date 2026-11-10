# Agrupamiento de huellas dactilares con k-means: rasgos, preprocesamiento y evaluación (informe didáctico para el Trabajo Final 2026)

La literatura no usa una única "receta" para agrupar huellas con k-means. Lo que hace es combinar tres familias de rasgos: (a) rasgos globales de nivel 1 sobre escala de grises (campo de orientación, frecuencia o distancia entre crestas, puntos singulares, FingerCode con Gabor); (b) rasgos locales de nivel 2 que se obtienen después de binarizar y adelgazar (minucias); y (c) descriptores genéricos de textura o forma (HOG, LBP, momentos de Hu). Cada familia responde de manera distinta a la traslación, la rotación, el ruido, la presión y el corte parcial. La decisión de diseño más importante no es el algoritmo de agrupamiento, sino cómo convertir cada huella en un vector de longitud fija y alineado. k-means trabaja con distancias euclídeas y no resuelve por sí mismo la alineación.

## TL;DR

- **Rasgos:** los rasgos clásicos de clasificación son el campo de orientación, los puntos singulares (índice de Poincaré) y la respuesta a filtros de Gabor (FingerCode, 80 sectores × 8 filtros = 640 valores). Todos se calculan sobre escala de grises, antes de binarizar. Las minucias (terminaciones y bifurcaciones), el esqueleto y la limpieza de minucias espurias se calculan después de binarizar y adelgazar.
- **k-means con huellas:** k-means necesita vectores de longitud fija y comparables entre sí. Por eso los trabajos que lo usan (por ejemplo, Wang et al. 2002 con campos direccionales y Liu, Jiang y Kot 2007 con orientación multiescala en teselado circular) construyen el vector alrededor de un punto de referencia (núcleo), o bien usan rasgos invariantes. Los conjuntos de minucias tienen tamaño variable y no se le pueden pasar directamente.
- **Evaluación:** con etiquetas verdaderas disponibles (3 identidades × 20), las métricas externas habituales son pureza, ARI, NMI/AMI y exactitud con asignación húngara, y conviene acompañarlas de la matriz de confusión. La silueta es interna y no usa etiquetas. Con k = 3 solo hay 3! = 6 asignaciones posibles entre clústeres e identidades.

## Hallazgos clave

1. **Clasificar no es lo mismo que identificar.** En la literatura, "clasificación de huellas" casi siempre significa asignar la huella a una clase de Henry (arco, arco en tienda, lazo izquierdo, lazo derecho, espiral) para reducir búsquedas en bases grandes.\[1\] En tu caso, agrupar impresiones por *identidad* se parece más a la indexación o al "matching" no supervisado. Si dos de tus tres huellas base tienen la misma clase de Henry (los lazos son la mayoría de las huellas), los rasgos que solo capturan la clase no alcanzarán para separarlas.
2. **El pipeline estándar** (Hong, Wan y Jain, 1998; Handbook of Fingerprint Recognition) es: normalización → campo de orientación → frecuencia de crestas → máscara de región → filtrado de Gabor → binarización → adelgazamiento → extracción de minucias (número de cruces) → poda de minucias espurias.\[2\]
3. **Sensibilidad a tus transformaciones:** la traslación afecta a todo rasgo que dependa de coordenadas absolutas o de una malla fija de bloques. La rotación desplaza todas las orientaciones en la misma cantidad. El ruido sal y pimienta y el gaussiano generan minucias espurias después de binarizar. La presión cambia el grosor de las crestas y el contraste. El corte parcial puede eliminar núcleos, deltas y parte de las minucias.
4. **Trabajos con k-means u otro agrupamiento:** Hong et al. (1998) ya usan agrupamiento por error cuadrático (6 clústeres) dentro del propio preprocesamiento.\[2\] Wang, Zhang y Wang (2002) combinan k-means y 3-NN sobre rasgos de campo direccional para 4 clases.\[3\] Liu, Jiang y Kot (2007) usan un k-means modificado para particionar bases de datos.\[4\] Hou et al. (2021, Universidad de Dali) usan un autoencoder más k-means, aglomerativo y BIRCH, y según su resumen "the CCAE achieves an accuracy of 97.3% on only 1000 unlabeled fingerprints in the NIST-DB4" (1000 imágenes elegidas al azar entre las 4000 de NIST-DB4, con 6,3 % de rechazo).

## Detalles

### 1. Conceptos previos: niveles de rasgos y por qué importa la alineación

- **Nivel 1 (global):** flujo de crestas, campo de orientación, frecuencia, puntos singulares y clase de Henry. La revisión de Galar et al. (2015, *Knowledge-Based Systems* 81:76–97) señala que la clasificación clásica se apoya en el mapa de orientación, los puntos singulares y los filtros de Gabor.\[5\]
- **Nivel 2 (local):** minucias. El modelo de coordenadas del FBI solo considera terminaciones y bifurcaciones, y describe cada minucia por su tipo, sus coordenadas x e y, y el ángulo θ de la tangente a la cresta respecto del eje horizontal (Maltoni, *A Tutorial on Fingerprint Recognition*, extraído del Handbook).\[6\] Hong, Wan y Jain (IEEE TPAMI 20(8):777, 1998) lo dicen textualmente: "A good quality fingerprint typically contains about 40–100 minutiae".
- **Nivel 3:** poros y contornos de cresta. Requieren alta resolución y en general no se usan en trabajos con k-means.
- **Escala física de referencia:** a 500 dpi (la resolución recomendada por el FBI), un desplazamiento del dedo de solo 2 mm equivale a unos 40 píxeles (Maltoni).\[2\]\[6\] Por eso la traslación no es un detalle menor para rasgos ligados a posiciones.

### 2. Pipeline estándar de preprocesamiento (pregunta 2)

| Etapa | Entrada | Operación | Salida | Dominio |
|---|---|---|---|---|
| 1. Normalización | Imagen gris I | Llevar media y varianza a valores deseados M₀ y VAR₀ | Imagen G | Gris |
| 2. Segmentación / máscara | G | Separar el área de huella del fondo o de zonas irrecuperables | Máscara R | Gris |
| 3. Campo de orientación | G | Gradientes por bloques w×w | Matriz θ(i,j) | Gris |
| 4. Frecuencia de crestas | G y θ | Firma x perpendicular a las crestas | Matriz f(i,j) | Gris |
| 5. Puntos singulares | θ | Índice de Poincaré | Núcleos y deltas (posición, tipo) | Sobre θ |
| 6. Mejora (Gabor) | G, θ, f, R | Filtro de Gabor sintonizado en cada bloque | Imagen mejorada E | Gris |
| 7. Binarización | E | Umbral global, local o adaptativo | Imagen binaria B | Binario |
| 8. Adelgazamiento | B | Esqueletización (Zhang-Suen, Guo-Hall, etc.) | Esqueleto S (crestas de 1 px) | Binario |
| 9. Extracción de minucias | S | Número de cruces (crossing number) | Lista {(x, y, θ, tipo)} | Binario |
| 10. Poda de espurias | Lista y S | Reglas estructurales (distancias, bordes, puentes) | Lista depurada | Binario |

**Fórmulas de las etapas 1, 3, 4 y 6 (Hong, Wan y Jain, IEEE TPAMI 20(8):777–789, 1998):**

- *Normalización:* G(i,j) = M₀ + √(VAR₀·(I(i,j) − M)² / VAR) si I(i,j) > M, y G(i,j) = M₀ − √(VAR₀·(I(i,j) − M)² / VAR) en caso contrario. Es una operación píxel a píxel que no cambia la claridad de las crestas y solo reduce la variación de los niveles de gris.\[2\]
- *Orientación por mínimos cuadrados* (bloques de 16×16): Vx = Σ 2·∂x(u,v)·∂y(u,v); Vy = Σ (∂x(u,v)² − ∂y(u,v)²); θ = ½·atan(Vy / Vx). Para suavizar se usa el campo vectorial Φx = cos(2θ), Φy = sin(2θ), se aplica un filtro pasa-bajos (5×5 por defecto) y se vuelve a θ = ½·atan(Φ'y / Φ'x). El ángulo se duplica porque 90° y 270° son la misma orientación de cresta.\[2\] Maltoni advierte que calcular θ directamente como el arcotangente de ∇y/∇x tiene problemas de discontinuidad alrededor de 90° y es muy sensible al ruido.\[6\] Bazen y Gerez (IEEE TPAMI 24(7):905–919, 2002) formalizan el enfoque de gradientes al cuadrado e introducen la *coherencia* como medida de confiabilidad de cada bloque.\[7\]
- *Frecuencia de crestas:* en una ventana orientada de 32×16 se calcula la "firma x" X[k] (el promedio de niveles de gris a lo largo de la cresta). Si T es el número medio de píxeles entre dos picos consecutivos, f = 1/T. A 500 dpi, los valores válidos están en el rango [1/25, 1/3], es decir, períodos de entre 3 y 25 píxeles. Fuera de ese rango, el bloque se marca como inválido y se interpola desde sus vecinos.\[2\]
- *Gabor par (simétrico):* h(x,y; φ,f) = exp{−½·[x_φ²/δx² + y_φ²/δy²]}·cos(2π·f·x_φ), con x_φ = x·cos φ + y·sin φ e y_φ = −x·sin φ + y·cos φ. Hong et al. usan δx = δy = 4,0 y una máscara de 11×11. Hay un compromiso explícito: con valores mayores el filtro es más robusto al ruido, pero crea más crestas espurias.\[2\]
- *Máscara de región:* este dato es interesante para un trabajo sobre k-means. Hong et al. caracterizan cada bloque con tres rasgos (amplitud, frecuencia y varianza de la firma x), agrupan 2000 patrones con un algoritmo de agrupamiento por error cuadrático en **6 clústeres** (4 recuperables y 2 irrecuperables) y usan los prototipos en un clasificador 1-NN. Si menos del 40 % de la imagen es recuperable, la imagen se rechaza.\[2\]

**Binarización y adelgazamiento.** Según el tutorial de Maltoni, la mayoría de los métodos de extracción de minucias convierten la imagen a binaria y luego la adelgazan hasta crestas de 1 píxel. También existen métodos que trabajan directamente en escala de grises (seguimiento de crestas de Maio y Maltoni), motivados porque (i) se pierde información al binarizar, (ii) binarizar y adelgazar es costoso y el adelgazamiento puede introducir muchas minucias espurias, y (iii) sin una mejora previa, la binarización da malos resultados en imágenes de baja calidad.\[6\] En la comparación de Khanyile, Tapamo y Dube (ISSA 2011), Zhang-Suen fue el algoritmo de adelgazamiento más rápido y Guo-Hall el que produjo la mejor calidad de esqueleto.\[8\]

**Número de cruces (Rutovitz):** CN(p) = ½·Σ_{i=1..8} |P_i − P_{i+1}|, con P₉ = P₁ y recorriendo los 8 vecinos del píxel p del esqueleto. CN = 1 indica una terminación, CN = 2 un píxel intermedio de cresta y CN ≥ 3 una bifurcación (Maltoni).\[6\]\[9\]

**Minucias espurias.** Zhao y Tang (*Pattern Recognition* 40:1270–1281, 2007) enumeran los tipos típicos: ruptura (break), espolón (spur), puente (bridge), fusión (merge), triángulo, escalera (ladder), lago, isla y arruga.\[10\] La poda puede ser estructural (sobre el esqueleto) o hacerse en el dominio de grises.\[6\]

### 3. Catálogo de rasgos: qué miden, cómo se calculan y su sensibilidad (pregunta 1)

**3.1 Campo de orientación (malla de bloques).**
*Qué mide:* el ángulo local θ ∈ [0°, 180°) de las crestas en cada bloque.
*Cálculo:* ver las fórmulas de la sección 2. Para usarlo como vector se suele codificar cada bloque como (cos 2θ, sin 2θ), lo que evita el salto entre 179° y 0°. Un ejemplo es una malla de 16×16 bloques, que da 512 valores.
*Antes o después de binarizar:* antes.
*Sensibilidad:* traslación: alta si la malla está fija a la imagen y baja si se centra en el núcleo. Rotación: rotar la imagen α suma α a todas las θ, y además mueve los bloques de lugar. Ruido gaussiano y sal y pimienta: moderada, y el suavizado vectorial la reduce. Presión: baja, porque la dirección del flujo casi no cambia. Corte parcial: los bloques fuera de la huella quedan vacíos o inválidos.

**3.2 Histograma de orientaciones.**
*Qué mide:* la distribución global (o por regiones) de θ, ponderada por la magnitud del gradiente o por la coherencia.
*Cálculo:* h[b] = Σ peso(i,j) para los bloques con θ(i,j) dentro del bin b, normalizado para que Σ h = 1.
*Sensibilidad:* el histograma global es prácticamente invariante a la traslación. Una rotación produce un desplazamiento circular de los bins. Esto es una propiedad geométrica y no un resultado de una fuente concreta. Para lograr invarianza a la rotación existen alternativas como alinear el bin máximo o usar la magnitud de la DFT del histograma, pero pierden información. El corte parcial cambia las proporciones.

**3.3 Distancia, frecuencia y densidad de crestas.**
*Qué mide:* la frecuencia local f = 1/T (T es la distancia entre crestas en píxeles). Maltoni la define como la inversa del número de crestas por unidad de longitud a lo largo de un segmento ortogonal a θ. Varía entre dedos y también entre regiones de un mismo dedo.\[6\]
*Cálculo:* con la firma x de Hong et al., en el dominio de Fourier (Kovacs-Vajna et al., bloques de 64×64) o con el modelo sinusoidal de Maio y Maltoni.\[6\] En la imagen binaria también se puede aproximar como "ancho de cresta + ancho de valle". Esto es lo que algunos trabajos llaman *inter-ridge width*, y es sensible al grosor de las crestas.
*Rasgo agregado:* media, desvío o histograma de f, o la "distancia media entre crestas". Liu, Jiang y Kot (2007) la usan como rasgo auxiliar de búsqueda.\[4\]
*Sensibilidad:* traslación y rotación: la media global es casi invariante. Ruido: moderada. Presión: puede modificar el ancho relativo de cresta y valle y, cuando se calcula sobre binario, la distancia medida. Corte parcial: cambia el promedio si las regiones perdidas tenían otra densidad. Escala: muy alta, porque cualquier remuestreo cambia f.

**3.4 Puntos singulares (núcleos y deltas) e índice de Poincaré.**
*Qué mide:* las regiones donde el flujo cambia de forma brusca. Los tipos son lazo (∩), delta (Δ) y espiral (O). El núcleo se define como el centro del lazo situado más al norte.\[6\]
*Cálculo (Kawagoe y Tojo, según Maltoni):* se recorre una curva cerrada C (por ejemplo, los 8 vecinos d₀…d₇) sobre el campo de orientación y se suman algebraicamente las diferencias de orientación entre elementos consecutivos. Antes de sumar, cada diferencia se lleva al intervalo (−90°, 90°]: P(i,j) = Σ_k δ(d_k, d_{k+1}). El resultado vale 0° si no hay singularidad, +180° si es un lazo, −180° si es un delta y +360° si es una espiral.\[6\] Normalizado por 360°, eso da +½ para un núcleo y −½ para un delta.\[11\]
*Rasgos derivados:* número de núcleos y deltas, sus posiciones relativas (por ejemplo, la distancia núcleo–delta) y el ángulo entre ellos.
*Sensibilidad:* traslación y rotación: el *tipo* y el *número* son invariantes, y la posición absoluta no. Ruido: el método de Poincaré produce muchas detecciones falsas en imágenes de baja calidad.\[12\] Una solución propuesta en la literatura es suavizar el campo de forma iterativa hasta obtener un número válido de singularidades.\[13\] Corte parcial: es crítico, porque si el corte elimina el delta o el núcleo, cambian el conteo y la clase inferida. En los arcos no hay núcleo propiamente dicho, y entonces se usa el punto de máxima curvatura.\[6\]

**3.5 Clases de Henry (arco, arco en tienda, lazo izquierdo, lazo derecho, espiral).**
*Qué mide:* el patrón global. Se infiere sobre todo del número y la posición relativa de núcleos y deltas: los arcos no tienen singularidades, los lazos tienen un núcleo y un delta, y las espirales tienen dos deltas.\[14\]
*Cálculo:* con reglas sobre los puntos singulares (Karu y Jain, 1996), con particionado del campo direccional (Cappelli, Lumini, Maio y Maltoni, IEEE TPAMI 21(5):402–421, 1999), con FingerCode más un clasificador (Jain, Prabhakar y Hong, IEEE TPAMI 21(4):348–359, 1999) o con redes neuronales.\[15\]\[16\]
*Distribución:* según International Security Journal ("Types of Fingerprints"), en distintas poblaciones los lazos van de alrededor del 52 % al 65 %, las espirales del 27 % al 40 % y los arcos del 4 % al 15 % (una muestra de 196 adultos dio "52.71% loops, 27.38% whorls and 15.28% arches"). El reparto 60–65 % / 30–35 % / 5 % se entiende mejor como "a widely used forensic average". Con estas proporciones, la probabilidad de que dos de tus tres huellas base compartan clase no es despreciable.
*Sensibilidad:* hereda la de los puntos singulares. Es robusta a la presión y frágil ante cortes que eliminen deltas.
*Limitación para tu objetivo:* es una variable categórica gruesa. Sirve para clasificar por patrón, no para distinguir identidades de la misma clase. La "clasificación continua" de Lumini, Maio y Maltoni (*Pattern Recognition Letters* 18(10):1027–1034, 1997) surgió precisamente porque las clases exclusivas son pocas y están desbalanceadas.\[17\]\[18\]

**3.6 FingerCode (banco de Gabor).**
*Qué mide:* la energía de las crestas por orientación en cada sector de un teselado circular alrededor del punto de referencia (núcleo).
*Cálculo (Jain, Prabhakar, Hong y Pankanti, CVPR 1999; Maltoni):* (1) se localiza el núcleo; (2) se tesela la región de interés en sectores (80 en la versión de matching); (3) se normaliza cada sector; (4) se filtra con 8 filtros de Gabor (0°, 22,5°, …, 157,5°) cuya frecuencia corresponde a la distancia media entre crestas; (5) en cada sector S_i y para cada filtro θ se calcula la desviación absoluta media: V_iθ = (1/n_i)·Σ_{(x,y)∈S_i} |F_iθ(x,y) − P_iθ|, donde P_iθ es la media de las respuestas en el sector. El resultado es un vector fijo de 80 × 8 = 640 valores, y la comparación entre huellas se hace con distancia euclídea.\[6\]\[19\]\[20\] Esto lo hace compatible por naturaleza con k-means.
*Sensibilidad:* traslación: invariante si el núcleo se detecta bien, porque el teselado se centra en él. Rotación: el método supone la huella vertical. La rotación se compensa calculando el código para varias rotaciones o desplazando circularmente los sectores y los filtros, y se toma la distancia mínima (tesis de Chikkerur, SUNY Buffalo).\[21\] Ruido: moderada, porque la AAD por sector promedia. Presión: moderada (contraste). Corte parcial: los sectores sin huella quedan vacíos y, si el corte elimina el núcleo, falla todo el vector.

**3.7 Minucias (terminaciones y bifurcaciones).**
*Qué mide:* la posición (x, y), el ángulo θ y el tipo de cada discontinuidad de cresta.
*Cálculo:* número de cruces sobre el esqueleto, seguido de la poda de espurias.
*Rasgos derivables para un vector fijo:* conteo total, conteo por tipo, densidad (minucias por área útil), histogramas de ángulos o de distancias entre pares, histograma espacial en una malla centrada en el núcleo, estructuras locales invariantes y descriptores tipo cilindro.
*Comparación clásica entre dos conjuntos:* dos minucias se emparejan si la distancia espacial cumple sd = √((x′ − x)² + (y′ − y)²) ≤ r₀ y la diferencia angular cumple dd = min(|θ′ − θ|, 360° − |θ′ − θ|) ≤ θ₀. Antes hay que estimar la alineación (Δx, Δy, θ), por ejemplo con la transformada de Hough (Maltoni).\[6\]
*Estructuras locales invariantes:* Jiang y Yau usan, para cada minucia m_i y sus dos vecinas más cercanas, el vector v_i = [d_ij, d_ik, θ_ij, θ_ik, φ_ij, φ_ik, n_ij, n_ik, t_i, t_j, t_k] (distancias, diferencias de dirección, conteos de crestas y tipos). Es invariante a traslación y rotación globales.\[6\]
*MCC (Minutia Cylinder-Code):* Cappelli, Ferrara y Maltoni (IEEE TPAMI 32(12):2128–2141, 2010) asocian a cada minucia un "cilindro" 3D de longitud fija, construido con distancias y ángulos relativos, que es invariante a traslación y rotación.\[22\]\[23\]\[24\] El *número* de cilindros sigue siendo variable por huella.
*Sensibilidad:* traslación y rotación: las coordenadas absolutas son muy sensibles y los rasgos relativos o los conteos lo son mucho menos. Ruido: muy sensible, porque el ruido sal y pimienta y el gaussiano crean islas, lagos y rupturas que el número de cruces detecta como minucias falsas. Presión: alta, porque mucha presión une crestas (puentes) y poca presión las corta (rupturas). Corte parcial: el conteo baja más o menos en proporción al área perdida.

**3.8 Esqueleto (adelgazamiento) como fuente de rasgos.**
*Qué mide:* la topología de las crestas con 1 píxel de ancho.
*Rasgos posibles:* longitud total del esqueleto por unidad de área (una forma de densidad), número de componentes conexas y, sobre todo, ser la entrada de las minucias.
*Sensibilidad:* hereda todos los problemas de la binarización. El algoritmo de adelgazamiento elegido cambia la cantidad de ramas espurias.\[8\]

**3.9 Momentos de Hu.**
*Qué mide:* la forma global de una distribución de intensidad o de una silueta binaria.
*Cálculo (Hu, IRE Trans. Information Theory 8:179–187, 1962):* momentos centrales μ_pq = Σ Σ (x − x̄)^p·(y − ȳ)^q·I(x,y); normalizados η_pq = μ_pq / μ₀₀^(1+(p+q)/2); invariantes φ₁ = η₂₀ + η₀₂, φ₂ = (η₂₀ − η₀₂)² + 4·η₁₁², …, hasta φ₇. Los siete son invariantes a traslación, escala y rotación (φ₇ cambia de signo con una reflexión).\[25\]
*Sensibilidad:* traslación, rotación y escala: invariantes por construcción, suponiendo una imagen continua y sin ruido.\[26\] Ruido: los momentos de orden alto son sensibles. Corte parcial: muy sensible, porque cambia la silueta. Observación: describen sobre todo la *forma del área de la huella* y su distribución de tinta, no la individualidad de las crestas. Es una limitación conceptual a tener en cuenta.

**3.10 HOG (histograma de gradientes orientados).**
*Qué mide:* histogramas locales de orientación del gradiente por celdas, normalizados por bloques.
*Cálculo:* magnitud m = √(Gx² + Gy²) y ángulo α = atan2(Gy, Gx); cada celda acumula m en bins de α; luego se normaliza por bloque (por ejemplo, v/√(‖v‖² + ε²)).
*Sensibilidad:* es parecido a una malla de orientación. Es sensible a la rotación de la imagen (la literatura de detección de vida lo señala de forma explícita)\[27\] y a la traslación si las celdas son fijas. La normalización por bloques lo hace robusto al contraste, y por lo tanto en parte a la presión.

**3.11 LBP (patrones binarios locales).**
*Qué mide:* la microtextura. Cada píxel se codifica comparándolo con P vecinos en un radio R.
*Cálculo:* LBP_{P,R} = Σ_{p=0}^{P−1} s(g_p − g_c)·2^p, con s(z) = 1 si z ≥ 0 y 0 en otro caso. La variante "uniforme invariante a rotación" (riu2) de Ojala et al. reduce los códigos a P + 2 valores, y el rasgo final es el histograma por subventanas.
*Sensibilidad:* invariante a cambios monótonos de iluminación.\[28\] La versión riu2 es invariante a rotaciones locales y el histograma global lo es a la traslación. Es sensible al ruido de alta frecuencia, sobre todo al sal y pimienta, porque cambia los signos de las comparaciones. En huellas se usa mucho en detección de vida (Nikam y Agarwal, 2008: un vector riu2 de 54 dimensiones) y en matchers híbridos con Gabor.\[29\]\[30\]

**Tabla resumen de sensibilidad (cualitativa; síntesis de las fuentes citadas y de la geometría de cada rasgo)**

| Rasgo | Dominio | Traslación | Rotación | Ruido gauss./S&P | Presión | Corte parcial |
|---|---|---|---|---|---|---|
| Malla de orientación fija | Gris | Alta | Alta | Media | Baja | Alta |
| Malla centrada en núcleo | Gris | Baja* | Alta | Media | Baja | Alta si falta el núcleo |
| Histograma global de θ | Gris | Muy baja | Desplazamiento circular | Media | Baja | Media |
| Frecuencia media de crestas | Gris | Muy baja | Muy baja | Media | Media | Baja-media |
| Nº/tipo de singularidades | Sobre θ | Invariante | Invariante | Alta (falsos) | Baja | Muy alta |
| Clase de Henry | Sobre θ | Invariante | Baja | Media | Baja | Alta |
| FingerCode | Gris | Baja* | Alta (requiere compensación) | Media | Media | Alta |
| Minucias (x, y) absolutas | Binario | Muy alta | Muy alta | Muy alta | Alta | Alta |
| Conteos y densidad de minucias | Binario | Invariante | Invariante | Muy alta | Alta | Media-alta |
| Estructuras locales / MCC | Binario | Invariante | Invariante | Alta | Alta | Media |
| Momentos de Hu | Gris o binario | Invariante | Invariante | Media | Media | Muy alta |
| HOG (celdas fijas) | Gris | Alta | Alta | Media | Baja | Alta |
| LBP riu2 (histograma) | Gris | Baja | Baja | Alta (S&P) | Baja-media | Media |

\*Depende de que el núcleo se detecte de forma estable en todas las impresiones.

### 4. ¿Qué se calcula antes y qué después de binarizar? (resumen de la pregunta 2)

- **Antes (escala de grises):** normalización, segmentación, campo de orientación, coherencia, frecuencia o distancia entre crestas (método de firma x o de Fourier), puntos singulares y clase de Henry (se calculan sobre θ), FingerCode y Gabor, HOG, LBP, histogramas de orientación y momentos de Hu sobre intensidades.
- **Después (binario o esqueleto):** esqueleto, minucias por número de cruces, conteos, densidades y ángulos de minucias, estructuras locales y MCC (dependen de las minucias), poda de espurias, ancho de cresta medido en la imagen binaria, y momentos de Hu sobre la silueta binaria.\[31\]
- **Casos ambiguos:** algunas mejoras producen directamente una salida binaria, y en esos casos la frontera entre mejora y binarización se diluye (Maltoni). Además, hay extractores de minucias que trabajan sobre grises sin binarizar (Maio y Maltoni).\[6\]

### 5. k-means aplicado a huellas (pregunta 3)

**5.1 El algoritmo.** k-means minimiza J = Σ_{k=1..K} Σ_{x∈C_k} ‖x − μ_k‖², alternando dos pasos: asignar cada punto al centroide más cercano y recalcular cada centroide como la media de su clúster.\[32\] Converge a un óptimo local, por lo que la inicialización importa.\[16\] k-means++ (Arthur y Vassilvitskii, SODA 2007, pp. 1027–1035) elige el primer centro al azar y cada centro siguiente con probabilidad D(x)² / Σ D(x)², donde D(x) es la distancia al centro más cercano ya elegido.\[33\] Con esa siembra, el resultado esperado es O(log k)-competitivo respecto del óptimo.\[34\] La otra alternativa habitual es repetir con varias semillas aleatorias y quedarse con la de menor J.

**5.2 Qué vectores se le pasan (alternativas encontradas en la literatura):**
- *Campo direccional y rasgos alrededor del núcleo:* Wang, Zhang y Wang, "Fingerprint classification by directional fields" (Proc. 4th IEEE ICMI 2002, pp. 395–399). Calculan campos direccionales, detectan núcleos, extraen rasgos definidos por ellos y usan "k-means classifier and 3-nearest neighbor" para distinguir arco, lazo izquierdo, lazo derecho y espiral.\[3\]\[35\] El texto menciona que el k-means fue "trained by 500" ejemplos.\[36\] La Tabla 1 de Hou et al. (arXiv 2109.05526), que lo incluye entre los "existing supervised learning methods", lo lista como "Directional field + kmeans | NIST 14 dataset (1000) | 4 | 89.5", es decir, 89,5 % con 4 clases sobre 1000 huellas de NIST-14. No pude verificar ese número en el texto original de Wang et al.
- *Orientación multiescala en teselado circular y distancia media entre crestas:* Liu, Jiang y Kot, "Efficient fingerprint search based on database clustering" (*Pattern Recognition* 40(6):1793–1803, 2007, DOI 10.1016/j.patcog.2006.11.007). Particionan la huella de forma no uniforme con un teselado circular, usan la orientación multiescala como rasgo principal y la distancia media entre crestas como rasgo auxiliar, y proponen "a modified K-means clustering technique … to partition the orientation feature space into clusters". La consulta se compara solo con los prototipos de los clústeres.\[37\]\[38\]\[39\] El trabajo previo de Jiang, Liu y Kot (IEEE TIFS 1(4):532–542, 2006) combina orientación, distancia dominante entre crestas y una medida de distancia propia para orientaciones.\[39\]\[40\]
- *Rasgos aprendidos y ensamble de agrupamientos:* Hou et al., "An Unsupervised Deep-Learning Method for Fingerprint Classification: the CCAE Network and the Hybrid Clustering Strategy" (arXiv 2109.05526). Recortan manualmente la zona del núcleo (de 512×512 a 200×200, luego reescalada a 256×256), aplican eliminación de ruido gaussiano, ecualización y binarización adaptativa, extraen un vector latente con un autoencoder convolucional y lo agrupan en 4 clases con k-means, aglomerativo y BIRCH. Rechazan las muestras en las que los métodos no coinciden. Informan un 96,4 % (CCAE_a, 5,1 % de rechazo) y un 97,3 % (CCAE_b, 6,3 % de rechazo) sobre 1000 imágenes de NIST-DB4.\[41\]
- *Agrupamiento dentro del preprocesamiento:* Hong et al. (1998), con 6 clústeres sobre (amplitud, frecuencia, varianza) por bloque, como ya se describió.\[2\]
- *Otros usos de k-means "con huellas" que no son agrupamiento por identidad:* clasificación de calidad (k-means jerárquico en clases buena, húmeda, seca y normal; *Neurocomputing*, 2012) y reducción de puntos en fusión cara-huella (arXiv 1002.2523).\[42\]\[43\] También aparecen muchos resultados sobre "fingerprinting" Wi-Fi, que es otro dominio.\[32\]\[44\]

**5.3 Normalización y estandarización del vector.** Como k-means usa distancia euclídea, los rasgos de escala grande dominan. Las alternativas habituales son:
- Estandarización z: z = (x − μ)/σ por columna. Es la opción por defecto en muchos trabajos.
- Escalado min–max: x′ = (x − min)/(max − min).
- Normalización por vector (L2): x′ = x/‖x‖. Es frecuente con histogramas.
- Ponderación por bloques: cuando se concatenan familias de distinto tamaño (por ejemplo, 640 valores de FingerCode y 3 conteos de singularidades), la familia más larga domina salvo que se reescale cada bloque, por ejemplo dividiendo por √(dimensión del bloque).
- Reducción de dimensión: PCA antes de k-means. Con 60 muestras y cientos de dimensiones, la dimensionalidad es alta respecto del tamaño de la muestra, y PCA o la selección de rasgos son alternativas comunes.
- Variables angulares: θ no puede promediarse como un escalar, porque la media de 1° y 179° no es 90°. Por eso se representan como (cos 2θ, sin 2θ).

**5.4 Problemas típicos.**
1. *Alineación:* una malla de bloques fija a la imagen hace que dos impresiones trasladadas del mismo dedo produzcan vectores muy distintos. Las alternativas encontradas son: centrar en el núcleo (FingerCode, Liu et al.), alinear por el centroide de la máscara de segmentación, registrar contra una plantilla de referencia o usar rasgos invariantes (histogramas, conteos, Hu, estructuras locales).
2. *Rotación:* el FingerCode supone una huella vertical y compensa rotando sectores.\[21\] Para las mallas de orientación existen dos alternativas: estimar un ángulo dominante y restarlo, o usar rasgos invariantes. Los histogramas de θ rotan de forma circular.
3. *Longitud variable de las minucias:* cada impresión tiene una cantidad distinta de minucias y, además, sin correspondencia de orden, así que la lista no puede ser un vector de k-means. Hay tres alternativas: (a) resumirlas en estadísticos o histogramas de longitud fija; (b) codificar las estructuras locales (por ejemplo, MCC) y luego agregarlas, al estilo de una bolsa de palabras; (c) dejar k-means y usar una matriz de similitud entre pares obtenida con un matcher, combinada con agrupamiento espectral, aglomerativo o k-medoides (estas técnicas aceptan distancias arbitrarias).
4. *Corte parcial:* los bloques o sectores sin huella necesitan un tratamiento explícito. Las alternativas son marcarlos con 0 y una máscara, imputarlos o excluirlos de la distancia. Esto último ya no es k-means estándar.
5. *Elección de k:* en tu caso k = 3 es conocido de antemano, así que no hace falta estimarlo. Cuando no se conoce, se usan el método del codo sobre J, la silueta media o el estadístico gap.
6. *Inicialización:* se usan k-means++ o múltiples reinicios. Con solo 60 puntos es barato repetir muchas veces.
7. *Supuesto de clústeres esféricos y de tamaño similar:* k-means favorece grupos convexos y de varianza parecida. Si una de tus transformaciones (por ejemplo, un corte fuerte) produce subgrupos alejados dentro de una misma identidad, k-means puede partir una identidad y fusionar otras dos.
8. *Factores de confusión del dataset sintético:* si cada identidad viene de una huella base distinta, rasgos triviales como la intensidad media, el tamaño del área o el contraste pueden separar las identidades sin capturar las crestas. Conviene saber esto al interpretar resultados, sea cual sea la opción elegida.

**5.5 Cómo se evalúa el agrupamiento.** Notación: N = 60 muestras, clústeres C_k e identidades verdaderas T_j, y n_kj = |C_k ∩ T_j| (la tabla de contingencia, que es la matriz de confusión antes de reordenar).
- **Pureza:** Pureza = (1/N)·Σ_k max_j n_kj. Va de 0 a 1. Aumenta artificialmente con más clústeres, pero con k fijo en 3 eso no es un problema.
- **Índice de Rand ajustado (ARI; Hubert y Arabie, 1985):** RI cuenta los pares de muestras en los que ambas particiones coinciden (juntos-juntos o separados-separados). ARI = (RI − E[RI]) / (max(RI) − E[RI]). Vale 1 en coincidencia perfecta y alrededor de 0 para un agrupamiento al azar, y puede ser negativo.
- **NMI:** NMI = I(U;V)/√(H(U)·H(V)). También existen variantes con la media aritmética o el máximo en el denominador, y conviene declarar cuál se usa. Vinh, Epps y Bailey (JMLR 11:2837–2854, 2010) muestran que las medidas basadas en información mutua tienen sesgo por azar y proponen la versión ajustada AMI = (I − E[I]) / (prom(H(U), H(V)) − E[I]).\[45\]\[46\]
- **Exactitud con asignación húngara:** Acc = max_π (1/N)·Σ_k n_{k,π(k)}, donde π recorre las biyecciones entre clústeres e identidades. El algoritmo húngaro (Kuhn, 1955) la resuelve en general, y con k = 3 basta probar las 3! = 6 permutaciones.
- **Matriz de confusión:** la tabla n_kj con las columnas reordenadas según π. Muestra *qué* identidades se confunden entre sí.
- **Silueta (Rousseeuw, 1987):** s(i) = (b(i) − a(i)) / max(a(i), b(i)), donde a(i) es la distancia media de i a su propio clúster y b(i) la distancia media al clúster vecino más cercano. Va de −1 a 1 y es una medida *interna* que no usa etiquetas. Mide si los grupos son compactos y están separados, no si coinciden con las identidades.
- **Estabilidad:** repetir con distintas semillas o submuestras y medir el ARI entre corridas es otra práctica habitual para estimar la variabilidad.

### 6. Paso a paso general, de la imagen al resultado de k-means (pregunta 4)

| Paso | Entrada | Qué hace | Salida | Alternativas existentes |
|---|---|---|---|---|
| 0. Generación o carga | Huella base por identidad | Aplicar traslación, rotación, ruido, presión y corte (sintético, al estilo de SFinGe: una "huella maestra" y varias impresiones) | 60 imágenes con etiqueta de identidad (que se guarda solo para evaluar) | Generador propio; SFinGe (Cappelli, 2004), usado en FVC2000–2006\[47\]\[48\] |
| 1. Normalización | Imagen gris | Media y varianza a M₀ y VAR₀ | Imagen normalizada | Fórmula de Hong; ecualización de histograma; CLAHE |
| 2. Eliminación de ruido | Imagen normalizada | Atenuar el ruido impulsivo y el gaussiano | Imagen filtrada | Filtro de mediana (S&P); gaussiano; omitir y confiar en Gabor |
| 3. Segmentación | Imagen | Máscara de primer plano | Máscara | Varianza por bloque; media, varianza y coherencia (Bazen y Gerez); clustering de bloques (Hong)\[49\] |
| 4. Orientación | Imagen y máscara | θ por bloques y coherencia | Matriz θ, matriz de coherencia | Gradientes al cuadrado; PCA de gradientes; Fourier\[7\] |
| 5. Frecuencia | Imagen y θ | f = 1/T por bloque | Matriz f | Firma x; Fourier; modelo sinusoidal |
| 6. Singularidades y referencia | θ | Poincaré → núcleo y deltas; punto de referencia | Coordenadas y tipos | Poincaré; Poincaré multiescala; máxima curvatura (arcos) |
| 7. Alineación | Imagen o rasgos y referencia | Trasladar o rotar a un marco común | Imagen o rasgos alineados | Núcleo; centroide de la máscara; ángulo dominante; ninguna (rasgos invariantes) |
| 8. Mejora | Imagen, θ, f | Gabor contextual | Imagen mejorada | Gabor (Hong); STFT; filtros direccionales |
| 9. Binarización* | Imagen mejorada | Umbral | Binaria | Otsu global; umbral local o adaptativo; salida binaria de la propia mejora |
| 10. Adelgazamiento* | Binaria | Esqueleto de 1 px | Esqueleto | Zhang-Suen; Guo-Hall; morfológico\[50\] |
| 11. Minucias* | Esqueleto | Número de cruces y poda | Lista {(x, y, θ, tipo)} | CN y reglas estructurales; extracción directa en grises\[31\] |
| 12. Vector fijo | Rasgos de los pasos 4 a 11 | Construir x ∈ ℝ^d | Matriz X (60 × d) | Malla θ (cos 2θ, sin 2θ); histograma θ; estadísticos de f; conteos de singularidades y minucias; FingerCode; HOG; LBP; Hu; concatenaciones |
| 13. Escalado y reducción | X | Estandarizar o reducir | X′ | z-score; min–max; L2; PCA; ponderación por bloques |
| 14. k-means | X′, k = 3 | Agrupar | Etiquetas de clúster y centroides | Inicialización k-means++ o aleatoria con n reinicios; alternativas: k-medoides, aglomerativo, espectral |
| 15. Evaluación | Etiquetas de clúster y verdaderas | Comparar | Pureza, ARI, NMI/AMI, Acc-húngara, matriz de confusión, silueta | Además: estabilidad entre semillas; visualización PCA 2D |

\*Solo es necesario si se usan rasgos de nivel 2 (minucias o esqueleto). Si solo se usan rasgos de nivel 1 o de textura, el pipeline puede terminar en el paso 8, o incluso en el 5.

## Advertencias y datos poco confiables

- **Cifra del 89,5 % de Wang et al. (2002):** solo la encontré de segunda mano, en la Tabla 1 de Hou et al. (2021), donde figura como "Directional field + kmeans | NIST 14 dataset (1000) | 4 | 89.5". No pude acceder al texto completo del paper para confirmar el vector exacto, el número de clústeres ni el protocolo. Lo mismo vale para los detalles internos del "k-means modificado" de Liu, Jiang y Kot (2007): solo verifiqué el resumen y fragmentos.
- **Resultados de Hou et al. (2021):** es un preprint de arXiv. El recorte del núcleo fue manual y la exactitud se informa excluyendo las muestras rechazadas (5,1–6,3 %),\[41\] así que no es comparable con métodos que no rechazan.
- **Frecuencias de clases de Henry:** el reparto 60–65 % de lazos, 30–35 % de espirales y ~5 % de arcos es, según International Security Journal, "a widely used forensic average". Los estudios poblacionales varían entre 52–65 % de lazos, 27–40 % de espirales y 4–15 % de arcos. La cifra de Karu y Jain está confirmada en su resumen original (*Pattern Recognition* 29(3):389–404, 1996): "For the NIST-4 database, classification accuracies of 85.4% for the five-class problem and 91.1% for the four-class problem (with arch and tented arch placed in the same category) were achieved". Además, el error con 4 clases baja a menos del 6 % si se rechaza el 10 % de las imágenes.
- **Fuentes de baja calidad:** un artículo de IIETA afirma que los "rasgos k-means" (centroides de intensidades de píxel) son estables ante la rotación.\[51\] No tiene evaluación rigurosa y conviene tratarlo con cautela. La afirmación de que HOG es sensible a la rotación aparece en texto de patentes\[27\] y es coherente con su definición, pero no proviene de un estudio controlado sobre huellas.
- **Validez externa:** un dataset sintético de 3 identidades × 20 impresiones derivadas de una sola base por identidad tiene una variabilidad intraclase artificial y controlada. Las conclusiones sobre qué rasgo "funciona" pueden no trasladarse a huellas reales, con distorsión elástica no lineal, piel seca o húmeda y distintos sensores. Esas condiciones son justamente las que Maltoni señala como el verdadero desafío.\[6\]
- **Tabla de sensibilidad:** es cualitativa y combina lo que dicen las fuentes con razonamiento geométrico sobre cada rasgo. No proviene de un experimento único que haya medido todos los rasgos frente a todas tus transformaciones.
- **Referencia de base:** la obra de consulta es el *Handbook of Fingerprint Recognition* (Maltoni, Maio, Jain y Feng, 3.ª ed., Springer, 2022, ISBN 978-3-030-83624-5).\[52\]\[53\] Las fórmulas y definiciones de este informe que se atribuyen a Maltoni provienen del tutorial de 2005 (LNCS 3161), que reproduce partes de la 1.ª edición (2003).\[6\]

## Fuentes

1. [Fingerprint Classification](https://link.springer.com/referenceworkentry/10.1007/978-1-4899-7488-4_56)
2. <https://www.cse.msu.edu/~rossarun/BiometricsTextBook/Papers/Fingerprint/HongFpImageEnhancement.pdf>
3. [Fingerprint classification by directional fields](https://ieeexplore.ieee.org/document/1167027/)
4. [Efficient fingerprint search based on database clustering - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0031320306004845)
5. [A Survey of Fingerprint Classification Part I: Taxonomies on Feature Extraction Methods and Learning Models](https://www.researchgate.net/publication/272297677_A_Survey_of_Fingerprint_Classification_Part_I_Taxonomies_on_Feature_Extraction_Methods_and_Learning_Models)
6. <https://cedar.buffalo.edu/~govind/CSE666/fall2007/FP_Tutorial.pdf>
7. [Sabih Gerez - Google Scholar](https://scholar.google.com/citations?user=b4ZyYDkAAAAJ&hl=en)
8. [A Comparative Study of Fingerprint Thinning Algorithms N.P. Khanyile](https://digifors.cs.up.ac.za/issa/2011/Proceedings/Research/Khanyile_Tapamo_Dube.pdf)
9. [(PDF) Preprocessing and postprocessing for skeleton-based fingerprint minutiae extraction](https://www.researchgate.net/publication/222823428_Preprocessing_and_postprocessing_for_skeleton-based_fingerprint_minutiae_extraction)
10. [Preprocessing and postprocessing for skeleton-based fingerprint minutiae extraction - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0031320306003797)
11. [Entropy-Based Clustering Algorithm for Fingerprint Singular Point Detection](https://doi.org/10.3390/e21080786)
12. [Fingerprint analysis and singular point detection - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0167865507001195)
13. [Enhanced Secure Algorithm for Fingerprint Recognition](https://arxiv.org/pdf/1402.4936)
14. [Types of Fingerprints: 8 Key Patterns Explained in Detail](https://internationalsecurityjournal.com/types-of-fingerprints/)
15. [(PDF) Fingerprint Classification by Directional Image Partitioning](https://www.academia.edu/10160269/Fingerprint_Classification_by_Directional_Image_Partitioning)
16. <https://arxiv.org/pdf/1211.4658>
17. [Fast and Accurate Fingerprint Indexing Based on Ridge Orientation and Frequency](https://www.researchgate.net/publication/224243391_Fast_and_Accurate_Fingerprint_Indexing_Based_on_Ridge_Orientation_and_Frequency)
18. [Continuous versus exclusive classification for fingerprint retrieval - Wikidata](https://www.wikidata.org/wiki/Q62609086)
19. [\[1909.09901\] Learning a Fixed-Length Fingerprint Representation](https://ar5iv.labs.arxiv.org/html/1909.09901)
20. [US20030169910A1 - Fingerprint matching using ridge feature maps - Google Patents](https://patents.google.com/patent/US20030169910A1/en)
21. [ONLINE FINGERPRINT VERIFICATION SYSTEM by Sharat S. Chikkerur](https://cedar.buffalo.edu/~govind/thesis_sc.pdf)
22. [A Generalized Approach for Cancellable Template and Its Realization for Minutia Cylinder-Code](https://arxiv.org/pdf/2203.01095)
23. [Minutia Cylinder-Code](https://biolab.csr.unibo.it/research.asp?organize=Activities&selObj=81&pathSubj=111%7C%7C8%7C%7C81)
24. [Minutia Cylinder-Code: a new representation and matching technique for fingerprint recognition - PubMed](https://pubmed.ncbi.nlm.nih.gov/20975113/)
25. [Methods and apparatus for data analysis](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8041541)
26. [System and method for extracting object information from digital images to evaluate for realism](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12488566)
27. [Liveness detection in fingerprint-based biometric systems](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11068693)
28. [Local Binary Pattern-based Fingerprint Matching](https://digforasp.uca.es/wp-content/uploads/2020/11/S1_2_ESCIM2020_oct7-1.pdf)
29. [(PDF) Local binary patterns for a hybrid fingerprint matcher](https://www.researchgate.net/publication/222853895_Local_binary_patterns_for_a_hybrid_fingerprint_matcher)
30. [Convolution Comparison Pattern: An Efficient Local Image Descriptor for Fingerprint Liveness Detection - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4742063/)
31. [Pattern Recognition 40 (2007) 1270–1281 www.elsevier.com/locate/pr](https://mmlab.ie.cuhk.edu.hk/archive/2007/PR07_fingerprint.pdf)
32. [K-Means Fingerprint Clustering for Low-Complexity Floor Estimation in Indoor Mobile Localization](https://arxiv.org/pdf/1509.01600)
33. [Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds](https://arxiv.org/pdf/1906.03671)
34. [k-means++: The Advantages of Careful Seeding](http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf)
35. [Review of Parameters of Fingerprint Classification Methods Based on Algorithmic Flow](https://link.springer.com/chapter/10.1007/978-3-642-22555-0_4)
36. [(PDF) Fingerprint classification by directional fields](https://www.researchgate.net/publication/3998208_Fingerprint_classification_by_directional_fields)
37. [Liu, Manhua, Jiang, Xudong, Chichung Kot, Alex (2007) Efficient fingerprint search based on database clustering. Pattern Recognition, 40 (6). 1793-1803 doi:10.1016/j.patcog.2006.11.007](https://www.mindat.org/reference.php?id=9363522)
38. [Efficient Fingerprint Search Based On Database Clustering: Manhua Liu, Xudong Jiang, Alex Chichung Kot](https://www.scribd.com/document/292019498/JiangX-D-PR-07-2P)
39. [Efficient fingerprint search based on database clustering](https://dl.acm.org/doi/10.1016/j.patcog.2006.11.007)
40. [Deep learning compact binary codes for fingerprint indexing](https://link.springer.com/article/10.1631/FITEE.1700420)
41. <https://arxiv.org/pdf/2109.05526>
42. [A hierarchical k-means clustering based fingerprint quality classification - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0925231212000331)
43. [Feature Level Fusion of Face and Fingerprint Biometrics](https://arxiv.org/pdf/1002.2523)
44. [From Global to Local: Cluster-Aware Learning for Wi-Fi Fingerprinting Indoor Localisation](https://arxiv.org/pdf/2601.05650)
45. [Information Theoretic Measures for Clusterings Comparison](https://jmlr.csail.mit.edu/papers/volume11/vinh10a/vinh10a.pdf)
46. [Information Theoretic Measures for Clusterings Comparison: Variants, Properties, Normalization and Correction for Chance](https://dl.acm.org/doi/10.5555/1756006.1953024)
47. [\[PDF\] SFinGe : an Approach to Synthetic Fingerprint Generation](https://www.semanticscholar.org/paper/SFinGe-:-an-Approach-to-Synthetic-Fingerprint-Cappelli/45ece11d63ac7494642d93f581d13dce0a07a219)
48. [Biometric System Laboratory University of Bologna - ITALY Raffaele Cappelli](https://www.nist.gov/system/files/documents/2021/04/19/cappelli_using_synthetic_data_to_evaluate_the_quality_of_minutiae_extraction_algorithms.pdf)
49. [Filter Design and Performance Evaluation for Fingerprint Image Segmentation](https://arxiv.org/html/1501.02113v1)
50. [(PDF) A Comparative Study of Fingerprint Thinning algorithms.](https://www.researchgate.net/publication/220803321_A_Comparative_Study_of_Fingerprint_Thinning_algorithms)
51. [Analysis Methods Used to Extract Fingerprints Features](https://www.iieta.org/download/file/fid/57990)
52. [Handbook of Fingerprint Recognition](https://link.springer.com/book/10.1007/978-3-030-83624-5)
53. [Fingerprint Classification and Indexing](https://link.springer.com/chapter/10.1007/978-3-030-83624-5_5)
