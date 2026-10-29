"""K-means para las huellas dactilares (aprendizaje NO supervisado), implementado desde cero.

Que hace este script
--------------------
1. Carga los vectores de huellas (72 numeros por huella, ver caracterizar_huellas.py).
2. Separa ENTRENAMIENTO (impresiones 1 a 3 de cada efecto e identidad) y PRUEBA (impresiones 4 y 5).
3. Aplica K-means (K = 3, una por cada identidad autorizada) SOLO con el entrenamiento
   para encontrar los 3 CENTROIDES.
4. Asigna cada cluster a una identidad.
5. CALIBRA el RADIO de aceptacion (umbral de rechazo): simula intrusos dejando una identidad
   afuera cada vez y busca la distancia que equilibra los dos errores posibles
   (falso rechazo de una persona autorizada y falsa aceptacion de un intruso).
6. Mide que tan bien funciona en entrenamiento y en prueba (matriz de confusion y metricas de error),
   guarda el modelo y dibuja los graficos.

Definiciones rapidas (ver el documento "Teoria metodos y forma implementada en el codigo.md")
- Vector: la lista de 72 numeros que describe a una huella (un punto en un espacio de 72 dimensiones).
- Cluster: un grupo de huellas que quedan cerca unas de otras.
- Centroide: el "centro" de un cluster = el promedio de todos sus vectores.
- Distancia euclidea: la distancia "en linea recta" entre dos vectores.

Los PASOS del algoritmo estan numerados (PASO 1 a PASO 5) igual que en el documento.

Uso:
    python src/kmeans_huellas.py
    python src/kmeans_huellas.py --semilla 3 --reinicios 20
    python src/kmeans_huellas.py --radio 0.25                       # fijar el umbral a mano (sin calibrar)
    python src/kmeans_huellas.py --excluir-entrenamiento matada     # entrenar sin ese efecto (se sigue evaluando)
"""
import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # dibujar a archivo, sin ventana
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA      # SOLO para dibujar en 2D; K-means NO lo usa

import metricas

RAIZ = Path(__file__).resolve().parent.parent
CARPETA = RAIZ / "caracteristicas_huellas"
RUTA_MODELO = CARPETA / "modelo_kmeans.npz"

# ----------------------------- parametros del metodo -----------------------------
K = 3                      # numero de clusters = numero de identidades autorizadas (lo fija el enunciado)
REINICIOS = 10             # cuantas veces se repite K-means desde cero con centroides iniciales distintos
MAX_ITER = 300             # tope de seguridad de iteraciones por corrida
IMPRESIONES_ENTRENAMIENTO = (1, 2, 3)   # el resto (4 y 5) queda para prueba
GRILLA_RADIO = np.round(np.arange(0.10, 0.601, 0.01), 2)   # radios candidatos para la calibracion


# =====================================================================================
#  Datos
# =====================================================================================
def cargar_datos():
    """Lee dataset.npz y clases.json. Devuelve X, etiqueta (0/1/2), efecto, archivo, numero de impresion, nombres."""
    ruta = CARPETA / "dataset.npz"
    if not ruta.exists():
        sys.exit(f"No existe {ruta}. Ejecuta antes: python src/caracterizar_huellas.py")
    d = np.load(ruta, allow_pickle=False)
    clases = json.loads((CARPETA / "clases.json").read_text(encoding="utf-8"))
    inv = {v: k for k, v in clases.items()}
    archivo = d["archivo"].astype(str)
    # el numero de impresion sale del nombre del archivo: ..._imp3.png -> 3
    imp = np.array([int(re.search(r"_imp(\d+)", a).group(1)) for a in archivo])
    return d["X"].astype(np.float64), d["y"].astype(int), d["efecto"].astype(str), archivo, imp, inv


def distancias_euclideas(A, B):
    """Matriz de distancias euclideas entre cada fila de A y cada fila de B.

    distancia(a, b) = raiz( (a1-b1)^2 + (a2-b2)^2 + ... + (a72-b72)^2 )
    """
    diferencia = A[:, None, :] - B[None, :, :]          # (n_A, n_B, dimensiones)
    return np.sqrt((diferencia ** 2).sum(axis=2))


# =====================================================================================
#  K-means desde cero
# =====================================================================================
def kmeans_una_vez(X, k, rng, max_iter=MAX_ITER):
    """Una corrida completa de K-means. Devuelve centroides, etiquetas, inercia, iteraciones e historial.

    El historial guarda, en cada vuelta, (centroides usados, asignacion resultante);
    sirve para dibujar el paso a paso (kmeans_iteraciones.png).
    """
    # PASO 1: elegir K huellas AL AZAR como centroides iniciales (sin repetir).
    C = X[rng.choice(len(X), size=k, replace=False)].copy()
    etiquetas = None
    historial = []
    for vuelta in range(1, max_iter + 1):
        # PASO 2: ASIGNAR cada huella al centroide mas cercano (distancia euclidea).
        D = distancias_euclideas(X, C)                  # (n huellas, k centroides)
        nuevas = D.argmin(axis=1)
        historial.append((C.copy(), nuevas.copy()))

        # PASO 4 (condicion de parada): si ninguna huella cambio de cluster respecto de la
        # vuelta anterior, los centroides ya no se moverian -> el algoritmo CONVERGIO.
        if etiquetas is not None and np.array_equal(nuevas, etiquetas):
            break
        etiquetas = nuevas

        # PASO 3: RECALCULAR cada centroide como el promedio de las huellas de su cluster.
        # Luego se vuelve al PASO 2 (las huellas se REASIGNAN al nuevo centroide mas cercano).
        for j in range(k):
            miembros = X[etiquetas == j]
            if len(miembros) > 0:
                C[j] = miembros.mean(axis=0)
            else:
                # cluster vacio: se lo reubica en la huella mas lejana a su centroide actual
                lejana = D[np.arange(len(X)), etiquetas].argmax()
                C[j] = X[lejana]
    # inercia = suma de las distancias^2 de cada huella a su centroide (cuanto mas chica, clusters mas compactos)
    inercia = float(((X - C[etiquetas]) ** 2).sum())
    return C, etiquetas, inercia, vuelta, historial


def kmeans(X, k=K, reinicios=REINICIOS, semilla=0):
    """PASO 5: repetir K-means varias veces y quedarse con la corrida de MENOR inercia.

    El resultado depende de los centroides iniciales (puede caer en un optimo local),
    por eso se hacen varios reinicios con azar distinto.
    """
    rng = np.random.default_rng(semilla)
    corridas = [kmeans_una_vez(X, k, rng) for _ in range(reinicios)]
    mejor = min(range(reinicios), key=lambda i: corridas[i][2])
    return corridas, mejor


# =====================================================================================
#  De cluster a identidad, umbral y clasificacion
# =====================================================================================
def asignar_cluster_a_identidad(etiquetas_cluster, y_real, k, n_identidades):
    """K-means no sabe que identidad es cada cluster; se lo decimos con las etiquetas del ENTRENAMIENTO.

    Se arma la tabla "cuantas huellas de cada identidad cayeron en cada cluster" y se
    elige la asignacion uno a uno que maximiza los aciertos (algoritmo hungaro).
    Devuelve un vector: mapeo[cluster] = etiqueta de identidad.
    """
    tabla = np.zeros((k, n_identidades), dtype=int)
    for c, i in zip(etiquetas_cluster, y_real):
        tabla[c, i] += 1
    filas, columnas = linear_sum_assignment(-tabla)      # el "-" porque busca minimizar
    mapeo = np.zeros(k, dtype=int)
    mapeo[filas] = columnas
    return mapeo


def calibrar_radio(X, y, imp, es_train, grilla, reinicios, semilla, efecto, excluidos):
    """Elige el RADIO de aceptacion (umbral de rechazo) simulando intrusos.

    No hay huellas de personas NO autorizadas en el dataset. Para estimar cuantos intrusos
    "se colarian" se hace una validacion dejando una identidad afuera ("leave-one-identity-out"):

      Para cada identidad i (las 3 veces):
        1. i hace de INTRUSO: se la saca del entrenamiento.
        2. Con las otras dos identidades se entrena K-means (K = 2) igual que el modelo real.
        3. Para cada radio R de la grilla se mide:
             FRR (falso rechazo)    = % de huellas de PRUEBA de las 2 identidades conocidas cuya
                                      distancia al centroide mas cercano supera R (se las rechaza sin querer);
             FAR (falsa aceptacion) = % de huellas del intruso i cuya distancia a su centroide mas
                                      cercano es <= R (el intruso entra).
      Se promedian las 3 vueltas y se elige el R donde FRR y FAR se IGUALAN (error de igual tasa, EER).

    Limitacion: los "intrusos" son las otras 2 personas del propio dataset; los intrusos reales
    pueden parecerse mas o menos a las personas autorizadas.
    Devuelve las curvas FRR y FAR (promedio) y el radio elegido.
    """
    n = len(grilla)
    frr, far, frr_sin = np.zeros((3, n)), np.zeros((3, n)), np.zeros((3, n))
    entrenable = es_train & ~np.isin(efecto, excluidos)
    for intruso in range(3):
        conocida = y != intruso
        corridas, mejor = kmeans(X[entrenable & conocida], 2, reinicios, semilla)
        Cc = corridas[mejor][0]
        d = distancias_euclideas(X, Cc).min(axis=1)      # distancia de cada huella a su centroide mas cercano
        prueba_conocida = ~es_train & conocida
        sin_excluidos = prueba_conocida & ~np.isin(efecto, excluidos)
        for i, R in enumerate(grilla):
            frr[intruso, i] = (d[prueba_conocida] > R).mean()
            frr_sin[intruso, i] = (d[sin_excluidos] > R).mean() if sin_excluidos.any() else np.nan
            far[intruso, i] = (d[~conocida] <= R).mean()
    frr, far, frr_sin = frr.mean(axis=0), far.mean(axis=0), np.nanmean(frr_sin, axis=0)
    elegido = int(np.argmin(np.abs(frr - far)))           # punto donde las dos curvas se cruzan (EER)
    return {"radios": grilla, "frr": frr, "far": far, "frr_sin_excluidos": frr_sin,
            "radio": float(grilla[elegido]), "eer_frr": float(frr[elegido]), "eer_far": float(far[elegido])}


def clasificar(X, C, mapeo, radios):
    """Para cada vector: (identidad del centroide mas cercano, distancia a ese centroide, fue_aceptada)."""
    D = distancias_euclideas(X, C)
    cercano = D.argmin(axis=1)
    distancia = D[np.arange(len(X)), cercano]
    aceptada = distancia <= radios[cercano]
    return mapeo[cercano], distancia, aceptada, cercano


def identificar_huella(vector, modelo=None):
    """Para la demo: vector de 72 numeros de UNA huella -> ("identidad_N" | "NO AUTORIZADO", distancia).

    El vector se obtiene con:  caracterizar_huella(preprocesar_huella(imagen))["orient"]
    """
    r = identificar_huella_detalle(vector, modelo)
    return r["resultado"], r["distancia"]


def identificar_huella_detalle(vector, modelo=None):
    """Igual que identificar_huella pero devuelve TODO el detalle de la decision (un diccionario):

      resultado          "identidad_N" si se acepta, "NO AUTORIZADO" si se rechaza
      identidad_cercana  identidad del centroide mas cercano (aunque se haya rechazado)
      distancia          distancia al centroide mas cercano
      radio              radio de aceptacion (umbral) que se uso
      aceptada           True si distancia <= radio
      distancias         {identidad: distancia a su centroide} para las 3 identidades
      margen             distancia al 2.o centroide menos distancia al 1.o (chico = caso dudoso entre dos personas)
    """
    if modelo is None:
        modelo = np.load(RUTA_MODELO, allow_pickle=False)
    nombres = [str(n) for n in modelo["nombres"]]
    d = np.linalg.norm(modelo["centroides"] - np.asarray(vector, dtype=np.float64), axis=1)
    j = int(d.argmin())
    radio = float(modelo["radios"][j])
    ident = nombres[int(modelo["mapeo"][j])]
    orden = np.sort(d)
    aceptada = bool(d[j] <= radio)
    return {"resultado": ident if aceptada else "NO AUTORIZADO", "identidad_cercana": ident,
            "distancia": float(d[j]), "radio": radio, "aceptada": aceptada,
            "distancias": {nombres[int(modelo["mapeo"][c])]: float(d[c]) for c in range(len(d))},
            "margen": float(orden[1] - orden[0]) if len(orden) > 1 else float("nan")}


# =====================================================================================
#  Graficos
# =====================================================================================
def colores_identidad(n):
    return [plt.cm.tab10(i) for i in range(n)]


def grafico_clusters(carpeta_g, X, y, es_train, C, mapeo, radio, etiquetas_cluster, nombres, pca):
    """Clusters, centroides y radio de umbral sobre el plano PCA 2D.

    El PCA es una proyeccion ortogonal: la "sombra" en el plano de una esfera de 72 dimensiones de radio R
    es exactamente un circulo de radio R. Una huella aceptada cae siempre DENTRO del circulo, pero una huella
    dentro del circulo puede estar fuera de la esfera real (esta lejos en las dimensiones que no se ven).
    """
    P = pca.transform(X)
    Pc = pca.transform(C)
    col = colores_identidad(len(nombres))
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    for j in range(len(C)):
        ident = mapeo[j]
        for entren, marca in ((True, "o"), (False, "^")):
            m = (etiquetas_cluster == j) & (es_train == entren)
            mal = m & (y != ident)                               # huellas asignadas a la identidad equivocada
            ax.scatter(P[m & ~mal, 0], P[m & ~mal, 1], s=50, marker=marca, color=col[ident],
                       edgecolor="black", linewidth=0.4, alpha=0.85)
            ax.scatter(P[mal, 0], P[mal, 1], s=70, marker=marca, color=col[ident],
                       edgecolor="red", linewidth=2.0, alpha=0.95)
        # circulo de umbral: sombra 2D de la esfera de aceptacion (radio R en 72 dimensiones)
        ax.add_patch(Circle(Pc[j], radio, fill=False, edgecolor=col[ident], linewidth=2, linestyle="--"))
        ax.scatter(*Pc[j], marker="X", s=330, color=col[ident], edgecolor="black", linewidth=1.8, zorder=5)
        ax.annotate(f"centroide {j}", Pc[j], xytext=(8, 10), textcoords="offset points", fontsize=8, weight="bold")
    from matplotlib.lines import Line2D
    ayuda = [Line2D([], [], marker="s", linestyle="", color=col[i], markeredgecolor="black", label=n)
             for i, n in enumerate(nombres)]
    ayuda += [Line2D([], [], marker="o", linestyle="", color="lightgray", markeredgecolor="black", label="entrenamiento"),
              Line2D([], [], marker="^", linestyle="", color="lightgray", markeredgecolor="black", label="prueba / sin entrenar"),
              Line2D([], [], marker="o", linestyle="", color="white", markeredgecolor="red", markeredgewidth=2, label="mal asignada"),
              Line2D([], [], marker="X", linestyle="", color="gray", markeredgecolor="black", markersize=12, label="centroide"),
              Line2D([], [], linestyle="--", color="gray", label=f"umbral R = {radio:.2f} (calibrado)")]
    ax.legend(handles=ayuda, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.set_title("K-means de huellas: clusters, centroides y umbral\n"
                 "(el radio se mide en 72 dimensiones; el circulo es su sombra en este plano 2D)", fontsize=10)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    fig.savefig(carpeta_g / "kmeans_clusters_2d.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def grafico_iteraciones(carpeta_g, X_train, historial, pca, max_paneles=6):
    """Paso a paso de K-means: centroides iniciales al azar -> reasignacion -> ... -> convergencia."""
    pasos = list(range(len(historial)))
    if len(pasos) > max_paneles:                                  # si hay muchas vueltas, se muestran repartidas
        pasos = sorted(set(np.linspace(0, len(historial) - 1, max_paneles).round().astype(int)))
    P = pca.transform(X_train)
    colores = [plt.cm.tab10(i) for i in range(K)]
    n = len(pasos)
    cols = min(3, n)
    filas = int(np.ceil(n / cols))
    fig, axes = plt.subplots(filas, cols, figsize=(5 * cols, 4.2 * filas), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, t in zip(axes.ravel(), pasos):
        ax.axis("on")
        C, lab = historial[t]
        Pc = pca.transform(C)
        for j in range(K):
            ax.scatter(P[lab == j, 0], P[lab == j, 1], s=35, color=colores[j], edgecolor="black", linewidth=0.3, alpha=0.85)
            ax.scatter(*Pc[j], marker="X", s=260, color=colores[j], edgecolor="black", linewidth=1.6, zorder=5)
        if t == 0:
            titulo = "PASO 1 y 2: centroides al azar\ny primera asignacion"
        elif t == len(historial) - 1:
            titulo = f"Vuelta {t + 1}: nadie cambia de cluster\n(CONVERGIO)"
        else:
            cambiaron = int((lab != historial[t - 1][1]).sum())
            titulo = f"Vuelta {t + 1}: PASO 3 (recalcular) y\nPASO 2 (reasignar) - cambiaron {cambiaron}"
        ax.set_title(titulo, fontsize=9)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.3)
    fig.suptitle("K-means paso a paso (solo huellas de entrenamiento; los colores son clusters, no identidades)", fontsize=11)
    fig.tight_layout()
    fig.savefig(carpeta_g / "kmeans_iteraciones.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# =====================================================================================
#  Programa principal
# =====================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--semilla", type=int, default=0, help="semilla del azar (resultados repetibles)")
    ap.add_argument("--reinicios", type=int, default=REINICIOS)
    ap.add_argument("--radio", type=float, default=None,
                    help="fijar el radio de aceptacion a mano (por defecto se CALIBRA con intrusos simulados)")
    ap.add_argument("--excluir-entrenamiento", default="",
                    help="efectos (separados por coma) que NO se usan para entrenar K-means; se siguen evaluando en prueba")
    a = ap.parse_args()

    X, y, efecto, archivo, imp, inv = cargar_datos()
    nombres = [inv[i] for i in sorted(inv) if i < K]               # identidad_1, identidad_2, identidad_3
    excluidos = [e.strip().lower() for e in a.excluir_entrenamiento.split(",") if e.strip()]
    desconocidos = [e for e in excluidos if e not in set(efecto)]
    if desconocidos:
        sys.exit(f"Efecto(s) desconocido(s): {desconocidos}. Disponibles: {sorted(set(efecto))}")
    es_impresion_train = np.isin(imp, IMPRESIONES_ENTRENAMIENTO)
    # es_train = huellas con las que REALMENTE se entrena (impresiones 1-3, menos los efectos excluidos)
    es_train = es_impresion_train & ~np.isin(efecto, excluidos)
    print(f"Huellas: {len(X)} | entrenamiento (impresiones {IMPRESIONES_ENTRENAMIENTO}): {es_train.sum()} | prueba: {(~es_impresion_train).sum()}"
          + (f" | efectos excluidos del entrenamiento: {', '.join(excluidos)}" if excluidos else ""))
    print(f"Vector de {X.shape[1]} numeros por huella | K = {K} | efectos: {', '.join(sorted(set(efecto)))}")

    # ---- PASOS 1 a 5: K-means SOLO con el entrenamiento ----
    corridas, mejor = kmeans(X[es_train], K, a.reinicios, a.semilla)
    print("\nK-means: reinicio | vueltas hasta converger | inercia")
    for i, (_, _, inercia, vueltas, _) in enumerate(corridas):
        print(f"   {i + 1:>2}  {'<- elegido' if i == mejor else '':<11}{vueltas:>8} {inercia:>20.5f}")
    C, lab_train, inercia, vueltas, historial = corridas[mejor]
    print(f"Elegida la corrida {mejor + 1}: converge en {vueltas} vueltas, inercia {inercia:.5f}")

    # ---- de cluster a identidad (con el entrenamiento) ----
    mapeo = asignar_cluster_a_identidad(lab_train, y[es_train], K, len(nombres))
    print("\nCluster -> identidad (decidido con el entrenamiento):")
    for j in range(K):
        print(f"   cluster {j} -> {nombres[mapeo[j]]:<12} huellas de entrenamiento: {(lab_train == j).sum():>2}")

    # ---- PASO 6: CALIBRAR el radio de aceptacion simulando intrusos ----
    cal = calibrar_radio(X, y, imp, es_impresion_train, GRILLA_RADIO, a.reinicios, a.semilla, efecto, excluidos)
    print("\nPASO 6 - calibracion del umbral (intruso simulado = una identidad dejada afuera, 3 vueltas):")
    print("   radio R | FRR autorizadas rechazadas | FAR intrusos aceptados")
    for i, R in enumerate(cal["radios"]):
        if int(round(R * 100)) % 4 == 2 and 0.18 <= R <= 0.50:
            print(f"    {R:.2f}   |        {cal['frr'][i] * 100:5.1f}%            |      {cal['far'][i] * 100:5.1f}%")
    print(f"   Cruce de las dos curvas (EER): R = {cal['radio']:.2f}  (FRR {cal['eer_frr'] * 100:.1f}% / FAR {cal['eer_far'] * 100:.1f}%)")
    radio = a.radio if a.radio is not None else cal["radio"]
    print(f"   Radio usado: {radio:.2f}" + ("  (fijado a mano con --radio)" if a.radio is not None else "  (calibrado)"))
    radios = np.full(K, radio)                      # el mismo radio para los 3 clusters

    # ---- evaluacion ----
    pred, dist, aceptada, cercano = clasificar(X, C, mapeo, radios)
    resultados = {}
    es_prueba = ~es_impresion_train
    for nombre, mascara in (("ENTRENAMIENTO", es_train), ("PRUEBA", es_prueba)):
        cm = metricas.matriz_confusion(y[mascara], pred[mascara], len(nombres))
        m = metricas.imprimir_reporte(f"{nombre}: identidad real vs identidad del cluster", cm, nombres)
        m["matriz_confusion"] = cm.tolist()
        m["falso_rechazo"] = float((~aceptada[mascara]).mean())     # conocidas que el umbral rechazaria
        resultados[nombre.lower()] = m
        print(f"  con el umbral: {(~aceptada[mascara]).sum()} de {mascara.sum()} huellas conocidas serian rechazadas "
              f"(falso rechazo = {m['falso_rechazo'] * 100:.1f}%)")
    print("\nExactitud por efecto (entrenamiento / prueba):")
    por_efecto = {}
    for e in sorted(set(efecto)):
        fila = []
        for mascara in (es_train, es_prueba):
            s = (efecto == e) & mascara
            fila.append(float((pred[s] == y[s]).mean()) if s.any() else float("nan"))
        por_efecto[e] = fila
        print(f"   {e:<12} {fila[0] * 100:>6.1f}%  /  {fila[1] * 100:>6.1f}%")
    print(f"\nNota: la FAR ({cal['eer_far'] * 100:.0f}% en el cruce) es una ESTIMACION con intrusos simulados (las otras personas del")
    print("dataset). No hay huellas de intrusos reales; con ellas habria que volver a calibrar el umbral.")

    # ---- guardar modelo y metricas ----
    CARPETA.mkdir(parents=True, exist_ok=True)
    np.savez(RUTA_MODELO, centroides=C, mapeo=mapeo, radios=radios, k=K, radio=radio,
             nombres=np.array(nombres), semilla=a.semilla)
    resultados.update({"k": K, "semilla": a.semilla, "reinicios": a.reinicios, "inercia": inercia,
                       "vueltas": vueltas, "radio_usado": radio, "radio_calibrado": cal["radio"],
                       "calibracion": {"radios": cal["radios"].tolist(), "frr": cal["frr"].tolist(), "far": cal["far"].tolist(),
                                       "frr_sin_excluidos": cal["frr_sin_excluidos"].tolist(), "eer_frr": cal["eer_frr"], "eer_far": cal["eer_far"]},
                       "excluidos_del_entrenamiento": excluidos,
                       "cluster_a_identidad": {int(j): nombres[mapeo[j]] for j in range(K)},
                       "exactitud_por_efecto_entrenamiento_prueba": por_efecto})
    (CARPETA / "metricas_kmeans.json").write_text(json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- graficos ----
    carpeta_g = CARPETA / "graficos"
    carpeta_g.mkdir(exist_ok=True)
    pca = PCA(n_components=2, random_state=0).fit(X)                # el PCA solo sirve para dibujar

    grafico_clusters(carpeta_g, X, y, es_train, C, mapeo, radio, cercano, nombres, pca)
    grafico_iteraciones(carpeta_g, X[es_train], historial, pca)
    import graficar_kmeans_3d                                       # PCA en 3D con una esfera de aceptacion por centroide
    graficar_kmeans_3d.generar()
    print(f"\nModelo: {RUTA_MODELO}\nMetricas: {CARPETA / 'metricas_kmeans.json'}\nGraficos en: {carpeta_g}")


if __name__ == "__main__":
    main()
