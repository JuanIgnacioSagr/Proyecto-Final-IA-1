"""K-means para las huellas dactilares (aprendizaje NO supervisado), implementado desde cero.

Que hace este script
--------------------
1. Carga los vectores de huellas (normalizados, ver caracterizar_huellas.py).
2. Separa ENTRENAMIENTO / VALIDACION / PRUEBA (60/20/20 por identidad y efecto, ver particion_huellas.py).
3. Aplica K-means (K = 3, una por cada identidad autorizada) SOLO con el entrenamiento
   para encontrar los 3 CENTROIDES.
4. Asigna cada cluster a una identidad.
5. Ajusta la REGLA DE DECISION (decision_huellas.py): PCA + distancia de Mahalanobis (T2) + residuo (Q). Los limites salen
   de la validacion y el UMBRAL lambda se CALIBRA con los intrusos de calibracion (dataset_huellas/intrusas/intruso_01..25):
   es el mayor valor con el que ninguno de ellos es aceptado.
6. Mide que tan bien funciona en entrenamiento, validacion y prueba (matrices de confusion y metricas de error; los
   intrusos 26-50 solo se usan aqui), guarda el modelo y dibuja los graficos.

Definiciones rapidas (ver el documento "Teoria metodos y forma implementada en el codigo.md")
- Vector: la lista de 104 numeros que describe a una huella (un punto en un espacio de 104 dimensiones).
- Cluster: un grupo de huellas que quedan cerca unas de otras.
- Centroide: el "centro" de un cluster = el promedio de todos sus vectores.
- Distancia euclidea: la distancia "en linea recta" entre dos vectores (la usa K-means para armar los clusters).
- T2 / Q: ver decision_huellas.py (lo que decide si una huella se acepta o se rechaza).

Los PASOS del algoritmo estan numerados (PASO 1 a PASO 5) igual que en el documento.

Uso:
    python src/kmeans_huellas.py
    python src/kmeans_huellas.py --semilla 3 --reinicios 20
    python src/kmeans_huellas.py --lam 1.5                          # fijar el umbral lambda a mano (sin calibrar)
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
from matplotlib.lines import Line2D
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA      # SOLO para dibujar en 2D; K-means NO lo usa

import caracterizar_huellas as CH
import decision_huellas as DH
import graficos_decision as GD
import metricas
import particion_huellas as PH
import preprocesar as P

RAIZ = Path(__file__).resolve().parent.parent
CARPETA = RAIZ / "dataset_huellas" / "caracteristicas"
RUTA_MODELO = CARPETA / "modelo_kmeans.npz"

# ----------------------------- parametros del metodo -----------------------------
K = 3                      # numero de clusters = numero de identidades autorizadas (lo fija el enunciado)
REINICIOS = 10             # cuantas veces se repite K-means desde cero con centroides iniciales distintos
MAX_ITER = 300             # tope de seguridad de iteraciones por corrida


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


def cargar_intrusos():
    """Vectores de los intrusos de dataset_huellas/intrusas/ (mismo camino que una huella nueva). -> X, es_calibracion, nombres."""
    rutas = sorted((RAIZ / "dataset_huellas" / "intrusas").glob("intruso_*.jpg"))
    if not rutas:
        sys.exit("No hay intrusos en dataset_huellas/intrusas/. Generalos con: python src/generar_dataset_huellas.py --solo intrusos")
    X = np.stack([CH.rasgo_huella(P.preprocesar_huella(P.a_gris(r))) for r in rutas])
    return X, np.array([PH.intruso_calibracion(r.name) for r in rutas]), [r.name for r in rutas]


def identificar_huella(vector, modelo=None):
    """Para la demo: vector de UNA huella (caracterizar_huellas.rasgo_huella) -> ("identidad_N" | "NO AUTORIZADO", puntaje s).

    El vector se obtiene con:  rasgo_huella(preprocesar_huella(imagen))
    """
    r = identificar_huella_detalle(vector, modelo)
    return r["resultado"], r["distancia"]


def identificar_huella_detalle(vector, modelo=None):
    """Igual que identificar_huella pero devuelve TODO el detalle de la decision (un diccionario):

      resultado          "identidad_N" si se acepta, "NO AUTORIZADO" si se rechaza
      identidad_cercana  identidad con menor T2 (aunque se haya rechazado)
      distancia          puntaje s de esa identidad = max(T2/limite_T2, Q/limite_Q)  (se compara con el umbral)
      radio              umbral lambda que se uso
      aceptada           True si distancia <= radio
      distancias         {identidad: puntaje s} para las 3 identidades
      margen             diferencia de T2/limite entre la 2.a y la 1.a identidad (chico = caso dudoso entre dos personas)
      t2, q              T2/limite (de la identidad cercana) y Q/limite: los dos ejes del grafico T2-Q
    """
    if modelo is None:
        modelo = np.load(RUTA_MODELO, allow_pickle=False)
    m = DH.desde_modelo(modelo)
    nombres = [str(n) for n in modelo["nombres"]]
    s, T2, Q = DH.puntajes(np.asarray(vector, dtype=np.float64)[None, :], m)
    j = int(T2[0].argmin())
    lam = float(modelo["radios"][j])
    ident = nombres[int(modelo["mapeo"][j])]
    orden = np.sort(T2[0] / m["t2_limite"])
    aceptada = bool(s[0, j] <= lam)
    return {"resultado": ident if aceptada else "NO AUTORIZADO", "identidad_cercana": ident,
            "distancia": float(s[0, j]), "radio": lam, "aceptada": aceptada,
            "distancias": {nombres[int(modelo["mapeo"][c])]: float(s[0, c]) for c in range(s.shape[1])},
            "margen": float(orden[1] - orden[0]) if len(orden) > 1 else float("nan"),
            "t2": float(T2[0, j] / m["t2_limite"]), "q": float(Q[0] / m["q_limite"])}


# =====================================================================================
#  Graficos
# =====================================================================================
def colores_identidad(n):
    return [plt.cm.tab10(i) for i in range(n)]


def _leyenda_particion():
    return [Line2D([], [], marker="o", linestyle="", color="lightgray", markeredgecolor="black", label="entrenamiento"),
            Line2D([], [], marker="s", linestyle="", color="lightgray", markeredgecolor="black", label="validación"),
            Line2D([], [], marker="^", linestyle="", color="lightgray", markeredgecolor="black", label="prueba"),
            Line2D([], [], marker="X", linestyle="", color=plt.cm.tab10(3), markeredgecolor="black", markersize=10, label="intruso"),
            Line2D([], [], marker="o", linestyle="", color="white", markeredgecolor="red", markeredgewidth=2, label="mal asignada")]


def grafico_clusters(carpeta_g, m, X, y, parte, mapeo, lam, cluster, nombres, Xi, cal):
    """Clusters sobre el plano PC1-PC2 del modelo con la ELIPSE de aceptacion (izquierda) y grafico T2-Q (derecha).

    La elipse es la sombra EXACTA del elipsoide T2 <= lambda*limite en las k componentes. Lo que el plano no muestra
    (el residuo Q) esta en el grafico de la derecha, donde la zona de aceptacion es un rectangulo.
    """
    col = colores_identidad(len(nombres))
    Z = (X - m["mu"]) @ m["V"]
    Zi = (Xi - m["mu"]) @ m["V"]
    fig, axs = plt.subplots(1, 2, figsize=(17, 7))
    ax = axs[0]
    marcas = {"entrenamiento": "o", "validacion": "s", "prueba": "^"}
    for j in range(len(m["Zc"])):
        ident = mapeo[j]
        ax.add_patch(GD.elipse_2d(m, j, lam, fill=True, fc=col[ident], alpha=0.10, ec="none"))
        ax.add_patch(GD.elipse_2d(m, j, lam, fill=False, ec=col[ident], lw=2.2, ls="--"))
        for nombre, marca in marcas.items():
            sel = (cluster == j) & (parte == nombre)
            mal = sel & (y != ident)
            ax.scatter(Z[sel & ~mal, 0], Z[sel & ~mal, 1], s=44, marker=marca, color=col[ident], edgecolor="black", linewidth=0.4, alpha=0.85)
            ax.scatter(Z[mal, 0], Z[mal, 1], s=80, marker=marca, color=col[ident], edgecolor="red", linewidth=2.0)
        ax.scatter(*m["Zc"][j][:2], marker="X", s=320, color=col[ident], edgecolor="black", linewidth=1.7, zorder=6)
        ax.annotate(nombres[ident].replace("identidad_", "Identidad "), m["Zc"][j][:2], xytext=(8, 10), textcoords="offset points", fontsize=9, weight="bold")
    ax.scatter(Zi[:, 0], Zi[:, 1], marker="X", s=60, color=plt.cm.tab10(3), edgecolor="black", linewidth=0.5, zorder=5)
    ax.set_aspect("equal", adjustable="datalim")
    ax.autoscale_view()
    ax.set_xlabel(f"PC1 ({m['varianza'][0] / m['varianza'].sum() * 100:.1f} %)")
    ax.set_ylabel(f"PC2 ({m['varianza'][1] / m['varianza'].sum() * 100:.1f} %)")
    ax.set_title(f"Elipse de aceptación (sombra exacta del elipsoide en {m['k']} componentes, {m['varianza_explicada'] * 100:.0f} % de la varianza)", fontsize=10.5)
    ax.grid(alpha=0.3)
    ax.legend(handles=_leyenda_particion(), loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    grupos = [(X[parte == "validacion"], dict(s=40, color="#1f77b4", edgecolor="black", linewidth=0.4, marker="s"), "autorizadas (validación)"),
              (X[parte == "prueba"], dict(s=40, color="#2ca02c", edgecolor="black", linewidth=0.4, marker="^"), "autorizadas (prueba)"),
              (Xi[cal], dict(s=60, color=plt.cm.tab10(3), edgecolor="black", linewidth=0.5, marker="X"), "intrusos (calibración)"),
              (Xi[~cal], dict(s=60, color="gold", edgecolor="black", linewidth=0.5, marker="X"), "intrusos (prueba)")]
    GD.dibujar_t2_q(axs[1], m, lam, grupos)
    axs[1].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"K-means de huellas: clusters, elipse de aceptación y umbral λ = {lam:.2f}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(carpeta_g / "kmeans_clusters_2d.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def grafico_iteraciones(carpeta_g, X_train, historial, pca, max_paneles=6):
    """Paso a paso de K-means: centroides iniciales al azar -> reasignacion -> ... -> convergencia."""
    pasos = list(range(len(historial)))
    if len(pasos) > max_paneles:                                  # si hay muchas vueltas, se muestran repartidas
        pasos = sorted(set(np.linspace(0, len(historial) - 1, max_paneles).round().astype(int)))
    P_ = pca.transform(X_train)
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
            ax.scatter(P_[lab == j, 0], P_[lab == j, 1], s=35, color=colores[j], edgecolor="black", linewidth=0.3, alpha=0.85)
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


def grafico_matrices(carpeta_g, matrices, cm_final, nombres, lam):
    """Matrices de confusion por FASE (entrenamiento, validacion, prueba: identificacion 3x3) y la final con intrusos (4x4)."""
    etiquetas = [n.replace("identidad_", "Id ") for n in nombres]
    fig, axs = plt.subplots(1, 4, figsize=(21, 5.2), gridspec_kw={"width_ratios": [1, 1, 1, 1.25]})
    for ax, (titulo, cm) in zip(axs[:3], matrices):
        _dibujar_cm(ax, cm, etiquetas, etiquetas, titulo)
    _dibujar_cm(axs[3], cm_final, etiquetas + ["Intruso"], etiquetas + ["NO AUTORIZADO"],
                f"PRUEBA con intrusos y umbral λ = {lam:.2f}\n(autorizadas de prueba + intrusos 26-50)")
    axs[3].axhline(2.5, color="black", lw=1.5)
    axs[3].axvline(2.5, color="black", lw=1.5)
    fig.suptitle("Matrices de confusión por fase", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(carpeta_g / "matrices_confusion_fases.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def _dibujar_cm(ax, cm, filas, columnas, titulo):
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(cm.max(), 1))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() * 0.55 else ("#1f2937" if i == j else "#d62728" if cm[i, j] else "#9ca3af"))
    ax.set_xticks(range(len(columnas)))
    ax.set_xticklabels(columnas, rotation=20, ha="right")
    ax.set_yticks(range(len(filas)))
    ax.set_yticklabels(filas)
    ax.set_xlabel("lo que dijo el sistema")
    ax.set_ylabel("lo que la huella ES")
    ax.set_title(titulo, fontsize=10.5, fontweight="bold")


# =====================================================================================
#  Programa principal
# =====================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--semilla", type=int, default=0, help="semilla del azar (resultados repetibles)")
    ap.add_argument("--reinicios", type=int, default=REINICIOS)
    ap.add_argument("--lam", type=float, default=None,
                    help="fijar el umbral lambda a mano (por defecto se CALIBRA con los intrusos de calibracion)")
    ap.add_argument("--excluir-entrenamiento", default="",
                    help="efectos (separados por coma) que NO se usan para entrenar; se siguen evaluando")
    a = ap.parse_args()

    X, y, efecto, archivo, imp, inv = cargar_datos()
    nombres = [inv[i] for i in sorted(inv) if i < K]               # identidad_1, identidad_2, identidad_3
    excluidos = [e.strip().lower() for e in a.excluir_entrenamiento.split(",") if e.strip()]
    desconocidos = [e for e in excluidos if e not in set(efecto)]
    if desconocidos:
        sys.exit(f"Efecto(s) desconocido(s): {desconocidos}. Disponibles: {sorted(set(efecto))}")
    parte = PH.particion(archivo)
    # es_train = huellas con las que REALMENTE se entrena (particion de entrenamiento, menos los efectos excluidos)
    es_train = (parte == "entrenamiento") & ~np.isin(efecto, excluidos)
    es_val, es_test = parte == "validacion", parte == "prueba"
    print(f"Huellas: {len(X)} | entrenamiento: {es_train.sum()} | validacion: {es_val.sum()} | prueba: {es_test.sum()}"
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

    # ---- PASO 6: regla de decision (PCA + T2 + Q) y umbral lambda ----
    m = DH.ajustar(X[es_train], lab_train, C)
    m = DH.fijar_limites(m, X[es_val])
    t2_teo, q_teo = DH.limites_teoricos(m, int(np.bincount(lab_train).min()))
    Xi, cal, nombres_i = cargar_intrusos()
    s_i, _, _ = DH.puntajes(Xi, m)
    s_min_i = s_i.min(axis=1)
    lam = a.lam if a.lam is not None else DH.calibrar_lambda(s_min_i[cal])
    print(f"\nPASO 6 - regla de decision: PCA con k = {m['k']} componentes ({m['varianza_explicada'] * 100:.1f} % de la varianza)")
    print(f"   limite T2 = {m['t2_limite']:.2f}, limite Q = {m['q_limite']:.4f}  (percentil {DH.PERCENTIL_LIMITE} de la validacion)")
    print(f"   referencia teorica: T2 (F de Fisher) = {t2_teo:.2f}, Q (Jackson-Mudholkar) = {q_teo:.4f}")
    print(f"   intrusos de calibracion ({int(cal.sum())}): puntaje minimo s = {s_min_i[cal].min():.2f}")
    print(f"   umbral lambda = {lam:.2f}" + ("  (fijado a mano con --lam)" if a.lam is not None else "  (mayor valor sin aceptar intrusos de calibracion)"))
    radios = np.full(K, lam)                         # el mismo umbral para los 3 clusters

    # ---- evaluacion por fase ----
    cluster, s_aut, acep_aut = DH.decidir(X, m, lam)
    pred = mapeo[cluster]
    resultados, matrices = {}, []
    for nombre, mascara in (("ENTRENAMIENTO", es_train), ("VALIDACION", es_val), ("PRUEBA", es_test)):
        cm = metricas.matriz_confusion(y[mascara], pred[mascara], len(nombres))
        mt = metricas.imprimir_reporte(f"{nombre}: identidad real vs identidad asignada", cm, nombres)
        mt["matriz_confusion"] = cm.tolist()
        mt["falso_rechazo"] = float((~acep_aut[mascara]).mean())
        resultados[nombre.lower()] = mt
        matrices.append((f"{nombre} (n={int(mascara.sum())})\nidentificación, exactitud {mt['exactitud'] * 100:.1f} %", cm))
        print(f"  con el umbral: {(~acep_aut[mascara]).sum()} de {mascara.sum()} huellas autorizadas serian rechazadas "
              f"(falso rechazo = {mt['falso_rechazo'] * 100:.1f}%)")
    print("\nExactitud de identificacion por efecto (entrenamiento / validacion / prueba):")
    por_efecto = {}
    for e in sorted(set(efecto)):
        fila = []
        for mascara in (es_train, es_val, es_test):
            sel = (efecto == e) & mascara
            fila.append(float((pred[sel] == y[sel]).mean()) if sel.any() else float("nan"))
        por_efecto[e] = fila
        print(f"   {e:<12} " + "  /  ".join(f"{v * 100:>6.1f}%" for v in fila))

    # intrusos: calibracion (no deben entrar) y prueba (FAR honesta)
    j_i = DH.estadisticos(Xi, m)[0].argmin(axis=1)
    acep_i = s_min_i <= lam
    far_cal, far_test = float(acep_i[cal].mean()), float(acep_i[~cal].mean())
    cm4 = np.zeros((4, 4), dtype=int)                # filas = realidad (3 identidades + intruso), columnas = lo que dijo
    for r, p_, ok in zip(y[es_test], pred[es_test], acep_aut[es_test]):
        cm4[r, p_ if ok else 3] += 1
    for ok, jj in zip(acep_i[~cal], j_i[~cal]):
        cm4[3, int(mapeo[jj]) if ok else 3] += 1
    print(f"\nINTRUSOS: calibracion {int(cal.sum())} (aceptados {int(acep_i[cal].sum())}) | prueba {int((~cal).sum())} "
          f"(aceptados {int(acep_i[~cal].sum())}, FAR = {far_test * 100:.1f} %)")
    print("Matriz de la prueba con intrusos (filas: Id1, Id2, Id3, Intruso; columnas: Id1, Id2, Id3, NO AUTORIZADO):")
    print(cm4)

    # ---- guardar modelo y metricas ----
    CARPETA.mkdir(parents=True, exist_ok=True)
    np.savez(RUTA_MODELO, centroides=C, mapeo=mapeo, radios=radios, k=K, radio=lam, regla="mahalanobis_residuo",
             nombres=np.array(nombres), semilla=a.semilla, **DH.a_diccionario(m))
    resultados.update({"k": K, "semilla": a.semilla, "reinicios": a.reinicios, "inercia": inercia, "vueltas": vueltas,
                       "regla": "PCA + T2 (Mahalanobis) + Q (residuo)", "lambda": lam, "lambda_calibrado": DH.calibrar_lambda(s_min_i[cal]),
                       "componentes": m["k"], "varianza_explicada": m["varianza_explicada"],
                       "limite_t2": m["t2_limite"], "limite_q": m["q_limite"], "limite_t2_teorico": t2_teo, "limite_q_teorico": q_teo,
                       "conjuntos": {"entrenamiento": int(es_train.sum()), "validacion": int(es_val.sum()), "prueba": int(es_test.sum()),
                                     "intrusos_calibracion": int(cal.sum()), "intrusos_prueba": int((~cal).sum())},
                       "intrusos": {"far_calibracion": far_cal, "far_prueba": far_test, "matriz_prueba_con_intrusos": cm4.tolist()},
                       "excluidos_del_entrenamiento": excluidos,
                       "cluster_a_identidad": {int(j): nombres[mapeo[j]] for j in range(K)},
                       "exactitud_por_efecto_entrenamiento_validacion_prueba": por_efecto})
    (CARPETA / "metricas_kmeans.json").write_text(json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- graficos ----
    carpeta_g = CARPETA / "graficos"
    carpeta_g.mkdir(exist_ok=True)
    pca_dibujo = PCA(n_components=2, random_state=0).fit(X[es_train])      # solo para dibujar el paso a paso de K-means
    grafico_clusters(carpeta_g, m, X, y, parte, mapeo, lam, cluster, nombres, Xi, cal)
    grafico_iteraciones(carpeta_g, X[es_train], historial, pca_dibujo)
    grafico_matrices(carpeta_g, matrices, cm4, nombres, lam)
    import graficar_kmeans_3d                                       # PCA en 3D con un elipsoide de aceptacion por identidad
    graficar_kmeans_3d.generar()
    print(f"\nModelo: {RUTA_MODELO}\nMetricas: {CARPETA / 'metricas_kmeans.json'}\nGraficos en: {carpeta_g}")


if __name__ == "__main__":
    main()
