"""Grafico de clusters con la huella nueva.

Muestra, sobre un plano PCA (el PCA solo sirve para DIBUJAR; K-means usa los 72 numeros):
  - las huellas del dataset, coloreadas por identidad;
  - los 3 centroides (X grande) y el circulo del umbral R alrededor de cada uno;
  - la huella nueva (estrella) y una linea punteada hasta el centroide MAS CERCANO, con la distancia real.

POSICION DE LOS PUNTOS (para que el dibujo sea coherente con la decision):
el PCA aplasta 72 dimensiones en 2 y hace que los puntos parezcan MAS CERCANOS de lo que estan (una huella
rechazada podia verse dentro del circulo). Por eso cada punto se dibuja a su distancia REAL (72 dimensiones)
del centroide mas cercano, en la direccion que muestra el plano PCA. Asi, la regla del algoritmo se ve tal cual:
    dentro del circulo de radio R  =  huella aceptada      /      fuera del circulo  =  huella rechazada
y la linea punteada mide exactamente la distancia que se escribe en el grafico.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")                                  # dibujar dentro de una ventana de Tkinter
import matplotlib.pyplot as plt                          # noqa: E402
import numpy as np                                       # noqa: E402
from matplotlib.figure import Figure                     # noqa: E402
from matplotlib.lines import Line2D                      # noqa: E402
from matplotlib.patches import Circle                    # noqa: E402
from sklearn.decomposition import PCA                    # noqa: E402

import config                                            # noqa: E402

sys.path.insert(0, str(config.SRC))
import kmeans_huellas as KH                              # noqa: E402

NOMBRES = {0: "Identidad 1", 1: "Identidad 2", 2: "Identidad 3"}


def construir_figura(modelo, vector, decision, origen_real=None):
    """Devuelve una matplotlib.figure.Figure lista para mostrar o guardar.

    modelo    el modelo K-means cargado (centroides, mapeo, radios, nombres)
    vector    los 72 numeros de la huella nueva
    decision  lo que devolvio identificar_huella_detalle
    """
    datos = np.load(KH.CARPETA / "dataset.npz", allow_pickle=False)
    X, y = datos["X"].astype(float), datos["y"].astype(int)       # huellas del dataset y su identidad (0, 1, 2)
    C = modelo["centroides"]                                      # (3, 72)
    mapeo = modelo["mapeo"].astype(int)                           # centroide -> identidad (0, 1, 2)
    R = float(modelo["radios"][0])

    pca = PCA(n_components=2, random_state=0).fit(X)              # mismo plano que kmeans_clusters_2d.png
    P_x = pca.transform(X)
    P_c = pca.transform(C)
    v72 = np.asarray(vector, dtype=float)
    colores = [plt.cm.tab10(i) for i in range(3)]

    def posicion_real(X72, proyectado):
        """Lleva cada punto a su distancia REAL (72 dim) del centroide mas cercano, conservando la direccion del plano.

        Devuelve (posicion a dibujar, indice del centroide mas cercano, distancia real). El centroide mas cercano
        y la distancia se calculan en las 72 dimensiones, igual que decide el algoritmo.
        """
        D = np.linalg.norm(X72[:, None, :] - C[None, :, :], axis=2)
        cerca = D.argmin(axis=1)
        dist = D[np.arange(len(X72)), cerca]
        v = proyectado - P_c[cerca]
        norma = np.linalg.norm(v, axis=1, keepdims=True)
        u = np.where(norma > 1e-9, v / np.maximum(norma, 1e-9), np.array([1.0, 0.0]))   # direccion en el plano
        return P_c[cerca] + u * dist[:, None], cerca, dist

    P_x, _, _ = posicion_real(X, P_x)                              # huellas del dataset
    p_all, j_all, d_all = posicion_real(v72[None, :], pca.transform(v72[None, :]))
    p, j, d72 = p_all[0], int(j_all[0]), d_all[0]                  # huella nueva

    fig = Figure(figsize=(8.6, 6.3), dpi=100)
    ax = fig.add_subplot(111)
    for ident in range(3):
        m = y == ident
        ax.scatter(P_x[m, 0], P_x[m, 1], s=26, color=colores[ident], alpha=0.35, edgecolor="none")
    for c in range(len(C)):
        ident = mapeo[c]
        ax.add_patch(Circle(P_c[c], R, fill=False, edgecolor=colores[ident], linewidth=1.8, linestyle="--", alpha=0.9))
        ax.scatter(*P_c[c], marker="X", s=330, color=colores[ident], edgecolor="black", linewidth=1.6, zorder=5)
        ax.annotate(NOMBRES[ident], P_c[c], xytext=(9, 11), textcoords="offset points", fontsize=8, weight="bold")

    aceptada = decision["aceptada"]
    color_punto = "limegreen" if aceptada else "red"
    # linea punteada hasta el cluster (centroide) mas cercano
    ax.plot([p[0], P_c[j, 0]], [p[1], P_c[j, 1]], linestyle=":", color="black", linewidth=2, zorder=6)
    ax.scatter(*p, marker="*", s=520, color=color_punto, edgecolor="black", linewidth=1.5, zorder=7)
    medio = (p + P_c[j]) / 2
    # la etiqueta va separada de la linea (con un hilo fino) para que no tape la estrella ni el centroide
    ax.annotate(f"d = {d72:.3f}\n(umbral R = {R:.2f})", medio, xytext=(46, 52), textcoords="offset points",
                fontsize=9, weight="bold", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.95),
                arrowprops=dict(arrowstyle="-", color="gray", lw=1), zorder=8)

    ayuda = [Line2D([], [], marker="o", linestyle="", color=colores[i], alpha=0.6, label=f"huellas {NOMBRES[i]}")
             for i in range(3)]
    ayuda += [Line2D([], [], marker="X", linestyle="", color="gray", markeredgecolor="black", markersize=11, label="centroide (centro del cluster)"),
              Line2D([], [], linestyle="--", color="gray", label=f"umbral de aceptación R = {R:.2f}"),
              Line2D([], [], marker="*", linestyle="", color=color_punto, markeredgecolor="black", markersize=15,
                     label="huella nueva (" + ("aceptada" if aceptada else "rechazada") + ")"),
              Line2D([], [], linestyle=":", color="black", label="distancia al cluster más cercano")]
    ax.legend(handles=ayuda, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    veredicto = decision["resultado"].replace("identidad_", "Identidad ") if aceptada else "NO AUTORIZADO"
    subtitulo = f"  |  origen real: {origen_real}" if origen_real else ""
    ax.set_title(f"Huella nueva → {veredicto}   (cluster más cercano: {NOMBRES[mapeo[j]]}){subtitulo}", fontsize=10)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    fig.text(0.08, 0.012, "Cada punto está dibujado a su distancia REAL (72 dimensiones) del centroide más cercano, en la dirección del plano PCA:\n"
                          "dentro del círculo = huella aceptada · fuera del círculo = huella rechazada (la línea punteada mide esa distancia).",
             fontsize=8, color="#444444", va="bottom")
    fig.subplots_adjust(left=0.08, right=0.69, top=0.92, bottom=0.14)
    return fig
