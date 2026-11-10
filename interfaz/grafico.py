"""Grafico de clusters con la huella nueva.

Solo el plano PC1-PC2 del modelo (la regla de decision esta en src/decision_huellas.py): las huellas del dataset por
identidad, la ELIPSE de aceptacion de cada identidad (sombra exacta del elipsoide T2 <= lambda * limite) y la huella nueva
(estrella) en su posicion real, con una linea punteada a la identidad mas cercana.
Una huella puede verse dentro de la elipse y estar rechazada: la rechazan las componentes que el plano no muestra.
"""
import sys

import matplotlib
matplotlib.use("TkAgg")                                  # dibujar dentro de una ventana de Tkinter
import matplotlib.pyplot as plt                          # noqa: E402
import numpy as np                                       # noqa: E402
from matplotlib.figure import Figure                     # noqa: E402
from matplotlib.lines import Line2D                      # noqa: E402

import config                                            # noqa: E402

sys.path.insert(0, str(config.SRC))
import decision_huellas as DH                            # noqa: E402
import graficos_decision as GD                           # noqa: E402
import kmeans_huellas as KH                              # noqa: E402

NOMBRES = {0: "Identidad 1", 1: "Identidad 2", 2: "Identidad 3"}


def construir_figura(modelo, vector, decision, origen_real=None):
    """Devuelve una matplotlib.figure.Figure lista para mostrar o guardar.

    modelo    el modelo K-means cargado (centroides, mapeo, radios, regla de decision)
    vector    los numeros de la huella nueva (rasgo_huella, ya normalizados)
    decision  lo que devolvio identificar_huella_detalle
    """
    datos = np.load(KH.CARPETA / "dataset.npz", allow_pickle=False)
    X, y = datos["X"].astype(float), datos["y"].astype(int)       # huellas del dataset y su identidad (0, 1, 2)
    m = DH.desde_modelo(modelo)
    mapeo = modelo["mapeo"].astype(int)                           # centroide -> identidad (0, 1, 2)
    lam = float(modelo["radios"][0])
    v = np.asarray(vector, dtype=float)
    Z = (X - m["mu"]) @ m["V"]
    z = (v - m["mu"]) @ m["V"]
    T2, _, _ = DH.estadisticos(v[None, :], m)
    j = int(T2[0].argmin())                                       # identidad mas cercana segun T2
    colores = [plt.cm.tab10(i) for i in range(3)]
    ev = m["varianza"] / m["varianza"].sum() * 100

    fig = Figure(figsize=(8.6, 6.3), dpi=100)
    ax = fig.add_subplot(111)
    for ident in range(3):
        sel = y == ident
        ax.scatter(Z[sel, 0], Z[sel, 1], s=26, color=colores[ident], alpha=0.35, edgecolor="none")
    for c in range(len(m["Zc"])):
        ident = mapeo[c]
        ax.add_patch(GD.elipse_2d(m, c, lam, fill=True, fc=colores[ident], alpha=0.08, ec="none"))
        ax.add_patch(GD.elipse_2d(m, c, lam, fill=False, ec=colores[ident], lw=1.8, ls="--"))
        ax.scatter(*m["Zc"][c][:2], marker="X", s=300, color=colores[ident], edgecolor="black", linewidth=1.5, zorder=5)
        ax.annotate(NOMBRES[ident], m["Zc"][c][:2], xytext=(9, 11), textcoords="offset points", fontsize=8, weight="bold")
    aceptada = decision["aceptada"]
    color_punto = "limegreen" if aceptada else "red"
    ax.plot([z[0], m["Zc"][j][0]], [z[1], m["Zc"][j][1]], linestyle=":", color="black", linewidth=1.8, zorder=6)
    ax.scatter(z[0], z[1], marker="*", s=480, color=color_punto, edgecolor="black", linewidth=1.5, zorder=7)
    ayuda = [Line2D([], [], marker="o", linestyle="", color=colores[i], alpha=0.6, label=f"huellas {NOMBRES[i]}") for i in range(3)]
    ayuda += [Line2D([], [], marker="X", linestyle="", color="gray", markeredgecolor="black", markersize=11, label="centro de la identidad"),
              Line2D([], [], linestyle="--", color="gray", label=f"elipse de aceptación (λ = {lam:.2f})"),
              Line2D([], [], marker="*", linestyle="", color=color_punto, markeredgecolor="black", markersize=15,
                     label="huella nueva (" + ("aceptada" if aceptada else "rechazada") + ")")]
    ax.legend(handles=ayuda, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    veredicto = decision["resultado"].replace("identidad_", "Identidad ") if aceptada else "NO AUTORIZADO"
    subtitulo = f"  |  origen real: {origen_real}" if origen_real else ""
    ax.set_title(f"Huella nueva → {veredicto}{subtitulo}\n(identidad más cercana: {NOMBRES[mapeo[j]]}; puntaje {decision['distancia']:.2f}"
                 f" {'≤' if aceptada else '>'} umbral λ = {lam:.2f})", fontsize=9.5)
    fig.text(0.08, 0.012, f"Plano PC1-PC2 (la decisión usa {m['k']} componentes). Elipse = sombra exacta de la zona de aceptación: si la estrella\n"
                          "se ve dentro pero está rechazada, la rechazan las componentes que este plano no muestra.",
             fontsize=8, color="#444444", va="bottom")
    fig.subplots_adjust(left=0.08, right=0.69, top=0.92, bottom=0.14)
    return fig
