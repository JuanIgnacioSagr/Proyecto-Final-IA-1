"""Grafico explicativo: distancia euclidea vs distancia de Mahalanobis (esfera contra elipsoide).

    python explicativos/euclidea_vs_mahalanobis.py     # crea explicativos/imagenes/euclidea_vs_mahalanobis.png

Una identidad (nube de huellas) que varia MUCHO en una direccion y POCO en la otra, y dos huellas nuevas A y B que estan a la
MISMA distancia euclidea del centro. La euclidea las trata igual (zona circular). Mahalanobis divide cada diferencia por cuanto
varia esa identidad en esa direccion: A (en la direccion donde varia mucho) es normal; B (donde varia poco) es raro.
Colores: tab10 (azul = identidad, verde = aceptada, rojo = rechazada).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402
from matplotlib.patches import Circle, Ellipse                    # noqa: E402

AZUL, VERDE, ROJO = plt.cm.tab10(0), plt.cm.tab10(2), plt.cm.tab10(3)
SALIDA = Path(__file__).resolve().parent / "imagenes"
plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]


def main():
    rng = np.random.default_rng(3)
    desv = np.array([1.5, 0.5])                                      # la identidad varia 3 veces mas en x que en y
    nube = rng.normal(0, 1, (120, 2)) * desv
    A, B = np.array([3.4, 0.0]), np.array([0.0, 3.4])                # misma distancia euclidea (3,4) al centro
    R_euc = 3.6
    T = 2.6                                                          # umbral de Mahalanobis (en "desviaciones")
    fig, axs = plt.subplots(1, 2, figsize=(14, 6.4), sharex=True, sharey=True)
    for ax in axs:
        ax.scatter(nube[:, 0], nube[:, 1], s=22, color=AZUL, alpha=0.55, edgecolor="none", label="huellas de la identidad")
        ax.scatter(0, 0, marker="X", s=260, color=AZUL, edgecolor="black", linewidth=1.5, zorder=5, label="centro (centroide)")
        ax.set_aspect("equal")
        ax.set_xlim(-5.2, 6.6)
        ax.set_ylim(-4.6, 4.6)
        ax.grid(alpha=0.25)
        ax.set_xlabel("dirección donde la identidad varía MUCHO")
    axs[0].set_ylabel("dirección donde la identidad varía POCO")

    ax = axs[0]
    ax.add_patch(Circle((0, 0), R_euc, fill=True, fc=VERDE, alpha=0.10, ec="none"))
    ax.add_patch(Circle((0, 0), R_euc, fill=False, ec=VERDE, lw=2.6, ls="--"))
    for p, n in ((A, "A"), (B, "B")):
        ax.scatter(*p, s=170, color=VERDE, edgecolor="black", linewidth=1.3, zorder=6, marker="o")
        ax.text(p[0] + 0.25, p[1] + 0.25, f"{n}", fontsize=14, fontweight="bold")
    ax.set_title("Distancia euclídea: zona de aceptación = círculo\nA y B están a la misma distancia (3,4): se aceptan las dos", fontsize=12, fontweight="bold")

    ax = axs[1]
    ax.add_patch(Ellipse((0, 0), 2 * T * desv[0], 2 * T * desv[1], fill=True, fc=VERDE, alpha=0.10, ec="none"))
    ax.add_patch(Ellipse((0, 0), 2 * T * desv[0], 2 * T * desv[1], fill=False, ec=VERDE, lw=2.6, ls="--"))
    ax.scatter(*A, s=170, color=VERDE, edgecolor="black", linewidth=1.3, zorder=6)
    ax.scatter(*B, s=190, color=ROJO, edgecolor="black", linewidth=1.3, zorder=6, marker="X")
    ax.text(A[0] + 0.25, A[1] + 0.25, "A: se aparta 2,3 desvíos\n(normal en esa dirección)", fontsize=10.5, fontweight="bold", color="#14532d")
    ax.text(B[0] + 0.3, B[1] - 0.1, "B: se aparta 6,8 desvíos\n(raro en esa dirección)", fontsize=10.5, fontweight="bold", color="#7f1d1d")
    ax.set_title("Distancia de Mahalanobis: zona de aceptación = elipse\nA se acepta y B se rechaza", fontsize=12, fontweight="bold")
    axs[0].legend(loc="lower left", fontsize=9)
    fig.suptitle("Euclídea contra Mahalanobis: la distancia tiene en cuenta cuánto varía cada identidad en cada dirección", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    SALIDA.mkdir(parents=True, exist_ok=True)
    fig.savefig(SALIDA / "euclidea_vs_mahalanobis.png", dpi=140, bbox_inches="tight")
    print("Guardado:", SALIDA / "euclidea_vs_mahalanobis.png", "| T2 de A =", round((A[0] / desv[0]) ** 2, 2), "| T2 de B =", round((B[1] / desv[1]) ** 2, 2))


if __name__ == "__main__":
    main()
