"""Cuanto se parecen entre si las identidades: T2 a la identidad propia contra T2 a la segunda mas cercana.

    python explicativos/margen_identidades.py      # crea explicativos/imagenes/margen_identidades.png

Cada punto es una huella autorizada de evaluacion (prueba del dataset + subidas a mano). Los dos ejes son la distancia de Mahalanobis
(T2, dividida por su limite) a la identidad MAS CERCANA (X) y a la SEGUNDA mas cercana (Y), calculadas en las k componentes del
modelo: no hay proyeccion a 2D, asi que nada se comprime.
Sobre la diagonal (Y = X) la huella esta a la misma distancia de dos personas: caso dudoso. Lejos de la diagonal la identificacion
es clara. Los circulos rojos son huellas identificadas con otra persona. Los intrusos (X rojas) se ven lejos de todas.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402
from matplotlib.lines import Line2D                               # noqa: E402

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import intrusos_huellas as IH                                     # noqa: E402

DH = IH.DH
COLORES, ROJO = IH.COLORES, IH.ROJO
plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]


def main():
    m, mapeo, lam, _ = IH.modelo()
    X, verdad, _ = IH.cargar()
    T2 = DH.estadisticos(X, m)[0] / m["t2_limite"]
    orden = np.sort(T2, axis=1)
    pred = mapeo[T2.argmin(axis=1)]
    aut, intr = verdad < 3, verdad == 3
    mal = aut & (pred != verdad)

    fig, ax = plt.subplots(figsize=(8.6, 8))
    lo, hi = float(orden[:, 0].min()) * 0.7, float(orden[:, 1].max()) * 1.5
    ax.plot([lo, hi], [lo, hi], color="black", lw=1.5, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.text(lo * 1.2, hi * 0.5, "identificación clara\n(arriba de la diagonal)", color="#2ca02c", fontweight="bold", va="top")
    for i in range(3):
        sel = aut & (verdad == i)
        ax.scatter(orden[sel, 0], orden[sel, 1], s=62, color=COLORES[i], edgecolor="black", linewidth=0.7, zorder=5, label=f"autorizada, identidad {i + 1}")
    ax.scatter(orden[mal, 0], orden[mal, 1], s=230, facecolor="none", edgecolor=ROJO, linewidth=2.2, zorder=6)
    ax.scatter(orden[intr, 0], orden[intr, 1], marker="X", s=80, color=ROJO, edgecolor="black", linewidth=0.6, zorder=4, label="intruso")
    ax.set_xlabel("T² / límite a la identidad más cercana")
    ax.set_ylabel("T² / límite a la segunda identidad más cercana")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25, which="both")
    marg = np.log10(orden[:, 1] / orden[:, 0])[aut]
    ax.set_title(f"Separación entre identidades: {int(mal.sum())} de {int(aut.sum())} autorizadas identificadas con otra persona\n"
                 f"margen mínimo: la 2.ª identidad está {10 ** marg.min():.1f} veces más lejos que la 1.ª", fontsize=11.5, fontweight="bold")
    h, _ = ax.get_legend_handles_labels()
    h.append(Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=ROJO, markeredgewidth=2, markersize=13))
    ax.legend(handles=h, labels=[f"autorizada, identidad {i + 1}" for i in range(3)] + ["intruso", "identificada con otra persona"],
              loc="lower right", fontsize=9)
    fig.tight_layout()
    IH.SALIDA.mkdir(parents=True, exist_ok=True)
    fig.savefig(IH.SALIDA / "margen_identidades.png", dpi=140, bbox_inches="tight")
    print(f"autorizadas identificadas con otra persona: {int(mal.sum())}/{int(aut.sum())} | cociente minimo 2.a/1.a identidad: {10 ** marg.min():.2f} (mediana {10 ** np.median(marg):.2f})")


if __name__ == "__main__":
    main()
