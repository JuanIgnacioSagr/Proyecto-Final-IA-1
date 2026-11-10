"""PCA bien hecho para el umbral de las huellas: scree, elipses reales y grafico T2-Q (identidades, transformaciones e intrusos).

    python explicativos/pca_correcto.py           # crea pca_correcto.png y pca3d_correcto.png en explicativos/imagenes/
    python explicativos/pca_correcto.py --mostrar # ademas abre la ventana 3D para girarla

Que hace distinto a los PCA anteriores (con circulo o esfera):
  1. El PCA y las covarianzas se ajustan SOLO con el entrenamiento y las variables ya estan estandarizadas (z-score).
  2. Se muestra cuanta varianza explica cada componente (scree) y cuantas se usan (k, las que explican ~90 %).
  3. La zona de aceptacion de cada identidad es un ELIPSOIDE (Mahalanobis), no una esfera: su sombra en el plano PC1-PC2 es
     una ELIPSE exacta, con la covarianza de ese bloque. No se reescalan ni se mueven los puntos: cada huella esta donde cae.
  4. Lo que el plano no muestra (el residuo Q: la parte de la huella que las k componentes no explican) tiene su propio
     eje en el grafico T2-Q. Ahi la zona de aceptacion es un rectangulo y la decision se ve EXACTA.
Los marcadores muestran la decision real del sistema (igual que la matriz de confusion). Un punto que se ve dentro de la
elipse y esta rechazado (circulo negro) lo rechaza su residuo Q o una componente que el plano no muestra.
"""
import sys
from pathlib import Path

import matplotlib

if "--mostrar" in sys.argv:
    matplotlib.use("TkAgg")
else:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402
from matplotlib.lines import Line2D                               # noqa: E402

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import intrusos_huellas as IH                                     # noqa: E402

KH, DH, GD, PH = IH.KH, IH.DH, IH.GD, IH.PH
COLORES, ROJO = IH.COLORES, IH.ROJO
plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]


def leyenda(extra=()):
    base = [Line2D([], [], marker="o", ls="", color="gray", alpha=0.35, label="dataset: originales y transformaciones (tenue)"),
            Line2D([], [], marker="o", ls="", color="gray", markeredgecolor="black", label="autorizada aceptada (color = identidad)"),
            Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor="gray", markeredgewidth=2, label="autorizada rechazada"),
            Line2D([], [], marker="X", ls="", color=ROJO, markeredgecolor="black", markersize=10, label="intruso rechazado"),
            Line2D([], [], marker="X", ls="", color="gold", markeredgecolor="black", markersize=11, label="intruso ACEPTADO por error")]
    return base + list(extra)


def main():
    m, mapeo, lam, C = IH.modelo()
    datos = np.load(KH.CARPETA / "dataset.npz", allow_pickle=True)
    Xd, yd = datos["X"].astype(float), datos["y"].astype(int)
    parte = PH.particion([str(a) for a in datos["archivo"]])
    X, verdad, origen = IH.cargar(incluir_calibracion=True)
    salida, s, _ = IH.decidir(X, m, mapeo, lam)
    aceptada = salida < 3
    aut, intr = verdad < 3, verdad == 3
    Z, Zd = (X - m["mu"]) @ m["V"], (Xd - m["mu"]) @ m["V"]
    ev = m["varianza"] / m["varianza"].sum() * 100
    T2, _, _ = DH.estadisticos(X, m)
    # dentro de la elipse (sombra 2D) segun la distancia de Mahalanobis restringida al plano PC1-PC2
    dentro2d = np.zeros(len(X), dtype=bool)
    r2 = GD.radio_mahalanobis(m, lam) ** 2
    for j in range(len(m["Zc"])):
        cov = GD.covarianza(m, j)[:2, :2]
        d = Z[:, :2] - m["Zc"][j][:2]
        dentro2d |= np.einsum("ij,jk,ik->i", d, np.linalg.inv(cov), d) <= r2
    engana = dentro2d & ~aceptada

    fig = plt.figure(figsize=(21, 7.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 1.25, 1.1], wspace=0.28)

    # --- 1) scree
    ax = fig.add_subplot(gs[0])
    n = min(len(ev), 30)
    ax.bar(range(1, n + 1), ev[:n], color=["#1f77b4" if i < m["k"] else "#c7c7c7" for i in range(n)])
    ax2 = ax.twinx()
    ax2.plot(range(1, n + 1), np.cumsum(ev)[:n], color="black", marker="o", ms=3.5)
    ax2.axhline(DH.VARIANZA_OBJETIVO * 100, color=ROJO, ls="--", lw=1.2)
    ax2.set_ylim(0, 102)
    ax2.set_ylabel("varianza acumulada (%)")
    ax.set_xlabel("componente principal")
    ax.set_ylabel("varianza explicada (%)")
    ax.set_title(f"1) Scree: se usan k = {m['k']} componentes\n({m['varianza_explicada'] * 100:.0f} % de la varianza; el resto es el residuo Q)", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.2)

    # --- 2) scores con elipse
    ax = fig.add_subplot(gs[1])
    for i in range(3):
        ax.scatter(Zd[yd == i, 0], Zd[yd == i, 1], s=20, color=COLORES[i], alpha=0.18, edgecolor="none", zorder=2)
    for j in range(len(m["Zc"])):
        ident = mapeo[j]
        ax.add_patch(GD.elipse_2d(m, j, lam, fill=True, fc=COLORES[ident], alpha=0.10, ec="none", zorder=1))
        ax.add_patch(GD.elipse_2d(m, j, lam, fill=False, ec=COLORES[ident], lw=2.4, ls="--", zorder=3))
        ax.scatter(*m["Zc"][j][:2], marker="X", s=330, color=COLORES[ident], edgecolor="black", linewidth=1.5, zorder=6)
        ax.text(m["Zc"][j][0], m["Zc"][j][1], f"  Identidad {ident + 1}", fontweight="bold", fontsize=9, va="bottom", zorder=9)
    for i in range(3):
        sel = verdad == i
        ax.scatter(Z[sel & aceptada, 0], Z[sel & aceptada, 1], s=52, color=COLORES[i], edgecolor="black", linewidth=0.7, zorder=7)
        ax.scatter(Z[sel & ~aceptada, 0], Z[sel & ~aceptada, 1], s=60, facecolor="white", edgecolor=COLORES[i], linewidth=2, zorder=7)
    ax.scatter(Z[intr & ~aceptada, 0], Z[intr & ~aceptada, 1], marker="X", s=85, color=ROJO, edgecolor="black", linewidth=0.6, zorder=8)
    ax.scatter(Z[intr & aceptada, 0], Z[intr & aceptada, 1], marker="X", s=140, color="gold", edgecolor="black", linewidth=1.2, zorder=9)
    ax.scatter(Z[engana, 0], Z[engana, 1], s=270, facecolor="none", edgecolor="black", linewidth=1.5, ls="--", zorder=10)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.2)
    ax.set_xlabel(f"PC1 ({ev[0]:.1f} %)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1f} %)")
    ax.set_title(f"2) Scores PC1-PC2 con la ELIPSE de aceptación (λ = {lam:.2f})\n{int(engana.sum())} rechazadas se ven dentro (círculo negro): las rechaza el residuo Q",
                 fontsize=11, fontweight="bold")

    # --- 3) T2 contra Q
    ax = fig.add_subplot(gs[2])
    grupos = [(Xd[parte == "entrenamiento"], dict(s=16, color="#bbbbbb", alpha=0.6, edgecolor="none"), "dataset (entrenamiento)"),
              (X[aut & aceptada], dict(s=48, color="#2ca02c", edgecolor="black", linewidth=0.5), "autorizada aceptada"),
              (X[aut & ~aceptada], dict(s=60, facecolor="white", edgecolor="#2ca02c", linewidth=2), "autorizada rechazada"),
              (X[intr & ~aceptada], dict(s=75, marker="X", color=ROJO, edgecolor="black", linewidth=0.5), "intruso rechazado"),
              (X[intr & aceptada], dict(s=120, marker="X", color="gold", edgecolor="black", linewidth=1), "intruso ACEPTADO")]
    GD.dibujar_t2_q(ax, m, lam, [g for g in grupos if len(g[0])], titulo="3) T² contra residuo Q: aquí la decisión es exacta")
    ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(f"PCA correcto para el umbral: {int((aut & aceptada).sum())} de {int(aut.sum())} autorizadas aceptadas · "
                 f"{int((intr & ~aceptada).sum())} de {int(intr.sum())} intrusos rechazados (incluye los {int((origen == 'intruso (calibracion)').sum())} de calibración)",
                 fontsize=14, fontweight="bold")
    IH.SALIDA.mkdir(parents=True, exist_ok=True)
    fig.savefig(IH.SALIDA / "pca_correcto.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # --- 3D: PC1-PC3 con elipsoides (dos vistas)
    fig = plt.figure(figsize=(18, 8.4))
    for n_, (elev, azim) in enumerate(((20, -58), (12, 32))):
        ax = fig.add_subplot(1, 2, n_ + 1, projection="3d")
        for i in range(3):
            ax.scatter(*Zd[yd == i, :3].T, s=12, color=COLORES[i], alpha=0.18, edgecolor="none", depthshade=False)
        for j in range(len(m["Zc"])):
            ident = mapeo[j]
            x, y_, z_ = GD.elipsoide_3d(m, j, lam)
            ax.plot_surface(x, y_, z_, color=COLORES[ident], alpha=0.07, linewidth=0, shade=False)
            ax.plot_wireframe(x, y_, z_, color=COLORES[ident], alpha=0.28, linewidth=0.45, rstride=3, cstride=3)
            ax.scatter(*m["Zc"][j][:3], marker="X", s=300, color=COLORES[ident], edgecolor="black", linewidth=1.5, depthshade=False)
            ax.text(*m["Zc"][j][:3], f"  Identidad {ident + 1}", fontweight="bold", fontsize=9)
        for i in range(3):
            sel = verdad == i
            ax.scatter(*Z[sel & aceptada, :3].T, s=44, color=COLORES[i], edgecolor="black", linewidth=0.6, depthshade=False)
            ax.scatter(*Z[sel & ~aceptada, :3].T, s=58, facecolor="white", edgecolor=COLORES[i], linewidth=1.8, depthshade=False)
        ax.scatter(*Z[intr & ~aceptada, :3].T, marker="X", s=80, color=ROJO, edgecolor="black", linewidth=0.6, depthshade=False)
        ax.scatter(*Z[intr & aceptada, :3].T, marker="X", s=140, color="gold", edgecolor="black", linewidth=1.2, depthshade=False)
        ax.set_box_aspect(tuple(np.ptp(np.vstack([Zd[:, :3], Z[:, :3]]), axis=0)))
        ax.set_xlabel(f"PC1 ({ev[0]:.1f} %)")
        ax.set_ylabel(f"PC2 ({ev[1]:.1f} %)")
        ax.set_zlabel(f"PC3 ({ev[2]:.1f} %)")
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"Vista {n_ + 1}", fontsize=12.5, fontweight="bold")
    fig.legend(handles=leyenda(), loc="lower center", ncol=3, fontsize=10, frameon=False)
    fig.suptitle(f"Elipsoides de aceptación en PC1-PC3 ({ev[:3].sum():.1f} % de la varianza; el residuo Q completa la decisión)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(IH.SALIDA / "pca3d_correcto.png", dpi=130, bbox_inches="tight")
    print(f"lambda={lam:.2f} | k={m['k']} ({m['varianza_explicada'] * 100:.1f} % de la varianza) | autorizadas aceptadas {int((aut & aceptada).sum())}/{int(aut.sum())} | "
          f"intrusos rechazados {int((intr & ~aceptada).sum())}/{int(intr.sum())} | rechazadas que se ven dentro de la elipse: {int(engana.sum())}")
    if "--mostrar" in sys.argv:
        plt.show()


if __name__ == "__main__":
    main()
