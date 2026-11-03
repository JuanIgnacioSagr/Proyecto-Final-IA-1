"""Curvas FRR y FAR en funcion del radio R: donde se cruzan se elige el umbral.

    python explicativos/curvas_umbral.py       # crea explicativos/imagenes/curvas_umbral.png

  FRR (falso rechazo)    = % de huellas autorizadas con distancia > R   (baja al agrandar R)
  FAR (falsa aceptacion) = % de huellas intrusas con distancia <= R     (sube al agrandar R)
Panel 1: calibracion con intrusos SIMULADOS (una identidad afuera), la que dio R = 0,31 (viene de metricas_kmeans.json).
Panel 2: con huellas REALES: autorizadas (prueba del dataset + subidas a mano) e intrusos (a mano + generados con Anguli).
Panel 3: solo lo que se probo a mano (15 autorizadas y 6 intrusas), donde 0,42 es el mayor radio sin aceptar intrusos.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import intrusos_huellas as IH                                     # noqa: E402

KH = IH.KH
AZUL, ROJO = plt.cm.tab10(0), plt.cm.tab10(3)
VERDE = plt.cm.tab10(2)
plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]


def cruce(radios, frr, far):
    i = int(np.argmin(np.abs(frr - far)))
    return radios[i], frr[i], far[i]


def panel(ax, radios, frr, far, titulo, marcar=(0.31, 0.42), nota=None):
    ax.plot(radios, frr * 100, color=AZUL, lw=2.8, label="FRR: autorizadas rechazadas")
    ax.plot(radios, far * 100, color=ROJO, lw=2.8, label="FAR: intrusos aceptados")
    r, f1, f2 = cruce(radios, frr, far)
    ax.scatter([r], [(f1 + f2) / 2 * 100], s=170, color="gold", edgecolor="black", linewidth=1.5, zorder=6)
    ax.annotate(f"se cruzan en R = {r:.2f}\n(FRR {f1 * 100:.0f} % · FAR {f2 * 100:.0f} %)", xy=(r, (f1 + f2) / 2 * 100),
                xytext=(radios.min() + 0.03 * (radios.max() - radios.min()), 9), fontsize=10, fontweight="bold", arrowprops=dict(arrowstyle="-|>", lw=1.6),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black"))
    for m, c, txt in [(0.31, VERDE, "R = 0,31"), (0.42, "black", "R = 0,42 (aplicado)")]:
        if m in marcar:
            ax.axvline(m, color=c, lw=1.6, ls="--")
            ax.text(m - 0.004 * (radios.max() - radios.min()) / 0.5, 99, txt, rotation=90, va="top", ha="right", fontsize=9, color=c, fontweight="bold")
    ax.set_xlabel("radio R (umbral de aceptación)")
    ax.set_ylabel("error (%)")
    ax.set_ylim(-2, 102)
    ax.set_xlim(radios.min(), radios.max())
    ax.set_title(titulo, fontsize=11.5, fontweight="bold")
    ax.grid(alpha=0.25)
    if nota:
        ax.text(0.98, 0.5, nota, transform=ax.transAxes, va="center", ha="right", fontsize=9, color="#374151")


def main():
    modelo = np.load(KH.RUTA_MODELO, allow_pickle=False)
    C, mapeo = modelo["centroides"], modelo["mapeo"].astype(int)
    X, verdad, origen = IH.cargar()
    _, d, _ = IH.decidir(X, C, mapeo, 1.0)
    cal = json.loads((KH.CARPETA / "metricas_kmeans.json").read_text(encoding="utf-8"))["calibracion"]
    rs, fr, fa = np.array(cal["radios"]), np.array(cal["frr"]), np.array(cal["far"])

    grilla = np.round(np.arange(0.10, 0.851, 0.01), 2)
    aut, intr = verdad < 3, verdad == 3
    frr_real = np.array([(d[aut] > r).mean() for r in grilla])
    far_real = np.array([(d[intr] <= r).mean() for r in grilla])
    mano = origen == "subida a mano"
    frr_m = np.array([(d[aut & mano] > r).mean() for r in grilla])
    far_m = np.array([(d[intr & mano] <= r).mean() for r in grilla])

    fig, axs = plt.subplots(1, 3, figsize=(19, 5.8))
    panel(axs[0], rs, fr, fa, "1) Intrusos SIMULADOS\n(una identidad afuera del entrenamiento)", marcar=(0.31,),
          nota="origen del R = 0,31 inicial")
    panel(axs[1], grilla, frr_real, far_real, f"2) Huellas REALES: {int(aut.sum())} autorizadas y {int(intr.sum())} intrusos",
          nota="autorizadas: prueba + a mano\nintrusos: a mano + Anguli")
    panel(axs[2], grilla, frr_m, far_m, f"3) Solo lo probado a mano: {int((aut & mano).sum())} autorizadas y {int((intr & mano).sum())} intrusos",
          nota="con poquitos datos las curvas\nson escalonadas")
    fig.suptitle("¿Dónde se cruzan FRR y FAR? Así se elige el umbral de rechazo", fontsize=15, fontweight="bold")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=11, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    IH.SALIDA.mkdir(parents=True, exist_ok=True)
    fig.savefig(IH.SALIDA / "curvas_umbral.png", dpi=150, bbox_inches="tight")
    for nombre, g, a, b in (("simulado", rs, fr, fa), ("real", grilla, frr_real, far_real), ("a mano", grilla, frr_m, far_m)):
        r, a1, b1 = cruce(g, a, b)
        print(f"{nombre:<9} cruce en R = {r:.2f}  (FRR {a1 * 100:.1f} %, FAR {b1 * 100:.1f} %)")
    i31, i42 = int(np.where(grilla == 0.31)[0][0]), int(np.where(grilla == 0.42)[0][0])
    print(f"real: R=0.31 FRR {frr_real[i31] * 100:.1f} % FAR {far_real[i31] * 100:.1f} % | R=0.42 FRR {frr_real[i42] * 100:.1f} % FAR {far_real[i42] * 100:.1f} %")
    print(f"a mano: R=0.42 FRR {frr_m[i42] * 100:.1f} % FAR {far_m[i42] * 100:.1f} %; mayor R con FAR=0:",
          grilla[far_m == 0].max())


if __name__ == "__main__":
    main()
