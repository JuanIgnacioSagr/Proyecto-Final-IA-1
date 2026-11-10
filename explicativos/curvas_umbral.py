"""Curvas FRR y FAR en funcion del umbral lambda: asi se calibra y asi se verifica.

    python explicativos/curvas_umbral.py       # crea explicativos/imagenes/curvas_umbral.png

  FRR (falso rechazo)    = % de huellas autorizadas con puntaje s > lambda   (baja al agrandar lambda)
  FAR (falsa aceptacion) = % de huellas intrusas con puntaje s <= lambda     (sube al agrandar lambda)
  s = max(T2/limite, Q/limite): ver src/decision_huellas.py (1 = percentil 95 de las autorizadas de validacion).
Panel 1: CALIBRACION. Autorizadas de validacion + intrusos 01-25. El lambda aplicado es el MAYOR con el que ningun intruso de
         calibracion es aceptado (el mismo criterio con el que se fijo R antes).
Panel 2: PRUEBA. Autorizadas de prueba + intrusos 26-50 (no se usaron para elegir lambda): muestra la FAR y la FRR reales.
Panel 3: lo subido a mano (autorizadas e intrusos de dataset_huellas/pruebas_manuales/), tambien fuera de la calibracion.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import intrusos_huellas as IH                                     # noqa: E402

KH, DH, PH = IH.KH, IH.DH, IH.PH
AZUL, ROJO = plt.cm.tab10(0), plt.cm.tab10(3)
plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]


def panel(ax, grilla, s_aut, s_int, titulo, lam, nota=None):
    frr = np.array([(s_aut > r).mean() for r in grilla])
    far = np.array([(s_int <= r).mean() for r in grilla])
    ax.plot(grilla, frr * 100, color=AZUL, lw=2.8, label="FRR: autorizadas rechazadas")
    ax.plot(grilla, far * 100, color=ROJO, lw=2.8, label="FAR: intrusos aceptados")
    ax.axvline(lam, color="black", lw=1.8, ls="--")
    i = int(np.argmin(np.abs(grilla - lam)))
    ax.scatter([lam], [far[i] * 100], s=110, color="gold", edgecolor="black", linewidth=1.5, zorder=6)
    ax.annotate(f"λ aplicado = {lam:.2f}\nFRR {frr[i] * 100:.1f} % · FAR {far[i] * 100:.1f} %", xy=(lam, max(far[i], frr[i]) * 100),
                xytext=(0.45, 0.55), textcoords="axes fraction", fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", lw=1.5), bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black"))
    ax.set_xscale("log")
    ax.set_xlabel("umbral λ (escala log)")
    ax.set_ylabel("error (%)")
    ax.set_ylim(-2, 102)
    ax.set_title(titulo, fontsize=11.5, fontweight="bold")
    ax.grid(alpha=0.25, which="both")
    if nota:
        ax.text(0.02, 0.97, nota, transform=ax.transAxes, va="top", ha="left", fontsize=9, color="#374151")
    return frr[i], far[i]


def main():
    m, mapeo, lam, _ = IH.modelo()
    datos = np.load(KH.CARPETA / "dataset.npz", allow_pickle=True)
    parte = PH.particion([str(a) for a in datos["archivo"]])
    s_dataset = DH.decidir(datos["X"].astype(float), m, lam)[1]
    Xi, cal, _ = KH.cargar_intrusos()
    s_i = DH.decidir(Xi, m, lam)[1]
    X, verdad, origen = IH.cargar()
    s_h = DH.decidir(X, m, lam)[1]
    mano = origen == "subida a mano"

    grilla = np.round(np.geomspace(0.3, 200, 400), 3)
    fig, axs = plt.subplots(1, 3, figsize=(19, 5.8))
    panel(axs[0], grilla, s_dataset[parte == "validacion"], s_i[cal],
          f"1) Calibración: {int((parte == 'validacion').sum())} autorizadas (validación) y {int(cal.sum())} intrusos",
          lam, "el λ aplicado es el mayor\nque no deja entrar a ningún\nintruso de calibración")
    frr_t, far_t = panel(axs[1], grilla, s_dataset[parte == "prueba"], s_i[~cal],
                         f"2) Prueba: {int((parte == 'prueba').sum())} autorizadas y {int((~cal).sum())} intrusos (no usados para elegir λ)",
                         lam, "FAR y FRR reales con huellas\nque el umbral no vio")
    panel(axs[2], grilla, s_h[mano & (verdad < 3)], s_h[mano & (verdad == 3)],
          f"3) Subidas a mano: {int((mano & (verdad < 3)).sum())} autorizadas y {int((mano & (verdad == 3)).sum())} intrusos", lam,
          "pocas huellas: curvas escalonadas")
    fig.suptitle("¿Dónde se elige el umbral λ y cómo se comporta con huellas nuevas?", fontsize=15, fontweight="bold")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=11, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    IH.SALIDA.mkdir(parents=True, exist_ok=True)
    fig.savefig(IH.SALIDA / "curvas_umbral.png", dpi=150, bbox_inches="tight")
    print(f"lambda aplicado {lam:.2f} | prueba: FRR {frr_t * 100:.1f} %, FAR {far_t * 100:.1f} % ({int((s_i[~cal] <= lam).sum())} de {int((~cal).sum())} intrusos aceptados)")
    for L in (2, 3, 4, 5, lam):
        print(f"  si lambda = {L:.2f}: FRR prueba {(s_dataset[parte == 'prueba'] > L).mean() * 100:.1f} % | FAR prueba {(s_i[~cal] <= L).mean() * 100:.1f} % | "
              f"FRR a mano {(s_h[mano & (verdad < 3)] > L).mean() * 100:.0f} %")


if __name__ == "__main__":
    main()
