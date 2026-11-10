"""El umbral de aceptacion en distancias T2: del percentil 95 de las autorizadas al umbral calibrado con intrusos.

    python explicativos/umbral_t2.py       # crea Fotos/umbral_t2.png

Dos paneles con la MISMA escala de T2 (distancia de Mahalanobis al cuadrado a la identidad mas cercana, en las k componentes):
  ARRIBA  autorizadas: entrenamiento y prueba (histograma: cada rectangulo es un intervalo de T2, alto = % de las huellas del grupo).
  ABAJO   intrusos: calibracion (los que fijan el umbral) y prueba (los que solo evaluan).
Lineas: el LIMITE inicial (percentil 95 de las autorizadas de validacion, puntaje 1) y el UMBRAL aplicado = lambda x limite.
El umbral es el mayor valor que no acepta a ningun intruso de calibracion, asi que queda justo por debajo del mas cercano.
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
SALIDA = AQUI / "imagenes" / "umbral_t2.png"
plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]


def curva(ax, t2, color, etiqueta, bordes):
    """Histograma: un rectangulo por intervalo de T2 (mismos intervalos en todos los grupos), alto = % de las huellas del grupo."""
    cuenta, _ = np.histogram(t2, bins=bordes)
    ax.bar(bordes[:-1], 100 * cuenta / len(t2), width=np.diff(bordes), align="edge", color=color, alpha=0.55,
           edgecolor="black", linewidth=0.8, label=etiqueta)


def main():
    m, mapeo, lam, _ = IH.modelo()
    datos = np.load(KH.CARPETA / "dataset.npz", allow_pickle=True)
    parte = PH.particion([str(a) for a in datos["archivo"]])
    T2d = DH.estadisticos(datos["X"].astype(float), m)[0].min(axis=1)
    Xi, cal, nombres = KH.cargar_intrusos()
    T2i = DH.estadisticos(Xi, m)[0].min(axis=1)

    limite = m["t2_limite"]                       # percentil 95 de las autorizadas de validacion
    umbral = lam * limite                         # T2 maximo que se acepta
    i_cerca = int(np.argmin(np.where(cal, T2i, np.inf)))          # intruso de calibracion mas cercano
    t2_cerca = float(T2i[i_cerca])
    grupos = {"ent": T2d[parte == "entrenamiento"], "pru": T2d[parte == "prueba"], "cal": T2i[cal], "ipr": T2i[~cal]}
    acep = {k: int((v <= umbral).sum()) for k, v in grupos.items()}

    todo = np.concatenate(list(grupos.values()))
    # intervalos de ancho constante en escala log (5 por decada) que ARRANCAN en el umbral: ningun rectangulo cruza la linea negra
    k0, k1 = int(np.floor(5 * np.log10(todo.min() / umbral))) - 1, int(np.ceil(5 * np.log10(todo.max() / umbral))) + 1
    x = umbral * 10 ** (np.arange(k0, k1 + 1) / 5)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 8.2), sharex=True, gridspec_kw={"hspace": 0.12})
    curva(a1, grupos["ent"], "#1f77b4", f"Entrenamiento (n = {len(grupos['ent'])}): {acep['ent']} aceptadas", x)
    curva(a1, grupos["pru"], "#2ca02c", f"Prueba (n = {len(grupos['pru'])}): {acep['pru']} aceptadas", x)
    curva(a2, grupos["cal"], "#d62728", f"Calibración (n = {len(grupos['cal'])}): {acep['cal']} aceptados", x)
    curva(a2, grupos["ipr"], "#e377c2", f"Prueba (n = {len(grupos['ipr'])}): {acep['ipr']} aceptados", x)

    for ax in (a1, a2):
        ax.axvspan(x[0], umbral, color="#2ca02c", alpha=0.07)
        ax.axvspan(umbral, x[-1], color="#d62728", alpha=0.06)
        ax.axvline(limite, color="#1f77b4", lw=2.2, ls="--")
        ax.axvline(umbral, color="black", lw=2.6, ls="--")
        ax.set_xscale("log")
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(bottom=0)
        ax.set_ylabel("% de las huellas del grupo")
    a2.axvline(t2_cerca, color="#d62728", lw=1.4, ls=":")

    a1.set_ylim(0, a1.get_ylim()[1] * 1.4)
    top = a1.get_ylim()[1]
    a1.annotate("", xy=(umbral, top * 0.80), xytext=(limite, top * 0.80), arrowprops=dict(arrowstyle="-|>", lw=2.2, color="#555555"))
    a1.text(limite * 0.95, top * 0.80, f"× {lam:.2f}", ha="right", va="center", fontsize=13, fontweight="bold", color="#555555")
    # etiquetas de las lineas, por encima del grafico para que no tapen las curvas
    etiquetas = ((f"límite inicial: percentil 95 de las\nautorizadas de validación, T² = {limite:.2f}", limite * 0.95, "right", "#1f77b4"),
                 (f"umbral aplicado = λ × límite\n= {lam:.2f} × {limite:.2f} = {umbral:.2f}", umbral * 1.05, "left", "black"))
    for ax_, (txt, xx, ha, col) in ((a1, e) for e in etiquetas):
        ax_.annotate(txt, xy=(xx, 1.0), xycoords=("data", "axes fraction"), xytext=(0, 4), textcoords="offset points",
                     ha=ha, va="bottom", fontsize=10, fontweight="bold", color=col, annotation_clip=False)
    a1.text(x[0] * 1.2, top * 0.97, "ACEPTADA (T² ≤ umbral)", color="#2ca02c", fontweight="bold", va="top")
    a1.text(x[-1] / 1.1, top * 0.97, "RECHAZADA (T² > umbral)", color="#d62728", fontweight="bold", ha="right", va="top")
    a1.legend(loc="upper right", bbox_to_anchor=(0.99, 0.86), fontsize=10)
    a2.legend(loc="upper left", fontsize=10)
    top2 = a2.get_ylim()[1]
    a2.annotate(f"intruso de calibración más cercano\nT² = {t2_cerca:.2f} (puntaje {t2_cerca / limite:.2f})\n→ el umbral queda justo por debajo",
                xy=(t2_cerca, top2 * 0.03), xytext=(0.13, top2 * 0.55), fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", lw=1.5), bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black"))
    a1.set_title("Autorizadas", loc="left", fontsize=11.5, fontweight="bold", pad=34)
    a2.set_title("Intrusos", loc="left", fontsize=11.5, fontweight="bold")
    a2.set_xlabel("T² a la identidad más cercana (distancia de Mahalanobis al cuadrado, escala log)")
    fig.suptitle(f"Umbral de aceptación en distancias T²: del percentil 95 ({limite:.2f}) al umbral calibrado con intrusos ({umbral:.2f})",
                 fontsize=13, fontweight="bold", y=1.03)
    SALIDA.parent.mkdir(exist_ok=True)
    fig.savefig(SALIDA, dpi=150, bbox_inches="tight")
    print(f"limite T2 = {limite:.3f} | umbral = {lam:.2f} x {limite:.3f} = {umbral:.3f} | intruso de calibracion mas cercano {nombres[i_cerca]}: T2 = {t2_cerca:.3f}")
    print("aceptadas:", {k: f"{acep[k]}/{len(v)}" for k, v in grupos.items()})


if __name__ == "__main__":
    main()
