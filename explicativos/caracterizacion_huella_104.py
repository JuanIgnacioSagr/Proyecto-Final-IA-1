"""Caracterizacion actual de las huellas: vector de 104 numeros = 72 de orientacion + 32 de energia de Gabor, normalizado.

    python explicativos/caracterizacion_huella_104.py       # crea explicativos/imagenes/caracterizacion_huella_104.png

Usa las funciones reales de src/caracterizar_huellas.py sobre una huella ya preprocesada:
  BLOQUE A  orientacion de las crestas (72): histograma global (24) + 4 cuadrantes de 12.
  BLOQUE B  energia de Gabor (32): 8 filtros orientados (0 a 157,5 grados) medidos en 4 regiones (2x2).
  NORMALIZACION  cada columna se centra y se divide por su desvio (calculados con el entrenamiento) y por 33: asi ningun numero domina.
Colores: azul = bloque de orientacion, naranja = bloque de Gabor (los mismos que en el resto de los graficos).
"""
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
import caracterizar_huellas as CH                                 # noqa: E402
import preprocesar as P                                           # noqa: E402

SALIDA = Path(__file__).resolve().parent / "imagenes"
AZUL, NARANJA, VERDE = (plt.cm.tab10(i) for i in (0, 1, 2))
HSV = plt.cm.hsv
plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]


def main():
    ruta = RAIZ / "dataset_huellas" / "procesadas" / "impresionPrimera" / "id1" / "primera_id1_imp1.png"
    binaria = P.a_gris(ruta)
    orient = CH.rasgo_orientacion(binaria)
    gabor = CH.rasgo_gabor_compacto(binaria)
    crudo = np.concatenate([orient, gabor]).astype(float)
    media, desvio = CH.cargar_normalizacion()
    norm = CH.normalizar(crudo, media, desvio)
    d = CH.cargar_dataset()
    Xn, y = d["X"], d["y"]
    t = CH._tinta(binaria)
    H, W = t.shape
    zona = CH._zona_huella(t)

    fig = plt.figure(figsize=(19, 12))
    gs = fig.add_gridspec(3, 12, height_ratios=[3.1, 1.6, 1.9], hspace=0.55, wspace=1.4)

    # ---- huella con rejilla 2x2 ----
    ax = fig.add_subplot(gs[0, 0:3])
    ax.imshow(binaria, cmap="gray")
    ax.axvline(W / 2, color=VERDE, lw=2.5)
    ax.axhline(H / 2, color=VERDE, lw=2.5)
    for (qx, qy, n) in [(0.25, 0.06, "1"), (0.75, 0.06, "2"), (0.25, 0.94, "3"), (0.75, 0.94, "4")]:
        ax.text(qx * W, qy * H, n, ha="center", va="center", fontsize=14, fontweight="bold",
                bbox=dict(boxstyle="circle,pad=0.25", fc="white", ec=VERDE, lw=2))
    ax.set_title("Huella procesada\ndividida en 4 regiones (2×2)", fontsize=12, fontweight="bold")
    ax.axis("off")

    # ---- bloque A: orientacion ----
    axg = fig.add_subplot(gs[0, 3:7])
    c = (np.arange(24) + 0.5) * 180 / 24
    axg.bar(c, orient[:24], width=180 / 24 * 0.9, color=[HSV(v / 180) for v in c], edgecolor="black", linewidth=0.4)
    axg.set_xlim(0, 180)
    axg.set_xticks([0, 45, 90, 135, 180])
    axg.set_xlabel("ángulo de las crestas (°)")
    axg.set_ylabel("fracción")
    axg.set_title("BLOQUE A · orientación global: 24 números", fontsize=11.5, fontweight="bold", color=AZUL)
    axg.grid(axis="y", alpha=0.25)
    sub = gs[1, 3:7].subgridspec(1, 4, wspace=0.45)
    for q in range(4):
        a = fig.add_subplot(sub[0, q])
        hq = orient[24 + 12 * q: 24 + 12 * (q + 1)]
        cc = (np.arange(12) + 0.5) * 180 / 12
        a.bar(cc, hq, width=15 * 0.88, color=[HSV(v / 180) for v in cc], edgecolor="black", linewidth=0.3)
        a.set_xticks([0, 90, 180])
        a.tick_params(labelsize=7)
        a.set_title(f"región {q + 1}", fontsize=9, color=VERDE, fontweight="bold")
    fig.text(0.435, 0.545, "BLOQUE A · orientación por región: 4 × 12 números (total del bloque: 72)", ha="center", fontsize=11.5,
             fontweight="bold", color=AZUL)

    # ---- bloque B: Gabor ----
    sub = gs[0, 7:12].subgridspec(2, 4, wspace=0.15, hspace=0.35)
    for i, k in enumerate(CH._GABOR):
        a = fig.add_subplot(sub[i // 4, i % 4])
        a.imshow(k, cmap="gray")
        a.set_title(f"{i * 22.5:g}°", fontsize=9)
        a.axis("off")
    fig.text(0.79, 0.915, "BLOQUE B · banco de 8 filtros de Gabor (cada uno detecta crestas en una dirección)", ha="center",
             fontsize=11, fontweight="bold", color=NARANJA)
    ax = fig.add_subplot(gs[1, 7:12])
    im = ax.imshow(gabor.reshape(4, 8), cmap="Oranges", aspect="auto")
    for i in range(4):
        for j in range(8):
            ax.text(j, i, f"{gabor.reshape(4, 8)[i, j]:.2f}", ha="center", va="center", fontsize=7.5)
    ax.set_xticks(range(8))
    ax.set_xticklabels([f"{j * 22.5:g}°" for j in range(8)], fontsize=8)
    ax.set_yticks(range(4))
    ax.set_yticklabels([f"región {i + 1}" for i in range(4)], fontsize=9)
    ax.set_title("energía de cada filtro en cada región: 4 × 8 = 32 números", fontsize=11, fontweight="bold", color=NARANJA)

    # ---- vector final ----
    ax = fig.add_subplot(gs[2, 0:12])
    ax.set_xlim(0, 104)
    ax.set_ylim(0, 3.6)
    m = max(abs(norm).max(), 1e-9)
    for i in range(104):
        col = plt.cm.coolwarm(0.5 + 0.5 * norm[i] / m)
        ax.add_patch(plt.Rectangle((i, 1.9), 1, 0.9, fc=col, ec="white", lw=0.3))
        ax.add_patch(plt.Rectangle((i, 0.9), 1, 0.6, fc=AZUL if i < 72 else NARANJA, ec="white", lw=0.3, alpha=0.85))
    ax.text(-0.8, 2.35, "vector\nnormalizado", ha="right", va="center", fontsize=9.5, fontweight="bold")
    ax.text(-0.8, 1.2, "bloque", ha="right", va="center", fontsize=9.5, fontweight="bold")
    ax.text(36, 0.35, "72 números de orientación", ha="center", fontsize=11, fontweight="bold", color=AZUL)
    ax.text(88, 0.35, "32 de Gabor", ha="center", fontsize=11, fontweight="bold", color=NARANJA)
    ax.text(52, 3.25, "Vector final de la huella: 72 + 32 = 104 números. Cada columna se normaliza: (valor − media) / desvío / 33, "
                      "con la media y el desvío del entrenamiento", ha="center", fontsize=11.5, fontweight="bold")
    ax.axis("off")

    fig.suptitle("Caracterización de una huella: 104 números (orientación de crestas + energía de Gabor)", fontsize=16, fontweight="bold", y=0.985)
    SALIDA.mkdir(parents=True, exist_ok=True)
    fig.savefig(SALIDA / "caracterizacion_huella_104.png", dpi=140, bbox_inches="tight")
    print("Guardado:", SALIDA / "caracterizacion_huella_104.png", "| dimension", len(norm))


if __name__ == "__main__":
    main()
