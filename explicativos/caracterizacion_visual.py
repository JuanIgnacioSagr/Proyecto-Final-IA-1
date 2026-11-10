"""Caracterizacion en imagenes: como se ven los numeros que describen una huella y un audio.

    python explicativos/caracterizacion_visual.py     # crea explicativos/imagenes/histograma_orientacion.png
                                                      #       y explicativos/imagenes/tramos_audio.png

HUELLA: se dibuja la huella procesada con una reticula; cada casillero muestra con una rayita hacia donde apuntan
las crestas (el color = el angulo, de 0 a 180 grados). Los histogramas cuentan cuanta huella hay en cada angulo:
uno global (24 barras) y uno por cuadrante (4 x 12 barras) = 72 numeros.
AUDIO: se dibuja la senal con sus tramos de 25 ms (uno cada 10 ms) y la matriz de 39 numeros por tramo
(13 MFCC, 13 delta, 13 delta-delta) que sale de ellos, y el vector fijo de 91 numeros.
"""
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402
import soundfile as sf                                            # noqa: E402
from matplotlib.patches import Rectangle                          # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
import caracterizar_audios as CA                                  # noqa: E402
import caracterizar_huellas as CH                                 # noqa: E402
import preprocesar as P                                           # noqa: E402

SALIDA = Path(__file__).resolve().parent / "imagenes"
AZUL, NARANJA, VERDE, ROJO = (plt.cm.tab10(i) for i in (0, 1, 2, 3))
plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
HSV = plt.cm.hsv


# =====================================================================================
#  HUELLA
# =====================================================================================
def huella(ruta=None):
    ruta = ruta or RAIZ / "dataset_huellas" / "procesadas" / "impresionPrimera" / "id1" / "primera_id1_imp1.png"
    binaria = P.a_gris(Path(ruta))
    t = CH._tinta(binaria)
    p = CH.PARAMS
    alfa, coh = P.campo_orientacion(cv2.GaussianBlur(t, (0, 0), p["suavizado_sigma"]), sigma=p["tensor_sigma"])
    peso = coh * t
    H, W = t.shape
    vector = CH.rasgo_orientacion(binaria)

    fig = plt.figure(figsize=(17, 9.6))
    gs = fig.add_gridspec(3, 6, height_ratios=[3.3, 1.5, 0.9], hspace=0.5, wspace=1.0)

    # ---- imagen con reticula y rayitas de orientacion ----
    ax = fig.add_subplot(gs[0, 0:2])
    ax.imshow(binaria, cmap="gray", alpha=0.55)
    nx, ny = 12, 16
    for gy in range(ny):
        for gx in range(nx):
            y0, y1, x0, x1 = gy * H // ny, (gy + 1) * H // ny, gx * W // nx, (gx + 1) * W // nx
            w = peso[y0:y1, x0:x1]
            if w.sum() < 1e-3 * w.size:
                continue
            c2, s2 = (w * np.cos(2 * alfa[y0:y1, x0:x1])).sum(), (w * np.sin(2 * alfa[y0:y1, x0:x1])).sum()
            ang = (np.arctan2(s2, c2) / 2) % np.pi
            fuerza = np.hypot(c2, s2) / (w.sum() + 1e-9)
            cx, cy, r = (x0 + x1) / 2, (y0 + y1) / 2, 0.40 * (x1 - x0) * (0.35 + 0.65 * fuerza)
            ax.plot([cx - r * np.cos(ang), cx + r * np.cos(ang)], [cy + r * np.sin(ang) * -1 * -1, cy - r * np.sin(ang)],
                    color=HSV(ang / np.pi), lw=2.6, solid_capstyle="round")
    for k in range(1, 2):
        ax.axvline(k * W / 2, color="black", lw=2.4)
        ax.axhline(k * H / 2, color="black", lw=2.4)
    for gx in range(1, nx):
        ax.axvline(gx * W / nx, color="white", lw=0.4, alpha=0.7)
    for gy in range(1, ny):
        ax.axhline(gy * H / ny, color="white", lw=0.4, alpha=0.7)
    ys, xs = np.where(t > 0.5)
    bx0, bx1, by0, by1 = xs.min() - 12, xs.max() + 12, ys.min() - 12, ys.max() + 12
    ax.set_xlim(bx0, bx1)
    ax.set_ylim(by1, by0)
    for (qx, qy, txt) in [(0.25, 0.0, "Q1"), (0.75, 0.0, "Q2"), (0.25, 1.0, "Q3"), (0.75, 1.0, "Q4")]:
        ax.text(bx0 + qx * (bx1 - bx0), by0 + qy * (by1 - by0) + (24 if qy == 0 else -24), txt, ha="center", fontsize=12, fontweight="bold", color="black",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.9))
    ax.set_title("Huella procesada con reticula:\ncada casillero = hacia donde apuntan las crestas", fontsize=11, fontweight="bold")
    ax.axis("off")

    # ---- barra de colores = angulo ----
    axc = fig.add_subplot(gs[0, 2])
    axc.imshow(np.linspace(0, 1, 256)[:, None], cmap="hsv", aspect="auto", extent=[0, 1, 180, 0])
    axc.set_xticks([])
    axc.set_yticks([0, 45, 90, 135, 180])
    axc.set_yticklabels(["0°", "45°", "90°", "135°", "180°"])
    axc.set_title("color =\nángulo", fontsize=10, fontweight="bold")

    # ---- histograma global (24 barras) ----
    axg = fig.add_subplot(gs[0, 3:6], projection=None)
    hg = vector[:24]
    cent = (np.arange(24) + 0.5) * 180 / 24
    axg.bar(cent, hg, width=180 / 24 * 0.9, color=[HSV(c / 180) for c in cent], edgecolor="black", linewidth=0.5)
    axg.set_xlim(0, 180)
    axg.set_xlabel("ángulo de las crestas (0° a 180°, 24 rangos de 7,5°)")
    axg.set_ylabel("fracción de la huella")
    axg.set_title("Histograma GLOBAL: 24 números\n(cuánta huella tiene crestas con cada orientación)", fontsize=11, fontweight="bold")
    axg.grid(axis="y", alpha=0.25)

    # ---- histogramas por cuadrante (4 x 12) ----
    for q in range(4):
        a = fig.add_subplot(gs[1, [0, 1, 2, 3][q] + (0 if q < 3 else 0): ([0, 1, 2, 3][q] + 1)]) if False else None
    posiciones = [(1, 0, 2), (1, 2, 4), (1, 4, 6)]
    cols = [(0, 1), (1, 2), (2, 3), (3, 4)]
    colocar = [gs[1, 0:2], gs[1, 2:4], gs[1, 4:6]]
    nombres = ["Q1 (arriba-izquierda)", "Q2 (arriba-derecha)", "Q3 (abajo-izquierda)", "Q4 (abajo-derecha)"]
    sub = fig.add_gridspec(1, 4, left=0.125, right=0.9, top=0.405, bottom=0.255, wspace=0.28)
    for q in range(4):
        a = fig.add_subplot(sub[0, q])
        hq = vector[24 + 12 * q: 24 + 12 * (q + 1)]
        c = (np.arange(12) + 0.5) * 180 / 12
        a.bar(c, hq, width=180 / 12 * 0.88, color=[HSV(v / 180) for v in c], edgecolor="black", linewidth=0.5)
        a.set_xlim(0, 180)
        a.set_xticks([0, 90, 180])
        a.set_title(nombres[q] + "\n12 números", fontsize=9.5, fontweight="bold")
        a.grid(axis="y", alpha=0.25)

    # ---- el vector de 72 como tira de colores ----
    ax = fig.add_subplot(gs[2, :])
    ax.set_xlim(0, 72)
    ax.set_ylim(0, 1)
    for i in range(72):
        if i < 24:
            ang = (i + 0.5) * 180 / 24
        else:
            ang = (((i - 24) % 12) + 0.5) * 180 / 12
        ax.add_patch(Rectangle((i, 0.25), 1, 0.5, fc=HSV(ang / 180), ec="white", lw=0.6, alpha=0.35 + 0.65 * min(1, vector[i] / (vector.max() + 1e-9))))
    for (a0, a1, txt) in [(0, 24, "global: 24"), (24, 36, "Q1: 12"), (36, 48, "Q2: 12"), (48, 60, "Q3: 12"), (60, 72, "Q4: 12")]:
        ax.plot([a0 + 0.1, a1 - 0.1], [0.15, 0.15], color="black", lw=2)
        ax.text((a0 + a1) / 2, 0.02, txt, ha="center", fontsize=10, fontweight="bold")
    ax.text(36, 0.88, "Vector final de la huella: 24 + 4 × 12 = 72 números (cada casillero, más intenso = valor más grande)",
            ha="center", fontsize=11, fontweight="bold")
    ax.axis("off")

    fig.suptitle("Caracterización de una huella: histograma de orientación de las crestas", fontsize=16, fontweight="bold")
    SALIDA.mkdir(parents=True, exist_ok=True)
    fig.savefig(SALIDA / "histograma_orientacion.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("Guardado:", SALIDA / "histograma_orientacion.png")


# =====================================================================================
#  AUDIO
# =====================================================================================
def audio(ruta=None):
    ruta = ruta or sorted((RAIZ / "dataset_audios" / "normalizados" / "reconocer").glob("*.wav"))[3]
    x, fs = sf.read(str(ruta), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    seq, vec = CA.caracterizar_audio(x, fs)
    p = CA.PARAMS
    t = np.arange(len(x)) / fs
    T = len(seq)
    n_t, salto = p["tramo_muestras"], p["salto_muestras"]

    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.1, 1.0, 2.6, 0.9], width_ratios=[3.2, 1.35], hspace=0.62, wspace=0.18)

    # ---- senal completa ----
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t, x, color=AZUL, lw=0.8)
    z0, z1 = 0.30 * len(x) / fs, 0.30 * len(x) / fs + 0.12
    ax.axvspan(z0, z1, color=NARANJA, alpha=0.35)
    ax.text(z1 + 0.01, ax.get_ylim()[1] * 0.6, "zoom", color=NARANJA, fontweight="bold")
    ax.set_title(f"Audio normalizado ({len(x) / fs:.2f} s, 16 kHz): la voz dice «reconocer»", fontsize=11, fontweight="bold")
    ax.set_xlabel("tiempo (s)")
    ax.set_xlim(0, t[-1])

    # ---- zoom con tramos de 25 ms cada 10 ms ----
    ax = fig.add_subplot(gs[1, 0])
    i0, i1 = int(z0 * fs), int(z1 * fs)
    ax.plot(t[i0:i1], x[i0:i1], color="black", lw=1.0)
    k = 0
    s0 = i0
    while s0 + n_t <= i1 + salto:
        fila = k % 3
        ax.add_patch(Rectangle((s0 / fs, -1.05 + fila * 0.0), n_t / fs, 2.1, fc=plt.cm.tab10(k % 10), alpha=0.0))
        ax.add_patch(Rectangle((s0 / fs, 1.12 + fila * 0.32), n_t / fs, 0.26, fc=plt.cm.tab10(k % 10), alpha=0.85, ec="black", lw=0.5))
        k += 1
        s0 += salto
    ax.set_ylim(x[i0:i1].min() * 1.2, ax.get_ylim()[1] + 1.1 * (x[i0:i1].max() - x[i0:i1].min()) + 0.2)
    ax.set_title("Zoom de 120 ms: los tramos de 25 ms se solapan, uno nuevo cada 10 ms", fontsize=11, fontweight="bold")
    ax.set_xlabel("tiempo (s)")
    ax.set_yticks([])

    # ---- matriz 39 x tramos ----
    ax = fig.add_subplot(gs[2, 0])
    im = ax.imshow(seq.T, aspect="auto", cmap="coolwarm", vmin=-np.abs(seq).max() * 0.6, vmax=np.abs(seq).max() * 0.6,
                   extent=[0, T * 0.01, 39, 0], interpolation="nearest")
    for y in (13, 26):
        ax.axhline(y, color="black", lw=1.6)
    for (ya, txt, c) in [(6.5, "13 MFCC\n(timbre)", AZUL), (19.5, "13 Δ\n(velocidad)", NARANJA), (32.5, "13 ΔΔ\n(aceleración)", VERDE)]:
        ax.text(-0.012, ya, txt, ha="right", va="center", fontsize=10, fontweight="bold", color=c, transform=ax.get_yaxis_transform())
    col = T // 2
    ax.add_patch(Rectangle((col * 0.01, 0), 0.01, 39, fill=False, ec="black", lw=2.6))
    ax.annotate("un TRAMO = una columna = 39 números", xy=((col + 1) * 0.01, 19.5), xytext=(0.62, 19.5), va="center",
                fontsize=11, fontweight="bold", ha="left", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black"),
                arrowprops=dict(arrowstyle="-|>", lw=2))
    ax.set_xlabel("tiempo (s)  —  cada columna = un tramo (uno cada 10 ms)")
    ax.set_yticks([])
    ax.set_title(f"Secuencia de MFCC: matriz de {T} tramos × 39 números (la usa el DTW)", fontsize=11, fontweight="bold", pad=6)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("valor del coeficiente")

    # ---- el vector de 91 ----
    ax = fig.add_subplot(gs[3, :])
    m = np.abs(vec).max()
    ax.imshow(vec[None, :], aspect="auto", cmap="coolwarm", vmin=-m * 0.6, vmax=m * 0.6, extent=[0, 91, 0, 1])
    for xline in (13, 26, 39, 52):
        ax.axvline(xline, color="black", lw=1.4)
    for (a0, a1, txt, c) in [(0, 13, "media 1.er cuarto", AZUL), (13, 26, "media 2.º cuarto", AZUL), (26, 39, "media 3.er cuarto", AZUL),
                             (39, 52, "media 4.º cuarto", AZUL), (52, 91, "desvío de los 39 coeficientes (cuánto varía en el audio)", VERDE)]:
        ax.text((a0 + a1) / 2, 1.08, txt, ha="center", fontsize=9.5, fontweight="bold", color=c)
    ax.set_yticks([])
    ax.set_xlabel("posición en el vector")
    ax.set_title("Vector fijo del audio: 4 × 13 + 39 = 91 números (la usa la distancia euclídea)", fontsize=11, fontweight="bold", pad=24)

    # ---- panel lateral: una columna como barras ----
    ax = fig.add_subplot(gs[1:3, 1])
    cols = [AZUL] * 13 + [NARANJA] * 13 + [VERDE] * 13
    ax.barh(np.arange(39), seq[col], color=cols, edgecolor="black", linewidth=0.4)
    ax.invert_yaxis()
    ax.set_title("Los 39 números de UN tramo", fontsize=11, fontweight="bold")
    ax.set_yticks([6, 19, 32])
    ax.set_yticklabels(["MFCC", "Δ", "ΔΔ"])
    ax.axvline(0, color="black", lw=0.8)
    ax.grid(axis="x", alpha=0.25)

    ax = fig.add_subplot(gs[0, 1])
    ax.axis("off")
    ax.text(0, 0.9, "Cada tramo:\n pre-énfasis → ventana → FFT\n → 40 filtros Mel → log → DCT\n → 13 MFCC (+ Δ y ΔΔ)", fontsize=10.5, va="top", linespacing=1.6)

    fig.suptitle("Caracterización de un audio: de la señal a los números", fontsize=16, fontweight="bold")
    fig.savefig(SALIDA / "tramos_audio.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("Guardado:", SALIDA / "tramos_audio.png", f"({T} tramos, vector {vec.shape[0]})")


if __name__ == "__main__":
    SALIDA.mkdir(parents=True, exist_ok=True)
    huella()
    audio()
