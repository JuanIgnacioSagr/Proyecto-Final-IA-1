"""DTW con audios REALES: cada tramo es un vector de 13 numeros (MFCC), no un numero suelto.

    python explicativos/dtw_real.py      # crea explicativos/imagenes/dtw_real.png

Muestra, con dos audios reales del comando "reconocer" (de personas distintas y con distinta duracion):
  - cada audio como una matriz (13 coeficientes x tramos): cada COLUMNA es un tramo;
  - la tabla de costos: la celda (i, j) es la distancia euclidea ENTRE LOS 13 NUMEROS del tramo i de A y del tramo j de B
    (se usan todos los componentes); la banda de Sakoe-Chiba marca hasta donde se permite alinear;
  - el camino de menor costo (el emparejamiento que elige el DTW);
  - un tramo de A con los tramos de B que le tocaron: se ve que se comparan los 13 coeficientes completos.
Usa las mismas funciones que el KNN (src/knn_audios.py): preparar_secuencia (13 MFCC, media restada, 1 tramo de cada 2) y dtw.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
import caracterizar_audios as CA                                  # noqa: E402
import knn_audios as KA                                           # noqa: E402

SALIDA = Path(__file__).resolve().parent / "imagenes"
AZUL, NARANJA, VERDE, ROJO = (plt.cm.tab10(i) for i in (0, 1, 2, 3))
plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]


def camino_dtw(a, b, banda=KA.BANDA):
    """Misma tabla que knn_audios.dtw, pero devolviendo tambien el camino (para dibujarlo)."""
    n, m = len(a), len(b)
    local = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2))
    ancho = max(int(banda * max(n, m)), abs(n - m))
    acum = np.full((n + 1, m + 1), np.inf)
    acum[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - ancho), min(m, i + ancho) + 1):
            acum[i, j] = local[i - 1, j - 1] + min(acum[i - 1, j - 1], acum[i - 1, j], acum[i, j - 1])
    i, j, cam = n, m, [(n - 1, m - 1)]
    while (i, j) != (1, 1):
        opc = [(acum[i - 1, j - 1], (i - 1, j - 1)), (acum[i - 1, j], (i - 1, j)), (acum[i, j - 1], (i, j - 1))]
        i, j = min(opc, key=lambda t: t[0])[1]
        cam.append((i - 1, j - 1))
    return local, acum[1:, 1:], cam[::-1], ancho, acum[n, m] / (n + m)


def elegir_audios():
    X, y, archivos, sesion = CA.cargar_dataset()
    seqs, _, _ = CA.cargar_secuencias()
    idx = [i for i in range(len(y)) if int(y[i]) == 0]
    por_sesion = {}
    for i in idx:
        por_sesion.setdefault(sesion[i], i)
    a_i = por_sesion.get("1_normal", idx[0])
    b_i = next((i for s, i in por_sesion.items() if s not in ("1_normal",) and len(seqs[i]) != len(seqs[a_i])), idx[1])
    return (archivos[a_i], sesion[a_i], KA.preparar_secuencia(seqs[a_i])), (archivos[b_i], sesion[b_i], KA.preparar_secuencia(seqs[b_i]))


def main():
    (fa, sa, A), (fb, sb, B) = elegir_audios()
    local, acum, cam, ancho, valor = camino_dtw(A, B)
    n, m = len(A), len(B)
    print(f"A: {Path(fa).name} ({sa}) {n} tramos | B: {Path(fb).name} ({sb}) {m} tramos | DTW = {valor:.3f} | banda = {ancho} tramos")

    fig = plt.figure(figsize=(17, 10.5))
    gs = fig.add_gridspec(3, 3, width_ratios=[1.35, 1.0, 1.0], height_ratios=[1, 1, 1.5], hspace=0.55, wspace=0.28)
    lim = np.abs(np.vstack([A, B])).max() * 0.7

    for fila, (mat, nombre, sesn, color) in enumerate(((A, "A", sa, AZUL), (B, "B", sb, NARANJA))):
        ax = fig.add_subplot(gs[fila, 0])
        ax.imshow(mat.T, aspect="auto", cmap="coolwarm", vmin=-lim, vmax=lim, interpolation="nearest")
        ax.set_title(f"Audio {nombre} ({sesn[2:]}): {len(mat)} tramos × 13 coeficientes", fontsize=11, fontweight="bold", color=color)
        ax.set_ylabel("coeficiente MFCC (1 a 13)")
        ax.set_xlabel("tramo (tiempo)")

    # ---- tabla de costos ----
    ax = fig.add_subplot(gs[2, 0])
    mostrar = np.where(np.isfinite(acum), local, np.nan)
    ax.imshow(mostrar.T, origin="lower", cmap="Blues", aspect="auto", interpolation="nearest")
    ax.plot([c[0] for c in cam], [c[1] for c in cam], color=ROJO, lw=2.8, label="camino de menor costo")
    ii = np.arange(n)
    ax.plot(ii, np.clip(ii + ancho, 0, m - 1), color="gray", lw=1.2, ls=":")
    ax.plot(ii, np.clip(ii - ancho, 0, m - 1), color="gray", lw=1.2, ls=":", label="banda permitida")
    veces = np.bincount([c[0] for c in cam], minlength=n)        # con cuantos tramos de B se empareja cada tramo de A
    i0 = int(np.argmax(veces))                                   # se muestra el tramo de A que se empareja con MAS tramos de B
    js = [c[1] for c in cam if c[0] == i0]
    for j in js:
        ax.scatter([i0], [j], s=90, facecolor="none", edgecolor=VERDE, linewidth=2.4, zorder=5)
    ax.set_xlabel("tramo de A")
    ax.set_ylabel("tramo de B")
    ax.set_title("Tabla de costos: celda (i, j) = distancia entre los 13 números\ndel tramo i de A y del tramo j de B", fontsize=10.5, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8.5)

    # ---- un tramo de A contra los tramos de B que le tocaron ----
    ax = fig.add_subplot(gs[0:2, 1:3])
    k = np.arange(1, 14)
    ancho_b = 0.8 / (len(js) + 1)
    ax.bar(k - 0.4 + ancho_b / 2, A[i0], ancho_b, color=AZUL, label=f"tramo {i0} de A", edgecolor="black", linewidth=0.4)
    for q, j in enumerate(js):
        ax.bar(k - 0.4 + ancho_b * (q + 1.5), B[j], ancho_b, color=plt.cm.Oranges(0.45 + 0.4 * q / max(1, len(js) - 1)),
               label=f"tramo {j} de B (d = {local[i0, j]:.2f})", edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(k)
    ax.set_xlabel("coeficiente MFCC (los 13 se usan en cada comparación)")
    ax.set_ylabel("valor del coeficiente")
    ax.set_title(f"El tramo {i0} de A se compara con {len(js)} tramo(s) de B: en cada par se usan los 13 coeficientes\n"
                 r"$d(i,j)=\sqrt{\sum_{c=1}^{13}(a_{i,c}-b_{j,c})^2}$", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.25)

    ax = fig.add_subplot(gs[2, 1:3])
    ax.axis("off")
    ax.text(0, 0.98, "Cómo se lee", fontsize=13, fontweight="bold", va="top")
    ax.text(0, 0.84,
            "• Cada tramo es un vector de 13 números. La celda (i, j) de la tabla usa los 13 (distancia euclídea entre los dos vectores): no se pierde ningún componente.\n"
            "• El DTW no compara un número con otro número: decide QUÉ TRAMO de A se empareja con QUÉ TRAMO de B.\n"
            "• Para decidir, llena la tabla (solo dentro de la banda, cerca de la diagonal) y toma el camino de menor costo acumulado:\n"
            "   en cada celda elige el menor de los 3 anteriores (diagonal, arriba o izquierda).\n"
            "• Puede usar un tramo de A varias veces (círculos verdes): la misma parte del sonido dicha más lenta en B.\n"
            "• El camino empieza en (0, 0) y termina en el último tramo de cada audio, y nunca retrocede.\n"
            f"• Resultado: DTW = suma de los costos del camino / (n + m) = {valor:.3f}.   Cuanto menor, más parecidos.",
            fontsize=10.5, va="top", linespacing=1.55)

    fig.suptitle("DTW con dos audios reales de «reconocer»: cada tramo es un vector, no un número", fontsize=15, fontweight="bold")
    SALIDA.mkdir(parents=True, exist_ok=True)
    fig.savefig(SALIDA / "dtw_real.png", dpi=140, bbox_inches="tight")
    print("Guardado:", SALIDA / "dtw_real.png")


if __name__ == "__main__":
    main()
