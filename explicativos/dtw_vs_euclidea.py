"""Grafico explicativo: distancia euclidea vs DTW.

    python explicativos/dtw_vs_euclidea.py       # guarda explicativos/imagenes/dtw_vs_euclidea.png

Dos "palabras" iguales dichas con distinto ritmo (la misma forma, desplazada y estirada en el tiempo):
  - euclidea: compara el tramo i de A SOLO con el tramo i de B (componente por componente);
  - DTW: cada tramo de A se empareja con el tramo de B que mejor le corresponde, respetando el orden.
Colores = los de los PCA (tab10): azul = audio A, naranja = audio B, verde = DTW, rojo = euclidea.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402
from matplotlib.patches import ConnectionPatch                    # noqa: E402

AZUL, NARANJA, VERDE, ROJO = (plt.cm.tab10(i) for i in (0, 1, 2, 3))
SALIDA = Path(__file__).resolve().parent / "imagenes"
plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]


def senal(n, picos, ancho=0.07):
    """Curva suave con picos en las posiciones relativas `picos` (0..1)."""
    t = np.linspace(0, 1, n)
    return sum(a * np.exp(-((t - p) ** 2) / (2 * ancho ** 2)) for p, a in picos)


def dtw(a, b):
    n, m = len(a), len(b)
    c = np.abs(a[:, None] - b[None, :])
    D = np.full((n, m), np.inf)
    D[0, 0] = c[0, 0]
    for i in range(n):
        for j in range(m):
            if i == j == 0:
                continue
            D[i, j] = c[i, j] + min(D[i - 1, j - 1] if i and j else np.inf, D[i - 1, j] if i else np.inf, D[i, j - 1] if j else np.inf)
    i, j, camino = n - 1, m - 1, [(n - 1, m - 1)]
    while (i, j) != (0, 0):
        opc = [(D[i - 1, j - 1] if i and j else np.inf, (i - 1, j - 1)), (D[i - 1, j] if i else np.inf, (i - 1, j)),
               (D[i, j - 1] if j else np.inf, (i, j - 1))]
        i, j = min(opc, key=lambda z: z[0])[1]
        camino.append((i, j))
    return D, camino[::-1]


def main():
    n = 40
    A = senal(n, [(0.25, 1.0), (0.55, 1.6), (0.80, 0.9)])
    B = senal(n, [(0.38, 1.0), (0.70, 1.6), (0.90, 0.9)])         # misma forma, dicha mas tarde / mas lenta
    # Las dos distancias se expresan en la MISMA unidad para poder compararlas: el costo medio por par de tramos emparejados,
    # (1/pares) * suma de |a_i - b_j| sobre los pares que usa cada metodo (la diagonal en la euclidea, el camino en DTW).
    D, camino = dtw(A, B)
    euc = float(np.mean([abs(A[i] - B[i]) for i in range(n)]))
    dtw_val = float(np.mean([abs(A[i] - B[j]) for i, j in camino]))

    fig = plt.figure(figsize=(17, 9.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.38, wspace=0.28)

    # ---- fila 1: euclidea (izq) y DTW (centro) ----
    for col, (titulo, color, pares, valor, formula) in enumerate([
        ("Distancia euclídea: tramo i de A contra tramo i de B", ROJO, [(i, i) for i in range(n)], euc,
         f"costo medio por par de tramos = {euc:.3f}"),
        ("DTW: cada tramo con su tramo correspondiente", VERDE, camino, dtw_val,
         f"costo medio por par de tramos = {dtw_val:.3f}")]):
        ax = fig.add_subplot(gs[0, col])
        t = np.arange(n)
        for (i, j) in pares[::1 if col else 1]:
            ax.add_artist(ConnectionPatch((i, A[i] + 2.2), (j, B[j]), "data", "data", color=color, alpha=0.45, lw=1.0))
        ax.plot(t, A + 2.2, color=AZUL, lw=2.6, label="audio A")
        ax.plot(t, B, color=NARANJA, lw=2.6, label="audio B (más lento)")
        ax.scatter(t, A + 2.2, color=AZUL, s=14, zorder=3)
        ax.scatter(t, B, color=NARANJA, s=14, zorder=3)
        ax.set_title(titulo, fontsize=11.5, fontweight="bold", color=color)
        ax.text(0.5, -0.24, formula, transform=ax.transAxes, ha="center", fontsize=13, color=color, fontweight="bold")
        ax.set_yticks([])
        ax.set_xlabel("tramo (tiempo)")
        ax.legend(loc="upper right", fontsize=9, frameon=False)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)

    # ---- fila 1 derecha: resumen ----
    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    ax.text(0, 1.0, "Qué cambia", fontsize=14, fontweight="bold", va="top")
    ax.text(0, 0.84, "Euclídea\n• compara posición por posición (i con i)\n• si una palabra se dice más rápido o lento,\n  los tramos no coinciden y la distancia crece\n• pide que los dos audios tengan el mismo largo\n  (por eso se usa el vector fijo de 91 números)",
            fontsize=10.5, color=ROJO, va="top", linespacing=1.5)
    ax.text(0, 0.38, "DTW\n• busca el mejor emparejamiento entre tramos,\n  estirando o encogiendo el tiempo\n• mantiene el orden (no se cruzan las líneas)\n• trabaja con secuencias de largo distinto\n  (usa los 39 números de cada tramo)",
            fontsize=10.5, color=VERDE, va="top", linespacing=1.5)

    # ---- fila 2: tablas de costo ----
    ax = fig.add_subplot(gs[1, 0])
    costo = np.abs(A[:, None] - B[None, :])
    ax.imshow(costo.T, origin="lower", cmap="Blues", aspect="auto", vmin=0, vmax=costo.max())
    ax.plot([0, n - 1], [0, n - 1], color=ROJO, lw=2.6, label="euclídea (diagonal)")
    ax.set_title("Matriz de costos: la euclídea solo usa la diagonal", fontsize=11, fontweight="bold")
    ax.set_xlabel("tramo de A")
    ax.set_ylabel("tramo de B")
    ax.legend(loc="upper left", fontsize=9)

    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(costo.T, origin="lower", cmap="Blues", aspect="auto", vmin=0, vmax=costo.max())
    ax.plot([c[0] for c in camino], [c[1] for c in camino], color=VERDE, lw=2.8, label="camino DTW (menor costo total)")
    ax.set_title("El DTW elige el camino de menor costo", fontsize=11, fontweight="bold")
    ax.set_xlabel("tramo de A")
    ax.set_ylabel("tramo de B")
    ax.legend(loc="upper left", fontsize=9)

    # ---- fila 2 derecha: resultado real en el KNN (exactitud en validacion) ----
    ax = fig.add_subplot(gs[1, 2])
    K = [1, 3, 5, 7, 9, 11, 13, 15]
    eu = [81.6, 94.7, 92.1, 92.1, 92.1, 92.1, 92.1, 94.7]
    dt = [94.7, 100, 100, 100, 100, 100, 100, 100]
    x = np.arange(len(K))
    ax.bar(x - 0.2, eu, 0.38, color=ROJO, label="euclídea (91 números)")
    ax.bar(x + 0.2, dt, 0.38, color=VERDE, label="DTW (secuencia)")
    ax.set_xticks(x)
    ax.set_xticklabels(K)
    ax.set_ylim(70, 103)
    ax.set_xlabel("K (vecinos)")
    ax.set_ylabel("exactitud en validación (%)")
    ax.set_title("Resultado real con los audios (38 de validación)", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Euclídea vs DTW: cómo se comparan dos audios", fontsize=16, fontweight="bold")
    SALIDA.mkdir(parents=True, exist_ok=True)
    fig.savefig(SALIDA / "dtw_vs_euclidea.png", dpi=150, bbox_inches="tight")
    print("Guardado:", SALIDA / "dtw_vs_euclidea.png", f"| euclidea={euc:.2f}  DTW={dtw_val:.3f}")


if __name__ == "__main__":
    main()
