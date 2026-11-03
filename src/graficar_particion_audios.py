"""Grafico del PCA de los audios separado en entrenamiento, validacion y prueba (KNN).

    python src/graficar_particion_audios.py            # guarda caracteristicas/graficos/pca_particion.png
    python src/graficar_particion_audios.py --mostrar  # ademas abre la ventana

Mismo PCA para todos los paneles (ejes iguales), calculado con los 197 audios estandarizados (91 numeros cada uno),
asi los tres conjuntos se pueden comparar a simple vista. Color = comando (como en el PCA de audios),
marcador = sesion (persona). Los conjuntos se separan por PERSONA, igual que en src/knn_audios.py.
"""
import argparse
import sys
from pathlib import Path

import matplotlib

if "--mostrar" in sys.argv:
    matplotlib.use("TkAgg")
else:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
import numpy as np                                    # noqa: E402
from matplotlib.lines import Line2D                   # noqa: E402
from sklearn.decomposition import PCA                 # noqa: E402
from sklearn.preprocessing import StandardScaler      # noqa: E402

import knn_audios as KA                               # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
CARPETA = RAIZ / "caracteristicas"
SALIDA = CARPETA / "graficos" / "pca_particion.png"
MARCADORES = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mostrar", action="store_true")
    ap.add_argument("--validacion", default=KA.SESIONES_VALIDACION)
    ap.add_argument("--prueba", default=KA.SESIONES_PRUEBA)
    a = ap.parse_args()

    d = np.load(CARPETA / "dataset.npz", allow_pickle=True)
    X, y, ses = d["X"].astype(float), d["y"].astype(int), d["sesion"].astype(str)
    nombres_cmd = ["reconocer", "identificar", "salir"]
    persona = np.array([s.split("_", 1)[1] if "_" in s else s for s in ses])

    pedidas_val = [p.strip() for p in a.validacion.split(",")]
    pedidas_test = [p.strip() for p in a.prueba.split(",")]
    es_val = np.isin(persona, pedidas_val)
    es_test = np.isin(persona, pedidas_test)
    es_train = ~(es_val | es_test)
    conjuntos = [("Entrenamiento", es_train, "#1f77b4"), ("Validación", es_val, "#ff7f0e"), ("Prueba", es_test, "#2ca02c")]

    pca = PCA(n_components=2, random_state=0)
    P = pca.fit_transform(StandardScaler().fit_transform(X))
    ev = pca.explained_variance_ratio_ * 100
    colores = [plt.cm.tab10(i) for i in range(3)]
    sesiones = sorted(set(persona))
    marc = {s: MARCADORES[i % len(MARCADORES)] for i, s in enumerate(sesiones)}

    margen = 0.6
    lim = (P[:, 0].min() - margen, P[:, 0].max() + margen), (P[:, 1].min() - margen, P[:, 1].max() + margen)

    fig, axs = plt.subplots(1, 4, figsize=(21, 5.6), sharex=True, sharey=True)
    for ax, (nombre, m, c) in zip(axs[:3], conjuntos):
        for cmd in range(3):
            for s in sesiones:
                k = m & (y == cmd) & (persona == s)
                if k.any():
                    ax.scatter(P[k, 0], P[k, 1], color=colores[cmd], marker=marc[s], s=52, alpha=0.9, edgecolor="black", linewidth=0.5)
        quienes = ", ".join(sorted(set(persona[m])))
        ax.set_title(f"{nombre}: {int(m.sum())} audios\n({quienes})", fontsize=11, color=c, fontweight="bold")
    # panel 4: los tres juntos, color por conjunto
    for nombre, m, c in conjuntos:
        axs[3].scatter(P[m, 0], P[m, 1], color=c, s=40, alpha=0.8, edgecolor="black", linewidth=0.4, label=f"{nombre} ({int(m.sum())})")
    axs[3].set_title("Los tres conjuntos juntos\n(color = conjunto)", fontsize=11, fontweight="bold")
    axs[3].legend(loc="best", fontsize=9)
    for ax in axs:
        ax.set_xlim(*lim[0])
        ax.set_ylim(*lim[1])
        ax.grid(alpha=0.25)
        ax.set_xlabel(f"PC1 ({ev[0]:.1f} %)")
    axs[0].set_ylabel(f"PC2 ({ev[1]:.1f} %)")

    h1 = [Line2D([], [], marker="o", linestyle="", color=colores[i], markeredgecolor="black", label=nombres_cmd[i]) for i in range(3)]
    h2 = [Line2D([], [], marker=marc[s], linestyle="", color="lightgray", markeredgecolor="black", label=s) for s in sesiones]
    leg = fig.legend(handles=h1, loc="lower center", ncol=3, title="Comando (color)", fontsize=9, bbox_to_anchor=(0.32, -0.02))
    fig.legend(handles=h2, loc="lower center", ncol=len(sesiones), title="Persona / sesión (marcador)", fontsize=9, bbox_to_anchor=(0.72, -0.02))
    fig.suptitle("PCA de los audios por conjunto de datos (separados por persona)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SALIDA, dpi=140, bbox_inches="tight")
    print("Guardado:", SALIDA)
    for nombre, m, _ in conjuntos:
        print(f"  {nombre:<14}{int(m.sum()):>4} audios | personas: {', '.join(sorted(set(persona[m])))}")
    if a.mostrar:
        plt.show()


if __name__ == "__main__":
    main()
