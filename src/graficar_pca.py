"""Graficos PCA 2D y 3D de los vectores de huellas o de audios.

    python src/graficar_pca.py --datos huellas
    python src/graficar_pca.py --datos audios
    python src/graficar_pca.py --datos huellas --rasgos fingercode
    python src/graficar_pca.py --datos audios --no-escalar

Lee el dataset.npz de caracteristicas_huellas/ o de caracteristicas/ y guarda
pca_2d.png y pca_3d.png en la carpeta graficos/ de esa misma carpeta.

- Huellas: color = identidad, marcador = efecto. Por defecto SIN estandarizar
  (el histograma de orientacion ya esta en proporciones comparables; estandarizar
  amplifica ruido y empeora la separacion).
- Audios: color = comando, marcador = sesion (hablante o condicion). Por defecto
  CON estandarizar (z-score), porque los rasgos MFCC tienen escalas muy distintas.
El PCA se ajusta con todos los datos solo para dibujar; no es un modelo de clasificacion.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parent.parent
CARPETAS = {"huellas": RAIZ / "caracteristicas_huellas", "audios": RAIZ / "caracteristicas"}
BLOQUES_HUELLAS = {"orient": "X", "fingercode": "X_fingercode", "minucias": "X_minucias"}
MARCADORES = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "p"]


def cargar(datos, rasgos):
    carpeta = CARPETAS[datos]
    ruta = carpeta / "dataset.npz"
    if not ruta.exists():
        sys.exit(f"No existe {ruta}. Genera antes el dataset con "
                 f"{'caracterizar_huellas.py' if datos == 'huellas' else 'caracterizar_audios.py'}.")
    d = np.load(ruta, allow_pickle=False)
    clases = json.loads((carpeta / "clases.json").read_text(encoding="utf-8"))
    inv = {v: k for k, v in clases.items()}
    if datos == "huellas":
        X = d[BLOQUES_HUELLAS[rasgos]]
        grupo = d["efecto"].astype(str)
        titulo_marca = "efecto"
    else:
        X = d["X"]
        grupo = d["sesion"].astype(str)
        titulo_marca = "sesion"
    y = d["y"]
    nombres = np.array([inv[int(v)] for v in y])
    return X.astype(np.float64), y, nombres, grupo, titulo_marca, carpeta


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datos", choices=["huellas", "audios"], required=True)
    ap.add_argument("--rasgos", choices=list(BLOQUES_HUELLAS), default="orient",
                    help="solo huellas: bloque de rasgos a proyectar (por defecto orient = histograma de orientacion)")
    ap.add_argument("--escalar", action=argparse.BooleanOptionalAction, default=None,
                    help="estandarizar (z-score) antes del PCA; por defecto: no en huellas, si en audios")
    a = ap.parse_args()
    if a.datos == "audios" and a.rasgos != "orient":
        ap.error("--rasgos solo se usa con --datos huellas")

    X, y, nombres, grupo, titulo_marca, carpeta = cargar(a.datos, a.rasgos)
    escalar = a.escalar if a.escalar is not None else (a.datos == "audios")
    if len(X) < 4:
        sys.exit("Hacen falta al menos 4 muestras para graficar PCA en 3D.")
    Z = StandardScaler().fit_transform(X) if escalar else X
    n = min(3, Z.shape[1])
    pca = PCA(n_components=n, random_state=0)
    P = pca.fit_transform(Z)
    var = pca.explained_variance_ratio_ * 100
    etiqueta = ("orient" if a.datos == "huellas" and a.rasgos == "orient" else a.rasgos) if a.datos == "huellas" else "MFCC"
    titulo = f"PCA {a.datos} ({etiqueta}, {'con' if escalar else 'sin'} z-score) - {len(X)} muestras"
    print(titulo)
    print("Varianza explicada: " + ", ".join(f"PC{i + 1} {v:.1f}%" for i, v in enumerate(var)) + f"  (suma {var.sum():.1f}%)")

    clases_u = sorted(set(nombres), key=lambda c: int(y[nombres == c][0]))
    colores = {c: plt.cm.tab10(i) for i, c in enumerate(clases_u)}
    grupos_u = sorted(set(grupo))
    marc = {g: MARCADORES[i % len(MARCADORES)] for i, g in enumerate(grupos_u)}

    def dibujar(ax, ejes, tres_d=False):
        for c in clases_u:
            for g in grupos_u:
                m = (nombres == c) & (grupo == g)
                if not m.any():
                    continue
                coords = [P[m, e] for e in ejes]
                ax.scatter(*coords, color=colores[c], marker=marc[g], s=45, alpha=0.85,
                           edgecolor="black", linewidth=0.4)
        ax.set_xlabel(f"PC{ejes[0] + 1} ({var[ejes[0]]:.1f}%)")
        ax.set_ylabel(f"PC{ejes[1] + 1} ({var[ejes[1]]:.1f}%)")
        if tres_d:
            ax.set_zlabel(f"PC{ejes[2] + 1} ({var[ejes[2]]:.1f}%)")

    def leyenda(fig, ax, x=1.02):
        from matplotlib.lines import Line2D
        h1 = [Line2D([], [], marker="o", linestyle="", color=colores[c], markeredgecolor="black", label=c) for c in clases_u]
        h2 = [Line2D([], [], marker=marc[g], linestyle="", color="lightgray", markeredgecolor="black", label=g) for g in grupos_u]
        l1 = ax.legend(handles=h1, title="clase", loc="upper left", bbox_to_anchor=(x, 1.0), fontsize=8)
        ax.add_artist(l1)
        ax.legend(handles=h2, title=titulo_marca, loc="upper left", bbox_to_anchor=(x, 0.55), fontsize=8)

    carpeta_g = carpeta / "graficos"
    carpeta_g.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    dibujar(ax, (0, 1))
    ax.grid(alpha=0.3)
    ax.set_title(titulo, fontsize=10)
    leyenda(fig, ax)
    fig.tight_layout()
    fig.savefig(carpeta_g / "pca_2d.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    if n >= 3:
        fig = plt.figure(figsize=(9, 6.5))
        ax = fig.add_subplot(111, projection="3d")
        dibujar(ax, (0, 1, 2), tres_d=True)
        ax.set_title(titulo, fontsize=10)
        leyenda(fig, ax, x=1.15)
        fig.savefig(carpeta_g / "pca_3d.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
    print(f"Guardado en {carpeta_g}: pca_2d.png" + (", pca_3d.png" if n >= 3 else ""))


if __name__ == "__main__":
    main()
