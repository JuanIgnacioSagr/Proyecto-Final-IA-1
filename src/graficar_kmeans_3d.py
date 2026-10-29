"""Grafico 3D de K-means: PCA en 3 dimensiones con una ESFERA de aceptacion alrededor de cada centroide.

    python src/graficar_kmeans_3d.py              # guarda caracteristicas_huellas/graficos/kmeans_clusters_3d.png
    python src/graficar_kmeans_3d.py --mostrar    # ademas abre una ventana interactiva (se puede girar con el mouse)
    python src/graficar_kmeans_3d.py --radio 0.31 # dibuja las esferas con otro radio (por ejemplo para comparar)

Que se ve
---------
- Una esfera translucida alrededor de cada uno de los 3 centroides (X grande): su radio es el umbral R del modelo.
  Una huella DENTRO de la esfera se acepta; FUERA se rechaza ("NO AUTORIZADO").
- Cada punto es una huella, coloreada por su identidad real. Lleno = aceptada, hueco = rechazada.
  Circulo = impresiones de entrenamiento (1-3), triangulo = prueba (4-5).

Como se ubican los puntos (importante)
--------------------------------------
El PCA aplasta 72 dimensiones en 3 y hace que los puntos parezcan mas cerca de lo que estan. Para que el dibujo sea
coherente con la decision, cada punto se coloca a su distancia REAL (72 dimensiones) del centroide mas cercano, en la
direccion que muestra el plano PCA. Asi, "dentro de la esfera" coincide exactamente con "distancia <= R".
Es el mismo criterio que usa el grafico de clusters de la interfaz.
"""
import argparse
import re
import sys
from pathlib import Path

import matplotlib

if "--mostrar" in sys.argv:
    matplotlib.use("TkAgg")        # ventana interactiva
else:
    matplotlib.use("Agg")          # solo guardar el archivo
import matplotlib.pyplot as plt                               # noqa: E402
import numpy as np                                            # noqa: E402
from matplotlib.lines import Line2D                           # noqa: E402
from sklearn.decomposition import PCA                         # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
CARPETA = RAIZ / "caracteristicas_huellas"
SALIDA = CARPETA / "graficos" / "kmeans_clusters_3d.png"


def _esfera(centro, radio, n=40):
    """Puntos (x, y, z) de la superficie de una esfera, para plot_surface."""
    u, v = np.mgrid[0:2 * np.pi:complex(n), 0:np.pi:complex(n // 2)]
    return (centro[0] + radio * np.cos(u) * np.sin(v),
            centro[1] + radio * np.sin(u) * np.sin(v),
            centro[2] + radio * np.cos(v))


def construir_figura(radio=None):
    """Arma la figura 3D con los datos y el modelo guardados. Devuelve (figura, resumen de aceptadas).

    radio: si se indica, se usa en lugar del radio guardado en el modelo (solo para el dibujo).
    """
    modelo = np.load(CARPETA / "modelo_kmeans.npz", allow_pickle=False)
    datos = np.load(CARPETA / "dataset.npz", allow_pickle=False)
    X, y = datos["X"].astype(float), datos["y"].astype(int)                 # 72 numeros por huella e identidad (0, 1, 2)
    archivo = datos["archivo"].astype(str)
    imp = np.array([int(re.search(r"_imp(\d+)", a).group(1)) for a in archivo])
    entrenamiento = np.isin(imp, (1, 2, 3))
    C, mapeo, R = modelo["centroides"], modelo["mapeo"].astype(int), float(modelo["radios"][0] if radio is None else radio)
    nombres = [str(n).replace("identidad_", "Identidad ") for n in modelo["nombres"]][:3]

    # ---- PCA a 3 dimensiones (solo para dibujar) ----
    pca = PCA(n_components=3, random_state=0).fit(X)
    P_x, P_c = pca.transform(X), pca.transform(C)

    # ---- cada punto a su distancia real del centroide mas cercano ----
    D = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)              # (huellas, centroides) en 72 dimensiones
    cerca = D.argmin(axis=1)
    dist = D[np.arange(len(X)), cerca]
    v = P_x - P_c[cerca]
    norma = np.linalg.norm(v, axis=1, keepdims=True)
    u = np.where(norma > 1e-9, v / np.maximum(norma, 1e-9), np.array([1.0, 0.0, 0.0]))
    pos = P_c[cerca] + u * dist[:, None]
    aceptada = dist <= R

    colores = [plt.cm.tab10(i) for i in range(3)]
    fig = plt.figure(figsize=(10.5, 7.6))
    ax = fig.add_subplot(111, projection="3d")

    # ---- esferas de aceptacion (una por centroide) ----
    for c in range(len(C)):
        ident = mapeo[c]
        x, yy, z = _esfera(P_c[c], R)
        ax.plot_surface(x, yy, z, color=colores[ident], alpha=0.07, linewidth=0, shade=False, antialiased=True)
        ax.plot_wireframe(x, yy, z, color=colores[ident], alpha=0.28, linewidth=0.45, rstride=4, cstride=4)
        ax.scatter(*P_c[c], marker="X", s=260, color=colores[ident], edgecolor="black", linewidth=1.6, depthshade=False)
        ax.text(P_c[c][0], P_c[c][1], P_c[c][2] + R * 1.08, nombres[ident], fontsize=9, weight="bold", ha="center")

    # ---- huellas ----
    for ident in range(3):
        for entren, marca in ((True, "o"), (False, "^")):
            m = (y == ident) & (entrenamiento == entren)
            for ok in (True, False):
                s = m & (aceptada == ok)
                if not s.any():
                    continue
                ax.scatter(pos[s, 0], pos[s, 1], pos[s, 2], marker=marca, s=46 if ok else 62,
                           facecolors=colores[ident] if ok else "none", edgecolors="black" if ok else colores[ident],
                           linewidths=0.6 if ok else 1.6, alpha=0.95, depthshade=False)

    # ---- ejes iguales (para que las esferas se vean redondas) ----
    todo = np.vstack([pos] + [P_c + R, P_c - R])
    centro = (todo.max(0) + todo.min(0)) / 2
    medio = (todo.max(0) - todo.min(0)).max() / 2
    ax.set_xlim(centro[0] - medio, centro[0] + medio)
    ax.set_ylim(centro[1] - medio, centro[1] + medio)
    ax.set_zlim(centro[2] - medio, centro[2] + medio)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2] * 100:.1f}%)")
    ax.view_init(elev=22, azim=-58)

    ayuda = [Line2D([], [], marker="o", linestyle="", color=colores[i], markeredgecolor="black", label=f"huellas {nombres[i]}") for i in range(3)]
    ayuda += [Line2D([], [], marker="X", linestyle="", color="gray", markeredgecolor="black", markersize=11, label="centroide"),
              Line2D([], [], marker="o", linestyle="", color="gray", markeredgecolor="black", label="aceptada (dentro de la esfera)"),
              Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor="gray", markeredgewidth=1.6, label="rechazada (fuera)"),
              Line2D([], [], marker="o", linestyle="", color="lightgray", markeredgecolor="black", label="entrenamiento"),
              Line2D([], [], marker="^", linestyle="", color="lightgray", markeredgecolor="black", label="prueba")]
    ax.legend(handles=ayuda, loc="upper left", bbox_to_anchor=(0.0, 0.98), fontsize=8)
    ax.set_title(f"K-means de huellas en 3D: esfera de aceptación de radio R = {R:.2f} alrededor de cada centroide", fontsize=11)
    fig.text(0.5, 0.015, "Cada punto se dibuja a su distancia REAL (72 dimensiones) del centroide más cercano, en la dirección del plano PCA: "
                         "dentro de la esfera = aceptada.", ha="center", fontsize=8, color="#444444")
    resumen = {"R": R, "aceptadas": int(aceptada.sum()), "total": int(len(X)), "pca": [float(v) for v in pca.explained_variance_ratio_]}
    return fig, resumen


def generar(mostrar=False, radio=None):
    """Guarda el PNG (y opcionalmente abre la ventana interactiva)."""
    fig, r = construir_figura(radio)
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SALIDA, dpi=130, bbox_inches="tight")
    print(f"Grafico 3D guardado: {SALIDA}")
    print(f"  esferas de radio R = {r['R']:.2f} | huellas dentro de su esfera: {r['aceptadas']} de {r['total']} "
          f"| PCA: " + ", ".join(f"{v * 100:.1f}%" for v in r["pca"]))
    if mostrar:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mostrar", action="store_true", help="abrir una ventana interactiva para girar el grafico")
    ap.add_argument("--radio", type=float, default=None, help="radio de las esferas (por defecto, el del modelo)")
    a = ap.parse_args()
    generar(mostrar=a.mostrar, radio=a.radio)
