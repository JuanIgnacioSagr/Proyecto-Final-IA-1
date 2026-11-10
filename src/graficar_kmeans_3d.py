"""Grafico 3D de K-means: PCA en 3 dimensiones con un ELIPSOIDE de aceptacion alrededor de cada identidad.

    python src/graficar_kmeans_3d.py              # guarda dataset_huellas/caracteristicas/graficos/kmeans_clusters_3d.png
    python src/graficar_kmeans_3d.py --mostrar    # ademas abre una ventana interactiva (se puede girar con el mouse)
    python src/graficar_kmeans_3d.py --lam 1.5    # dibuja los elipsoides con otro umbral lambda (para comparar)

Que se ve
---------
- Un elipsoide translucido alrededor de cada identidad (X grande): es la sombra en las 3 primeras componentes de la zona
  de aceptacion  T2 <= lambda * limite  (decision_huellas.py). Su forma sale de como se estira cada identidad.
- Cada punto es una huella en su posicion REAL en PC1-PC3 (no se recoloca ni se reescala), coloreada por su identidad real.
  Lleno = aceptada por el sistema, hueco = rechazada. Circulo = entrenamiento, cuadrado = validacion, triangulo = prueba.
  Las X rojas son los intrusos (los de calibracion y los de prueba).
- Una huella puede verse dentro de un elipsoide y estar rechazada: la rechaza lo que el plano NO muestra (otras componentes
  o el residuo Q). Esa parte esta exacta en el grafico T2-Q de kmeans_clusters_2d.png.
"""
import argparse
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

import decision_huellas as DH                                 # noqa: E402
import graficos_decision as GD                                # noqa: E402
import kmeans_huellas as KH                                   # noqa: E402
import particion_huellas as PH                                # noqa: E402

CARPETA = KH.CARPETA
SALIDA = CARPETA / "graficos" / "kmeans_clusters_3d.png"
MARCAS = {"entrenamiento": "o", "validacion": "s", "prueba": "^"}


def construir_figura(lam=None):
    """Arma la figura 3D con los datos y el modelo guardados. Devuelve (figura, resumen)."""
    modelo = np.load(KH.RUTA_MODELO, allow_pickle=False)
    datos = np.load(CARPETA / "dataset.npz", allow_pickle=False)
    X, y = datos["X"].astype(float), datos["y"].astype(int)
    parte = PH.particion(datos["archivo"].astype(str))
    m = DH.desde_modelo(modelo)
    mapeo = modelo["mapeo"].astype(int)
    lam_modelo = float(modelo["radios"][0])
    lam = lam_modelo if lam is None else lam
    nombres = [str(n).replace("identidad_", "Identidad ") for n in modelo["nombres"]][:3]
    Xi, cal, _ = KH.cargar_intrusos()

    Z = (X - m["mu"]) @ m["V"]
    Zi = (Xi - m["mu"]) @ m["V"]
    _, s, acep = DH.decidir(X, m, lam)
    _, s_i, acep_i = DH.decidir(Xi, m, lam)
    colores = [plt.cm.tab10(i) for i in range(3)]
    ev = m["varianza"] / m["varianza"].sum() * 100

    fig = plt.figure(figsize=(10.5, 7.6))
    ax = fig.add_subplot(111, projection="3d")
    for j in range(len(m["Zc"])):
        ident = mapeo[j]
        x, yy, z = GD.elipsoide_3d(m, j, lam)
        ax.plot_surface(x, yy, z, color=colores[ident], alpha=0.07, linewidth=0, shade=False, antialiased=True)
        ax.plot_wireframe(x, yy, z, color=colores[ident], alpha=0.28, linewidth=0.45, rstride=4, cstride=4)
        ax.scatter(*m["Zc"][j][:3], marker="X", s=260, color=colores[ident], edgecolor="black", linewidth=1.6, depthshade=False)
        ax.text(m["Zc"][j][0], m["Zc"][j][1], m["Zc"][j][2], "  " + nombres[ident], fontsize=9, weight="bold")
    for ident in range(3):
        for nombre, marca in MARCAS.items():
            sel = (y == ident) & (parte == nombre)
            for ok in (True, False):
                t = sel & (acep == ok)
                if t.any():
                    ax.scatter(*Z[t, :3].T, marker=marca, s=46 if ok else 62, facecolors=colores[ident] if ok else "none",
                               edgecolors="black" if ok else colores[ident], linewidths=0.6 if ok else 1.6, alpha=0.95, depthshade=False)
    ax.scatter(*Zi[:, :3].T, marker="X", s=48, color=plt.cm.tab10(3), edgecolor="black", linewidth=0.5, depthshade=False)

    todo = np.vstack([Z[:, :3], Zi[:, :3]])
    ax.set_box_aspect(tuple(np.ptp(todo, axis=0)))
    ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)")
    ax.set_zlabel(f"PC3 ({ev[2]:.1f}%)")
    ax.view_init(elev=22, azim=-58)
    ayuda = [Line2D([], [], marker="o", linestyle="", color=colores[i], markeredgecolor="black", label=f"huellas {nombres[i]}") for i in range(3)]
    ayuda += [Line2D([], [], marker="X", linestyle="", color="gray", markeredgecolor="black", markersize=11, label="centro de la identidad"),
              Line2D([], [], marker="o", linestyle="", color="gray", markeredgecolor="black", label="aceptada"),
              Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor="gray", markeredgewidth=1.6, label="rechazada"),
              Line2D([], [], marker="o", linestyle="", color="lightgray", markeredgecolor="black", label="entrenamiento"),
              Line2D([], [], marker="s", linestyle="", color="lightgray", markeredgecolor="black", label="validación"),
              Line2D([], [], marker="^", linestyle="", color="lightgray", markeredgecolor="black", label="prueba"),
              Line2D([], [], marker="X", linestyle="", color=plt.cm.tab10(3), markeredgecolor="black", markersize=10, label="intruso")]
    ax.legend(handles=ayuda, loc="upper left", bbox_to_anchor=(0.0, 0.98), fontsize=8)
    ax.set_title(f"K-means de huellas en 3D: elipsoide de aceptación (umbral λ = {lam:.2f}) alrededor de cada identidad", fontsize=11)
    fig.text(0.5, 0.015, "Elipsoide = zona T² ≤ λ·límite proyectada en PC1-PC3. Lo que rechaza a un punto que se ve adentro está en las otras "
                         "componentes o en el residuo Q (ver el grafico T²-Q).", ha="center", fontsize=8, color="#444444")
    resumen = {"lam": lam, "aceptadas": int(acep.sum()), "total": int(len(X)), "intrusos_aceptados": int(acep_i.sum()),
               "intrusos": int(len(Xi)), "pca": [float(v) for v in ev[:3] / 100]}
    return fig, resumen


def generar(mostrar=False, lam=None):
    """Guarda el PNG (y opcionalmente abre la ventana interactiva)."""
    fig, r = construir_figura(lam)
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SALIDA, dpi=130, bbox_inches="tight")
    print(f"Grafico 3D guardado: {SALIDA}")
    print(f"  umbral lambda = {r['lam']:.2f} | autorizadas aceptadas: {r['aceptadas']} de {r['total']} | intrusos aceptados: "
          f"{r['intrusos_aceptados']} de {r['intrusos']} | PCA: " + ", ".join(f"{v * 100:.1f}%" for v in r["pca"]))
    if mostrar:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mostrar", action="store_true", help="abrir una ventana interactiva para girar el grafico")
    ap.add_argument("--lam", type=float, default=None, help="umbral lambda de los elipsoides (por defecto, el del modelo)")
    a = ap.parse_args()
    generar(mostrar=a.mostrar, lam=a.lam)
