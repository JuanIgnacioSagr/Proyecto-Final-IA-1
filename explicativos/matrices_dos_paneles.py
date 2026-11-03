"""Matrices de confusion de dos paneles, una por cada algoritmo (los valores salen de los json de metricas).

    python explicativos/matrices_dos_paneles.py

  K-means (huellas): ENTRENAMIENTO (impresiones 1-3) y PRUEBA (impresiones 4-5)   -> matriz_kmeans_dos_paneles.png
  KNN (voz):         VALIDACION (mama, joaco) y PRUEBA (maxi, mili)                -> matriz_knn_dos_paneles.png
El K-means no tiene conjunto de validacion; el KNN no se entrena (su "entrenamiento" es la base de ejemplos),
por eso cada algoritmo muestra sus dos etapas con sentido.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
import metricas                                                   # noqa: E402

SALIDA = Path(__file__).resolve().parent / "imagenes"
plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]


def dos_paneles(titulo, nombres, izq, der, archivo):
    fig, ejes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    for ax, (nombre, cm) in zip(ejes, (izq, der)):
        cm = np.array(cm)
        m = metricas.metricas_clasificacion(cm)
        metricas.dibujar_confusion(ax, cm, nombres, f"{nombre} ({int(cm.sum())} muestras)\nexactitud {m['exactitud'] * 100:.1f} %  ·  F1 {m['f1_macro']:.3f}")
        ax.title.set_fontsize(11.5)
        ax.title.set_fontweight("bold")
    fig.suptitle(titulo, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    SALIDA.mkdir(parents=True, exist_ok=True)
    fig.savefig(SALIDA / archivo, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Guardado:", SALIDA / archivo)


def main():
    k = json.loads((RAIZ / "caracteristicas_huellas" / "metricas_kmeans.json").read_text(encoding="utf-8"))
    dos_paneles("K-means (huellas): matrices de confusión", ["identidad 1", "identidad 2", "identidad 3"],
                ("Entrenamiento", k["entrenamiento"]["matriz_confusion"]), ("Prueba", k["prueba"]["matriz_confusion"]),
                "matriz_kmeans_dos_paneles.png")
    a = json.loads((RAIZ / "caracteristicas" / "metricas_knn.json").read_text(encoding="utf-8"))
    dos_paneles(f"KNN (voz, {a['distancia'].upper()}, K = {a['k']}): matrices de confusión", ["reconocer", "identificar", "salir"],
                ("Validación", a["validacion"]["matriz_confusion"]), ("Prueba", a["prueba"]["matriz_confusion"]),
                "matriz_knn_dos_paneles.png")


if __name__ == "__main__":
    main()
