"""Matrices de confusion del KNN (voz) en las tres etapas: entrenamiento, validacion y prueba.

    python explicativos/matrices_audios.py      # crea explicativos/imagenes/matrices_confusion_audios.png

Como se obtiene cada una (KNN con DTW y K = 3, lo elegido en la validacion):
  ENTRENAMIENTO  El KNN no "aprende" nada, solo guarda ejemplos; si cada audio se comparara contra el conjunto que
                 lo incluye se encontraria a si mismo y daria 100 % sin sentido. Por eso se clasifica cada audio
                 de entrenamiento usando como referencia SOLO las otras sesiones (personas) del entrenamiento:
                 "dejar una persona afuera" (leave-one-session-out). Es un control, no la prueba final.
  VALIDACION     38 audios (mama, joaco), referencia = los 137 del entrenamiento. Con esto se eligio DTW y K = 3.
  PRUEBA         22 audios (maxi, mili), una sola vez, referencia = entrenamiento + validacion (175 audios).
Validacion y prueba salen de dataset_audios/caracteristicas/metricas_knn.json (las mismas matrices que calcula src/knn_audios.py).
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
import caracterizar_audios as CA                                  # noqa: E402
import knn_audios as KA                                           # noqa: E402
import metricas                                                   # noqa: E402

SALIDA = Path(__file__).resolve().parent / "imagenes"
plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]


def main():
    res = json.loads((KA.CARPETA / "metricas_knn.json").read_text(encoding="utf-8"))
    k, dist = res["k"], res["distancia"]
    nombres = ["reconocer", "identificar", "salir"]

    X, y, archivos, sesion = CA.cargar_dataset()
    X = X.astype(np.float64)
    seqs, _, _ = CA.cargar_secuencias()
    seqs_prep = [KA.preparar_secuencia(s) for s in seqs]
    clases = np.unique(y)
    yi = np.searchsorted(clases, y)
    persona = np.array([s.split("_", 1)[1] if "_" in s else s for s in sesion])
    en_val = np.isin(persona, [p.split("_", 1)[-1] for p in res["sesiones_validacion"]])
    en_test = np.isin(persona, [p.split("_", 1)[-1] for p in res["sesiones_prueba"]])
    idx_train = np.flatnonzero(~(en_val | en_test))

    # ---- entrenamiento: cada audio contra las otras personas del entrenamiento ----
    print(f"Calculando {dist.upper()} entre los {len(idx_train)} audios de entrenamiento (puede tardar un poco)...")
    D = KA.distancias(dist, idx_train, idx_train, X, seqs_prep).astype(float)
    pers_tr = persona[idx_train]
    D[pers_tr[:, None] == pers_tr[None, :]] = np.inf          # no se puede usar a la misma persona como referencia
    pred = KA.knn_predecir(D, yi[idx_train], k, len(clases))
    cm_tr = metricas.matriz_confusion(yi[idx_train], pred, len(clases))
    cm_val = np.array(res["validacion"]["matriz_confusion"])
    cm_te = np.array(res["prueba"]["matriz_confusion"])

    fig, ejes = plt.subplots(1, 3, figsize=(17, 5.4))
    titulos = []
    for ax, cm, nombre, detalle in zip(ejes, (cm_tr, cm_val, cm_te), ("Entrenamiento", "Validación", "Prueba"),
                                       (f"{len(idx_train)} audios (1 persona afuera cada vez)", f"{int(cm_val.sum())} audios (mama, joaco)",
                                        f"{int(cm_te.sum())} audios (maxi, mili)")):
        m = metricas.metricas_clasificacion(cm)
        metricas.dibujar_confusion(ax, cm, nombres, f"{nombre}: {detalle}\nexactitud {m['exactitud'] * 100:.1f} %  ·  F1 {m['f1_macro']:.3f}")
        print(f"{nombre:<14} exactitud {m['exactitud'] * 100:.1f} %  F1 {m['f1_macro']:.3f}\n{cm}")
    fig.suptitle(f"KNN de voz ({dist.upper()}, K = {k}): matrices de confusión por etapa", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    SALIDA.mkdir(parents=True, exist_ok=True)
    fig.savefig(SALIDA / "matrices_confusion_audios.png", dpi=150, bbox_inches="tight")
    print("Guardado:", SALIDA / "matrices_confusion_audios.png")


if __name__ == "__main__":
    main()
