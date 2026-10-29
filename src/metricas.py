"""Metricas de clasificacion y matriz de confusion (compartido por K-means y KNN).

Todo esta implementado "a mano" con numpy para poder explicarlo; la
verificacion contra scikit-learn se hizo aparte (ver el documento de teoria).

Vocabulario (para cada clase C):
    VP (verdaderos positivos) : eran C y se predijo C
    FP (falsos positivos)     : NO eran C y se predijo C
    FN (falsos negativos)     : eran C y se predijo otra cosa
    precision    = VP / (VP + FP)   "de lo que dije que era C, cuanto lo era"
    exhaustividad (recall) = VP / (VP + FN)   "de lo que era C, cuanto encontre"
    F1           = 2 * precision * recall / (precision + recall)   (promedio armonico)
    exactitud    = aciertos / total ;  tasa de error = 1 - exactitud
"""
import numpy as np


def matriz_confusion(y_real, y_pred, n_clases):
    """Matriz (n_clases x n_clases): FILA = clase real, COLUMNA = clase predicha.

    La diagonal son los aciertos; lo que queda fuera de la diagonal son los errores
    (la celda [i, j] cuenta cuantos de la clase i se confundieron con la clase j).
    """
    cm = np.zeros((n_clases, n_clases), dtype=int)
    for r, p in zip(y_real, y_pred):
        cm[int(r), int(p)] += 1
    return cm


def metricas_clasificacion(cm):
    """Calcula las metricas a partir de la matriz de confusion.

    Devuelve un dict con exactitud, tasa de error, y por clase: precision,
    recall, F1 y soporte (cuantas muestras reales hay de esa clase); mas el
    F1 promedio simple entre clases ("macro").
    """
    cm = np.asarray(cm, dtype=float)
    total = cm.sum()
    vp = np.diag(cm)
    fp = cm.sum(axis=0) - vp          # columna: predichos como C que no eran C
    fn = cm.sum(axis=1) - vp          # fila: eran C pero se predijo otra cosa
    # np.divide con where evita dividir por cero cuando una clase no tiene muestras
    precision = np.divide(vp, vp + fp, out=np.zeros_like(vp), where=(vp + fp) > 0)
    recall = np.divide(vp, vp + fn, out=np.zeros_like(vp), where=(vp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(vp), where=(precision + recall) > 0)
    exactitud = float(vp.sum() / total) if total else 0.0
    con_datos = cm.sum(axis=1) > 0     # el promedio macro ignora clases sin muestras
    return {"exactitud": exactitud,
            "tasa_error": 1.0 - exactitud,
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1": f1.tolist(),
            "soporte": cm.sum(axis=1).astype(int).tolist(),
            "f1_macro": float(f1[con_datos].mean()) if con_datos.any() else 0.0,
            "n": int(total)}


def imprimir_reporte(titulo, cm, nombres):
    """Imprime la matriz de confusion y la tabla de metricas en texto."""
    m = metricas_clasificacion(cm)
    ancho = max(len(n) for n in nombres) + 2
    print(f"\n{titulo}  (n={m['n']})")
    print("  matriz de confusion (fila = real, columna = predicho):")
    print("  " + " " * ancho + "".join(f"{n[:11]:>12}" for n in nombres))
    for i, n in enumerate(nombres):
        print("  " + f"{n:<{ancho}}" + "".join(f"{v:>12d}" for v in cm[i]))
    print(f"  exactitud {m['exactitud'] * 100:.1f}%   tasa de error {m['tasa_error'] * 100:.1f}%   F1 macro {m['f1_macro']:.3f}")
    print("  " + f"{'clase':<{ancho}}{'precision':>10}{'recall':>10}{'F1':>10}{'muestras':>10}")
    for i, n in enumerate(nombres):
        print("  " + f"{n:<{ancho}}{m['precision'][i]:>10.3f}{m['recall'][i]:>10.3f}{m['f1'][i]:>10.3f}{m['soporte'][i]:>10d}")
    return m


def dibujar_confusion(ax, cm, nombres, titulo):
    """Dibuja la matriz de confusion: color = porcentaje de la fila, texto = cantidad (y %)."""
    cm = np.asarray(cm)
    sumas = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm, sumas, out=np.zeros(cm.shape, dtype=float), where=sumas > 0)
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n{pct[i, j] * 100:.0f}%", ha="center", va="center",
                    color="white" if pct[i, j] > 0.55 else "black", fontsize=10)
    ax.set_xticks(range(len(nombres)))
    ax.set_yticks(range(len(nombres)))
    ax.set_xticklabels(nombres, rotation=20, ha="right")
    ax.set_yticklabels(nombres)
    ax.set_xlabel("predicho")
    ax.set_ylabel("real")
    ax.set_title(titulo, fontsize=10)
