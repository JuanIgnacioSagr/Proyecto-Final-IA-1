"""Dibujo de la regla de decision de huellas (PCA + T2 + Q): elipses reales y grafico T2 contra Q.

Como se dibuja bien un PCA cuando el umbral es un elipsoide (ver decision_huellas.py):
  - Cada huella se proyecta con los MISMOS ejes del modelo (ajustados solo con entrenamiento).
  - La zona de aceptacion de una identidad es el elipsoide  T2 <= lambda * limite_T2  en las k componentes. Su sombra en el
    plano PC_a-PC_b es EXACTAMENTE la elipse con la covarianza de ese bloque (la proyeccion de un elipsoide es una elipse).
  - Lo que el PCA descarta (residuo Q) se muestra con su propio eje: en el grafico T2-Q la zona de aceptacion es un rectangulo.
    Una huella puede verse dentro de la elipse y estar rechazada por su residuo: ese caso queda marcado.
"""
import numpy as np
from matplotlib.patches import Ellipse

import decision_huellas as DH


def covarianza(m, j):
    """Covarianza de la identidad j en las k componentes (la inversa de la precision guardada)."""
    return np.linalg.inv(m["precision"][j])


def radio_mahalanobis(m, lam):
    """Radio del elipsoide de aceptacion, en unidades de Mahalanobis: T2 <= lambda * limite_T2."""
    return float(np.sqrt(lam * m["t2_limite"]))


def elipse_2d(m, j, lam, a=0, b=1, **estilo):
    """Parche matplotlib: sombra en el plano de las componentes a y b del elipsoide de aceptacion de la identidad j."""
    cov = covarianza(m, j)[np.ix_([a, b], [a, b])]
    w, V = np.linalg.eigh(cov)
    r = radio_mahalanobis(m, lam)
    angulo = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
    return Ellipse(m["Zc"][j][[a, b]], 2 * r * np.sqrt(w[1]), 2 * r * np.sqrt(w[0]), angle=angulo, **estilo)


def elipsoide_3d(m, j, lam, n=36):
    """Superficie (x, y, z) del elipsoide de aceptacion de la identidad j en las 3 primeras componentes."""
    cov = covarianza(m, j)[:3, :3]
    w, V = np.linalg.eigh(cov)
    r = radio_mahalanobis(m, lam)
    u, v = np.mgrid[0:2 * np.pi:complex(n), 0:np.pi:complex(n // 2)]
    esf = np.stack([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)], axis=-1)
    pts = m["Zc"][j][:3] + (esf * (r * np.sqrt(w))) @ V.T
    return pts[..., 0], pts[..., 1], pts[..., 2]


def puntos_t2_q(X, m):
    """(T2 / limite_T2 de su identidad mas cercana, Q / limite_Q) por huella: las coordenadas del grafico T2-Q."""
    T2, Q, _ = DH.estadisticos(X, m)
    return T2.min(axis=1) / m["t2_limite"], Q / m["q_limite"]


def dibujar_t2_q(ax, m, lam, grupos, titulo="T² contra residuo Q: zona de aceptación exacta"):
    """Grafico T2 contra Q con la zona de aceptacion (rectangulo). `grupos` = [(X, estilo_scatter, etiqueta), ...]."""
    todos = []
    for X, estilo, etiqueta in grupos:
        t, q = puntos_t2_q(X, m)
        todos += [t, q]
        ax.scatter(t, q, label=etiqueta, **estilo)
    valores = np.concatenate(todos)
    positivos = valores[valores > 0]
    piso = float(positivos.min()) * 0.8 if len(positivos) else 1e-3
    tope = max(float(valores.max()) * 1.08, lam * 1.3)
    if m.get("usar_q", True):
        ax.fill_between([piso, lam], piso, lam, color="#2ca02c", alpha=0.10)
        ax.plot([lam, lam, piso], [piso, lam, lam], color="black", lw=2.2, ls="--")
    else:                                                # la decision solo usa T2: el borde es una linea vertical
        ax.fill_between([piso, lam], piso, tope, color="#2ca02c", alpha=0.10)
        ax.axvline(lam, color="black", lw=2.2, ls="--")
    ax.text(piso * 1.3, piso * 1.3, "ACEPTADA", color="#2ca02c", fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(piso, tope)
    ax.set_ylim(piso, tope)
    ax.set_xlabel("T² / límite  (qué tan lejos del centro de su identidad, dentro del PCA)")
    ax.set_ylabel("Q / límite  (lo que el PCA no explica)" + ("" if m.get("usar_q", True) else ": solo informativo, no se usa"))
    ax.set_title(titulo, fontsize=11.5, fontweight="bold")
    ax.grid(alpha=0.25, which="both")
