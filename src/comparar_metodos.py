"""Compara los metodos de filtrado/binarizacion A-E sobre las 90 huellas.

A: mediana 3x3 + Otsu                          (base simple)
B: gaussiano + umbral adaptativo
C: mediana + Gabor orientado + Otsu
D: C + limpieza de motas/agujeros              (candidato)
E: A + limpieza de motas/agujeros

Metricas (todas calculadas sobre las 90 imagenes, para elegir UN pipeline
valido para cualquier efecto):
  motas      : componentes de tinta < 20 px dentro de la huella (menor = mejor)
  misma/otra : similitud coseno media del campo de orientacion entre
               impresiones de la MISMA identidad / de identidades DISTINTAS
  separ.     : misma - otra (mayor = el preprocesamiento conserva lo que
               distingue a cada identidad; es solo una pista, la prueba real
               es el K-means posterior)
"""
import time
from itertools import combinations

import cv2
import numpy as np

import preprocesar as P

FIG = P.SALIDA / "_figuras"
MUESTRAS = ["primera_id1_imp1", "rotada_id1_imp2", "trasladada_id1_imp2",
            "matada_id1_imp1", "rayada_id1_imp2", "ruido_id1_imp1"]


def motas(binaria):
    ink = (binaria == 0).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    # Se ignoran los bordes del lienzo (la mascara ya limita el area)
    return int(np.sum(st[1:, cv2.CC_STAT_AREA] < 20))


def mapa(binaria):
    """Campo de orientacion de crestas (cos2a, sin2a ponderado por coherencia),
    reducido 8x. Es una huella digital 'de flujo' comparable entre impresiones."""
    f = cv2.GaussianBlur((binaria == 0).astype(np.float32), (0, 0), 1.5)
    alfa, coh = P.campo_orientacion(f, sigma=8.0)
    v = np.concatenate([(np.cos(2 * alfa) * coh)[::8, ::8].ravel(),
                        (np.sin(2 * alfa) * coh)[::8, ::8].ravel()])
    return v / (np.linalg.norm(v) + 1e-9)


def main():
    rutas = sorted(P.ENTRADA.glob("*/id*/*.jpg"))
    info = [(r.parent.parent.name.lower(), r.parent.name) for r in rutas]  # (efecto, id)
    efectos = sorted({e for e, _ in info})
    print(f"{len(rutas)} imagenes, efectos: {efectos}\n")

    resultados, salidas = {}, {}
    for met in P.METODOS:
        t0 = time.time()
        outs = [P.preprocesar_huella(r, met) for r in rutas]
        dt = (time.time() - t0) / len(rutas)
        salidas[met] = outs
        mo = np.array([motas(o) for o in outs])
        V = np.stack([mapa(o) for o in outs])
        C = V @ V.T
        misma, otra = [], []
        for i, j in combinations(range(len(rutas)), 2):
            (misma if info[i][1] == info[j][1] else otra).append(C[i, j])
        por_efecto = {e: float(mo[[k for k, (ef, _) in enumerate(info) if ef == e]].mean()) for e in efectos}
        resultados[met] = (por_efecto, np.mean(misma), np.mean(otra), dt)

    print("Motas sueltas por imagen (media por efecto)")
    print("met  " + "  ".join(f"{e[:10]:>10}" for e in efectos) + "   misma   otra  separ.  s/img")
    for met in P.METODOS:
        pe, a, b, dt = resultados[met]
        print(f"{met}    " + "  ".join(f"{pe[e]:10.1f}" for e in efectos)
              + f"   {a:5.3f}  {b:5.3f}  {a - b:5.3f}  {dt:5.2f}")

    # Panel visual: filas = imagenes de muestra, columnas = original + metodos
    FIG.mkdir(parents=True, exist_ok=True)
    nombres = [r.stem for r in rutas]
    filas = []
    for m in MUESTRAS:
        k = nombres.index(m)
        orig = cv2.resize(P.a_gris(rutas[k]), (205, 300))
        cols = [orig] + [cv2.resize(salidas[met][k], (205, 300)) for met in P.METODOS]
        filas.append(np.hstack(cols))
    panel = np.vstack(filas)
    cabecera = np.full((24, panel.shape[1]), 255, np.uint8)
    for i, t in enumerate(["original"] + [f"metodo {m}" for m in P.METODOS]):
        cv2.putText(cabecera, t, (i * 205 + 50, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
    P.guardar_png(FIG / "comparacion_metodos.png", np.vstack([cabecera, panel]))
    print(f"\nFigura: {FIG / 'comparacion_metodos.png'}")


if __name__ == "__main__":
    main()
