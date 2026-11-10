"""Compara rasgos candidatos para las huellas (solo mide, no cambia el modelo).

    python explicativos/evaluar_rasgos.py

Para cada conjunto de rasgos mide, con la particion de particion_huellas.py (entrenamiento 60 %, validacion 20 %,
prueba 20 %; el z-score y los centroides salen SOLO del entrenamiento) y con la distancia euclidea al centroide de cada identidad:
  ident. valid.    % de las huellas de VALIDACION cuyo centroide mas cercano es el de su identidad (con esto se ELIGE el conjunto)
  ident. prueba    igual con la prueba (solo informativo)
  separacion       distancia minima entre centroides / radio propio mediano (mas alto = identidades mas separadas)
  R*               criterio con el que se fijo el umbral: mayor R que no deja entrar a ningun intruso de CALIBRACION (01-25)
  FRR valid.       con ese R*: autorizadas de validacion rechazadas
  FAR prueba       con ese R*: intrusos de prueba (26-50) aceptados
  ambiguas         autorizadas de validacion que quedan dentro de R* de dos o mas centroides a la vez
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import caracterizar_huellas as CH                                 # noqa: E402
import particion_huellas as PH                                   # noqa: E402
import preprocesar as P                                           # noqa: E402

AQUI = Path(__file__).resolve().parent


# --------------------------------------------------------------------------------------
#  Rasgos candidatos (entrada: imagen binaria preprocesada, crestas = 0)
# --------------------------------------------------------------------------------------
def _campo(b):
    t = CH._tinta(b)
    alfa, coh = P.campo_orientacion(cv2.GaussianBlur(t, (0, 0), CH.PARAMS["suavizado_sigma"]), sigma=CH.PARAMS["tensor_sigma"])
    return t, alfa, coh


def _regiones(shape, g):
    H, W = shape
    return [(slice(gy * H // g, (gy + 1) * H // g), slice(gx * W // g, (gx + 1) * W // g)) for gy in range(g) for gx in range(g)]


def orient_rejilla(g):
    """Histograma de orientacion global (24 bins) + g x g regiones de 12 bins. g = 2 es el rasgo actual (72 numeros)."""
    def f(b):
        t, alfa, coh = _campo(b)
        peso = coh * t
        v = [CH._histograma(alfa, peso, 24)]
        v += [CH._histograma(alfa[sl], peso[sl], 12) for sl in _regiones(t.shape, g)]
        return np.concatenate(v)
    return f


def crestas(b):
    """3x3 regiones: periodo de cresta (FFT), densidad de tinta y coherencia media -> 27 numeros."""
    t, alfa, coh = _campo(b)
    zona = CH._zona_huella(t)
    suave = cv2.GaussianBlur(t, (0, 0), CH.PARAMS["suavizado_sigma"])
    v = []
    for sl in _regiones(t.shape, 3):
        z = zona[sl]
        if z.sum() > 400:
            v += [P.periodo_crestas(suave[sl], z.astype(np.float32)) / 14.0, float(t[sl][z].mean()), float(coh[sl][z].mean())]
        else:
            v += [0.0, 0.0, 0.0]
    return np.array(v, dtype=np.float32)


def poincare(b, bloque=16):
    """Nucleos (indice +1/2) y deltas (-1/2) del campo de orientacion, contados en una rejilla 3x3 -> 18 numeros."""
    t, alfa, coh = _campo(b)
    zona = CH._zona_huella(t)
    H, W = t.shape
    h, w = H // bloque, W // bloque
    c = cv2.resize(np.cos(2 * alfa) * coh, (w, h), interpolation=cv2.INTER_AREA)
    s = cv2.resize(np.sin(2 * alfa) * coh, (w, h), interpolation=cv2.INTER_AREA)
    th = 0.5 * np.arctan2(s, c)
    zb = cv2.resize(zona.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) > 0.9
    vec = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    nucleos, deltas = np.zeros((h, w)), np.zeros((h, w))
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if not zb[i - 1:i + 2, j - 1:j + 2].all():
                continue
            a = [th[i + di, j + dj] for di, dj in vec]
            a.append(a[0])
            tot = sum(((a[k + 1] - a[k] + np.pi / 2) % np.pi) - np.pi / 2 for k in range(8)) / (2 * np.pi)
            if tot > 0.3:
                nucleos[i, j] = 1
            elif tot < -0.3:
                deltas[i, j] = 1
    v = []
    for m in (nucleos, deltas):
        for sl in _regiones(m.shape, 3):
            v.append(float(m[sl].sum()))
    return np.array(v, dtype=np.float32)


def gabor_compacto(b):
    """Energia de Gabor (8 orientaciones) en 2x2 regiones -> 32 numeros (FingerCode reducido)."""
    t = CH._tinta(b)
    zona = CH._zona_huella(t)
    resp = [np.abs(cv2.filter2D(t, cv2.CV_32F, k)) for k in CH._GABOR]
    v = []
    for sl in _regiones(t.shape, 2):
        z = zona[sl]
        v += [float(R[sl][z].mean()) if z.sum() > 50 else 0.0 for R in resp]
    return np.array(v, dtype=np.float32)


CANDIDATOS = {
    "orient2": orient_rejilla(2), "orient3": orient_rejilla(3), "orient4": orient_rejilla(4),
    "crestas": crestas, "poincare": poincare, "gabor": gabor_compacto,
    "fingercode": CH.rasgo_fingercode, "minucias": CH.rasgo_minucias,
}
CONJUNTOS = [                                       # (nombre, bloques, z-score)
    ("BASE: 72 orientacion (sin z-score)", ["orient2"], False),
    ("72 orientacion (z-score)", ["orient2"], True),
    ("72 + gabor (el actual, 104)", ["orient2", "gabor"], True),
    ("orientacion 3x3 (132)", ["orient3"], True),
    ("orientacion 4x4 (216)", ["orient4"], True),
    ("gabor compacto (32)", ["gabor"], True),
    ("fingercode (256)", ["fingercode"], True),
    ("minucias (18)", ["minucias"], True),
    ("crestas (27)", ["crestas"], True),
    ("poincare (18)", ["poincare"], True),
    ("72 + fingercode", ["orient2", "fingercode"], True),
    ("72 + minucias", ["orient2", "minucias"], True),
    ("72 + gabor + minucias", ["orient2", "gabor", "minucias"], True),
    ("72 + gabor + crestas", ["orient2", "gabor", "crestas"], True),
    ("72 + gabor + poincare", ["orient2", "gabor", "poincare"], True),
    ("orient3 + gabor", ["orient3", "gabor"], True),
    ("orient3 + gabor + minucias + crestas", ["orient3", "gabor", "minucias", "crestas"], True),
    ("72 + gabor + minucias + crestas + poincare", ["orient2", "gabor", "minucias", "crestas", "poincare"], True),
]


# --------------------------------------------------------------------------------------
#  Datos: dataset (particionado) + intrusos de calibracion y de prueba + huellas subidas a mano
# --------------------------------------------------------------------------------------
def cargar_imagenes():
    imgs = CH._listar(CH.ENTRADA)
    rel = [i.relative_to(CH.ENTRADA).as_posix() for i in imgs]
    dat = [P.a_gris(i) for i in imgs]
    y = np.array([int(i.parent.name[2:]) - 1 for i in imgs])
    parte = PH.particion(rel)
    intr, cal = [], []
    for ruta in sorted((RAIZ / "dataset_huellas" / "intrusas").glob("intruso_*.jpg")):
        intr.append(P.preprocesar_huella(P.a_gris(ruta)))
        cal.append(PH.intruso_calibracion(ruta.name))
    return dat, y, parte, intr, np.array(cal)


def medir(nombre, Fd, yd, parte, Fi, cal, z):
    tr, va, te = parte == "entrenamiento", parte == "validacion", parte == "prueba"
    Fd, Fi = Fd.astype(float), Fi.astype(float)
    if z:
        mu, sd = Fd[tr].mean(0), Fd[tr].std(0)
        sd = np.maximum(sd, 0.1 * np.median(sd) + 1e-9)             # evita inflar columnas casi constantes
        Fd, Fi = (Fd - mu) / sd, (Fi - mu) / sd
    C = np.stack([Fd[tr & (yd == i)].mean(0) for i in range(3)])
    dist = lambda F: np.linalg.norm(F[:, None] - C[None], axis=2)       # noqa: E731
    Dd, Di = dist(Fd), dist(Fi)
    acc_v = (Dd[va].argmin(1) == yd[va]).mean() * 100
    acc_t = (Dd[te].argmin(1) == yd[te]).mean() * 100
    propio = np.median(np.linalg.norm(Fd[tr] - C[yd[tr]], axis=1))
    sep = min(np.linalg.norm(C[a] - C[b]) for a in range(3) for b in range(a + 1, 3)) / propio
    R = Di.min(1)[cal].min()                                            # mayor R sin intrusos de calibracion aceptados (exclusivo)
    frr = (Dd[va].min(1) >= R).mean() * 100
    far = (Di.min(1)[~cal] < R).mean() * 100
    amb = int(((Dd[va] < R).sum(1) >= 2).sum())
    print(f"{nombre:46s} dim {Fd.shape[1]:4d} | ident. valid. {acc_v:5.1f}% prueba {acc_t:5.1f}% | sep {sep:4.2f} | "
          f"R* {R:6.3f} | FRR valid. {frr:5.1f}% | FAR prueba {far:5.1f}% | ambiguas {amb:2d}/{int(va.sum())}")


def main():
    dat, yd, parte, intr, cal = cargar_imagenes()
    print(f"dataset {len(dat)} (entrenamiento {int((parte == 'entrenamiento').sum())}, validacion {int((parte == 'validacion').sum())}, "
          f"prueba {int((parte == 'prueba').sum())}) | intrusos: {int(cal.sum())} de calibracion, {int((~cal).sum())} de prueba\n")
    bloques_d, bloques_i = {}, {}
    for n, f in CANDIDATOS.items():
        bloques_d[n] = np.stack([f(b) for b in dat])
        bloques_i[n] = np.stack([f(b) for b in intr])
    for nombre, bl, z in CONJUNTOS:
        medir(nombre, np.hstack([bloques_d[b] for b in bl]), yd, parte, np.hstack([bloques_i[b] for b in bl]), cal, z)


if __name__ == "__main__":
    main()
