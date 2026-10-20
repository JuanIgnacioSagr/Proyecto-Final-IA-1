"""Preprocesamiento universal de huellas dactilares sinteticas.

Una unica funcion publica, `preprocesar_huella`, sirve para cualquier imagen
(cualquier identidad, con o sin traslacion / rotacion / ruido / rayones): no
sabe que efecto trae y todo se estima de la propia imagen.

Pasos: gris -> normalizacion -> mascara ROI -> alineacion (traslacion +
rotacion) -> filtrado -> binarizacion.

Convencion de salida: crestas = 0 (negro), fondo = 255 (blanco), lienzo fijo
CANVAS_HW para todas las huellas.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

CANVAS_HW = (448, 320)  # alto, ancho del lienzo de salida
METODO_DEFECTO = "D"
METODOS = ("A", "B", "C", "D", "E")

RAIZ = Path(__file__).resolve().parent.parent
ENTRADA = RAIZ / "huellasDactilares"
SALIDA = RAIZ / "huellasProcesadas"


# --------------------------------------------------------------------------
# Entrada / salida
# --------------------------------------------------------------------------
def a_gris(entrada):
    """Acepta ruta o ndarray (gris o color) y devuelve uint8 2D."""
    if isinstance(entrada, (str, Path)):
        datos = np.fromfile(str(entrada), dtype=np.uint8)  # robusto con tildes en Windows
        g = cv2.imdecode(datos, cv2.IMREAD_GRAYSCALE)
        if g is None:
            raise ValueError(f"No se pudo leer la imagen: {entrada}")
        return g
    g = np.asarray(entrada)
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    return g.astype(np.uint8)


def guardar_png(ruta, img):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError(f"No se pudo codificar {ruta}")
    buf.tofile(str(ruta))


# --------------------------------------------------------------------------
# 1. Normalizacion de contraste
# --------------------------------------------------------------------------
def normalizar(g):
    """Estira el rango por percentiles 1-99 (no altera imagenes ya binarias)."""
    lo, hi = np.percentile(g, (1, 99))
    if hi - lo < 10:
        return g
    return np.clip((g.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# 2. Mascara ROI (region con crestas)
# --------------------------------------------------------------------------
def mascara_roi(g):
    """Mascara binaria uint8 {0,1} de la zona de huella.

    La mediana 3x3 elimina la sal y pimienta del fondo; la densidad local de
    tinta separa la huella (~20-30 % de tinta, incluso con crestas finas) del
    fondo (~0 %).
    """
    tinta = (255 - cv2.medianBlur(g, 3)).astype(np.float32) / 255.0
    dens = cv2.blur(tinta, (21, 21))
    m = (dens > 0.05).astype(np.uint8)
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, el)
    m = ndimage.binary_fill_holes(m).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, el)
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:  # imagen vacia: usar todo el cuadro
        return np.ones_like(m)
    mayor = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (etiquetas == mayor).astype(np.uint8)


# --------------------------------------------------------------------------
# 3. Alineacion (traslacion + rotacion)
# --------------------------------------------------------------------------
def _geometria(mask):
    """(angulo_eje_mayor, excentricidad, (cx, cy)) de la huella.

    Angulo en grados con x a la derecha e y hacia abajo, en (-90, 90].
    Si la huella toca el borde del cuadro (esta cortada por una traslacion) los
    momentos de la mascara se sesgan; en ese caso se ajusta una elipse solo a
    la parte del contorno que NO esta sobre el borde.
    """
    h, w = mask.shape
    M = cv2.moments(mask, binaryImage=True)
    mu20, mu02, mu11 = M["mu20"] / M["m00"], M["mu02"] / M["m00"], M["mu11"] / M["m00"]
    ang = 0.5 * np.degrees(np.arctan2(2 * mu11, mu20 - mu02))
    d = np.sqrt(4 * mu11 ** 2 + (mu20 - mu02) ** 2)
    l1, l2 = (mu20 + mu02 + d) / 2, (mu20 + mu02 - d) / 2
    ecc = float(np.sqrt(l1 / max(l2, 1e-9)))
    centro = (M["m10"] / M["m00"], M["m01"] / M["m00"])

    cont, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = max(cont, key=len)[:, 0, :]
    libres = pts[(pts[:, 0] > 2) & (pts[:, 0] < w - 3) & (pts[:, 1] > 2) & (pts[:, 1] < h - 3)]
    if len(libres) < len(pts) * 0.95 and len(libres) >= 30:  # cortada por el borde
        (ex, ey), (d1, d2), a = cv2.fitEllipse(libres.astype(np.float32).reshape(-1, 1, 2))
        mayor, menor = max(d1, d2), min(d1, d2)
        if 1.1 < mayor / menor < 2.5 and -0.25 * w < ex < 1.25 * w and -0.25 * h < ey < 1.25 * h:
            a_mayor = a if d1 >= d2 else a + 90.0
            a_mayor = (a_mayor + 90.0) % 180.0 - 90.0
            return a_mayor, mayor / menor, (ex, ey)
    return ang, ecc, centro


def alinear(g, mask):
    """Centra la huella en el lienzo y pone su eje mayor vertical.

    Devuelve (imagen_alineada, mascara_alineada, angulo_aplicado).
    """
    ang, excentricidad, (cx, cy) = _geometria(mask)
    # Corrige a vertical (+-90 deg); rango (-45, 45] para evitar el volteo de 180.
    delta = ang - 90.0 if ang > 0 else ang + 90.0
    # Casi redonda, ya vertical, o angulo absurdo (> 35 deg: mas que cualquier
    # giro esperable de un dedo) => no rotar.
    if excentricidad < 1.15 or abs(delta) < 2.0 or abs(delta) > 35.0:
        delta = 0.0
    h, w = CANVAS_HW
    M = cv2.getRotationMatrix2D((cx, cy), delta, 1.0)
    M[0, 2] += w / 2.0 - cx
    M[1, 2] += h / 2.0 - cy
    g2 = cv2.warpAffine(g, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)
    m2 = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return g2, m2, delta


# --------------------------------------------------------------------------
# 4. Filtrado
# --------------------------------------------------------------------------
def campo_orientacion(f, sigma=6.0):
    """Angulo del gradiente dominante por pixel (alfa, en [-pi/2, pi/2]) y
    coherencia en [0, 1]. Tensor de estructura suavizado (metodo de Bazen-Gerez)."""
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    gxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma)
    gyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma)
    gxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma)
    alfa = 0.5 * np.arctan2(2 * gxy, gxx - gyy)
    coh = np.sqrt((gxx - gyy) ** 2 + 4 * gxy ** 2) / (gxx + gyy + 1e-6)
    return alfa, coh


def periodo_crestas(f, mask):
    """Periodo medio de cresta (px) con el pico radial de la FFT."""
    h, w = f.shape
    x = (f - f[mask > 0].mean()) * mask
    x = x * np.outer(np.hanning(h), np.hanning(w))
    mag = np.abs(np.fft.fftshift(np.fft.fft2(x)))
    yy, xx = np.indices(mag.shape)
    r = np.hypot((yy - h / 2) / h, (xx - w / 2) / w)  # ciclos/pixel
    bins = np.linspace(1 / 16.0, 1 / 4.0, 60)  # periodos entre 4 y 16 px
    idx = np.digitize(r.ravel(), bins)
    suma = np.bincount(idx, weights=mag.ravel(), minlength=len(bins) + 1)[1:len(bins)]
    cuenta = np.bincount(idx, minlength=len(bins) + 1)[1:len(bins)]
    perfil = suma / np.maximum(cuenta, 1)
    frec = 0.5 * (bins[:-1] + bins[1:])[int(np.argmax(perfil))]
    return float(np.clip(1.0 / frec, 5.0, 14.0))


def gabor_orientado(g, mask, n_orient=16):
    """Realce de crestas con Gabor orientado (Hong, Wan y Jain, 1998).

    Cada pixel se filtra con el Gabor sintonizado a su orientacion local, de
    modo que rayones y cortes se rellenan siguiendo la direccion de la cresta.
    Devuelve una imagen float32 (tinta alta) y la coherencia.
    """
    tinta = (255 - g).astype(np.float32) / 255.0
    tinta = cv2.medianBlur((tinta * 255).astype(np.uint8), 3).astype(np.float32) / 255.0
    alfa, coh = campo_orientacion(tinta)
    lam = periodo_crestas(tinta, mask.astype(np.float32))
    sigma = 0.5 * lam
    r = int(np.ceil(2.5 * sigma))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    env = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    resp = []
    for i in range(n_orient):
        a = np.pi * i / n_orient
        k = env * np.cos(2 * np.pi * (xx * np.cos(a) + yy * np.sin(a)) / lam)
        k -= k.mean()
        k /= np.abs(k).sum()
        resp.append(cv2.filter2D(tinta, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT))
    resp = np.stack(resp)
    idx = np.round((alfa % np.pi) / np.pi * n_orient).astype(int) % n_orient
    out = np.take_along_axis(resp, idx[None], axis=0)[0]
    return out, coh


def limpiar(ink, mask, min_px=30):
    """Quita motas de tinta y rellena huecos blancos menores a min_px dentro de la mascara."""
    ink = ink.astype(np.uint8)
    for valor in (1, 0):  # 1: motas de tinta, 0: agujeros
        x = ink if valor == 1 else 1 - ink
        n, et, st, _ = cv2.connectedComponentsWithStats(x, connectivity=8)
        chicos = np.isin(et, 1 + np.where(st[1:, cv2.CC_STAT_AREA] < min_px)[0])
        if valor == 1:
            ink[chicos] = 0
        else:
            ink[chicos & (mask > 0)] = 1
    return ink


# --------------------------------------------------------------------------
# 5. Binarizacion + filtrado por metodo
# --------------------------------------------------------------------------
def _otsu(img_u8, mask):
    vals = img_u8[mask > 0]
    if vals.size == 0:
        vals = img_u8.ravel()
    t, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return t


def filtrar_y_binarizar(g, mask, metodo=METODO_DEFECTO):
    """g: gris alineado, mask: mascara alineada. Devuelve tinta {0,1}."""
    if metodo in ("A", "E"):
        f = cv2.medianBlur(g, 3)
        ink = (f <= _otsu(f, mask)).astype(np.uint8)
    elif metodo == "B":
        f = cv2.GaussianBlur(g, (3, 3), 0)
        ink = (cv2.adaptiveThreshold(f, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 21, 4)).astype(np.uint8)
    elif metodo in ("C", "D"):
        out, _ = gabor_orientado(g, mask)
        o8 = np.clip((out - out.min()) / (np.ptp(out) + 1e-9) * 255, 0, 255).astype(np.uint8)
        ink = (o8 >= _otsu(o8, mask)).astype(np.uint8)
    else:
        raise ValueError(f"metodo desconocido: {metodo}")
    ink = ink * mask
    if metodo in ("D", "E"):
        ink = limpiar(ink, mask)
    return ink


# --------------------------------------------------------------------------
# API publica
# --------------------------------------------------------------------------
def preprocesar_huella(entrada, metodo=METODO_DEFECTO, devolver_pasos=False):
    """Preprocesa UNA huella cualquiera.

    entrada: ruta o ndarray. Devuelve imagen uint8 CANVAS_HW con crestas=0 y
    fondo=255 (o un dict con los pasos intermedios si devolver_pasos=True).
    """
    gris = a_gris(entrada)
    norm = normalizar(gris)
    mask = mascara_roi(norm)
    alin, mask2, ang = alinear(norm, mask)
    ink = filtrar_y_binarizar(alin, mask2, metodo)
    binaria = np.where(ink > 0, 0, 255).astype(np.uint8)
    if devolver_pasos:
        return {"gris": gris, "normalizada": norm, "mascara": mask * 255,
                "alineada": alin, "mascara_alineada": mask2 * 255,
                "angulo": ang, "binaria": binaria}
    return binaria


def procesar_carpeta(metodo=METODO_DEFECTO, entrada=ENTRADA, salida=SALIDA):
    """Procesa huellasDactilares/<efecto>/id<N>/*.jpg -> huellasProcesadas/ (espejo, PNG)."""
    n = 0
    for ruta in sorted(entrada.glob("*/id*/*.jpg")):
        rel = ruta.relative_to(entrada).with_suffix(".png")
        guardar_png(salida / rel, preprocesar_huella(ruta, metodo))
        n += 1
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--metodo", default=METODO_DEFECTO, choices=METODOS)
    args = ap.parse_args()
    print(f"{procesar_carpeta(args.metodo)} imagenes procesadas (metodo {args.metodo}) -> {SALIDA}")
