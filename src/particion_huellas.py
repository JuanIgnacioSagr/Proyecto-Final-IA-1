"""Particion de las huellas en ENTRENAMIENTO / VALIDACION / PRUEBA (y de los intrusos en CALIBRACION / PRUEBA).

Las huellas autorizadas se reparten dentro de cada identidad y de cada efecto (primera, debil, medio), ordenadas por numero
de impresion: las primeras 60 % para entrenar, el 20 % siguiente para validar y el 20 % final para la prueba.
Asi las tres particiones tienen las mismas identidades y los mismos niveles de filtro (primera 3/1/1, debil 15/5/5, medio 15/5/5).

  entrenamiento : K-means, PCA, media/desvio del z-score y covarianzas de cada identidad
  validacion    : fija los limites de T2 y Q (95 % de las autorizadas que el modelo no vio)
  prueba        : se usa una sola vez, para informar el resultado
Intrusos (dataset_huellas/intrusas/intruso_01..50.jpg): 01-25 CALIBRAN el umbral (mayor valor que no acepta a ninguno), 26-50 solo EVALUAN.
"""
import re
from pathlib import Path

import numpy as np

FRACCION_ENTRENAMIENTO = 0.6
FRACCION_VALIDACION = 0.2
N_INTRUSOS_CALIBRACION = 25


def _clave(archivo):
    """'impresionPrimera/id1/primera_id1_imp3.png' -> (efecto, identidad, numero de impresion)."""
    partes = Path(str(archivo)).as_posix().split("/")
    efecto = re.sub(r"^impresion", "", partes[0].lower())
    return efecto, partes[1], int(re.search(r"_imp(\d+)", partes[-1]).group(1))


def particion(archivos):
    """-> array de 'entrenamiento' / 'validacion' / 'prueba', uno por archivo (mismo orden que `archivos`)."""
    claves = [_clave(a) for a in archivos]
    salida = np.empty(len(claves), dtype=object)
    grupos = {}
    for i, (efecto, ident, n) in enumerate(claves):
        grupos.setdefault((efecto, ident), []).append((n, i))
    for lista in grupos.values():
        lista.sort()
        n = len(lista)
        n_train = int(round(FRACCION_ENTRENAMIENTO * n))
        n_val = int(round(FRACCION_VALIDACION * n))
        for pos, (_, i) in enumerate(lista):
            salida[i] = "entrenamiento" if pos < n_train else ("validacion" if pos < n_train + n_val else "prueba")
    return salida.astype(str)


def es_entrenamiento(archivos):
    return particion(archivos) == "entrenamiento"


def numero_intruso(nombre):
    """'intruso_07.jpg' -> 7."""
    return int(re.search(r"intruso_(\d+)", str(nombre)).group(1))


def intruso_calibracion(nombre):
    """True si el intruso sirve para CALIBRAR el umbral (01-25); False si es de prueba (26-50)."""
    return numero_intruso(nombre) <= N_INTRUSOS_CALIBRACION
