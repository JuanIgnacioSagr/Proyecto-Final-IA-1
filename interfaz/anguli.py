"""Generacion de huellas con Anguli usando la libreria subprocess.

subprocess permite que Python EJECUTE Y CONTROLE otro programa (aca Anguli.exe) como si lo escribieras
en la consola: se le pasan argumentos, se espera a que termine, se lee su salida y su codigo de retorno
(0 = termino bien) y se puede cortar si tarda demasiado (timeout).

Anguli genera MUCHAS carpetas y subcarpetas (Impression_1/fp_1/..., Fingerprints/, ...). Por eso cada vez
trabaja en una carpeta TEMPORAL: de ahi se toma solo la imagen que interesa, se la mueve a la carpeta
unica de huellas con el nombre "id2_<fecha>.jpg" o "intruso_<fecha>.jpg", y se borra todo lo demas.

Dos formas de usar Anguli (modo consola):
  identidad k : Anguli -indir <dir con fp_1/k.jpg> -impr -ni 1 -outdir <dir> [filtros]
                -> una IMPRESION NUEVA de la huella autorizada k (la misma persona, con un filtro nuevo)
  intruso     : Anguli -num 1 -cdist natural -ni 1 -outdir <dir> -seed <al azar> [filtros]
                -> una huella maestra NUEVA (otra persona, no autorizada) y una impresion suya
"""
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import config

# CREATE_NO_WINDOW evita que se abra una ventana de consola negra cada vez que se llama a Anguli
SIN_VENTANA = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0

TIPOS = {"id1": "Identidad 1", "id2": "Identidad 2", "id3": "Identidad 3", "intruso": "Intruso"}


class AnguliError(Exception):
    """Algo salio mal al ejecutar Anguli (no esta, fallo, tardo demasiado, no genero imagen)."""


# ----------------------------------------------------------------------------------------
#  Filtros (las distorsiones que aplica Anguli a la impresion)
#  Cada NIVEL combina las cuatro distorsiones a la vez; lo que cambia entre niveles es hasta donde llegan:
#     rot     : giro maximo en grados            trans   : desplazamiento maximo en pixeles
#     scratch : cantidad maxima de rayones       noise   : nivel maximo de ruido
#  Cada distorsion se da como un RANGO (minimo, maximo) para el valor maximo que se le pasa a Anguli.
#
#  Por que rangos y no un valor fijo: con los mismos parametros Anguli genera SIEMPRE LA MISMA IMAGEN (la semilla
#  -seed solo funciona al crear huellas maestras, no impresiones). Sorteando el valor maximo dentro del rango del
#  nivel cada huella sale distinta y sigue perteneciendo al nivel (los rangos de niveles distintos no se pisan).
#  El nivel FUERTE llega a lo que tiene el dataset (giro 30, traslacion 60).
# ----------------------------------------------------------------------------------------
FILTROS = {
    "Débil": {"rot": (3, 7), "trans": (6, 14), "scratch": (1, 3), "noise": (2, 4)},
    "Medio": {"rot": (12, 18), "trans": (25, 35), "scratch": (6, 10), "noise": (6, 10)},
    "Fuerte": {"rot": (26, 30), "trans": (52, 60), "scratch": (12, 15), "noise": (12, 15)},
}
FILTRO_DEFECTO = "Medio"
# Niveles que muestra la interfaz. El dataset y el K-means se entrenaron SOLO con Debil y Medio (ver src/generar_dataset_huellas.py):
# con "Fuerte" (giro 30 y traslacion 60 a la vez) rechaza a la mayoria de las autorizadas, asi que no se ofrece en la demo.
# "Fuerte" sigue en FILTROS por si se quiere probar a mano.
FILTROS_INTERFAZ = ("Débil", "Medio")


def sortear_filtro(nivel):
    """Parametros concretos para Anguli: un valor al azar dentro del rango de cada distorsion del nivel."""
    r = FILTROS[nivel]
    return {"rot": random.randint(*r["rot"]), "trans": random.randint(*r["trans"]),
            "scratch": (0, random.randint(*r["scratch"])), "noise": (0, random.randint(*r["noise"]))}


def describir_filtro(p):
    """Texto corto para mostrar: 'giro 12, desplaz. 20, rayones 0-5, ruido 0-6'."""
    return (f"giro {p['rot']}, desplaz. {p['trans']}, "
            f"rayones {p['scratch'][0]}-{p['scratch'][1]}, ruido {p['noise'][0]}-{p['noise'][1]}")


def _argumentos_filtro(p):
    # -scratch y -noise llevan DOS numeros en un solo argumento ('0 5'); al pasar una lista a subprocess
    # cada elemento es un argumento, asi que no hace falta poner comillas.
    return ["-rot", str(p["rot"]), "-trans", str(p["trans"]),
            "-scratch", f"{p['scratch'][0]} {p['scratch'][1]}",
            "-noise", f"{p['noise'][0]} {p['noise'][1]}"]


# ----------------------------------------------------------------------------------------
#  Generacion
# ----------------------------------------------------------------------------------------
def generar_huella(tipo, nombre_filtro, exe=None, carpeta_destino=None):
    """Genera UNA huella con Anguli y la deja en la carpeta de huellas.

    tipo           'id1' | 'id2' | 'id3' | 'intruso'
    nombre_filtro  una clave de FILTROS
    Devuelve un dict: ruta (Path de la imagen final), tipo, filtro (parametros usados), segundos.
    Lanza AnguliError si algo falla.
    """
    if tipo not in TIPOS:
        raise AnguliError(f"Tipo de huella desconocido: {tipo}")
    exe = Path(exe) if exe else config.ruta_anguli()
    if not exe.exists():
        raise AnguliError(f"No se encontró Anguli.exe en:\n{exe}")
    carpeta_destino = Path(carpeta_destino) if carpeta_destino else config.CARPETA_HUELLAS
    carpeta_destino.mkdir(parents=True, exist_ok=True)

    filtro = sortear_filtro(nombre_filtro)
    semilla = random.randrange(1, 2_000_000_000)       # semilla nueva = resultado nuevo en cada llamada
    # Impresion que se toma de la corrida: -ni N crea las impresiones 1..N y cada una es distinta (la 1 siempre es igual
    # con los mismos parametros, pero la 2, la 3... no). Pedir hasta 8 no tarda mas, y asi hay mas variedad de huellas.
    n_imp = 1 if tipo == "intruso" else random.randint(1, 8)
    tmp = Path(tempfile.mkdtemp(prefix="anguli_"))     # carpeta temporal donde Anguli deja sus subcarpetas
    salida = tmp / "salida"
    t0 = time.time()
    try:
        if tipo == "intruso":
            # huella maestra NUEVA (otra persona) + una impresion suya
            args = ["-num", "1", "-cdist", "natural", "-ni", "1", "-outdir", str(salida),
                    "-numT", "1", "-seed", str(semilla), "-itype", "jpg"]
        else:
            # impresion nueva de la huella autorizada k: se arma una entrada con SOLO esa huella (fp_1/k.jpg)
            k = tipo[-1]
            base = config.CARPETA_BASE / f"{k}.jpg"
            if not base.exists():
                raise AnguliError(f"No se encontró la huella base de la identidad {k}:\n{base}")
            entrada = tmp / "entrada" / "fp_1"
            entrada.mkdir(parents=True)
            shutil.copy(base, entrada / f"{k}.jpg")
            args = ["-indir", str(tmp / "entrada"), "-impr", "-ni", str(n_imp), "-outdir", str(salida),
                    "-numT", "1", "-seed", str(semilla), "-itype", "jpg"]
        comando = [str(exe)] + args + _argumentos_filtro(filtro)

        try:
            # cwd = carpeta de Anguli: necesita las carpetas Densitymaps/, Filterbank/ y noise_blob/ que estan al lado
            r = subprocess.run(comando, cwd=str(exe.parent), capture_output=True, text=True,
                               errors="replace", timeout=config.TIEMPO_MAX_ANGULI_S, creationflags=SIN_VENTANA)
        except subprocess.TimeoutExpired:
            raise AnguliError(f"Anguli tardó más de {config.TIEMPO_MAX_ANGULI_S} s y se canceló.")
        except OSError as e:
            raise AnguliError(f"No se pudo ejecutar Anguli: {e}")
        if r.returncode != 0:
            raise AnguliError(f"Anguli terminó con error (código {r.returncode}).\n{(r.stdout + r.stderr)[-400:]}")

        # la impresion elegida esta en salida/Impression_<n_imp>/fp_1/<k>.jpg
        carpeta_imp = salida / f"Impression_{n_imp}"
        imagenes = sorted(carpeta_imp.rglob("*.jpg")) if carpeta_imp.exists() else []
        if not imagenes:
            raise AnguliError("Anguli no generó ninguna imagen (revisá los parámetros).")

        # se mueve a la carpeta unica, con el nombre de la identidad o intruso
        marca = time.strftime("%Y%m%d_%H%M%S")
        destino = carpeta_destino / f"{tipo}_{marca}.jpg"
        n = 1
        while destino.exists():                        # por si se generan dos en el mismo segundo
            destino = carpeta_destino / f"{tipo}_{marca}_{n}.jpg"
            n += 1
        shutil.move(str(imagenes[0]), str(destino))
        return {"ruta": destino, "tipo": tipo, "filtro": filtro, "segundos": time.time() - t0}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)         # se borran TODAS las subcarpetas temporales de Anguli
