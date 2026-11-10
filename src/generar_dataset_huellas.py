"""Genera con Anguli el dataset NUEVO de huellas: impresiones con filtros Debil y Medio (sin Fuerte) e intrusos.

    python src/generar_dataset_huellas.py                      # 50 impresiones por identidad + 50 intrusos
    python src/generar_dataset_huellas.py --solo impresiones   # solo las impresiones de las 3 identidades
    python src/generar_dataset_huellas.py --solo intrusos      # solo los intrusos

Resultado (se saltea lo que ya existe, asi se puede cortar y retomar):
    dataset_huellas/base/ImpresionDebil/id<K>/debil_id<K>_imp006.jpg ... imp030.jpg   (25 por identidad)
    dataset_huellas/base/ImpresionMedio/id<K>/medio_id<K>_imp031.jpg ... imp055.jpg   (25 por identidad)
    dataset_huellas/intrusas/intruso_01.jpg ... intruso_50.jpg   (impares: Debil, pares: Medio)
Las impresiones 1 a 5 de cada identidad ya estan en dataset_huellas/base/impresionPrimera (sin filtro fuerte).
Cada impresion es una distorsion NUEVA de la huella maestra de la identidad (Fingerprints/fp_1/<K>.jpg); cada intruso es
una persona nueva (huella maestra nueva). Los filtros son los de interfaz/anguli.py (FILTROS); aqui no se usa el nivel Fuerte.
"""
import argparse
import shutil
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "interfaz"))
import anguli as AN                                       # noqa: E402

DATASET = RAIZ / "dataset_huellas" / "base"
INTRUSAS = RAIZ / "dataset_huellas" / "intrusas"
NIVELES = {"Débil": ("ImpresionDebil", "debil", 6), "Medio": ("ImpresionMedio", "medio", 31)}   # nivel -> carpeta, prefijo, 1.a impresion
POR_NIVEL = 25                                            # 25 Debil + 25 Medio = 50 por identidad
N_INTRUSOS = 50
REINTENTOS = 3


def generar(tipo, nivel, destino):
    """Una huella con Anguli movida a `destino`; reintenta si Anguli falla. Devuelve True si quedo generada."""
    tmp = destino.parent / "_tmp"
    for intento in range(1, REINTENTOS + 1):
        try:
            r = AN.generar_huella(tipo, nivel, carpeta_destino=tmp)
        except AN.AnguliError as e:
            print(f"  ! {destino.name}: intento {intento}: {str(e)[:120]}")
            continue
        destino.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(r["ruta"]), str(destino))
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"  ok {destino.relative_to(RAIZ)} ({r['segundos']:.0f} s)", flush=True)
        return True
    return False


def impresiones():
    fallidas = 0
    for k in (1, 2, 3):
        for nivel, (carpeta, prefijo, primera) in NIVELES.items():
            for n in range(primera, primera + POR_NIVEL):
                destino = DATASET / carpeta / f"id{k}" / f"{prefijo}_id{k}_imp{n:03d}.jpg"
                if not destino.exists() and not generar(f"id{k}", nivel, destino):
                    fallidas += 1
    return fallidas


def intrusos():
    fallidos = 0
    niveles = list(NIVELES)
    for i in range(1, N_INTRUSOS + 1):
        destino = INTRUSAS / f"intruso_{i:02d}.jpg"
        if not destino.exists() and not generar("intruso", niveles[(i - 1) % 2], destino):
            fallidos += 1
    return fallidos


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--solo", choices=["impresiones", "intrusos"], default=None)
    a = ap.parse_args()
    fallos = 0
    if a.solo in (None, "impresiones"):
        fallos += impresiones()
    if a.solo in (None, "intrusos"):
        fallos += intrusos()
    total_i = len(list(DATASET.glob("Impresion*/id*/*.jpg")))
    total_x = len(list(INTRUSAS.glob("*.jpg")))
    print(f"Listo: {total_i} impresiones nuevas y {total_x} intrusos | fallidas: {fallos}")


if __name__ == "__main__":
    main()
