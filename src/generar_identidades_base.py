"""Genera con Anguli las 3 IDENTIDADES BASE (huellas maestras) y sus 5 impresiones "primera".

    python src/generar_identidades_base.py                           # deja todo en dataset_huellas/base/
    python src/generar_identidades_base.py --destino otra_carpeta    # para probar sin tocar el dataset
    python src/generar_identidades_base.py --clases left,whirl,arch --semillas 101,202,303

Por que clases distintas: las tres identidades del primer dataset eran las tres "Whorl" (espirales) y se parecian mucho,
asi que sus clusters quedaban cerca. Aca cada identidad es de una clase de huella distinta (Anguli: arch, tarch, left,
right, dloop, whirl), lo que separa mas los patrones de crestas. Los intrusos (dataset_huellas/intrusas/) NO se tocan.

Por cada identidad k se hace UNA corrida de Anguli (una huella maestra nueva + 5 impresiones, con los mismos parametros
que usaba el dataset: giro 30, traslacion 16, 0-20 rayones, ruido 0-20) y se deja:
    <destino>/Fingerprints/fp_1/<k>.jpg                          huella maestra (la usa interfaz/anguli.py para las impresiones nuevas)
    <destino>/Meta Info/fp_1/<k>.txt                             tipo de huella y puntos singulares
    <destino>/impresionPrimera/id<k>/primera_id<k>_imp<1..5>.jpg
Despues hay que generar el resto con src/generar_dataset_huellas.py.
"""
import argparse
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "interfaz"))
import anguli as AN                                       # noqa: E402
import config                                             # noqa: E402

PARAMS_PRIMERA = {"rot": 30, "trans": 16, "scratch": (0, 20), "noise": (0, 20)}
CLASES_VALIDAS = ("natural", "arch", "tarch", "right", "left", "dloop", "whirl")


def generar_identidad(k, clase, semilla, destino):
    exe = config.ruta_anguli()
    tmp = Path(tempfile.mkdtemp(prefix="anguli_base_"))
    salida = tmp / "salida"
    try:
        comando = [str(exe), "-num", "1", "-cdist", clase, "-ni", "5", "-meta", "-outdir", str(salida), "-numT", "1",
                   "-seed", str(semilla), "-itype", "jpg"] + AN._argumentos_filtro(PARAMS_PRIMERA)
        r = subprocess.run(comando, cwd=str(exe.parent), capture_output=True, text=True, errors="replace",
                           timeout=config.TIEMPO_MAX_ANGULI_S * 2, creationflags=AN.SIN_VENTANA)
        if r.returncode != 0:
            raise SystemExit(f"Anguli fallo con la identidad {k}: {(r.stdout + r.stderr)[-300:]}")
        maestra = next((salida / "Fingerprints").rglob("*.jpg"))
        (destino / "Fingerprints" / "fp_1").mkdir(parents=True, exist_ok=True)
        shutil.copy(maestra, destino / "Fingerprints" / "fp_1" / f"{k}.jpg")
        meta = next((salida / "Meta Info").rglob("*.txt"), None)
        if meta is not None:
            (destino / "Meta Info" / "fp_1").mkdir(parents=True, exist_ok=True)
            shutil.copy(meta, destino / "Meta Info" / "fp_1" / f"{k}.txt")
        carpeta = destino / "impresionPrimera" / f"id{k}"
        carpeta.mkdir(parents=True, exist_ok=True)
        for n in range(1, 6):
            imagen = next((salida / f"Impression_{n}").rglob("*.jpg"))
            shutil.copy(imagen, carpeta / f"primera_id{k}_imp{n}.jpg")
        tipo = meta.read_text(encoding="utf-8", errors="replace").splitlines()[0] if meta else "?"
        print(f"  identidad {k}: clase pedida '{clase}' (semilla {semilla}) -> Anguli: {tipo}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--destino", type=Path, default=RAIZ / "dataset_huellas" / "base")
    ap.add_argument("--clases", default="left,whirl,arch", help="clase de huella de cada identidad (3, separadas por coma)")
    ap.add_argument("--semillas", default=None, help="semilla de cada identidad (3 numeros); por defecto al azar")
    a = ap.parse_args()
    clases = [c.strip() for c in a.clases.split(",")]
    if len(clases) != 3 or any(c not in CLASES_VALIDAS for c in clases):
        sys.exit(f"--clases necesita 3 valores de {CLASES_VALIDAS}")
    semillas = [int(s) for s in a.semillas.split(",")] if a.semillas else [random.randrange(1, 2_000_000_000) for _ in range(3)]
    print("Generando las 3 identidades base en", a.destino)
    for k in (1, 2, 3):
        generar_identidad(k, clases[k - 1], semillas[k - 1], a.destino)
    print("Semillas usadas:", semillas)


if __name__ == "__main__":
    main()
