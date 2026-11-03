"""Genera huellas de INTRUSOS con Anguli (personas nuevas que no estan en el sistema) para evaluar el rechazo.

    python explicativos/generar_intrusos.py            # 18 intrusos (6 por nivel de filtro: debil, medio, fuerte)
    python explicativos/generar_intrusos.py --n 30     # otra cantidad (se reparte entre los 3 niveles)

Cada intruso es una huella maestra NUEVA de Anguli (otra persona) con un filtro al azar del nivel indicado.
Se guardan en explicativos/intrusos_generados/ (nombre intruso_<nivel>_<n>.jpg) y se saltean los que ya existen,
asi se puede cortar y retomar. Tarda unos 10-20 s por intruso.
"""
import argparse
import shutil
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "interfaz"))
import anguli as AN                                    # noqa: E402

DESTINO = Path(__file__).resolve().parent / "intrusos_generados"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=18)
    a = ap.parse_args()
    DESTINO.mkdir(parents=True, exist_ok=True)
    niveles = list(AN.FILTROS)
    por_nivel = a.n // len(niveles)
    for nivel in niveles:
        for i in range(1, por_nivel + 1):
            final = DESTINO / f"intruso_{nivel.lower().replace('é', 'e')}_{i:02d}.jpg"
            if final.exists():
                continue
            tmp = DESTINO / "_tmp"
            try:
                r = AN.generar_huella("intruso", nivel, carpeta_destino=tmp)
            except AN.AnguliError as e:
                print(f"  ! {final.name}: {e}")
                continue
            shutil.move(str(r["ruta"]), str(final))
            print(f"  ok {final.name} ({r['segundos']:.0f} s)")
    shutil.rmtree(DESTINO / "_tmp", ignore_errors=True)
    print("Listo:", len(list(DESTINO.glob('*.jpg'))), "intrusos en", DESTINO)


if __name__ == "__main__":
    main()
