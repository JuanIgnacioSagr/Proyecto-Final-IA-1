"""Graba el dataset de audios para el KNN de voz.

Guarda WAV crudos (16 kHz, mono, 16 bit, duracion fija) en
    audios/<comando>/<comando>_s<sesion>_<NNN>.wav

Uso:
    python src/grabar_audios.py --listar
    python src/grabar_audios.py                       # recorre los 4 comandos
    python src/grabar_audios.py --comando reconocer --sesion 2_rapido --n 40

En la consola: ENTER = grabar, r = borrar la ultima y repetir,
s = pasar al siguiente comando, q = salir.
"""
import argparse
import re
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

FS = 16000          # Hz
DUR = 1.5           # segundos
N_POR_COMANDO = 40
COMANDOS = ("reconocer", "identificar", "salir", "otros")
PICO_MIN = 0.02     # por debajo: casi silencio
PICO_MAX = 0.99     # por encima: saturacion

SALIDA = Path(__file__).resolve().parent.parent / "audios"


def siguiente_numero(carpeta, comando, sesion):
    patron = re.compile(rf"{comando}_s{re.escape(sesion)}_(\d+)\.wav$")
    nums = [int(m.group(1)) for f in carpeta.glob("*.wav") if (m := patron.match(f.name))]
    return max(nums, default=0) + 1


def grabar(dur, dispositivo):
    x = sd.rec(int(dur * FS), samplerate=FS, channels=1, dtype="float32", device=dispositivo)
    sd.wait()
    return x[:, 0]


def cuenta_regresiva():
    for t in ("3", "2", "1"):
        print(f"  {t}...", end="\r", flush=True)
        time.sleep(0.35)
    print("  >>> HABLA <<<", flush=True)


def sesion_comando(comando, sesion, n, dur, dispositivo):
    """Devuelve 'q' si el usuario quiere salir del programa, '' en otro caso."""
    carpeta = SALIDA / comando
    carpeta.mkdir(parents=True, exist_ok=True)
    num = siguiente_numero(carpeta, comando, sesion)
    hechos, ultimo = 0, None
    indicacion = "silencio, ruido o OTRA palabra" if comando == "otros" else comando.upper()
    print(f"\n=== {comando.upper()} (sesion {sesion}, {n} muestras) ===")
    while hechos < n:
        op = input(f"Di: {indicacion}  [{hechos + 1}/{n}]  ENTER=grabar  r=repetir  s=siguiente  q=salir > ").strip().lower()
        if op == "q":
            return "q"
        if op == "s":
            return ""
        if op == "r":
            if ultimo is not None and ultimo.exists():
                ultimo.unlink()
                hechos -= 1
                num -= 1
                print(f"  borrado {ultimo.name}")
                ultimo = None
            else:
                print("  no hay nada que borrar")
            continue
        cuenta_regresiva()
        x = grabar(dur, dispositivo)
        pico = float(np.max(np.abs(x)))
        if pico < PICO_MIN and comando != "otros":
            print(f"  [!] casi silencio (pico {pico:.3f}); no se guardo. Acerca el microfono y repite.")
            continue
        if pico > PICO_MAX:
            print(f"  [!] saturado (pico {pico:.3f}); se guardo igual, conviene repetir con 'r'.")
        ruta = carpeta / f"{comando}_s{sesion}_{num:03d}.wav"
        sf.write(str(ruta), x, FS, subtype="PCM_16")
        print(f"  guardado {ruta.name}  (pico {pico:.2f})")
        ultimo, num, hechos = ruta, num + 1, hechos + 1
    return ""


def resumen():
    print("\nResumen del dataset:")
    for c in COMANDOS:
        archivos = sorted((SALIDA / c).glob("*.wav"))
        por_sesion = {}
        for f in archivos:
            m = re.fullmatch(rf"{c}_s(.+)_\d+\.wav", f.name)
            k = m.group(1) if m else "?"
            por_sesion[k] = por_sesion.get(k, 0) + 1
        det = ", ".join(f"s{k}: {v}" for k, v in sorted(por_sesion.items())) or "-"
        print(f"  {c:<12} {len(archivos):>3}   ({det})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--comando", choices=COMANDOS, help="si se omite, recorre todos")
    ap.add_argument("--sesion", default="1",
                    help="nombre de la tanda: letras, numeros, - o _ (ej. 1, 2_rapido, 3_lejos); "
                         "sirve para separar train/test despues")
    ap.add_argument("--n", type=int, default=N_POR_COMANDO, help="muestras por comando en esta pasada")
    ap.add_argument("--dur", type=float, default=DUR)
    ap.add_argument("--dispositivo", type=int, default=None, help="ID del microfono (ver --listar)")
    ap.add_argument("--listar", action="store_true", help="muestra los dispositivos de audio y sale")
    args = ap.parse_args()

    if args.listar:
        print(sd.query_devices())
        print(f"\nPor defecto (entrada, salida): {sd.default.device}")
        return

    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.sesion):
        ap.error("--sesion solo admite letras, numeros, '-' y '_'")

    comandos = [args.comando] if args.comando else list(COMANDOS)
    for c in comandos:
        if sesion_comando(c, args.sesion, args.n, args.dur, args.dispositivo) == "q":
            break
    resumen()


if __name__ == "__main__":
    main()
