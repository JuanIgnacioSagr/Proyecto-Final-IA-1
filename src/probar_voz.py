"""Prueba en vivo del reconocimiento de voz: graba un audio y dice QUE ENTENDIO.

Es el paso previo a la aplicacion final. Hace exactamente lo mismo que se hizo con el dataset:

    grabar (16 kHz, mono, 1,5 s)  ->  normalizar_audio (filtro, recorte de silencios, volumen)
        ->  caracterizar_audio (MFCC)  ->  KNN con distancia euclidea (modelo_knn.npz)  ->  comando

Uso:
    python src/probar_voz.py                          # graba, clasifica y repite
    python src/probar_voz.py --dispositivo usb        # elegir microfono por numero o por parte del nombre
    python src/probar_voz.py --comando salir          # vas a decir "salir" todas las veces: cuenta los aciertos
    python src/probar_voz.py --guardar                # guarda cada grabacion en pruebas_voz/
    python src/probar_voz.py --archivo mi_audio.wav   # clasifica un WAV existente (sin microfono)

En la consola: ENTER = grabar, q = salir.

IMPORTANTE: el sistema solo conoce 3 comandos (reconocer, identificar, salir). Si decis otra cosa, o si hay
ruido, igual elige el MAS PARECIDO: todavia no existe la clase "otros" que le permita decir "no es un comando".
Los votos de los vecinos sirven como indicador de confianza.
"""
import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

import grabar_audios as G
import knn_audios as KA

RAIZ = Path(__file__).resolve().parent.parent
CARPETA_PRUEBAS = RAIZ / "pruebas_voz"


def resolver_dispositivo(texto):
    """'3' -> 3 ; 'usb' -> el microfono de entrada cuyo nombre contiene 'usb' ; None -> el de por defecto."""
    if texto is None:
        return None
    if texto.isdigit():
        return int(texto)
    candidatos = [(i, d) for i, d in enumerate(sd.query_devices())
                  if d["max_input_channels"] > 0 and texto.lower() in d["name"].lower()]
    if not candidatos:
        sys.exit(f"No hay ningun microfono con '{texto}' en el nombre. Mira la lista con: python src/grabar_audios.py --listar")
    # el mismo microfono aparece con varios "host API" (MME, DirectSound, ...); se prefiere MME, el de siempre
    candidatos.sort(key=lambda c: sd.query_hostapis(c[1]["hostapi"])["name"] != "MME")
    return candidatos[0][0]


def nombre_microfono(dispositivo):
    idx = dispositivo if dispositivo is not None else sd.default.device[0]
    return f"{idx}: {sd.query_devices(idx)['name']}"


def mostrar_resultado(r):
    """Imprime, en lenguaje simple, que entendio el sistema y por que."""
    ancho = {"alta": "seguro: los 3 vecinos coinciden", "media": "POCO SEGURO: gano por mayoria",
             "baja": "NO SEGURO: no hubo mayoria"}[r["confianza"]]
    print(f"\n  >>> ENTENDI: {r['comando'].upper()}   ({ancho})")
    votos = ", ".join(f"{c} {v}" for c, v in r["votos"].items())
    print(f"      votos de los {len(r['vecinos'])} vecinos mas cercanos: {votos}")
    print("      vecinos (audio del dataset, su comando, distancia):")
    for archivo, comando, dist in r["vecinos"]:
        print(f"        {Path(archivo).name:<34} {comando:<12} {dist:6.2f}")


def procesar(x, fs, modelo, esperado=None):
    """Clasifica una grabacion. Devuelve el comando entendido, o None si no se escucho voz."""
    pico = float(np.max(np.abs(x))) if len(x) else 0.0
    if pico < G.PICO_MIN:
        print(f"\n  No escuche nada (volumen muy bajo, pico {pico:.3f}). Acercate al microfono y volve a intentar.")
        return None
    r = KA.clasificar_audio_detalle(x, fs, modelo)
    info = r["info"]
    if info.get("sin_voz"):
        print("\n  No detecte voz clara en la grabacion (solo ruido o silencio). Volve a intentar.")
        return None
    if pico > G.PICO_MAX:
        print("  [!] la grabacion saturo (muy fuerte); hablale un poco mas lejos al microfono.")
    print(f"  (pico {pico:.2f}; voz detectada: {info['dur_despues_s']:.2f} s de {info['dur_antes_s']:.2f} s)", end="")
    mostrar_resultado(r)
    if esperado is not None:
        if r["comando"] == esperado:
            print(f"      OK: coincide con lo esperado ({esperado}).")
        else:
            print(f"      X: se esperaba '{esperado}'.")
    return r["comando"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dispositivo", default=None, help="numero o parte del nombre del microfono (ver grabar_audios.py --listar)")
    ap.add_argument("--comando", choices=["reconocer", "identificar", "salir"], default=None,
                    help="comando que vas a decir en esta tanda; permite contar aciertos")
    ap.add_argument("--guardar", action="store_true", help="guarda cada grabacion en pruebas_voz/")
    ap.add_argument("--archivo", type=Path, default=None, help="clasificar un WAV existente (no usa el microfono)")
    ap.add_argument("--dur", type=float, default=G.DUR, help="segundos de grabacion (por defecto igual que el dataset)")
    a = ap.parse_args()

    if not KA.RUTA_MODELO.exists():
        sys.exit(f"No existe el modelo {KA.RUTA_MODELO}. Generalo con: python src/knn_audios.py")
    modelo = np.load(KA.RUTA_MODELO, allow_pickle=False)
    print(f"Modelo: KNN con distancia {modelo['distancia']}, K = {int(modelo['k'])}, "
          f"{len(modelo['y_ref'])} audios de referencia. Comandos: {', '.join(str(n) for n in modelo['nombres'])}.")

    # ---- modo archivo: sin microfono ----
    if a.archivo is not None:
        x, fs = sf.read(str(a.archivo), dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        print(f"Archivo: {a.archivo} ({len(x) / fs:.2f} s, {fs} Hz)")
        procesar(x, fs, modelo, a.comando)
        return

    # ---- modo microfono ----
    dispositivo = resolver_dispositivo(a.dispositivo)
    print(f"Microfono: {nombre_microfono(dispositivo)}")
    if a.comando:
        print(f"Vas a decir: {a.comando.upper()} (se cuentan los aciertos).")
    print("Aviso: solo conoce 3 comandos; si decis otra cosa elegira el mas parecido.\n")
    if a.guardar:
        CARPETA_PRUEBAS.mkdir(exist_ok=True)

    total, aciertos, entendidos = 0, 0, Counter()
    while True:
        try:
            op = input("ENTER = grabar, q = salir > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if op == "q":
            break
        G.cuenta_regresiva()
        x = G.grabar(a.dur, dispositivo)
        if a.guardar:
            ruta = CARPETA_PRUEBAS / f"prueba_{time.strftime('%Y%m%d_%H%M%S')}{'_' + a.comando if a.comando else ''}.wav"
            sf.write(str(ruta), x, G.FS, subtype="PCM_16")
        entendido = procesar(x, G.FS, modelo, a.comando)
        if entendido is not None:
            total += 1
            entendidos[entendido] += 1
            aciertos += entendido == a.comando

    if a.comando and total:
        print(f"\nResumen: dijiste '{a.comando}' {total} veces y lo entendio bien {aciertos} ({aciertos / total * 100:.0f}%).")
        print("         lo que entendio: " + ", ".join(f"{c} x{n}" for c, n in entendidos.most_common()))
    elif total:
        print(f"\nResumen: {total} intentos. " + ", ".join(f"{c} x{n}" for c, n in entendidos.most_common()))


if __name__ == "__main__":
    main()
