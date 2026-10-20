"""Acondicionamiento (filtrado + normalizacion) de los audios del dataset de voz.

    audios/<comando>/<comando>_s<sesion>_<NNN>.wav          (solo lectura)
        -> audios_normalizados/<comando>/<mismo nombre>.wav

Pasos por archivo: mono 16 kHz -> quitar DC -> pasa-banda Butterworth
(SOS, fase cero) -> recorte de silencios (VAD por energia) -> normalizacion RMS
del tramo con voz a -20 dBFS con limitador de pico.
NO hace pre-enfasis, MFCC ni nada de caracterizacion.

La carpeta `audios/` nunca se modifica. Al volver a ejecutar solo se procesan
los archivos nuevos (o cuyo original cambio); si cambian los parametros del
pipeline hay que usar --forzar para no mezclar criterios.

Uso:
    python src/normalizar_audios.py               # procesa lo pendiente
    python src/normalizar_audios.py --simular     # solo muestra que haria
    python src/normalizar_audios.py --forzar      # reprocesa todo

`normalizar_audio(x, fs)` es reutilizable para condicionar el audio del
microfono en la demo con exactamente el mismo codigo.
"""
import argparse
import csv
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import ndimage, signal

RAIZ = Path(__file__).resolve().parent.parent
ENTRADA = RAIZ / "audios"
SALIDA = RAIZ / "audios_normalizados"

PARAMS = {
    "version": 2,
    "fs": 16000,
    "banda_hz": [80, 7000],
    "orden_butterworth": 4,
    "vad_tramo_ms": 20,
    "vad_salto_ms": 10,
    "vad_margen_ms": 100,
    "vad_min_racha_ms": 60,        # rachas activas mas cortas = clic/golpe, no voz
    "vad_sobre_piso_db": 12,       # umbral >= piso de ruido + 12 dB
    "vad_bajo_ref_db": 30,         # umbral >= referencia - 30 dB
    "vad_min_dinamica_db": 15,     # si referencia - piso < 15 dB: sin voz
    "rms_objetivo_dbfs": -20.0,
    "ganancia_max_db": 30.0,
    "pico_max": 0.99,
}

CAMPOS = ["archivo", "comando", "sesion", "src_bytes", "src_mtime_ns",
          "dur_antes_s", "dur_despues_s", "pico_antes", "pico_despues",
          "rms_voz_antes_dbfs", "rms_voz_despues_dbfs", "ganancia_db",
          "sin_voz", "saturado_origen", "limitado", "procesado_en"]


# --------------------------------------------------------------------------
# Pipeline (funciones puras)
# --------------------------------------------------------------------------
def _a_mono_16k(x, fs):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if fs != PARAMS["fs"]:
        g = math.gcd(int(fs), PARAMS["fs"])
        x = signal.resample_poly(x, PARAMS["fs"] // g, int(fs) // g).astype(np.float32)
    return x


def _pasabanda(x):
    sos = signal.butter(PARAMS["orden_butterworth"], PARAMS["banda_hz"],
                        btype="band", fs=PARAMS["fs"], output="sos")
    if len(x) < 64:  # sosfiltfilt necesita algo de largo
        return x
    return signal.sosfiltfilt(sos, x).astype(np.float32)


def _energia_db(x):
    """Energia (dB) por tramo de vad_tramo_ms con salto vad_salto_ms."""
    fs = PARAMS["fs"]
    n = int(fs * PARAMS["vad_tramo_ms"] / 1000)
    h = int(fs * PARAMS["vad_salto_ms"] / 1000)
    if len(x) < n:
        return np.array([10 * np.log10(np.mean(x ** 2) + 1e-12)]), n, h
    idx = np.arange(0, len(x) - n + 1, h)
    e = np.array([np.mean(x[i:i + n] ** 2) for i in idx])
    return 10 * np.log10(e + 1e-12), n, h


def _rachas_largas(activo, min_tramos):
    """Indices de tramos activos que pertenecen a rachas de >= min_tramos consecutivos
    (descarta clics y golpes aislados, que duran 1-3 tramos)."""
    etiquetas, n = ndimage.label(activo)
    if n == 0:
        return np.array([], dtype=int)
    largos = np.bincount(etiquetas.ravel())
    ok = largos >= min_tramos
    ok[0] = False
    return np.where(ok[etiquetas])[0]


def _vad(x):
    """Devuelve (inicio, fin) en muestras del tramo con voz, o None si no hay.

    La referencia es el percentil 95 de la energia (no el maximo) y se exige una
    racha minima de tramos activos, para que un golpe o clic aislado no cuente como voz.
    """
    e, n, h = _energia_db(x)
    ref, piso = np.percentile(e, 95), np.percentile(e, 10)
    if ref - piso < PARAMS["vad_min_dinamica_db"]:
        return None
    umbral = max(piso + PARAMS["vad_sobre_piso_db"], ref - PARAMS["vad_bajo_ref_db"])
    activos = _rachas_largas(e > umbral, int(round(PARAMS["vad_min_racha_ms"] / PARAMS["vad_salto_ms"])))
    if len(activos) == 0:
        return None
    margen = int(PARAMS["fs"] * PARAMS["vad_margen_ms"] / 1000)
    ini = max(0, activos[0] * h - margen)
    fin = min(len(x), activos[-1] * h + n + margen)
    return ini, fin, activos[0] * h, min(len(x), activos[-1] * h + n)


def _dbfs(v):
    return 20 * np.log10(max(float(v), 1e-9))


def normalizar_audio(x, fs):
    """x: audio (ndarray, mono o estereo) y su fs. Devuelve (y_float32, info)."""
    x = _a_mono_16k(x, fs)
    info = {"dur_antes_s": len(x) / PARAMS["fs"],
            "pico_antes": float(np.max(np.abs(x))) if len(x) else 0.0,
            "saturado_origen": bool(len(x) and np.max(np.abs(x)) >= 0.999)}
    x = x - np.mean(x) if len(x) else x                       # 1. quitar DC
    x = _pasabanda(x)                                         # 2. pasa-banda
    voz = _vad(x) if len(x) else None                         # 3. VAD
    info["sin_voz"] = voz is None
    if voz is None:
        seg, nucleo = x, x                                    # no recortar
    else:
        ini, fin, n_ini, n_fin = voz
        seg, nucleo = x[ini:fin], x[n_ini:n_fin]
    rms = float(np.sqrt(np.mean(nucleo ** 2))) if len(nucleo) else 0.0   # 4. RMS de la voz
    info["rms_voz_antes_dbfs"] = _dbfs(rms)
    g = 10 ** (PARAMS["rms_objetivo_dbfs"] / 20) / max(rms, 1e-9)
    g = min(g, 10 ** (PARAMS["ganancia_max_db"] / 20))
    pico = float(np.max(np.abs(seg))) if len(seg) else 0.0
    info["limitado"] = bool(pico * g > PARAMS["pico_max"])
    if info["limitado"]:
        g = PARAMS["pico_max"] / pico
    y = (seg * g).astype(np.float32)
    info.update(ganancia_db=_dbfs(g), dur_despues_s=len(y) / PARAMS["fs"],
                pico_despues=float(np.max(np.abs(y))) if len(y) else 0.0,
                rms_voz_despues_dbfs=_dbfs(rms * g))
    return y, info


# --------------------------------------------------------------------------
# Manejo de carpetas, registro e incrementalidad
# --------------------------------------------------------------------------
def _leer_registro(ruta):
    if not ruta.exists():
        return {}
    with open(ruta, newline="", encoding="utf-8") as f:
        return {r["archivo"]: r for r in csv.DictReader(f)}


def _escribir_registro(ruta, filas):
    tmp = ruta.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS)
        w.writeheader()
        for k in sorted(filas):
            w.writerow({c: filas[k].get(c, "") for c in CAMPOS})
    tmp.replace(ruta)


def _sesion(comando, stem):
    m = re.fullmatch(rf"{re.escape(comando)}_s(.+)_\d+", stem)
    return m.group(1) if m else "?"


def _pendientes(entrada, salida, registro, forzar):
    nuevos, cambiados = [], []
    for src in sorted(entrada.glob("*/*.wav")):
        rel = f"{src.parent.name}/{src.name}"
        st = src.stat()
        fila = registro.get(rel)
        if forzar or fila is None or not (salida / rel).exists():
            nuevos.append(src)
        elif (str(st.st_size) != fila["src_bytes"] or str(st.st_mtime_ns) != fila["src_mtime_ns"]):
            cambiados.append(src)
    return nuevos, cambiados


def procesar(entrada=ENTRADA, salida=SALIDA, forzar=False, simular=False):
    if not entrada.exists():
        sys.exit(f"No existe la carpeta de entrada: {entrada}")
    ruta_par, ruta_reg = salida / "_parametros.json", salida / "_registro.csv"

    if ruta_par.exists():
        previos = json.loads(ruta_par.read_text(encoding="utf-8"))
        if previos != PARAMS and not forzar:
            sys.exit("Los parametros del pipeline cambiaron respecto de la corrida anterior.\n"
                     "Para no mezclar criterios, ejecuta con --forzar (reprocesa todo; los\n"
                     "originales en audios/ no se tocan).")
    registro = {} if forzar else _leer_registro(ruta_reg)
    nuevos, cambiados = _pendientes(entrada, salida, registro, forzar)
    total = len(list(entrada.glob("*/*.wav")))
    huerfanos = [k for k in registro if not (entrada / k).exists()]
    pendientes = nuevos + cambiados
    print(f"Originales: {total} | a procesar: {len(pendientes)} "
          f"({len(nuevos)} nuevos, {len(cambiados)} modificados) | ya hechos: {total - len(pendientes)}"
          + (f" | huerfanos (original borrado, salida conservada): {len(huerfanos)}" if huerfanos else ""))
    if simular:
        for p in pendientes[:15]:
            print("  pendiente:", f"{p.parent.name}/{p.name}")
        if len(pendientes) > 15:
            print(f"  ... y {len(pendientes) - 15} mas")
        print("(--simular: no se escribio nada)")
        return registro
    if not pendientes:
        return registro

    salida.mkdir(parents=True, exist_ok=True)
    ruta_par.write_text(json.dumps(PARAMS, indent=2), encoding="utf-8")
    flags = {"sin_voz": 0, "saturado_origen": 0, "limitado": 0}
    for src in pendientes:
        rel = f"{src.parent.name}/{src.name}"
        x, fs = sf.read(str(src), dtype="float32", always_2d=False)
        y, info = normalizar_audio(x, fs)
        dst = salida / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dst), y, PARAMS["fs"], subtype="PCM_16")
        st = src.stat()
        registro[rel] = {"archivo": rel, "comando": src.parent.name,
                         "sesion": _sesion(src.parent.name, src.stem),
                         "src_bytes": st.st_size, "src_mtime_ns": st.st_mtime_ns,
                         "procesado_en": datetime.now().isoformat(timespec="seconds"),
                         **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in info.items()}}
        for k in flags:
            flags[k] += int(info[k])
    _escribir_registro(ruta_reg, registro)
    print(f"Procesados {len(pendientes)} -> {salida}")
    print("Avisos:", ", ".join(f"{k}={v}" for k, v in flags.items()),
          "(detalle por archivo en _registro.csv)")
    return registro


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--entrada", type=Path, default=ENTRADA)
    ap.add_argument("--salida", type=Path, default=SALIDA)
    ap.add_argument("--simular", action="store_true", help="muestra que haria, sin escribir")
    ap.add_argument("--forzar", action="store_true", help="reprocesa todo (necesario si cambian los parametros)")
    a = ap.parse_args()
    procesar(a.entrada, a.salida, a.forzar, a.simular)
