"""Caracterizacion (MFCC) de los audios normalizados + etiquetas.

    audios_normalizados/<comando>/<comando>_s<sesion>_<NNN>.wav     (solo lectura)
        -> caracteristicas/

caracteristicas/
    etiquetas.csv   una fila por audio: archivo, comando, etiqueta, sesion
    clases.json     numero de cada comando (reconocer=0, identificar=1, salir=2, otros=3)
    dataset.npz     LISTO PARA EL KNN: X (N x 91), y (N), archivo, comando, sesion
    secuencias/     un .npy por audio: matriz (tramos x 39) de MFCC+delta+delta-delta (para DTW)
    _control.json   uso interno: parametros y audios ya procesados

Pasos (informe, Fase 3): pre-enfasis 0,97 -> tramos de 25 ms / salto 10 ms con
ventana Hamming -> FFT 512 -> banco Mel de 40 filtros (80-7600 Hz) -> log ->
DCT-II sin C0 -> 13 MFCC -> delta y delta-delta = 39 numeros por tramo.
No hace z-score, PCA ni clasificacion: eso va dentro del pipeline del KNN,
ajustado solo con los datos de entrenamiento.

Al volver a ejecutar solo se procesan los audios nuevos (o cuyo WAV
normalizado cambio). Si cambian los parametros hay que usar --forzar.

Uso:
    python src/caracterizar_audios.py              # procesa lo pendiente
    python src/caracterizar_audios.py --simular    # solo muestra que haria
    python src/caracterizar_audios.py --forzar     # reprocesa todo

En Python:
    from caracterizar_audios import cargar_dataset, cargar_secuencias
    X, y, archivos, sesiones = cargar_dataset()
"""
import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.fft import dct

RAIZ = Path(__file__).resolve().parent.parent
ENTRADA = RAIZ / "audios_normalizados"
SALIDA = RAIZ / "caracteristicas"

PARAMS = {
    "version": 1,
    "fs": 16000,
    "preenfasis": 0.97,
    "tramo_muestras": 400,        # 25 ms
    "salto_muestras": 160,        # 10 ms
    "n_fft": 512,
    "n_mel": 40,
    "fmin_hz": 80,
    "fmax_hz": 7600,
    "n_mfcc": 13,                 # sin C0
    "delta_ventana": 2,
    "segmentos_vector": 4,        # cuartos temporales para el vector fijo X
}
CLASES_BASE = ["reconocer", "identificar", "salir", "otros"]
DIM_X = PARAMS["segmentos_vector"] * PARAMS["n_mfcc"] + 3 * PARAMS["n_mfcc"]   # 52 + 39 = 91


# --------------------------------------------------------------------------
# Caracterizacion (funciones puras, reutilizables en la demo)
# --------------------------------------------------------------------------
def _hz_a_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _mel_a_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def banco_mel():
    """Matriz (n_mel, n_fft/2+1) de filtros triangulares equiespaciados en Mel."""
    p = PARAMS
    f_fft = np.linspace(0, p["fs"] / 2, p["n_fft"] // 2 + 1)
    puntos = _mel_a_hz(np.linspace(_hz_a_mel(p["fmin_hz"]), _hz_a_mel(p["fmax_hz"]), p["n_mel"] + 2))
    H = np.zeros((p["n_mel"], len(f_fft)))
    for m in range(1, p["n_mel"] + 1):
        izq, cen, der = puntos[m - 1], puntos[m], puntos[m + 1]
        H[m - 1] = np.maximum(0.0, np.minimum((f_fft - izq) / (cen - izq), (der - f_fft) / (der - cen)))
    return H


_BANCO = banco_mel()


def _delta(C):
    k = PARAMS["delta_ventana"]
    pad = np.pad(C, ((k, k), (0, 0)), mode="edge")
    den = 2.0 * sum(i * i for i in range(1, k + 1))
    return sum(i * (pad[k + i:len(C) + k + i] - pad[k - i:len(C) + k - i]) for i in range(1, k + 1)) / den


def secuencia_mfcc(x):
    """x: audio float mono a 16 kHz -> matriz (tramos, 39) = [MFCC13, d13, dd13]."""
    p = PARAMS
    x = np.asarray(x, dtype=np.float64)
    x = np.append(x[0], x[1:] - p["preenfasis"] * x[:-1]) if len(x) else x
    n = p["tramo_muestras"]
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    idx = np.arange(0, len(x) - n + 1, p["salto_muestras"])
    ventana = signal.get_window("hamming", n)
    tramos = np.stack([x[i:i + n] * ventana for i in idx])
    potencia = np.abs(np.fft.rfft(tramos, p["n_fft"])) ** 2 / p["n_fft"]
    log_mel = np.log(potencia @ _BANCO.T + 1e-10)
    C = dct(log_mel, type=2, axis=1, norm="ortho")[:, 1:p["n_mfcc"] + 1]       # sin C0
    D = _delta(C)
    return np.hstack([C, D, _delta(D)]).astype(np.float32)


def vector_fijo(seq):
    """Vector de largo fijo (91) para un KNN euclideo rapido: media de los 13 MFCC en
    cada cuarto del audio (conserva algo de orden temporal) + desvio de los 39 coeficientes."""
    C = seq[:, :PARAMS["n_mfcc"]]
    if len(C) < PARAMS["segmentos_vector"]:
        C = np.repeat(C, PARAMS["segmentos_vector"], axis=0)
    medias = [s.mean(axis=0) for s in np.array_split(C, PARAMS["segmentos_vector"])]
    return np.concatenate(medias + [seq.std(axis=0)]).astype(np.float32)


def caracterizar_audio(x, fs=16000):
    """Audio ya normalizado (ver normalizar_audios.normalizar_audio) -> (secuencia, vector)."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if fs != PARAMS["fs"]:
        g = math.gcd(int(fs), PARAMS["fs"])
        x = signal.resample_poly(x, PARAMS["fs"] // g, int(fs) // g).astype(np.float32)
    seq = secuencia_mfcc(x)
    return seq, vector_fijo(seq)


def cmn(seq):
    """Resta la media de cada coeficiente en el audio (normalizacion cepstral de media)."""
    return seq - seq.mean(axis=0, keepdims=True)


# --------------------------------------------------------------------------
# Carga de lo ya generado (para el KNN)
# --------------------------------------------------------------------------
def cargar_dataset(carpeta=SALIDA):
    """-> X (N,91), y (N,), archivos (N,), sesiones (N,)"""
    d = np.load(Path(carpeta) / "dataset.npz", allow_pickle=False)
    return d["X"], d["y"], d["archivo"], d["sesion"]


def cargar_secuencias(carpeta=SALIDA):
    """-> lista de matrices (tramos,39), y (N,), archivos (N,)"""
    carpeta = Path(carpeta)
    d = np.load(carpeta / "dataset.npz", allow_pickle=False)
    seqs = [np.load(carpeta / "secuencias" / Path(a).with_suffix(".npy")) for a in d["archivo"]]
    return seqs, d["y"], d["archivo"]


# --------------------------------------------------------------------------
# Procesamiento incremental
# --------------------------------------------------------------------------
def _sesion(comando, stem):
    m = re.fullmatch(rf"{re.escape(comando)}_s(.+)_\d+", stem)
    return m.group(1) if m else "?"


def _escribir_atomico(ruta, escribir):
    tmp = ruta.with_name(ruta.name + ".tmp")
    with open(tmp, "wb") as f:
        escribir(f)
    tmp.replace(ruta)


def _asignar_clases(salida, comandos):
    """clases.json estable: los numeros ya asignados no cambian; los nuevos se agregan al final."""
    ruta = salida / "clases.json"
    clases = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}
    if not clases:
        clases = {c: i for i, c in enumerate(CLASES_BASE)}
    for c in sorted(comandos):
        if c not in clases:
            clases[c] = max(clases.values()) + 1
    ruta.write_text(json.dumps(clases, indent=2, ensure_ascii=False), encoding="utf-8")
    return clases


def procesar(entrada=ENTRADA, salida=SALIDA, forzar=False, simular=False):
    if not entrada.exists():
        sys.exit(f"No existe la carpeta de entrada: {entrada}")
    ruta_ctl = salida / "_control.json"
    control = {"parametros": PARAMS, "archivos": {}}
    if ruta_ctl.exists():
        previo = json.loads(ruta_ctl.read_text(encoding="utf-8"))
        if previo["parametros"] != PARAMS and not forzar:
            sys.exit("Los parametros de caracterizacion cambiaron respecto de la corrida anterior.\n"
                     "Para no mezclar criterios, ejecuta con --forzar (reprocesa todo; las\n"
                     "carpetas de audio no se tocan).")
        if not forzar:
            control = previo

    wavs = sorted(entrada.glob("*/*.wav"))
    pendientes = []
    for w in wavs:
        rel = f"{w.parent.name}/{w.name}"
        st = w.stat()
        sello = {"bytes": st.st_size, "mtime_ns": st.st_mtime_ns}
        previo = control["archivos"].get(rel)
        npy = salida / "secuencias" / Path(rel).with_suffix(".npy")
        if forzar or previo is None or previo != sello or not npy.exists():
            pendientes.append((w, rel, sello))
    existentes = {f"{w.parent.name}/{w.name}" for w in wavs}
    huerfanos = [k for k in control["archivos"] if k not in existentes]
    print(f"Audios normalizados: {len(wavs)} | a procesar: {len(pendientes)} | ya hechos: {len(wavs) - len(pendientes)}"
          + (f" | sin original (excluidos del dataset, .npy conservado): {len(huerfanos)}" if huerfanos else ""))
    if simular:
        for _, rel, _ in pendientes[:15]:
            print("  pendiente:", rel)
        if len(pendientes) > 15:
            print(f"  ... y {len(pendientes) - 15} mas")
        print("(--simular: no se escribio nada)")
        return
    if not wavs:
        print("No hay audios normalizados (ejecuta primero normalizar_audios.py).")
        return

    (salida / "secuencias").mkdir(parents=True, exist_ok=True)
    control["parametros"] = PARAMS
    for w, rel, sello in pendientes:
        x, fs = sf.read(str(w), dtype="float32")
        seq, _ = caracterizar_audio(x, fs)
        npy = salida / "secuencias" / Path(rel).with_suffix(".npy")
        npy.parent.mkdir(parents=True, exist_ok=True)
        _escribir_atomico(npy, lambda f, s=seq: np.save(f, s))
        control["archivos"][rel] = sello

    # etiquetas.csv y dataset.npz: se reconstruyen completos con los audios que existen hoy
    comandos = {w.parent.name for w in wavs}
    clases = _asignar_clases(salida, comandos)
    filas = []
    for w in wavs:
        rel = f"{w.parent.name}/{w.name}"
        seq = np.load(salida / "secuencias" / Path(rel).with_suffix(".npy"))
        filas.append((rel, w.parent.name, clases[w.parent.name], _sesion(w.parent.name, w.stem), vector_fijo(seq)))
    _escribir_atomico(salida / "etiquetas.csv", lambda f: f.write(
        ("archivo,comando,etiqueta,sesion\n" + "".join(f"{a},{c},{e},{s}\n" for a, c, e, s, _ in filas)).encode("utf-8")))
    _escribir_atomico(salida / "dataset.npz", lambda f: np.savez(
        f, X=np.stack([v for *_, v in filas]).astype(np.float32),
        y=np.array([e for _, _, e, _, _ in filas], dtype=np.int64),
        archivo=np.array([a for a, *_ in filas]), comando=np.array([c for _, c, *_ in filas]),
        sesion=np.array([s for _, _, _, s, _ in filas])))
    ruta_ctl.write_text(json.dumps(control, indent=1), encoding="utf-8")

    y = np.array([e for _, _, e, _, _ in filas])
    cuentas = ", ".join(f"{c}={int((y == i).sum())}" for c, i in clases.items() if (y == i).any())
    print(f"Procesados {len(pendientes)} -> {salida}")
    print(f"Dataset: {len(filas)} audios, X de {DIM_X} numeros por audio | {cuentas}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--entrada", type=Path, default=ENTRADA)
    ap.add_argument("--salida", type=Path, default=SALIDA)
    ap.add_argument("--simular", action="store_true", help="muestra que haria, sin escribir")
    ap.add_argument("--forzar", action="store_true", help="reprocesa todo (necesario si cambian los parametros)")
    a = ap.parse_args()
    procesar(a.entrada, a.salida, a.forzar, a.simular)
