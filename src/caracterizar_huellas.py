"""Caracterizacion (vectorizacion) de las huellas preprocesadas + etiquetas.

    dataset_huellas/procesadas/<efecto>/id<N>/<nombre>.png        (solo lectura)
        -> dataset_huellas/caracteristicas/

dataset_huellas/caracteristicas/
    etiquetas.csv   una fila por huella: archivo, efecto, identidad, etiqueta
    clases.json     numero de cada identidad (identidad_1=0, identidad_2=1, identidad_3=2, intruso=3)
    dataset.npz     X (N x 104, rasgo principal NORMALIZADO) + X_crudo (N x 104, sin normalizar)
                    + X_fingercode (N x 256) + X_minucias (N x 18), y, archivo, efecto, identidad
    normalizacion.npz  media y desvio de cada columna (calculados con las huellas de entrenamiento) para normalizar
    vectores/       un .npz por huella con sus bloques de rasgos
    graficos/       lo llena graficar_pca.py
    _control.json   uso interno: parametros y huellas ya procesadas

Rasgo principal X (104 numeros) = 72 de orientacion + 32 de energia de Gabor:
  - orientacion (72): campo de orientacion de las crestas (tensor de estructura) -> histograma
    de orientacion ponderado por coherencia y tinta: global de 24 bins + 4 cuadrantes (2x2)
    de 12 bins, cada uno normalizado a suma 1.
  - Gabor compacto (32): energia de 8 filtros de Gabor en 4 cuadrantes (2x2) (FingerCode reducido).
Cada columna se NORMALIZA con z-score (media y desvio de las huellas de entrenamiento, ver particion_huellas.py) y
se divide por ESCALA_Z para que las distancias queden en el mismo orden que antes (R cerca de 0,4).
Con eso identifica mejor que los 72 solos (ver explicativos/evaluar_rasgos.py).
Bloques opcionales: FingerCode completo (energia de 8 filtros de Gabor en 4 anillos x 8
sectores) y densidad de minucias (terminaciones y bifurcaciones por zona 3x3).
OJO: si se agregan huellas, la media y el desvio cambian y hay que volver a entrenar K-means.

Al volver a ejecutar solo se procesan las huellas nuevas (o cuya imagen cambio).
Si cambian los parametros hay que usar --forzar.

Uso:
    python src/caracterizar_huellas.py              # procesa lo pendiente
    python src/caracterizar_huellas.py --simular    # solo muestra que haria
    python src/caracterizar_huellas.py --forzar     # reprocesa todo

En Python:
    from caracterizar_huellas import cargar_dataset, caracterizar_huella
    d = cargar_dataset();  X, y = d["X"], d["y"]
    bloques = caracterizar_huella(preprocesar_huella("huella.jpg"))   # para la demo
"""
import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize

import particion_huellas as PH
import preprocesar as P

RAIZ = Path(__file__).resolve().parent.parent
ENTRADA = RAIZ / "dataset_huellas" / "procesadas"
SALIDA = RAIZ / "dataset_huellas" / "caracteristicas"

PARAMS = {
    "version": 1,
    "suavizado_sigma": 1.5,       # suavizado de la huella binaria antes del campo de orientacion
    "tensor_sigma": 4.0,          # suavizado del tensor de estructura
    "bins_global": 24,
    "bins_region": 12,
    "rejilla_orientacion": 2,     # 2x2 cuadrantes
    "fingercode_anillos": 4,
    "fingercode_sectores": 8,
    "fingercode_orientaciones": 8,
    "fingercode_periodo_px": 9.0,
    "gabor_compacto_rejilla": 2,  # 2x2 regiones x 8 orientaciones = 32 numeros (se suman al rasgo principal)
    "minucias_rejilla": 3,
    "area_ventana_px": 25,
    "area_umbral": 0.05,
}
CLASES_BASE = ["identidad_1", "identidad_2", "identidad_3", "intruso"]
DIM_ORIENT = PARAMS["bins_global"] + PARAMS["rejilla_orientacion"] ** 2 * PARAMS["bins_region"]   # 24 + 48 = 72
DIM_GC = PARAMS["gabor_compacto_rejilla"] ** 2 * PARAMS["fingercode_orientaciones"]                # 4 * 8 = 32
DIM_X = DIM_ORIENT + DIM_GC                                                                        # 104 (rasgo principal)
ESCALA_Z = 33.0                         # divisor final: deja las distancias en el orden de magnitud anterior (R ~ 0,4)
RUTA_NORM = SALIDA / "normalizacion.npz"
DIM_FC = PARAMS["fingercode_anillos"] * PARAMS["fingercode_sectores"] * PARAMS["fingercode_orientaciones"]  # 256
DIM_MIN = 2 * PARAMS["minucias_rejilla"] ** 2                                                              # 18


# --------------------------------------------------------------------------
# Rasgos (funciones puras; reutilizables en la demo)
# --------------------------------------------------------------------------
def _tinta(binaria):
    return (np.asarray(binaria) == 0).astype(np.float32)


def _zona_huella(tinta):
    """Mascara booleana de la zona con crestas (densidad local de tinta)."""
    v = PARAMS["area_ventana_px"]
    return cv2.blur(tinta, (v, v)) > PARAMS["area_umbral"]


def _histograma(alfa, peso, bins):
    h, _ = np.histogram((alfa % np.pi).ravel(), bins=bins, range=(0, np.pi), weights=peso.ravel())
    return h / (h.sum() + 1e-9)


def rasgo_orientacion(binaria):
    """72 numeros: histograma de orientacion global (24) + por cuadrante 2x2 (4 x 12)."""
    p = PARAMS
    t = _tinta(binaria)
    alfa, coh = P.campo_orientacion(cv2.GaussianBlur(t, (0, 0), p["suavizado_sigma"]), sigma=p["tensor_sigma"])
    peso = coh * t
    H, W = t.shape
    g = p["rejilla_orientacion"]
    v = [_histograma(alfa, peso, p["bins_global"])]
    for gy in range(g):
        for gx in range(g):
            sl = (slice(gy * H // g, (gy + 1) * H // g), slice(gx * W // g, (gx + 1) * W // g))
            v.append(_histograma(alfa[sl], peso[sl], p["bins_region"]))
    return np.concatenate(v).astype(np.float32)


def _banco_gabor():
    p = PARAMS
    lam = p["fingercode_periodo_px"]
    sig = 0.5 * lam
    r = int(2.5 * sig) + 1
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    env = np.exp(-(xx ** 2 + yy ** 2) / (2 * sig ** 2))
    ks = []
    for i in range(p["fingercode_orientaciones"]):
        a = np.pi * i / p["fingercode_orientaciones"]
        k = env * np.cos(2 * np.pi * (xx * np.cos(a) + yy * np.sin(a)) / lam)
        k -= k.mean()
        ks.append(k / np.abs(k).sum())
    return ks


_GABOR = _banco_gabor()


def rasgo_fingercode(binaria):
    """256 numeros: energia de Gabor (8 orientaciones) en 4 anillos x 8 sectores centrados en la imagen."""
    p = PARAMS
    t = _tinta(binaria)
    H, W = t.shape
    resp = [np.abs(cv2.filter2D(t, cv2.CV_32F, k)) for k in _GABOR]
    yy, xx = np.mgrid[:H, :W]
    r = np.hypot(yy - H / 2, xx - W / 2)
    th = (np.arctan2(yy - H / 2, xx - W / 2) + np.pi) / (2 * np.pi)
    rmax = min(H, W) / 2 * 1.15
    zona = _zona_huella(t)
    na, ns = p["fingercode_anillos"], p["fingercode_sectores"]
    v = []
    for i in range(na):
        for j in range(ns):
            m = (r >= i * rmax / na) & (r < (i + 1) * rmax / na) & (th >= j / ns) & (th < (j + 1) / ns) & zona
            for R in resp:
                v.append(R[m].mean() if m.sum() > 20 else 0.0)
    return np.array(v, dtype=np.float32)


def rasgo_gabor_compacto(binaria):
    """32 numeros: energia de Gabor (8 orientaciones) en 2x2 regiones de la huella (FingerCode reducido)."""
    g = PARAMS["gabor_compacto_rejilla"]
    t = _tinta(binaria)
    H, W = t.shape
    zona = _zona_huella(t)
    resp = [np.abs(cv2.filter2D(t, cv2.CV_32F, k)) for k in _GABOR]
    v = []
    for gy in range(g):
        for gx in range(g):
            sl = (slice(gy * H // g, (gy + 1) * H // g), slice(gx * W // g, (gx + 1) * W // g))
            z = zona[sl]
            v += [float(R[sl][z].mean()) if z.sum() > 50 else 0.0 for R in resp]
    return np.array(v, dtype=np.float32)


def rasgo_minucias(binaria):
    """18 numeros: terminaciones y bifurcaciones por 1000 px de huella, en una rejilla 3x3."""
    p = PARAMS
    t = _tinta(binaria)
    H, W = t.shape
    sk = skeletonize(t > 0).astype(np.uint8)
    vecinos = cv2.filter2D(sk, -1, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT) - sk
    term, bif = (sk == 1) & (vecinos == 1), (sk == 1) & (vecinos >= 3)
    zona = _zona_huella(t)
    g = p["minucias_rejilla"]
    v = []
    for gy in range(g):
        for gx in range(g):
            sl = (slice(gy * H // g, (gy + 1) * H // g), slice(gx * W // g, (gx + 1) * W // g))
            a = max(int(zona[sl].sum()), 1)
            v += [term[sl].sum() / a * 1000.0, bif[sl].sum() / a * 1000.0]
    return np.array(v, dtype=np.float32)


def caracterizar_huella(binaria):
    """Imagen binaria preprocesada (crestas=0, fondo=255) -> dict con los bloques de rasgos."""
    binaria = np.asarray(binaria)
    return {"orient": rasgo_orientacion(binaria),
            "gabor_c": rasgo_gabor_compacto(binaria),
            "fingercode": rasgo_fingercode(binaria),
            "minucias": rasgo_minucias(binaria)}


# --------------------------------------------------------------------------
# Vector que usa el modelo (104 numeros normalizados)
# --------------------------------------------------------------------------
def vector_crudo(bloques):
    """Une los bloques del rasgo principal (72 de orientacion + 32 de Gabor) SIN normalizar."""
    return np.concatenate([bloques["orient"], bloques["gabor_c"]]).astype(np.float64)


def calcular_normalizacion(X_crudo, entrenamiento):
    """Media y desvio por columna usando solo las huellas de entrenamiento. El desvio tiene un piso para no inflar columnas casi constantes."""
    mu, sd = X_crudo[entrenamiento].mean(axis=0), X_crudo[entrenamiento].std(axis=0)
    return mu, np.maximum(sd, 0.1 * np.median(sd) + 1e-9)


def normalizar(X_crudo, media, desvio):
    return (np.asarray(X_crudo, dtype=np.float64) - media) / desvio / ESCALA_Z


_NORM_CACHE = {}


def cargar_normalizacion(ruta=None):
    ruta = Path(ruta or RUTA_NORM)
    if not ruta.exists():
        sys.exit(f"No existe {ruta}. Ejecuta antes: python src/caracterizar_huellas.py")
    sello = ruta.stat().st_mtime_ns
    if _NORM_CACHE.get("sello") != (ruta, sello):
        d = np.load(ruta, allow_pickle=False)
        _NORM_CACHE.update(sello=(ruta, sello), media=d["media"], desvio=d["desvio"])
    return _NORM_CACHE["media"], _NORM_CACHE["desvio"]


def rasgo_huella(binaria):
    """Imagen binaria preprocesada -> vector de 104 numeros NORMALIZADO, el que entiende el modelo K-means."""
    media, desvio = cargar_normalizacion()
    crudo = np.concatenate([rasgo_orientacion(binaria), rasgo_gabor_compacto(binaria)])
    return normalizar(crudo, media, desvio)


# --------------------------------------------------------------------------
# Carga de lo ya generado
# --------------------------------------------------------------------------
def cargar_dataset(carpeta=SALIDA):
    """-> dict con X (N,104 normalizado), X_crudo (N,104), X_fingercode (N,256), X_minucias (N,18), y, archivo, efecto, identidad"""
    d = np.load(Path(carpeta) / "dataset.npz", allow_pickle=False)
    return {k: d[k] for k in d.files}


# --------------------------------------------------------------------------
# Procesamiento incremental
# --------------------------------------------------------------------------
def _efecto(nombre_carpeta):
    return re.sub(r"^impresion", "", nombre_carpeta.lower())


def _escribir_atomico(ruta, escribir):
    tmp = ruta.with_name(ruta.name + ".tmp")
    with open(tmp, "wb") as f:
        escribir(f)
    tmp.replace(ruta)


def _asignar_clases(salida, identidades):
    """clases.json estable: lo ya asignado no cambia; identidades nuevas se agregan al final."""
    ruta = salida / "clases.json"
    clases = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}
    if not clases:
        clases = {c: i for i, c in enumerate(CLASES_BASE)}
    for n in sorted(identidades):
        nombre = f"identidad_{n}"
        if nombre not in clases:
            clases[nombre] = max(clases.values()) + 1
    ruta.write_text(json.dumps(clases, indent=2, ensure_ascii=False), encoding="utf-8")
    return clases


def _listar(entrada):
    return sorted(p for p in entrada.glob("*/id*/*.png") if re.fullmatch(r"id\d+", p.parent.name))


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
                     "carpetas de huellas no se tocan).")
        if not forzar:
            control = previo

    imagenes = _listar(entrada)
    pendientes = []
    for img in imagenes:
        rel = img.relative_to(entrada).as_posix()
        st = img.stat()
        sello = {"bytes": st.st_size, "mtime_ns": st.st_mtime_ns}
        npz = salida / "vectores" / Path(rel).with_suffix(".npz")
        if forzar or control["archivos"].get(rel) != sello or not npz.exists():
            pendientes.append((img, rel, sello))
    existentes = {i.relative_to(entrada).as_posix() for i in imagenes}
    huerfanos = [k for k in control["archivos"] if k not in existentes]
    print(f"Huellas preprocesadas: {len(imagenes)} | a procesar: {len(pendientes)} | ya hechas: {len(imagenes) - len(pendientes)}"
          + (f" | sin original (excluidas del dataset, vector conservado): {len(huerfanos)}" if huerfanos else ""))
    if simular:
        for _, rel, _ in pendientes[:15]:
            print("  pendiente:", rel)
        if len(pendientes) > 15:
            print(f"  ... y {len(pendientes) - 15} mas")
        print("(--simular: no se escribio nada)")
        return
    if not imagenes:
        print("No hay huellas preprocesadas (ejecuta primero preprocesar.py).")
        return

    (salida / "vectores").mkdir(parents=True, exist_ok=True)
    control["parametros"] = PARAMS
    for img, rel, sello in pendientes:
        bloques = caracterizar_huella(P.a_gris(img))
        npz = salida / "vectores" / Path(rel).with_suffix(".npz")
        npz.parent.mkdir(parents=True, exist_ok=True)
        _escribir_atomico(npz, lambda f, b=bloques: np.savez(f, **b))
        control["archivos"][rel] = sello

    # etiquetas.csv y dataset.npz: se reconstruyen completos con las huellas que existen hoy
    identidades = {int(i.parent.name[2:]) for i in imagenes}
    clases = _asignar_clases(salida, identidades)
    filas = []
    for img in imagenes:
        rel = img.relative_to(entrada).as_posix()
        b = np.load(salida / "vectores" / Path(rel).with_suffix(".npz"))
        ident = int(img.parent.name[2:])
        filas.append((rel, _efecto(img.parent.parent.name), ident, clases[f"identidad_{ident}"],
                      vector_crudo(b), b["fingercode"], b["minucias"]))
    # normalizacion: media y desvio SOLO con las huellas de entrenamiento (ver particion_huellas.py)
    crudo = np.stack([r[4] for r in filas])
    media, desvio = calcular_normalizacion(crudo, PH.es_entrenamiento([r[0] for r in filas]))
    _escribir_atomico(salida / "normalizacion.npz", lambda f: np.savez(f, media=media, desvio=desvio, escala=ESCALA_Z))
    _escribir_atomico(salida / "etiquetas.csv", lambda f: f.write(
        ("archivo,efecto,identidad,etiqueta\n" + "".join(f"{a},{e},{i},{y}\n" for a, e, i, y, *_ in filas)).encode("utf-8")))
    _escribir_atomico(salida / "dataset.npz", lambda f: np.savez(
        f, X=normalizar(crudo, media, desvio), X_crudo=crudo, X_fingercode=np.stack([r[5] for r in filas]),
        X_minucias=np.stack([r[6] for r in filas]),
        y=np.array([r[3] for r in filas], dtype=np.int64), identidad=np.array([r[2] for r in filas], dtype=np.int64),
        archivo=np.array([r[0] for r in filas]), efecto=np.array([r[1] for r in filas])))
    ruta_ctl.write_text(json.dumps(control, indent=1), encoding="utf-8")

    cuentas = ", ".join(f"identidad_{n}={sum(1 for r in filas if r[2] == n)}" for n in sorted(identidades))
    efectos = sorted({r[1] for r in filas})
    print(f"Procesadas {len(pendientes)} -> {salida}")
    print(f"Dataset: {len(filas)} huellas, X de {DIM_X} numeros (+ fingercode {DIM_FC}, minucias {DIM_MIN}) | {cuentas}")
    print(f"Efectos presentes: {', '.join(efectos)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--entrada", type=Path, default=ENTRADA)
    ap.add_argument("--salida", type=Path, default=SALIDA)
    ap.add_argument("--simular", action="store_true", help="muestra que haria, sin escribir")
    ap.add_argument("--forzar", action="store_true", help="reprocesa todo (necesario si cambian los parametros)")
    a = ap.parse_args()
    procesar(a.entrada, a.salida, a.forzar, a.simular)
