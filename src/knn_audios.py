"""KNN (K vecinos mas cercanos) para los comandos de voz (aprendizaje SUPERVISADO), implementado desde cero.

Que hace este script
--------------------
1. Carga los audios ya caracterizados (91 numeros por audio y la secuencia MFCC completa).
2. Los divide en TRES conjuntos, separados por PERSONA/sesion (nunca al azar):
     - entrenamiento: los ejemplos de referencia con su comando conocido;
     - validacion: se usa para ELEGIR K y la distancia (audios que el modelo no vio);
     - prueba: se usa UNA sola vez al final para medir el poder de generalizacion real.
3. Clasifica cada audio con KNN: calcula la distancia a todos los de referencia, toma los K
   mas cercanos y elige el comando mas frecuente entre ellos.
4. Usa la distancia EUCLIDEA sobre el vector de 91 numeros (decision del proyecto). DTW sigue disponible con
   --distancia dtw solo para comparar (explicativos/dtw_*.py), pero ya no es la que usa el modelo.
5. Muestra matrices de confusion y metricas de error, guarda el modelo y dibuja los graficos.

Los PASOS estan numerados igual que en el documento "Teoria metodos y forma implementada en el codigo.md".

Uso:
    python src/knn_audios.py
    python src/knn_audios.py --validacion mama,joaco --prueba maxi,mili --distancia euclidea
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # dibujar a archivo, sin ventana
import matplotlib.pyplot as plt
import numpy as np

import caracterizar_audios as CA
import metricas
import normalizar_audios as NA

RAIZ = Path(__file__).resolve().parent.parent
CARPETA = RAIZ / "dataset_audios" / "caracteristicas"
RUTA_MODELO = CARPETA / "modelo_knn.npz"

# ----------------------------- parametros del metodo -----------------------------
VALORES_K = (1, 3, 5, 7, 9, 11, 13, 15)   # K impares: evitan la mayoria de los empates
N_MFCC = 13              # DTW usa los 13 MFCC (sin deltas) de cada tramo
SUBMUESTREO = 2          # DTW usa 1 de cada 2 tramos (acelera el calculo sin perder la forma)
BANDA = 0.25             # banda de Sakoe-Chiba: DTW solo alinea tramos a menos del 25 % de distancia temporal
SESIONES_VALIDACION = "mama,joaco"
SESIONES_PRUEBA = "maxi,mili"


# =====================================================================================
#  Distancias
# =====================================================================================
def distancias_euclideas(A, B, bloque=4000):
    """Matriz de distancias euclideas entre cada fila de A y cada fila de B.

    distancia(a, b) = raiz( (a1-b1)^2 + (a2-b2)^2 + ... )
    Se procesa por bloques para no gastar demasiada memoria.
    """
    salida = np.empty((len(A), len(B)))
    for i in range(0, len(A), bloque):
        dif = A[i:i + bloque, None, :] - B[None, :, :]
        salida[i:i + bloque] = np.sqrt((dif ** 2).sum(axis=2))
    return salida


def preparar_secuencia(seq):
    """Secuencia (tramos x 39) -> los 13 MFCC con la media restada (CMN) y 1 de cada 2 tramos.

    Restar la media de cada coeficiente en el audio quita el efecto del microfono y del volumen.
    """
    return CA.cmn(seq[:, :N_MFCC])[::SUBMUESTREO]


def dtw(a, b, banda=BANDA):
    """Distancia DTW (Dynamic Time Warping) entre dos secuencias a (n x d) y b (m x d).

    Idea: comparar dos palabras "estirando" o "encogiendo" el tiempo para alinear sus partes.
    Se llena una tabla acum[i][j] = costo minimo de alinear los primeros i tramos de a con los
    primeros j tramos de b:

        acum[i][j] = d(i, j) + min( acum[i-1][j-1],   # avanzar en las dos
                                    acum[i-1][j],     # avanzar solo en a
                                    acum[i][j-1] )    # avanzar solo en b

    donde d(i, j) es la distancia euclidea entre el tramo i de a y el tramo j de b.
    El resultado se divide por (n + m) para que no dependa del largo de los audios.
    """
    n, m = len(a), len(b)
    local = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2))   # d(i, j) para todos los pares
    ancho = max(int(banda * max(n, m)), abs(n - m))                        # banda de Sakoe-Chiba
    acum = np.full((n + 1, m + 1), np.inf)
    acum[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - ancho), min(m, i + ancho) + 1):
            acum[i, j] = local[i - 1, j - 1] + min(acum[i - 1, j - 1], acum[i - 1, j], acum[i, j - 1])
    return acum[n, m] / (n + m)


def matriz_dtw(consultas, referencias):
    """Matriz de distancias DTW entre cada secuencia de `consultas` y cada una de `referencias`."""
    D = np.empty((len(consultas), len(referencias)))
    for i, q in enumerate(consultas):
        for j, r in enumerate(referencias):
            D[i, j] = dtw(q, r)
    return D


# =====================================================================================
#  KNN desde cero
# =====================================================================================
def knn_predecir(D, y_ref, k, n_clases):
    """PASO 2: dada la matriz de distancias D (consultas x referencias), predice la clase de cada consulta.

    Para cada consulta:
      a) se ordenan las referencias de la mas cercana a la mas lejana y se toman las K primeras (los K vecinos);
      b) cada vecino VOTA por su clase; gana la clase con mas votos;
      c) si hay empate, gana la clase del vecino mas cercano entre las empatadas.
    """
    orden = np.argsort(D, axis=1, kind="stable")[:, :k]
    pred = np.zeros(len(D), dtype=int)
    for i, vecinos in enumerate(orden):
        votos = np.bincount(y_ref[vecinos], minlength=n_clases)
        empatadas = np.flatnonzero(votos == votos.max())
        if len(empatadas) == 1:
            pred[i] = empatadas[0]
        else:
            for v in vecinos:                       # los vecinos ya vienen ordenados por cercania
                if y_ref[v] in empatadas:
                    pred[i] = y_ref[v]
                    break
    return pred


def ajustar_zscore(X_ref):
    """Media y desvio de cada numero del vector, calculados SOLO con el conjunto de referencia.

    Si se calcularan con todos los datos (incluida la prueba) se "filtraria" informacion
    de la prueba hacia el entrenamiento (fuga de datos) y la exactitud saldria inflada.
    """
    return X_ref.mean(axis=0), X_ref.std(axis=0) + 1e-9


def distancias(distancia, idx_consulta, idx_ref, X, seqs_prep):
    """Matriz de distancias consulta x referencia con la distancia pedida ('euclidea' o 'dtw')."""
    if distancia == "euclidea":
        mu, sd = ajustar_zscore(X[idx_ref])
        return distancias_euclideas((X[idx_consulta] - mu) / sd, (X[idx_ref] - mu) / sd)
    return matriz_dtw([seqs_prep[i] for i in idx_consulta], [seqs_prep[i] for i in idx_ref])


# =====================================================================================
#  Modelo guardado y uso en la demo
# =====================================================================================
def guardar_modelo(distancia, k, y, clases, nombres, X, seqs_prep, archivos):
    """Guarda el modelo final: referencia = TODOS los audios (la prueba ya se uso para evaluar).

    Se guardan tambien los nombres de archivo de la referencia para poder mostrar, al clasificar,
    cuales fueron los vecinos mas cercanos.
    """
    mu, sd = ajustar_zscore(X)
    largos = np.array([len(s) for s in seqs_prep])
    np.savez(RUTA_MODELO, distancia=distancia, k=k, clases=clases, nombres=np.array(nombres),
             y_ref=y, X_ref=X, mu=mu, sd=sd, seq_concat=np.concatenate(seqs_prep).astype(np.float32),
             seq_largo=largos, banda=BANDA, submuestreo=SUBMUESTREO, archivos=np.array(archivos))


def clasificar_audio(x, fs=16000, modelo=None, normalizar=True):
    """Para la demo: audio del microfono -> (comando, distancia al vecino mas cercano, votos).

    normalizar=True aplica el mismo acondicionamiento que se uso con el dataset
    (normalizar_audios.normalizar_audio) antes de caracterizar.
    """
    r = clasificar_audio_detalle(x, fs, modelo, normalizar)
    return r["comando"], r["distancia"], r["votos"]


def clasificar_audio_detalle(x, fs=16000, modelo=None, normalizar=True):
    """Igual que clasificar_audio pero devuelve TODO el detalle de la decision (un diccionario):

      comando    el comando elegido
      votos      {comando: cantidad de vecinos que votaron por el}
      confianza  'alta' si votaron los K vecinos por el mismo comando, 'media' si gano por mayoria,
                 'baja' si no hubo mayoria (empate)
      vecinos    [(archivo, comando, distancia)] de los K mas cercanos, del mas cercano al mas lejano
      distancia  distancia al vecino mas cercano
      info       datos del acondicionamiento (pico, si se detecto voz, duracion antes/despues, ...)

    Aviso: el KNN SIEMPRE elige uno de los comandos conocidos; la distancia no sirve para rechazar
    lo que no es un comando (silencio y ruido quedan mas cerca que algunos comandos reales).
    """
    if modelo is None:
        modelo = np.load(RUTA_MODELO, allow_pickle=False)
    info = {}
    if normalizar:
        x, info = NA.normalizar_audio(x, fs)
        fs = NA.PARAMS["fs"]
    seq, vec = CA.caracterizar_audio(x, fs)
    y_ref = modelo["y_ref"]
    n_clases = int(modelo["clases"].max()) + 1
    if str(modelo["distancia"]) == "euclidea":
        d = np.linalg.norm((modelo["X_ref"] - modelo["mu"]) / modelo["sd"] - (vec - modelo["mu"]) / modelo["sd"], axis=1)
    else:
        q = preparar_secuencia(seq)
        corte = np.concatenate([[0], np.cumsum(modelo["seq_largo"])])
        d = np.array([dtw(q, modelo["seq_concat"][corte[i]:corte[i + 1]].astype(np.float64)) for i in range(len(y_ref))])
    k = int(modelo["k"])
    pred = knn_predecir(d[None, :], y_ref, k, n_clases)[0]
    vecinos = np.argsort(d, kind="stable")[:k]
    votos = np.bincount(y_ref[vecinos], minlength=n_clases)
    nombres = [str(n) for n in modelo["nombres"]]
    idx = {int(c): nombres[i] for i, c in enumerate(modelo["clases"])}
    archivos = [str(a) for a in modelo["archivos"]] if "archivos" in modelo.files else ["?"] * len(y_ref)
    ganadores = int(votos[int(pred)])
    confianza = "alta" if ganadores == k else ("media" if ganadores * 2 > k else "baja")
    return {"comando": idx[int(pred)], "distancia": float(d[vecinos[0]]),
            "votos": {idx[int(c)]: int(votos[c]) for c in modelo["clases"]},
            "confianza": confianza,
            "vecinos": [(archivos[v], idx[int(y_ref[v])], float(d[v])) for v in vecinos],
            "info": info}


# =====================================================================================
#  Programa principal
# =====================================================================================
def elegir_sesiones(disponibles, pedidas, para):
    """'mama,joaco' -> ['1_mama', '1_joaco'] (acepta el nombre con o sin el prefijo '1_')."""
    elegidas = []
    for t in [p.strip() for p in pedidas.split(",") if p.strip()]:
        coinciden = [s for s in disponibles if s == t or s.endswith("_" + t)]
        if not coinciden:
            sys.exit(f"La sesion '{t}' ({para}) no existe. Sesiones disponibles: {', '.join(disponibles)}")
        elegidas += coinciden
    return elegidas


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--validacion", default=SESIONES_VALIDACION, help="sesiones (personas) para validar K y la distancia")
    ap.add_argument("--prueba", default=SESIONES_PRUEBA, help="sesiones (personas) reservadas para la prueba final")
    ap.add_argument("--distancia", choices=["euclidea", "dtw"], default="euclidea",
                    help="euclidea (por defecto, la que usa el proyecto) o dtw (solo para comparar)")
    a = ap.parse_args()

    # ---------------- datos ----------------
    X, y, archivos, sesion = CA.cargar_dataset()
    X = X.astype(np.float64)
    seqs, _, _ = CA.cargar_secuencias()
    seqs_prep = [preparar_secuencia(s) for s in seqs]
    inv = {v: k for k, v in json.loads((CARPETA / "clases.json").read_text(encoding="utf-8")).items()}
    clases = np.unique(y)                                  # las clases que realmente hay (p. ej. sin 'otros')
    nombres = [inv[int(c)] for c in clases]
    yi = np.searchsorted(clases, y)                        # etiquetas 0..n-1 compactas
    n_clases = len(clases)

    # ---------------- PASO 1: tres conjuntos separados por persona ----------------
    disponibles = sorted(set(sesion))
    ses_val = elegir_sesiones(disponibles, a.validacion, "validacion")
    ses_test = elegir_sesiones(disponibles, a.prueba, "prueba")
    if set(ses_val) & set(ses_test):
        sys.exit("Una misma sesion no puede estar en validacion y en prueba.")
    idx_val = np.flatnonzero(np.isin(sesion, ses_val))
    idx_test = np.flatnonzero(np.isin(sesion, ses_test))
    idx_train = np.flatnonzero(~np.isin(sesion, ses_val + ses_test))
    print("PASO 1 - tres conjuntos separados por persona/sesion (asi la exactitud no se infla):")
    for nombre, idx, ses in (("entrenamiento", idx_train, sorted(set(sesion[idx_train]))),
                             ("validacion", idx_val, ses_val), ("prueba", idx_test, ses_test)):
        cuenta = ", ".join(f"{n}={int((yi[idx] == i).sum())}" for i, n in enumerate(nombres))
        print(f"   {nombre:<14}{len(idx):>4} audios  [{cuenta}]  sesiones: {', '.join(ses)}")

    # ---------------- PASOS 2 y 3: elegir distancia y K con la VALIDACION ----------------
    ks = [k for k in VALORES_K if k <= len(idx_train)]
    distancias_probar = [a.distancia]
    tabla, D_val = {}, {}
    print("\nPASO 2 y 3 - exactitud en VALIDACION (referencia = entrenamiento) segun K:")
    print("   distancia   " + "".join(f"K={k:<5}" for k in ks))
    for d in distancias_probar:
        D_val[d] = distancias(d, idx_val, idx_train, X, seqs_prep)
        tabla[d] = [float((knn_predecir(D_val[d], yi[idx_train], k, n_clases) == yi[idx_val]).mean()) for k in ks]
        print(f"   {d:<12}" + "".join(f"{v * 100:<7.1f}" for v in tabla[d]))
    # criterio: mayor exactitud; empate -> K mas chico; empate -> euclidea (mas simple y rapida)
    mejor = min(((d, k) for d in distancias_probar for k in ks),
                key=lambda dk: (-tabla[dk[0]][ks.index(dk[1])], dk[1], dk[0] != "euclidea"))
    dist_elegida, k_elegido = mejor
    print(f"\nElegido con la validacion: distancia {dist_elegida}, K = {k_elegido} "
          f"(exactitud en validacion {tabla[dist_elegida][ks.index(k_elegido)] * 100:.1f}%)")

    resultados = {"distancia": dist_elegida, "k": k_elegido, "valores_k": ks,
                  "exactitud_validacion_por_k": tabla,
                  "conjuntos": {"entrenamiento": len(idx_train), "validacion": len(idx_val), "prueba": len(idx_test)},
                  "sesiones_validacion": ses_val, "sesiones_prueba": ses_test}

    # ---------------- evaluacion detallada de la validacion ----------------
    pred_val = knn_predecir(D_val[dist_elegida], yi[idx_train], k_elegido, n_clases)
    cm_val = metricas.matriz_confusion(yi[idx_val], pred_val, n_clases)
    resultados["validacion"] = metricas.imprimir_reporte("VALIDACION", cm_val, nombres)
    resultados["validacion"]["matriz_confusion"] = cm_val.tolist()

    # ---------------- PASO 4: PRUEBA, una sola vez, con referencia = entrenamiento + validacion ----------------
    idx_ref = np.concatenate([idx_train, idx_val])
    print("\nPASO 4 - prueba final (una sola vez; referencia = entrenamiento + validacion)...")
    D_test = distancias(dist_elegida, idx_test, idx_ref, X, seqs_prep)
    pred_test = knn_predecir(D_test, yi[idx_ref], k_elegido, n_clases)
    cm_test = metricas.matriz_confusion(yi[idx_test], pred_test, n_clases)
    resultados["prueba"] = metricas.imprimir_reporte("PRUEBA", cm_test, nombres)
    resultados["prueba"]["matriz_confusion"] = cm_test.tolist()

    print("\nExactitud por sesion (persona):")
    por_sesion = {}
    for nombre, idx, pred in (("validacion", idx_val, pred_val), ("prueba", idx_test, pred_test)):
        for s in sorted(set(sesion[idx])):
            m = sesion[idx] == s
            por_sesion[s] = float((pred[m] == yi[idx][m]).mean())
            print(f"   {nombre:<11}{s:<10} {int(m.sum()):>3} audios  exactitud {por_sesion[s] * 100:5.1f}%")
    resultados["exactitud_por_sesion"] = por_sesion
    if len(idx_test) < 40:
        print(f"\nAviso: la prueba tiene solo {len(idx_test)} audios; cada error pesa {100 / len(idx_test):.1f} puntos "
              "porcentuales, asi que la exactitud de prueba es una estimacion poco precisa.")

    # ---------------- guardar modelo y metricas ----------------
    guardar_modelo(dist_elegida, k_elegido, y, clases, nombres, X, seqs_prep, archivos)
    (CARPETA / "metricas_knn.json").write_text(json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------------- graficos ----------------
    carpeta_g = CARPETA / "graficos"
    carpeta_g.mkdir(exist_ok=True)
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4.6))
    metricas.dibujar_confusion(ejes[0], cm_val, nombres, f"KNN - validacion (exactitud {resultados['validacion']['exactitud'] * 100:.1f}%)")
    metricas.dibujar_confusion(ejes[1], cm_test, nombres, f"KNN - prueba (exactitud {resultados['prueba']['exactitud'] * 100:.1f}%)")
    fig.tight_layout()
    fig.savefig(carpeta_g / "knn_confusion.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nModelo: {RUTA_MODELO}\nMetricas: {CARPETA / 'metricas_knn.json'}\nGraficos en: {carpeta_g}")


if __name__ == "__main__":
    main()
