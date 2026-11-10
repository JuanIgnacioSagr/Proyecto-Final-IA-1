"""Regla de decision de las huellas: PCA + distancia de Mahalanobis (T2) + residuo (Q).

Es el estandar del control multivariado con PCA. Cada huella (vector normalizado de caracterizar_huellas.py) se describe con
dos numeros respecto de cada identidad j:
  T2_j  distancia de Mahalanobis, DENTRO del espacio de las k componentes principales, al centro de la identidad j.
        Mide cuanto se aparta la huella de lo normal para esa identidad, teniendo en cuenta como se estira (covarianza).
        Su zona de aceptacion es un ELIPSOIDE; su sombra en el plano PC1-PC2 es exactamente una elipse.
  Q     residuo (SPE): la parte de la huella que las k componentes NO explican (distancia a ese subespacio).
        Es lo que un grafico PCA no muestra y por donde los intrusos se separan.
Puntaje por identidad:   s_j = max( T2_j / T2_limite ,  Q / Q_limite )
La huella se identifica con la identidad de menor T2 y se ACEPTA si su puntaje s <= lambda. Los limites son el percentil
95 de las autorizadas de VALIDACION (huellas que el modelo no vio) y lambda se calibra con el criterio de siempre:
el mayor valor con el que ningun intruso de calibracion es aceptado.

En lugar de "distancia euclidea <= R" (una esfera, cuya sombra en un PCA es siempre un circulo) el umbral pasa a ser
"s <= lambda": una zona elipsoidal en las k componentes MAS un limite del residuo. Se dibuja exacta con el grafico T2 contra Q.
"""
import numpy as np
from scipy import stats

K_MAX = 15                  # tope de componentes (cada identidad tiene solo ~33 huellas de entrenamiento)
VARIANZA_OBJETIVO = 0.90    # se eligen las componentes necesarias para explicar este % de la varianza
ENCOGIMIENTO = 0.3          # cuanto se mezcla la covarianza de cada identidad con la conjunta (la estabiliza)
PERCENTIL_LIMITE = 95       # limites de T2 y Q = percentil de las autorizadas de validacion
PASO_LAMBDA = 0.01
USAR_Q = False             # False: la regla es UNA sola distancia (T2 de Mahalanobis) contra UN umbral; True: ademas el residuo Q (puntaje = max(T2, Q))


def ajustar(X_ent, etiquetas_cluster, C, k=None):
    """Ajusta PCA y covarianzas SOLO con el entrenamiento. C = centroides de K-means (en el espacio completo)."""
    mu = X_ent.mean(axis=0)
    _, S, Vt = np.linalg.svd(X_ent - mu, full_matrices=False)
    var = S ** 2 / (len(X_ent) - 1)
    acumulada = np.cumsum(var) / var.sum()
    if k is None:
        k = int(min(K_MAX, np.searchsorted(acumulada, VARIANZA_OBJETIVO) + 1))
    V = Vt[:k].T                                              # (dimensiones, k)
    Z = (X_ent - mu) @ V
    Zc = (np.asarray(C, dtype=float) - mu) @ V               # centroides en el espacio PCA
    desv = [Z[etiquetas_cluster == j] - Zc[j] for j in range(len(C))]
    conjunta = sum(d.T @ d for d in desv) / max(len(Z) - len(C), 1)
    precision = []
    for d in desv:
        propia = d.T @ d / max(len(d) - 1, 1)
        cov = (1 - ENCOGIMIENTO) * propia + ENCOGIMIENTO * conjunta + 1e-9 * np.eye(k)
        precision.append(np.linalg.inv(cov))
    return {"mu": mu, "V": V, "Zc": Zc, "precision": np.stack(precision), "k": k, "varianza": var,
            "varianza_explicada": float(acumulada[k - 1]), "t2_limite": 1.0, "q_limite": 1.0, "usar_q": USAR_Q}


def estadisticos(X, m):
    """Devuelve T2 (n, identidades), Q (n,) y las coordenadas en las k componentes Z (n, k)."""
    Xc = np.asarray(X, dtype=float) - m["mu"]
    Z = Xc @ m["V"]
    Q = ((Xc - Z @ m["V"].T) ** 2).sum(axis=1)
    T2 = np.stack([np.einsum("ij,jk,ik->i", Z - m["Zc"][j], m["precision"][j], Z - m["Zc"][j])
                   for j in range(len(m["Zc"]))], axis=1)
    return T2, Q, Z


def fijar_limites(m, X_val):
    """Limites = percentil 95 de las autorizadas de validacion (T2 al centro mas cercano y Q)."""
    T2, Q, _ = estadisticos(X_val, m)
    m["t2_limite"] = float(np.percentile(T2.min(axis=1), PERCENTIL_LIMITE))
    m["q_limite"] = float(np.percentile(Q, PERCENTIL_LIMITE))
    return m


def puntajes(X, m):
    """s_j = max(T2_j / limite_T2, Q / limite_Q) para cada identidad; devuelve (s, T2, Q)."""
    T2, Q, _ = estadisticos(X, m)
    if not m.get("usar_q", True):                     # solo Mahalanobis: el puntaje es T2 dividido por su limite
        return T2 / m["t2_limite"], T2, Q
    return np.maximum(T2 / m["t2_limite"], (Q / m["q_limite"])[:, None]), T2, Q


def decidir(X, m, lam):
    """Para cada huella: (cluster mas cercano por T2, puntaje s de ese cluster, aceptada)."""
    s, T2, Q = puntajes(X, m)
    j = T2.argmin(axis=1)
    sj = s[np.arange(len(s)), j]
    return j, sj, sj <= lam


def calibrar_lambda(s_intrusos):
    """El mayor lambda (de a 0,01) con el que NINGUN intruso de calibracion es aceptado (criterio con el que se fijo R)."""
    return float(np.floor(np.min(s_intrusos) / PASO_LAMBDA - 1e-9) * PASO_LAMBDA)


def limites_teoricos(m, n_por_identidad):
    """Valores de referencia de los manuales: T2 por F de Fisher y Q por Jackson-Mudholkar (no se usan para decidir)."""
    k, n = m["k"], n_por_identidad
    t2 = k * (n + 1) * (n - 1) / (n * (n - k)) * stats.f.ppf(0.95, k, n - k) if n > k else float("nan")
    r = m["varianza"][k:]
    th1, th2, th3 = r.sum(), (r ** 2).sum(), (r ** 3).sum()
    h0 = 1 - 2 * th1 * th3 / (3 * th2 ** 2)
    c = stats.norm.ppf(0.95)
    q = th1 * (c * np.sqrt(2 * th2 * h0 ** 2) / th1 + 1 + th2 * h0 * (h0 - 1) / th1 ** 2) ** (1 / h0)
    return float(t2), float(q)


def a_diccionario(m):
    """Solo arreglos y numeros: se guardan dentro de modelo_kmeans.npz con el prefijo 'dec_'."""
    return {f"dec_{k}": np.asarray(v) for k, v in m.items()}


def desde_modelo(modelo):
    """Reconstruye el diccionario del ajuste a partir de modelo_kmeans.npz (o None si el modelo no lo trae)."""
    llaves = getattr(modelo, "files", list(modelo))
    if "dec_mu" not in llaves:
        return None
    m = {k[4:]: modelo[k] for k in llaves if k.startswith("dec_")}
    for k in ("t2_limite", "q_limite", "varianza_explicada"):
        m[k] = float(m[k])
    m["k"] = int(m["k"])
    m["usar_q"] = bool(m["usar_q"]) if "usar_q" in m else True      # modelos viejos: con Q
    return m
