"""Exporta los datos y la regla de decision de las huellas a un .mat para graficar en MATLAB.

    python explicativos/exportar_matlab.py      # crea explicativos/datos_huellas_matlab.mat

Variables (indices desde 1, como usa MATLAB):
  Xd, yd      dataset completo normalizado (originales y transformaciones); yd = 1..3
  X, verdad   huellas de evaluacion + intrusos (1..3 = identidad, 4 = intruso); origen: 1 prueba, 2 a mano, 3 intruso prueba, 4 intruso calibracion
  mu, V       media (1 x d) y componentes principales (d x k) del modelo
  Zc          centros de las 3 identidades en las k componentes (3 x k); mapeo (3 x 1) = identidad de cada centro
  Prec        matrices de precision (inversa de la covarianza) de cada identidad (3 x k x k)
  t2_limite, q_limite, usar_q, lambda   limites, si la regla usa tambien Q (1) o solo T2 (0), y umbral
"""
import sys
from pathlib import Path

import numpy as np
from scipy.io import savemat

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import intrusos_huellas as IH                                     # noqa: E402

KH = IH.KH
CODIGO = {"dataset (prueba)": 1, "subida a mano": 2, "intruso (prueba)": 3, "intruso (calibracion)": 4}


def main():
    m, mapeo, lam, _ = IH.modelo()
    X, verdad, origen = IH.cargar(incluir_calibracion=True)
    datos = np.load(KH.CARPETA / "dataset.npz", allow_pickle=True)
    ruta = AQUI / "datos_huellas_matlab.mat"
    savemat(ruta, {"X": X, "verdad": (verdad + 1).astype(float).reshape(-1, 1),
                   "origen": np.array([CODIGO[o] for o in origen], dtype=float).reshape(-1, 1),
                   "Xd": datos["X"].astype(float), "yd": (datos["y"].astype(int) + 1).astype(float).reshape(-1, 1),
                   "mu": m["mu"].reshape(1, -1), "V": m["V"], "Zc": m["Zc"], "Prec": m["precision"],
                   "mapeo": (mapeo + 1).astype(float).reshape(-1, 1),
                   "t2_limite": m["t2_limite"], "q_limite": m["q_limite"], "usar_q": float(m.get("usar_q", True)), "lambda": lam})
    print(f"Guardado {ruta} | X {X.shape}, Xd {datos['X'].shape}, k = {m['k']}, lambda = {lam:.2f}")


if __name__ == "__main__":
    main()
