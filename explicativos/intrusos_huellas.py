"""Huellas con INTRUSOS: matriz de confusion de 4 salidas y grafico de los intrusos fuera del umbral.

    python explicativos/intrusos_huellas.py      # crea 2 imagenes en explicativos/imagenes/

COMO SE CALCULA (etapa: evaluacion del K-means, despues de entrenar y de fijar el umbral R)
  1. Cada huella se procesa igual que en el agente: preprocesar -> vector de 72 numeros.
  2. Se calcula la distancia euclidea (en las 72 dimensiones, NO en el PCA) a los 3 centroides y se toma el mas cercano.
  3. Si distancia <= R  ->  se acepta con la identidad de ese cluster; si no  ->  "NO AUTORIZADO".
  4. Se compara con la verdad conocida. Filas = lo que la huella ES (identidad 1, 2, 3 o intruso);
     columnas = lo que el sistema DIJO (identidad 1, 2, 3 o NO AUTORIZADO).
Huellas usadas
  autorizadas : las 36 impresiones de prueba del dataset (4 y 5) + las autorizadas que subiste a mano (pruebas_huellas/)
  intrusos    : los que subiste a mano + los generados con Anguli (explicativos/generar_intrusos.py)
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402
from matplotlib.lines import Line2D                               # noqa: E402
from matplotlib.patches import Circle                             # noqa: E402
from sklearn.decomposition import PCA                             # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
import caracterizar_huellas as CH                                 # noqa: E402
import kmeans_huellas as KH                                       # noqa: E402
import preprocesar as P                                           # noqa: E402

AQUI = Path(__file__).resolve().parent
SALIDA = AQUI / "imagenes"
COLORES = [plt.cm.tab10(i) for i in range(3)]
ROJO = plt.cm.tab10(3)
plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
CLASES = ["Identidad 1", "Identidad 2", "Identidad 3", "Intruso"]
SALIDAS = ["Identidad 1", "Identidad 2", "Identidad 3", "NO AUTORIZADO"]


def vector_de(ruta):
    return CH.rasgo_orientacion(P.preprocesar_huella(P.a_gris(Path(ruta)))).astype(float)


def cargar():
    """Devuelve X (vectores de 72), verdad (0,1,2 = identidades, 3 = intruso) y el origen de cada huella."""
    d = np.load(KH.CARPETA / "dataset.npz", allow_pickle=True)
    imp = np.array([int(str(a).split("_imp")[1].split(".")[0]) for a in d["archivo"]])
    prueba = np.isin(imp, (4, 5))
    X = [d["X"][prueba].astype(float)]
    verdad = [d["y"][prueba].astype(int)]
    origen = [np.array(["dataset (prueba)"] * int(prueba.sum()))]

    vec, ver, ori = [], [], []
    with open(RAIZ / "pruebas_huellas" / "registro.csv", encoding="utf-8") as f:
        for fila in csv.DictReader(f):
            ruta = RAIZ / "huellas_nuevas" / fila["archivo"]
            if not ruta.exists():
                continue
            vec.append(vector_de(ruta))
            ver.append(3 if fila["verdad"] == "intruso" else int(fila["verdad"][-1]) - 1)
            ori.append("subida a mano")
    X.append(np.array(vec))
    verdad.append(np.array(ver))
    origen.append(np.array(ori))

    gen = sorted((AQUI / "intrusos_generados").glob("*.jpg"))
    X.append(np.array([vector_de(r) for r in gen]).reshape(-1, 72))
    verdad.append(np.full(len(gen), 3))
    origen.append(np.array(["generada con Anguli"] * len(gen)))
    return np.vstack(X), np.concatenate(verdad), np.concatenate(origen)


def decidir(X, C, mapeo, R):
    """Devuelve (salida 0..3, distancia al centroide mas cercano, indice de ese centroide)."""
    D = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)
    j = D.argmin(axis=1)
    d = D[np.arange(len(X)), j]
    salida = np.where(d <= R, mapeo[j], 3)
    return salida, d, j


def matriz(verdad, salida):
    M = np.zeros((4, 4), dtype=int)
    for v, s in zip(verdad, salida):
        M[v, s] += 1
    return M


def dibujar_matriz(ax, M, titulo):
    ax.imshow(M, cmap="Blues", vmin=0, vmax=M.max())
    for i in range(4):
        for j in range(4):
            ok = (i == j)
            ax.text(j, i, M[i, j], ha="center", va="center", fontsize=15, fontweight="bold",
                    color="white" if M[i, j] > M.max() * 0.55 else ("#1f2937" if ok else ROJO if M[i, j] else "#9ca3af"))
    ax.set_xticks(range(4))
    ax.set_xticklabels(SALIDAS, rotation=20, ha="right")
    ax.set_yticks(range(4))
    ax.set_yticklabels(CLASES)
    ax.set_xlabel("lo que dijo el sistema")
    ax.set_ylabel("lo que la huella ES")
    ax.set_title(titulo, fontsize=11.5, fontweight="bold")
    ax.axhline(2.5, color="black", lw=1.5)
    ax.axvline(2.5, color="black", lw=1.5)


def metricas(M):
    n_aut, n_int = M[:3].sum(), M[3].sum()
    bien_aut = int(np.trace(M[:3, :3]))
    rech = int(M[:3, 3].sum())
    mal_id = int(M[:3, :3].sum() - bien_aut)
    ia = int(M[3, :3].sum())
    return dict(n_aut=int(n_aut), n_int=int(n_int), bien=bien_aut, rechazadas=rech, mal_id=mal_id, intrusos_aceptados=ia,
                frr=rech / n_aut, far=ia / n_int, exactitud=(bien_aut + int(M[3, 3])) / M.sum())


def main():
    modelo = np.load(KH.RUTA_MODELO, allow_pickle=False)
    C, mapeo, R = modelo["centroides"], modelo["mapeo"].astype(int), float(modelo["radios"][0])
    X, verdad, origen = cargar()
    print(f"autorizadas: {(verdad < 3).sum()}  | intrusos: {(verdad == 3).sum()}  | R aplicado = {R:.2f}")

    # ------------------------------------------------ matriz de confusion
    fig, axs = plt.subplots(1, 2, figsize=(14, 6.2))
    for ax, r in zip(axs, (R, 0.31)):
        sal, d, _ = decidir(X, C, mapeo, r)
        M = matriz(verdad, sal)
        m = metricas(M)
        dibujar_matriz(ax, M, f"Radio R = {r:.2f}{' (aplicado)' if r == R else ''}\nFRR {m['frr']*100:.1f} %  ·  FAR {m['far']*100:.1f} %  ·  exactitud {m['exactitud']*100:.1f} %")
        print(f"R={r:.2f}  FRR={m['frr']*100:.1f}% ({m['rechazadas']}/{m['n_aut']})  mal_identificadas={m['mal_id']}  "
              f"FAR={m['far']*100:.1f}% ({m['intrusos_aceptados']}/{m['n_int']})  exactitud={m['exactitud']*100:.1f}%")
        print(M)
    fig.suptitle(f"Matriz de confusión de huellas con intrusos ({int((verdad < 3).sum())} autorizadas y {int((verdad == 3).sum())} intrusos)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    SALIDA.mkdir(parents=True, exist_ok=True)
    fig.savefig(SALIDA / "matriz_confusion_intrusos.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------ intrusos fuera del umbral
    sal, d, j = decidir(X, C, mapeo, R)
    datos = np.load(KH.CARPETA / "dataset.npz", allow_pickle=True)
    Xd, yd = datos["X"].astype(float), datos["y"].astype(int)
    pca = PCA(2, random_state=0).fit(Xd)
    Pc = pca.transform(C)

    def posicion(X72):
        D = np.linalg.norm(X72[:, None, :] - C[None, :, :], axis=2)
        jj = D.argmin(axis=1)
        dd = D[np.arange(len(X72)), jj]
        v = pca.transform(X72) - Pc[jj]
        n = np.linalg.norm(v, axis=1, keepdims=True)
        u = np.where(n > 1e-9, v / np.maximum(n, 1e-9), np.array([1.0, 0.0]))
        return Pc[jj] + u * dd[:, None]

    fig = plt.figure(figsize=(17, 7.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.2)
    ax = fig.add_subplot(gs[0])
    pd = posicion(Xd)
    for i in range(3):
        ax.scatter(pd[yd == i, 0], pd[yd == i, 1], s=22, color=COLORES[i], alpha=0.22, edgecolor="none")
    for c in range(3):
        ident = mapeo[c]
        ax.add_patch(Circle(Pc[c], R, fill=True, fc=COLORES[ident], alpha=0.08, ec="none"))
        ax.add_patch(Circle(Pc[c], R, fill=False, ec=COLORES[ident], lw=2.2, ls="--"))
        ax.scatter(*Pc[c], marker="X", s=330, color=COLORES[ident], edgecolor="black", linewidth=1.6, zorder=5)
        ax.text(Pc[c][0], Pc[c][1] + R + 0.03, f"Identidad {ident + 1}", ha="center", fontweight="bold", fontsize=10)
    pos = posicion(X)
    aut = verdad < 3
    ax.scatter(pos[aut, 0], pos[aut, 1], s=60, marker="o", facecolor="none", edgecolor="black", linewidth=1.2, zorder=6)
    for i in range(3):
        m = verdad == i
        ax.scatter(pos[m, 0], pos[m, 1], s=34, marker="o", color=COLORES[i], zorder=7)
    dentro = (verdad == 3) & (d <= R)
    fuera = (verdad == 3) & (d > R)
    ax.scatter(pos[fuera, 0], pos[fuera, 1], marker="X", s=95, color=ROJO, edgecolor="black", linewidth=0.7, zorder=8)
    ax.scatter(pos[dentro, 0], pos[dentro, 1], marker="X", s=130, color="gold", edgecolor="black", linewidth=1.3, zorder=9)
    ax.set_aspect("equal")
    ax.set_xlabel("posición según el plano PCA (cada punto a su distancia REAL del centroide)")
    ax.set_title(f"Intrusos y autorizadas respecto del umbral R = {R:.2f}", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.2)
    leg = [Line2D([], [], marker="o", ls="", color="gray", label=f"autorizadas ({int(aut.sum())}), color = identidad"),
           Line2D([], [], marker="X", ls="", color=ROJO, markeredgecolor="black", markersize=10, label=f"intrusos RECHAZADOS ({int(fuera.sum())})"),
           Line2D([], [], marker="X", ls="", color="gold", markeredgecolor="black", markersize=11, label=f"intrusos ACEPTADOS por error ({int(dentro.sum())})"),
           Line2D([], [], marker="o", ls="", color="lightgray", alpha=0.5, label="dataset de entrenamiento y prueba (tenue)")]
    ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=9, framealpha=0.95)

    ax = fig.add_subplot(gs[1])
    grupos = [("Autorizadas\n(dataset prueba)", (verdad < 3) & (origen == "dataset (prueba)"), "#2ca02c"),
              ("Autorizadas\n(subidas a mano)", (verdad < 3) & (origen == "subida a mano"), "#98df8a"),
              ("Intrusos\n(subidos a mano)", (verdad == 3) & (origen == "subida a mano"), "#ff9896"),
              ("Intrusos\n(Anguli)", (verdad == 3) & (origen == "generada con Anguli"), ROJO)]
    rng = np.random.default_rng(0)
    for k, (nombre, m, c) in enumerate(grupos):
        ax.scatter(rng.normal(k, 0.07, int(m.sum())), d[m], s=55, color=c, edgecolor="black", linewidth=0.6, zorder=3)
        ax.text(k, 0.015, f"{int((d[m] > R).sum())} de {int(m.sum())} rechazadas", ha="center", va="bottom", fontsize=9.5,
                fontweight="bold", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray"))
    ax.axhline(R, color="black", lw=2.2, ls="--")
    ax.axhspan(R, max(d.max() * 1.05, R + 0.1), color=ROJO, alpha=0.08)
    ax.text(3.45, R + 0.02, f"umbral R = {R:.2f}", ha="right", fontweight="bold")
    ax.text(-0.45, d.max() * 1.04, "RECHAZADA (NO AUTORIZADO)", ha="left", va="top", color=ROJO, fontweight="bold")
    ax.text(-0.45, R - 0.03, "ACEPTADA", ha="left", va="top", color="#2ca02c", fontweight="bold")
    ax.set_xticks(range(4))
    ax.set_xticklabels([g[0] for g in grupos])
    ax.set_ylim(0, max(d.max() * 1.08, R + 0.1))
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylabel("distancia al centroide más cercano (72 números)")
    ax.set_title("Distancia de cada huella al centroide más cercano", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Los intrusos quedan, en su mayoría, fuera del umbral", fontsize=15, fontweight="bold")
    fig.savefig(SALIDA / "intrusos_fuera_umbral.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Guardado en", SALIDA)


if __name__ == "__main__":
    main()
