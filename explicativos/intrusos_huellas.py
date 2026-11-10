"""Huellas con INTRUSOS: matriz de confusion de 4 salidas y grafico de los intrusos contra el umbral.

    python explicativos/intrusos_huellas.py      # crea 2 imagenes en explicativos/imagenes/

COMO SE DECIDE (etapa: evaluacion, despues de entrenar K-means y de fijar el umbral lambda; ver src/decision_huellas.py)
  1. Cada huella se procesa igual que en el agente: preprocesar -> rasgo_huella (vector normalizado).
  2. Se calcula T2 (Mahalanobis dentro de las k componentes del PCA) a cada identidad y Q (residuo); la identidad es la de menor T2.
  3. Puntaje s = max(T2 / limite_T2, Q / limite_Q). Si s <= lambda se acepta con esa identidad; si no -> "NO AUTORIZADO".
  4. Se compara con la verdad conocida. Filas = lo que la huella ES (identidad 1, 2, 3 o intruso);
     columnas = lo que el sistema DIJO (identidad 1, 2, 3 o NO AUTORIZADO).
Huellas usadas en la EVALUACION (nada de esto se uso para entrenar ni para calibrar)
  autorizadas : las de PRUEBA del dataset (20 %) + las autorizadas que se subieron a mano (dataset_huellas/pruebas_manuales/)
  intrusos    : los 25 de prueba (dataset_huellas/intrusas/intruso_26..50) + los subidos a mano
Los intrusos 01-25 se usan SOLO para calibrar lambda; aparecen en los graficos marcados como "calibracion".
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402
from matplotlib.lines import Line2D                               # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
import caracterizar_huellas as CH                                 # noqa: E402
import decision_huellas as DH                                     # noqa: E402
import graficos_decision as GD                                    # noqa: E402
import kmeans_huellas as KH                                       # noqa: E402
import particion_huellas as PH                                    # noqa: E402
import preprocesar as P                                           # noqa: E402

AQUI = Path(__file__).resolve().parent
SALIDA = AQUI / "imagenes"
COLORES = [plt.cm.tab10(i) for i in range(3)]
ROJO = plt.cm.tab10(3)
plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
CLASES = ["Identidad 1", "Identidad 2", "Identidad 3", "Intruso"]
SALIDAS = ["Identidad 1", "Identidad 2", "Identidad 3", "NO AUTORIZADO"]


def vector_de(ruta):
    """Vector normalizado de una huella nueva (el mismo camino que usa la interfaz)."""
    return CH.rasgo_huella(P.preprocesar_huella(P.a_gris(Path(ruta)))).astype(float)


def modelo():
    """(ajuste de la regla de decision m, mapeo cluster->identidad, lambda aplicado, centroides)."""
    mod = np.load(KH.RUTA_MODELO, allow_pickle=False)
    return DH.desde_modelo(mod), mod["mapeo"].astype(int), float(mod["radios"][0]), mod["centroides"]


def cargar(incluir_calibracion=False):
    """Devuelve X (vectores), verdad (0,1,2 = identidades, 3 = intruso) y el origen de cada huella.

    origen: 'dataset (prueba)', 'subida a mano', 'intruso (prueba)' y, si se pide, 'intruso (calibracion)'.
    """
    d = np.load(KH.CARPETA / "dataset.npz", allow_pickle=True)
    prueba = PH.particion([str(a) for a in d["archivo"]]) == "prueba"
    X, verdad = [d["X"][prueba].astype(float)], [d["y"][prueba].astype(int)]
    origen = [np.array(["dataset (prueba)"] * int(prueba.sum()))]

    vec, ver, ori = [], [], []
    with open(RAIZ / "dataset_huellas" / "pruebas_manuales" / "registro.csv", encoding="utf-8") as f:
        for fila in csv.DictReader(f):
            ruta = RAIZ / "dataset_huellas" / "pruebas_manuales" / "huellas_nuevas" / fila["archivo"]
            if not ruta.exists() or not fila["verdad"]:
                continue
            vec.append(vector_de(ruta))
            ver.append(3 if fila["verdad"] == "intruso" else int(fila["verdad"][-1]) - 1)
            ori.append("subida a mano")
    X.append(np.array(vec).reshape(-1, CH.DIM_X))
    verdad.append(np.array(ver, dtype=int))
    origen.append(np.array(ori))

    Xi, cal, _ = KH.cargar_intrusos()
    usar = np.ones(len(Xi), dtype=bool) if incluir_calibracion else ~cal
    X.append(Xi[usar])
    verdad.append(np.full(int(usar.sum()), 3))
    origen.append(np.where(cal[usar], "intruso (calibracion)", "intruso (prueba)"))
    return np.vstack(X), np.concatenate(verdad), np.concatenate(origen)


def decidir(X, m, mapeo, lam):
    """Devuelve (salida 0..3, puntaje s de la identidad mas cercana, indice del cluster mas cercano)."""
    j, s, acep = DH.decidir(X, m, lam)
    return np.where(acep, mapeo[j], 3), s, j


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
    m, mapeo, lam, C = modelo()
    X, verdad, origen = cargar()
    aut, intr = verdad < 3, verdad == 3
    print(f"autorizadas: {aut.sum()}  | intrusos: {intr.sum()}  | umbral lambda aplicado = {lam:.2f}")

    # ------------------------------------------------ matriz de confusion
    fig, axs = plt.subplots(1, 2, figsize=(14, 6.2))
    for ax, r in zip(axs, (lam, 1.0)):
        sal, _, _ = decidir(X, m, mapeo, r)
        M = matriz(verdad, sal)
        mt = metricas(M)
        etiqueta = " (aplicado, calibrado con intrusos)" if r == lam else " (solo el límite del 95 % de validación)"
        dibujar_matriz(ax, M, f"Umbral λ = {r:.2f}{etiqueta}\nFRR {mt['frr'] * 100:.1f} %  ·  FAR {mt['far'] * 100:.1f} %  ·  exactitud {mt['exactitud'] * 100:.1f} %")
        print(f"lambda={r:.2f}  FRR={mt['frr'] * 100:.1f}% ({mt['rechazadas']}/{mt['n_aut']})  mal_identificadas={mt['mal_id']}  "
              f"FAR={mt['far'] * 100:.1f}% ({mt['intrusos_aceptados']}/{mt['n_int']})  exactitud={mt['exactitud'] * 100:.1f}%")
        print(M)
    fig.suptitle(f"Matriz de confusión de huellas con intrusos, solo evaluación ({int(aut.sum())} autorizadas y {int(intr.sum())} intrusos)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    SALIDA.mkdir(parents=True, exist_ok=True)
    fig.savefig(SALIDA / "matriz_confusion_intrusos.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------ intrusos contra el umbral
    Xc, vc, oc = cargar(incluir_calibracion=True)
    _, s, _ = decidir(Xc, m, mapeo, lam)
    fig, axs = plt.subplots(1, 2, figsize=(17, 7.2), gridspec_kw={"width_ratios": [1.1, 1]})
    grupos = [("Autorizadas\n(dataset prueba)", (vc < 3) & (oc == "dataset (prueba)"), "#2ca02c"),
              ("Autorizadas\n(subidas a mano)", (vc < 3) & (oc == "subida a mano"), "#98df8a"),
              ("Intrusos\n(subidos a mano)", (vc == 3) & (oc == "subida a mano"), "#ff9896"),
              ("Intrusos\n(calibración)", (vc == 3) & (oc == "intruso (calibracion)"), "#d9a0a0"),
              ("Intrusos\n(prueba)", (vc == 3) & (oc == "intruso (prueba)"), ROJO)]
    rejilla = [(Xc[g[1]], dict(s=48, color=g[2], edgecolor="black", linewidth=0.5), g[0].replace("\n", " ")) for g in grupos if g[1].any()]
    GD.dibujar_t2_q(axs[0], m, lam, rejilla, titulo=(f"T² contra Q: a la izquierda de la línea = aceptada (λ = {lam:.2f})" if not m.get("usar_q", True) else f"T² contra Q: dentro del rectángulo = aceptada (λ = {lam:.2f})"))
    axs[0].legend(fontsize=8, loc="lower right")

    ax = axs[1]
    rng = np.random.default_rng(0)
    for k, (nombre, sel, color) in enumerate(grupos):
        ax.scatter(rng.normal(k, 0.07, int(sel.sum())), s[sel], s=55, color=color, edgecolor="black", linewidth=0.6, zorder=3)
        ax.text(k, s.max() * 1.06, f"{int((s[sel] > lam).sum())} de {int(sel.sum())}\nrechazadas", ha="center", va="bottom", fontsize=9,
                fontweight="bold", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray"))
    ax.axhline(lam, color="black", lw=2.2, ls="--")
    ax.axhspan(lam, s.max() * 1.05, color=ROJO, alpha=0.08)
    ax.text(len(grupos) - 0.55, lam, f"umbral λ = {lam:.2f}", ha="right", va="bottom", fontweight="bold")
    ax.set_yscale("log")
    ax.set_xticks(range(len(grupos)))
    ax.set_xticklabels([g[0] for g in grupos], fontsize=9)
    ax.set_ylabel("puntaje s = T²/límite  (escala log)" if not m.get("usar_q", True) else "puntaje s = máx(T²/límite, Q/límite)  (escala log)")
    ax.set_ylim(s.min() * 0.8, s.max() * 1.9)
    ax.set_title("Puntaje de cada huella (arriba de la línea = rechazada)", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.25, which="both")
    fig.suptitle("Los intrusos quedan fuera del umbral", fontsize=15, fontweight="bold")
    fig.savefig(SALIDA / "intrusos_fuera_umbral.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Guardado en", SALIDA)


if __name__ == "__main__":
    main()
