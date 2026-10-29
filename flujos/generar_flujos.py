"""Genera los diagramas de flujo del informe (en lugar de los flujos en texto plano).

    python flujos/generar_flujos.py            # crea los PNG en flujos/imagenes/
    python flujos/generar_flujos.py --svg      # ademas guarda cada diagrama en SVG (vectorial, para el PDF)

Colores: los mismos que los graficos de PCA (paleta tab10 de matplotlib):
    azul   = huellas (y "reconocer")        naranja = voz (e "identificar")
    verde  = resultado aceptado (y "salir") rojo    = rechazo / NO AUTORIZADO
    violeta y marron = pasos intermedios
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Circle  # noqa: E402

SALIDA = Path(__file__).resolve().parent / "imagenes"

# Paleta tab10 (la de los PCA)
AZUL, NARANJA, VERDE, ROJO, VIOLETA, MARRON = (plt.cm.tab10(i) for i in (0, 1, 2, 3, 4, 5))
GRIS = "#5b6573"
TEXTO = "#1f2937"
FONDO = "#ffffff"

plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]


def _oscuro(c, f=0.78):
    return tuple(v * f for v in matplotlib.colors.to_rgb(c))


def _claro(c, f=0.88):
    return tuple(1 - (1 - v) * (1 - f) for v in matplotlib.colors.to_rgb(c))


def lienzo(w, h, titulo=None, subtitulo=None):
    """Figura vacia con ejes en coordenadas 0..w x 0..h (h hacia arriba)."""
    fig = plt.figure(figsize=(w / 2.2, h / 2.2), facecolor=FONDO)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    if titulo:
        ax.text(0.5, h - 0.45, titulo, fontsize=17, fontweight="bold", color=TEXTO, va="center")
    if subtitulo:
        ax.text(0.5, h - 1.0, subtitulo, fontsize=10.5, color=GRIS, va="center")
    return fig, ax


def _ajustar(texto, ancho, tam, factor=0.56):
    """Reduce el tamano de letra para que la linea mas larga entre en `ancho` (1 unidad del lienzo ~ 32,7 pt)."""
    largo = max(len(linea) for linea in texto.split("\n"))
    return min(tam, ancho * 32.7 * 0.90 / (largo * factor))


def caja(ax, x, y, w, h, titulo, detalle=None, color=AZUL, relleno=True, tam=11.5, tam_d=9):
    """Caja redondeada con sombra suave. (x, y) = esquina inferior izquierda."""
    tam = _ajustar(titulo, w, tam, 0.58)
    if detalle:
        tam_d = _ajustar(detalle, w, tam_d, 0.50)
    ax.add_patch(FancyBboxPatch((x + 0.07, y - 0.07), w, h, boxstyle="round,pad=0,rounding_size=0.25",
                                fc=(0, 0, 0, 0.10), ec="none", zorder=1))
    if relleno:
        fc, ec, ct, cd = color, _oscuro(color, 0.85), "white", (1, 1, 1, 0.92)
    else:
        fc, ec, ct, cd = _claro(color, 0.9), color, _oscuro(color, 0.7), _oscuro(color, 0.8)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.25", fc=fc, ec=ec, lw=1.6, zorder=2))
    if detalle:
        ax.text(x + w / 2, y + h * 0.66, titulo, ha="center", va="center", fontsize=tam, fontweight="bold", color=ct, zorder=3)
        ax.text(x + w / 2, y + h * 0.30, detalle, ha="center", va="center", fontsize=tam_d, color=cd, zorder=3, linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, titulo, ha="center", va="center", fontsize=tam, fontweight="bold", color=ct, zorder=3)


def rombo(ax, cx, cy, w, h, texto, color=VIOLETA, tam=10.5):
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    ax.add_patch(Polygon([(px + 0.07, py - 0.07) for px, py in pts], closed=True, fc=(0, 0, 0, 0.10), ec="none", zorder=1))
    ax.add_patch(Polygon(pts, closed=True, fc=color, ec=_oscuro(color, 0.85), lw=1.6, zorder=2))
    ax.text(cx, cy, texto, ha="center", va="center", fontsize=tam, fontweight="bold", color="white", zorder=3, linespacing=1.2)


def flecha(ax, p1, p2, color=GRIS, texto=None, curva=0.0, lw=2.2, desp=(0, 0.22)):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=16, lw=lw, color=color,
                                 connectionstyle=f"arc3,rad={curva}", zorder=4, shrinkA=0, shrinkB=0))
    if texto:
        ax.text((p1[0] + p2[0]) / 2 + desp[0], (p1[1] + p2[1]) / 2 + desp[1], texto, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=color, zorder=5,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none"))


def circulo_num(ax, x, y, n, color):
    ax.add_patch(Circle((x, y), 0.32, fc="white", ec=color, lw=2.2, zorder=6))
    ax.text(x, y, str(n), ha="center", va="center", fontsize=11, fontweight="bold", color=color, zorder=7)


def guardar(fig, nombre, svg):
    SALIDA.mkdir(parents=True, exist_ok=True)
    fig.savefig(SALIDA / f"{nombre}.png", dpi=170, facecolor=FONDO)
    if svg:
        fig.savefig(SALIDA / f"{nombre}.svg", facecolor=FONDO)
    plt.close(fig)
    print("  ->", SALIDA / f"{nombre}.png")


# =====================================================================================
# 1) Flujo general del agente (seccion 1.1)
# =====================================================================================
def flujo_general(svg):
    fig, ax = lienzo(21, 5.2, "Flujo general del agente", "Las mismas etapas para huellas y para voz; cambian las técnicas de cada una")
    etapas = [("Percepción", "huella: imagen\nvoz: audio 1,5 s", AZUL),
              ("Acondicionamiento", "huella: normalizar\ny alinear\nvoz: volumen, silencios", NARANJA),
              ("Filtrado", "huella: Gabor + Otsu\nvoz: pasa-banda\n80–7000 Hz", VERDE),
              ("Vectorización", "huella: 72 números\nvoz: MFCC, 91 números", VIOLETA),
              ("Clasificación", "huella: K-means\nvoz: KNN + DTW", ROJO),
              ("Acción", "reconocer, identificar,\nsalir", MARRON)]
    w, h, sep, x0, y0 = 2.95, 2.5, 0.52, 0.5, 0.8
    for i, (t, d, c) in enumerate(etapas):
        x = x0 + i * (w + sep)
        caja(ax, x, y0, w, h, t, d, c, tam=11.5, tam_d=8.8)
        circulo_num(ax, x + 0.1, y0 + h, i + 1, c)
        if i < len(etapas) - 1:
            flecha(ax, (x + w + 0.04, y0 + h / 2), (x + w + sep - 0.04, y0 + h / 2), GRIS)
    guardar(fig, "01_flujo_general", svg)


# =====================================================================================
# 2) Regla de decision de la huella (seccion 1.2 h)
# =====================================================================================
def regla_huella(svg):
    fig, ax = lienzo(21, 7.6, "Regla de decisión para una huella nueva", "identificar_huella() en src/kmeans_huellas.py")
    caja(ax, 0.6, 3.0, 3.3, 1.8, "Vector de la huella", "72 números\n(histograma de orientación)", AZUL)
    caja(ax, 4.9, 3.0, 3.6, 1.8, "Distancia a los 3 centroides", "distancia euclídea\n3 valores", NARANJA)
    caja(ax, 9.5, 3.0, 3.3, 1.8, "Centroide más cercano", "d = menor distancia", VERDE)
    rombo(ax, 15.0, 3.9, 3.8, 2.7, "¿d ≤ R?\nR = 0,42", VIOLETA)
    caja(ax, 17.5, 5.55, 3.2, 1.4, "Identidad del cluster", "asignada con el húngaro", VERDE)
    caja(ax, 17.5, 0.85, 3.2, 1.4, "NO AUTORIZADO", None, ROJO, tam=12.5)
    flecha(ax, (3.94, 3.9), (4.86, 3.9))
    flecha(ax, (8.54, 3.9), (9.46, 3.9))
    flecha(ax, (12.84, 3.9), (13.1, 3.9))
    flecha(ax, (15.0 + 0.8, 4.9), (17.46, 6.25), VERDE, "sí", curva=-0.15, desp=(-0.25, 0.3))
    flecha(ax, (15.0 + 0.8, 2.9), (17.46, 1.55), ROJO, "no", curva=0.15, desp=(-0.25, -0.3))
    guardar(fig, "02_regla_decision_huella", svg)


# =====================================================================================
# 3) DTW (seccion 1.3 e): tabla de costos con el camino de alineacion
# =====================================================================================
def _dtw(a, b):
    n, m = len(a), len(b)
    costo = np.abs(a[:, None] - b[None, :])
    D = np.full((n, m), np.inf)
    D[0, 0] = costo[0, 0]
    for i in range(n):
        for j in range(m):
            if i == j == 0:
                continue
            prev = [D[i - 1, j - 1] if i and j else np.inf, D[i - 1, j] if i else np.inf, D[i, j - 1] if j else np.inf]
            D[i, j] = costo[i, j] + min(prev)
    i, j, camino = n - 1, m - 1, [(n - 1, m - 1)]
    while (i, j) != (0, 0):
        opciones = [(D[i - 1, j - 1] if i and j else np.inf, (i - 1, j - 1)),
                    (D[i - 1, j] if i else np.inf, (i - 1, j)),
                    (D[i, j - 1] if j else np.inf, (i, j - 1))]
        i, j = min(opciones, key=lambda t: t[0])[1]
        camino.append((i, j))
    return D, camino[::-1]


def dtw(svg):
    a = np.array([1, 2, 4, 6, 6, 5, 3, 2.0])             # "palabra" dicha rapido
    b = np.array([1, 1, 2, 3, 4, 6, 6, 6, 5, 4, 3, 2.0])  # la misma, mas lenta
    D, camino = _dtw(a, b)
    n, m = len(a), len(b)
    fig, ax = lienzo(21, 9.2, "Distancia DTW: alinear dos audios aunque duren distinto",
                     "Cada celda = costo de emparejar un tramo del audio A con uno del audio B; el camino marcado es el alineamiento de menor costo")
    c = 0.62                      # lado de celda
    ox, oy = 2.8, 0.9
    norm = plt.Normalize(D.min(), D.max())
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("t", [_claro(AZUL, 0.92), _claro(AZUL, 0.45), AZUL])
    en_camino = set(camino)
    for i in range(n):
        for j in range(m):
            x, y = ox + j * c, oy + (n - 1 - i) * c
            ax.add_patch(plt.Rectangle((x, y), c, c, fc=cmap(norm(D[i, j])), ec="white", lw=1.5, zorder=2))
            if (i, j) in en_camino:
                ax.add_patch(plt.Rectangle((x + 0.03, y + 0.03), c - 0.06, c - 0.06, fc=NARANJA, ec=_oscuro(NARANJA), lw=1.4, zorder=3))
    for j in range(m):
        ax.text(ox + j * c + c / 2, oy + n * c + 0.28, f"{b[j]:g}", ha="center", va="center", fontsize=8.5, color=GRIS)
    for i in range(n):
        ax.text(ox - 0.3, oy + (n - 1 - i) * c + c / 2, f"{a[i]:g}", ha="center", va="center", fontsize=8.5, color=GRIS)
    ax.text(ox + m * c / 2, oy + n * c + 0.75, "tramos del audio B (más lento)", ha="center", fontsize=10, fontweight="bold", color=AZUL)
    ax.text(1.2, oy + n * c / 2,"tramos del\naudio A\n(más rápido)", ha="center", va="center", fontsize=10, fontweight="bold", color=AZUL, linespacing=1.3)
    # recurrencia
    x0 = 11.2
    caja(ax, x0, 5.6, 9.3, 1.55, "Se llena la tabla celda por celda", "D[i][j] = d(tramo_A[i], tramo_B[j]) + mínimo( D[i-1][j-1],  D[i-1][j],  D[i][j-1] )", VERDE, tam=11.5, tam_d=10)
    caja(ax, x0, 3.35, 9.3, 1.55, "Resultado", "DTW(A, B) = D[última][última] / (largo A + largo B)", VIOLETA, tam=11.5, tam_d=10.5)
    caja(ax, x0, 1.1, 9.3, 1.55, "Cuanto menor, más parecidos", "la misma palabra dicha más rápido o más lento da una distancia baja", ROJO, tam=11.5, tam_d=10)
    flecha(ax, (x0 + 4.65, 5.55), (x0 + 4.65, 4.95), GRIS)
    flecha(ax, (x0 + 4.65, 3.3), (x0 + 4.65, 2.7), GRIS)
    guardar(fig, "03_dtw", svg)


# =====================================================================================
# 4) Clasificar un audio nuevo (seccion 1.3 h)
# =====================================================================================
def clasificar_audio(svg):
    fig, ax = lienzo(21, 9.0, "Qué se hace con un audio nuevo", "clasificar_audio_detalle() en src/knn_audios.py")
    pasos = [("Audio nuevo", "1,5 s, micrófono", AZUL),
             ("Acondicionar", "continua, pasa-banda,\nsilencios, volumen", NARANJA),
             ("Caracterizar", "secuencia de MFCC\n(tramos × 39)", VERDE),
             ("DTW contra\ntodos", "197 audios\nde referencia", VIOLETA)]
    w, h, y = 3.5, 2.1, 4.9
    for i, (t, d, c) in enumerate(pasos):
        x = 0.7 + i * (w + 0.55)
        caja(ax, x, y, w, h, t, d, c, tam=11.2, tam_d=9)
        if i < 3:
            flecha(ax, (x + w + 0.03, y + h / 2), (x + w + 0.52, y + h / 2))
    # segunda fila
    caja(ax, 0.7, 1.55, 3.9, 2.0, "3 vecinos", "los 3 ejemplos con\nmenor distancia", AZUL)
    caja(ax, 5.5, 1.55, 3.9, 2.0, "Votación", "gana el comando más votado\n(empate: el más cercano)", NARANJA)
    rombo(ax, 12.6, 2.55, 3.9, 2.3, "¿los 3\ncoinciden?", VIOLETA)
    caja(ax, 16.0, 3.2, 4.4, 1.5, "Confianza ALTA", "se ejecuta el comando", VERDE)
    caja(ax, 16.0, 0.8, 4.4, 1.5, "Confianza MEDIA", "2 contra 1: no se ejecuta", ROJO)
    # cierre de la fila 1 hacia la fila 2
    xf = 0.7 + 3 * (w + 0.55) + w / 2
    flecha(ax, (xf, y - 0.03), (xf, 4.35), GRIS)
    ax.plot([xf, 2.65], [4.35, 4.35], color=GRIS, lw=2.2, zorder=4)
    flecha(ax, (2.65, 4.35), (2.65, 3.6), GRIS)
    flecha(ax, (4.64, 2.55), (5.46, 2.55))
    flecha(ax, (9.44, 2.55), (10.6, 2.55))
    flecha(ax, (13.5, 3.2), (15.96, 3.95), VERDE, "sí", curva=-0.12, desp=(-0.1, 0.35))
    flecha(ax, (13.5, 1.95), (15.96, 1.55), ROJO, "no", curva=0.12, desp=(-0.1, -0.35))
    guardar(fig, "04_clasificar_audio", svg)


# =====================================================================================
# 5) y 7) Carriles huella / voz hacia la interfaz
# =====================================================================================
def carriles(svg, nombre, titulo, subtitulo, huella, voz, destino, ancho=21):
    fig, ax = lienzo(ancho, 8.4, titulo, subtitulo)
    n = max(len(huella), len(voz))
    w, h = 3.55, 1.7
    sep = 0.75
    x0 = 0.6
    tam_u = min(_ajustar(t, w, 10.5, 0.58) for t, _ in huella + voz)          # mismo tamano de titulo en todas las cajas
    tam_du = min(_ajustar(d, w, 8.6, 0.50) for _, d in huella + voz)
    for fila, (items, color, y, rotulo) in enumerate(((huella, AZUL, 4.6, "HUELLAS"), (voz, NARANJA, 1.4, "VOZ"))):
        ax.text(x0, y + h + 0.3, rotulo, fontsize=11, fontweight="bold", color=color)
        for i, (t, d) in enumerate(items):
            x = x0 + i * (w + sep)
            caja(ax, x, y, w, h, t, d, color, tam=tam_u, tam_d=tam_du)
            if i < len(items) - 1:
                flecha(ax, (x + w + 0.03, y + h / 2), (x + w + sep - 0.03, y + h / 2), color)
    xe = x0 + n * (w + sep) - sep + 0.9
    caja(ax, xe, 3.0, ancho - xe - 0.5, 1.9, destino[0], destino[1], VERDE, tam=11.5, tam_d=9)
    ult_h = x0 + (len(huella) - 1) * (w + sep) + w
    ult_v = x0 + (len(voz) - 1) * (w + sep) + w
    flecha(ax, (ult_h + 0.03, 4.6 + h / 2), (xe - 0.03, 4.55), AZUL, curva=-0.12)
    flecha(ax, (ult_v + 0.03, 1.4 + h / 2), (xe - 0.03, 3.4), NARANJA, curva=0.12)
    guardar(fig, nombre, svg)


def arquitectura(svg):
    carriles(svg, "05_arquitectura_codigo", "Organización del código",
             "Cada etapa es un script que deja su resultado en una carpeta; el siguiente lo lee",
             [("huellasDactilares/", "90 imágenes de Anguli"), ("preprocesar.py", "acondicionamiento\ny filtrado"),
              ("caracterizar_huellas.py", "vector de 72 números"), ("kmeans_huellas.py", "K-means desde cero\n+ umbral R")],
             [("grabar_audios.py", "audios de 1,5 s"), ("normalizar_audios.py", "acondicionamiento\ny filtrado"),
              ("caracterizar_audios.py", "MFCC, 39 y 91 números"), ("knn_audios.py", "KNN + DTW desde cero")],
             ("interfaz/app.py", "aplicación completa\n(usa los modelos de las\ndos ramas)"))


def ejecucion(svg):
    carriles(svg, "07_como_ejecutar", "Cómo ejecutar el proyecto", "Comandos en el orden de cada rama",
             [("python src/preprocesar.py", "huellas procesadas"), ("python src/caracterizar_huellas.py", "vectores de 72"),
              ("python src/kmeans_huellas.py", "modelo y umbral")],
             [("python src/normalizar_audios.py", "audios acondicionados"), ("python src/caracterizar_audios.py", "MFCC y vectores"),
              ("python src/knn_audios.py", "modelo KNN")],
             ("python interfaz/app.py", "aplicación completa"))


# =====================================================================================
# 6) Flujo de la interfaz (seccion 2.1 f)
# =====================================================================================
def interfaz(svg):
    fig, ax = lienzo(21, 9.2, "Flujo de trabajo de la interfaz y conexión con los otros programas", "Qué módulo de src/ usa cada paso")
    pasos = [("Generar huella", "anguli.py\nejecuta Anguli (subprocess)\n\nidentidad: huella base + filtro\nintruso: persona nueva\n\n→ imagen .jpg en huellasConID/", AZUL),
             ("Decir un comando", "micrófono → servicios.py\n→ knn_audios.clasificar_\naudio_detalle()\n\n(acondiciona con\nnormalizar_audios y calcula\nlos MFCC)\n\n→ comando + confianza", NARANJA),
             ("«Reconocer»", "imagen\n→ preprocesar.\npreprocesar_huella()\n→ caracterizar_huellas\n→ kmeans_huellas.\nidentificar_huella_detalle()\n\n→ AUTORIZADO /\nNO AUTORIZADO", VIOLETA),
             ("«Identificar»", "solo si fue autorizada\n\nmuestra la identidad\n\ny, opcional, el gráfico de\nclusters (grafico.py)", VERDE),
             ("«Salir»", "cierra el programa", ROJO)]
    w, h, sep, x0, y0 = 3.5, 6.2, 0.62, 0.55, 0.9
    tam_cuerpo = min(_ajustar(d, w - 0.3, 8.8, 0.50) for _, d, _ in pasos)
    for i, (t, d, c) in enumerate(pasos):
        x = x0 + i * (w + sep)
        ax.add_patch(FancyBboxPatch((x + 0.07, y0 - 0.07), w, h, boxstyle="round,pad=0,rounding_size=0.25", fc=(0, 0, 0, 0.10), ec="none", zorder=1))
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0,rounding_size=0.25", fc=_claro(c, 0.9), ec=c, lw=1.8, zorder=2))
        ax.add_patch(FancyBboxPatch((x, y0 + h - 1.2), w, 1.2, boxstyle="round,pad=0,rounding_size=0.25", fc=c, ec=c, lw=1.8, zorder=3))
        ax.add_patch(plt.Rectangle((x + 0.02, y0 + h - 1.2), w - 0.04, 0.5, fc=c, ec="none", zorder=3))
        ax.text(x + w / 2 + 0.3, y0 + h - 0.6, t, ha="center", va="center", fontsize=_ajustar(t, w - 1.1, 11.5, 0.58), fontweight="bold", color="white", zorder=4)
        circulo_num(ax, x + 0.5, y0 + h - 0.6, i + 1, c)
        ax.text(x + w / 2, y0 + (h - 1.2) / 2, d, ha="center", va="center", fontsize=tam_cuerpo, color=_oscuro(c, 0.55), zorder=4, linespacing=1.35)
        if i < len(pasos) - 1:
            flecha(ax, (x + w + 0.03, y0 + h - 0.6), (x + w + sep - 0.03, y0 + h - 0.6), GRIS)
    guardar(fig, "06_flujo_interfaz", svg)


GENERADORES = [flujo_general, regla_huella, dtw, clasificar_audio, arquitectura, interfaz, ejecucion]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--svg", action="store_true", help="guardar tambien en SVG")
    a = ap.parse_args()
    print("Generando diagramas en", SALIDA)
    for g in GENERADORES:
        g(a.svg)
    sys.exit(0)
