"""Como evoluciona UNA huella en el preprocesamiento, en dos graficos.

    python explicativos/etapas_preprocesado.py                 # elige una huella con giro, ruido y rayones
    python explicativos/etapas_preprocesado.py ruta/huella.jpg # o una huella cualquiera

  1) preproceso_1_acondicionamiento.png : original -> normalizacion -> mascara -> alineacion
  2) preproceso_2_filtrado.png          : acondicionada -> orientacion de crestas -> Gabor -> Otsu -> limpieza (resultado final)
Usa las funciones reales de src/preprocesar.py (no es una maqueta).
"""
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
import preprocesar as P                                           # noqa: E402

SALIDA = Path(__file__).resolve().parent / "imagenes"
AZUL, NARANJA, VERDE, ROJO, VIOLETA = (plt.cm.tab10(i) for i in (0, 1, 2, 3, 4))
plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]


def elegir_huella():
    """La impresion de 'primera' (giro, traslacion, rayones y ruido) de la identidad 2 con mas giro."""
    mejor, mejor_ang = None, -1
    for ruta in sorted((RAIZ / "huellasDactilares" / "impresionPrimera" / "id2").glob("*.jpg")):
        g = P.normalizar(P.a_gris(ruta))
        _, _, ang = P.alinear(g, P.mascara_roi(g))
        if abs(ang) > mejor_ang:
            mejor, mejor_ang = ruta, abs(ang)
    return mejor


def titulo(ax, n, texto, color):
    ax.set_title(f"{n}  {texto}", fontsize=11.5, fontweight="bold", color=color, loc="left")
    ax.axis("off")


def flecha_entre(fig, ax1, ax2):
    p1, p2 = ax1.get_position(), ax2.get_position()
    fig.patches.append(plt.matplotlib.patches.FancyArrowPatch((p1.x1 + 0.004, (p1.y0 + p1.y1) / 2), (p2.x0 - 0.004, (p2.y0 + p2.y1) / 2),
                       transform=fig.transFigure, arrowstyle="-|>", mutation_scale=22, lw=3, color="#5b6573"))


# =====================================================================================
def figura_1(ruta, pasos):
    gris, norm, mask, alin, mask2, ang = (pasos[k] for k in ("gris", "normalizada", "mascara", "alineada", "mascara_alineada", "angulo"))
    mask01 = (mask > 0).astype(np.uint8)
    a_eje, ecc, (cx, cy) = P._geometria(mask01)

    fig, axs = plt.subplots(1, 4, figsize=(19, 6.6))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.82, bottom=0.06, wspace=0.13)

    axs[0].imshow(gris, cmap="gray", vmin=0, vmax=255)
    titulo(axs[0], "1", "Imagen original\n(en gris, con giro, rayones y ruido)", AZUL)

    axs[1].imshow(norm, cmap="gray", vmin=0, vmax=255)
    titulo(axs[1], "2", "Normalización de contraste\n(se estira el rango de grises)", NARANJA)
    ins = axs[1].inset_axes([0.05, 0.02, 0.9, 0.17])
    ins.hist(gris.ravel(), bins=64, range=(0, 255), color=AZUL, alpha=0.6, label="antes")
    ins.hist(norm.ravel(), bins=64, range=(0, 255), color=NARANJA, alpha=0.6, label="después")
    ins.set_yscale("log")
    ins.set_yticks([])
    ins.tick_params(labelsize=7)
    ins.legend(fontsize=7, loc="upper center", ncol=2, frameon=False)

    axs[2].imshow(norm, cmap="gray", vmin=0, vmax=255)
    rojo = np.zeros((*mask01.shape, 4))
    rojo[mask01 > 0] = (*VERDE[:3], 0.45)
    axs[2].imshow(rojo)
    cont, _ = cv2.findContours(mask01, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in cont:
        axs[2].plot(c[:, 0, 0], c[:, 0, 1], color=VERDE, lw=2.4)
    t = np.radians(a_eje)
    L = 170
    axs[2].plot([cx - L * np.cos(t), cx + L * np.cos(t)], [cy - L * np.sin(t), cy + L * np.sin(t)], color=ROJO, lw=2.6, ls="--")
    axs[2].plot(cx, cy, "o", color=ROJO, ms=9, mec="white", mew=1.5)
    axs[2].text(0.5, 0.02, "verde = zona con crestas · rojo = centro y eje mayor", transform=axs[2].transAxes, ha="center",
                fontsize=9, color="white", bbox=dict(boxstyle="round,pad=0.25", fc="black", alpha=0.65))
    titulo(axs[2], "3", "Máscara de la zona con crestas\n(se separa la huella del fondo)", VERDE)

    axs[3].imshow(alin, cmap="gray", vmin=0, vmax=255)
    axs[3].axvline(alin.shape[1] / 2, color=ROJO, lw=1.2, ls="--", alpha=0.8)
    axs[3].axhline(alin.shape[0] / 2, color=ROJO, lw=1.2, ls="--", alpha=0.8)
    axs[3].text(0.5, 0.02, f"girada {ang:+.1f}° y centrada en el lienzo", transform=axs[3].transAxes, ha="center", fontsize=9,
                color="white", bbox=dict(boxstyle="round,pad=0.25", fc="black", alpha=0.65))
    titulo(axs[3], "4", "Alineación\n(se centra y se endereza)", VIOLETA)

    for i in range(3):
        flecha_entre(fig, axs[i], axs[i + 1])
    fig.suptitle(f"Acondicionamiento de una huella ({ruta.name})", fontsize=16, fontweight="bold", y=0.97)
    fig.savefig(SALIDA / "preproceso_1_acondicionamiento.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("Guardado:", SALIDA / "preproceso_1_acondicionamiento.png", f"(giro aplicado {ang:+.1f} grados)")


# =====================================================================================
def figura_2(ruta, pasos):
    alin, mask2, binaria = pasos["alineada"], (pasos["mascara_alineada"] > 0).astype(np.uint8), pasos["binaria"]
    out, coh = P.gabor_orientado(alin, mask2)
    o8 = np.clip((out - out.min()) / (np.ptp(out) + 1e-9) * 255, 0, 255).astype(np.uint8)
    T = P._otsu(o8, mask2)
    ink_sin = ((o8 >= T) * mask2).astype(np.uint8)
    ink_con = (binaria == 0).astype(np.uint8)

    tinta = (255 - cv2.medianBlur(alin, 3)).astype(np.float32) / 255.0
    alfa, _ = P.campo_orientacion(tinta)
    n, et, st, _ = cv2.connectedComponentsWithStats(ink_sin, connectivity=8)
    chicos = np.where(st[1:, cv2.CC_STAT_AREA] < 30)[0] + 1

    fig, axs = plt.subplots(2, 3, figsize=(17, 12.2))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.03, wspace=0.08, hspace=0.18)

    axs[0, 0].imshow(alin, cmap="gray", vmin=0, vmax=255)
    titulo(axs[0, 0], "1", "Huella acondicionada\n(sale de la alineación)", AZUL)

    axs[0, 1].imshow(alin, cmap="gray", vmin=0, vmax=255, alpha=0.55)
    H, W = alin.shape
    paso = 18
    for y in range(paso // 2, H, paso):
        for x in range(paso // 2, W, paso):
            if mask2[y, x]:
                a = alfa[y, x] + np.pi / 2                    # direccion de la cresta (perpendicular al gradiente)
                dx, dy = 0.45 * paso * np.cos(a), 0.45 * paso * np.sin(a)
                axs[0, 1].plot([x - dx, x + dx], [y - dy, y + dy], color=plt.cm.hsv((a % np.pi) / np.pi), lw=2.2, solid_capstyle="round")
    titulo(axs[0, 1], "2", "Orientación de las crestas en cada zona\n(el filtro se sintoniza a esa dirección; color = ángulo)", NARANJA)

    axs[0, 2].imshow(out, cmap="gray")
    titulo(axs[0, 2], "3", "Respuesta del filtro de Gabor orientado\n(realza crestas, rellena cortes y rayones)", VERDE)

    ax = axs[1, 0]
    ax.hist(o8[mask2 > 0], bins=64, color=VERDE, alpha=0.85)
    ax.axvline(T, color=ROJO, lw=2.8)
    ax.text(T + 4, ax.get_ylim()[1] * 0.85, f"umbral de Otsu = {T:.0f}\n(el que mejor separa\nclaro de oscuro)", color=ROJO, fontsize=10, fontweight="bold")
    ax.text(T - 6, ax.get_ylim()[1] * 0.5, "valle\n→ fondo", ha="right", fontsize=10)
    ax.text(T + 60, ax.get_ylim()[1] * 0.5, "cresta →", ha="left", fontsize=10)
    ax.set_yticks([])
    ax.set_xlabel("intensidad de la respuesta de Gabor (0-255)")
    ax.set_title("4  Binarización: se elige el umbral con Otsu", fontsize=11.5, fontweight="bold", color=ROJO, loc="left")

    img = np.where(ink_sin > 0, 0, 255).astype(np.uint8)
    rgb = np.stack([img] * 3, axis=-1)
    for k in chicos:
        rgb[et == k] = (255, 80, 80)
    axs[1, 1].imshow(rgb)
    axs[1, 1].text(0.5, 0.02, f"{len(chicos)} motas sueltas (en rojo)", transform=axs[1, 1].transAxes, ha="center", fontsize=9.5,
                   color="white", bbox=dict(boxstyle="round,pad=0.25", fc="black", alpha=0.65))
    titulo(axs[1, 1], "5", "Imagen en blanco y negro\n(solo con Otsu)", ROJO)

    axs[1, 2].imshow(binaria, cmap="gray", vmin=0, vmax=255)
    axs[1, 2].text(0.5, 0.02, "0 motas · crestas negras sobre fondo blanco", transform=axs[1, 2].transAxes, ha="center", fontsize=9.5,
                   color="white", bbox=dict(boxstyle="round,pad=0.25", fc="black", alpha=0.65))
    titulo(axs[1, 2], "6", "Limpieza de motas y agujeros\n= HUELLA PROCESADA (la que se vectoriza)", VIOLETA)

    fig.suptitle(f"Filtrado y binarización de la huella ({ruta.name})", fontsize=16, fontweight="bold", y=0.965)
    fig.savefig(SALIDA / "preproceso_2_filtrado.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Guardado:", SALIDA / "preproceso_2_filtrado.png", f"(Otsu = {T:.0f}, motas antes de limpiar = {len(chicos)})")


def main():
    ruta = Path(sys.argv[1]) if len(sys.argv) > 1 else elegir_huella()
    pasos = P.preprocesar_huella(ruta, devolver_pasos=True)
    SALIDA.mkdir(parents=True, exist_ok=True)
    figura_1(ruta, pasos)
    figura_2(ruta, pasos)


if __name__ == "__main__":
    main()
