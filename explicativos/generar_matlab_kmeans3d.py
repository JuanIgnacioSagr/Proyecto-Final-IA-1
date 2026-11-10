"""Genera un script de MATLAB AUTOCONTENIDO (con los datos incluidos) para ver el K-means de las huellas en 3D con sus ovalos.

    python explicativos/generar_matlab_kmeans3d.py     # crea explicativos/kmeans_huellas_3d.m

El .m muestra las huellas en las 3 primeras componentes del PCA del modelo, los 3 centroides y, alrededor de cada uno, el
ELIPSOIDE de aceptacion (sombra en 3D de la zona de Mahalanobis T2 <= lambda * limite, calculada en 6 componentes).
Se copia y se pega en MATLAB (no necesita ningun otro archivo).
"""
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
import decision_huellas as DH                                     # noqa: E402
import kmeans_huellas as KH                                       # noqa: E402
import particion_huellas as PH                                    # noqa: E402

SALIDA = Path(__file__).resolve().parent / "kmeans_huellas_3d.m"


def mat(a, fmt="%.4f"):
    return "[ ...\n" + "".join("  " + " ".join(fmt % v for v in fila) + "; ...\n" for fila in a) + "];"


def main():
    modelo = np.load(KH.RUTA_MODELO, allow_pickle=False)
    m = DH.desde_modelo(modelo)
    lam, mapeo = float(modelo["radios"][0]), modelo["mapeo"].astype(int)
    X, y, efecto, archivo, imp, inv = KH.cargar_datos()
    parte = PH.particion(archivo)
    Xi, cal, _ = KH.cargar_intrusos()
    Z = (X - m["mu"]) @ m["V"][:, :3]
    Zi = (Xi - m["mu"]) @ m["V"][:, :3]
    acepta_a = DH.decidir(X, m, lam)[2]
    acepta_i = DH.decidir(Xi, m, lam)[2]
    codigo_parte = np.array([{"entrenamiento": 1, "validacion": 2, "prueba": 3}[p] for p in parte])
    Zc = m["Zc"][:, :3]
    cov3 = np.stack([np.linalg.inv(m["precision"][j])[:3, :3] for j in range(len(Zc))])
    r = float(np.sqrt(lam * m["t2_limite"]))
    ev = m["varianza"][:3] / m["varianza"].sum() * 100

    L = []
    L.append("%% K-means de las huellas en 3D (PCA) con los OVALOS de aceptacion (elipsoides de Mahalanobis).")
    L.append("%% Pegar y ejecutar en MATLAB (los datos estan incluidos). Girar con el mouse.")
    L.append("%% Cada ovalo es la sombra en 3D de la zona T2 <= lambda*limite (calculada en %d componentes del PCA)." % m["k"])
    L.append("clear; clc; close all;")
    L.append("lambda = %.2f;   %% umbral (T2 <= lambda*limite_T2 = %.2f)" % (lam, lam * m["t2_limite"]))
    L.append("r = %.4f;      %% radio de Mahalanobis del ovalo = sqrt(lambda*limite_T2)" % r)
    L.append("varexp = [%s];   %% varianza explicada PC1 PC2 PC3 (porcentaje)" % " ".join("%.1f" % v for v in ev))
    L.append("ident_de_centro = [%s];   %% identidad (1..3) que le corresponde a cada centroide" % " ".join(str(int(v) + 1) for v in mapeo))
    L.append("Zc = " + mat(Zc))
    for j in range(len(Zc)):
        L.append("C%d = %s" % (j + 1, mat(cov3[j], "%.6f")))
    L.append("Z = " + mat(Z))
    L.append("ident = [%s];   %% identidad real de cada huella" % " ".join(str(int(v) + 1) for v in y))
    L.append("parte = [%s];   %% 1 entrenamiento, 2 validacion, 3 prueba" % " ".join(str(int(v)) for v in codigo_parte))
    L.append("acepta = [%s];   %% 1 si la regla (6 componentes) la acepta" % " ".join(str(int(v)) for v in acepta_a))
    L.append("Zi = " + mat(Zi))
    L.append("acepta_i = [%s];   %% intrusos: 1 si la regla los acepta (error)" % " ".join(str(int(v)) for v in acepta_i))
    L.append("""
col = [0.12 0.47 0.71; 1.00 0.50 0.05; 0.17 0.63 0.17];      % identidad 1, 2, 3 (los colores de los PCA)
rojo = [0.84 0.15 0.16];
figure('Color','w','Position',[80 60 1000 780]); hold on; grid on; axis equal;
[sx,sy,sz] = sphere(40);
for j = 1:3
    id = ident_de_centro(j);
    Cj = eval(sprintf('C%d', j));                    % covarianza del centro j en las 3 primeras componentes
    [V,D] = eig(Cj);
    P = [sx(:) sy(:) sz(:)] * (r*sqrt(D)) * V';      % esfera -> elipsoide
    X = reshape(P(:,1),size(sx)) + Zc(j,1);
    Y = reshape(P(:,2),size(sx)) + Zc(j,2);
    W = reshape(P(:,3),size(sx)) + Zc(j,3);
    surf(X,Y,W,'FaceColor',col(id,:),'FaceAlpha',0.10,'EdgeColor','none');
    mesh(X,Y,W,'EdgeColor',col(id,:),'EdgeAlpha',0.18,'FaceColor','none');
    plot3(Zc(j,1),Zc(j,2),Zc(j,3),'X','MarkerSize',18,'LineWidth',3.5,'Color',col(id,:)*0.7);
    text(Zc(j,1),Zc(j,2),Zc(j,3)+max(diag(r*sqrt(D)))*1.1,sprintf('Identidad %d',id),'FontWeight','bold','HorizontalAlignment','center');
end
marc = {'o','s','^'};                                 % entrenamiento, validacion, prueba
nomb = {'entrenamiento','validacion','prueba'};
for p = 1:3
    for id = 1:3
        k = (parte==p) & (ident==id);
        scatter3(Z(k,1),Z(k,2),Z(k,3),38,col(id,:),marc{p},'filled','MarkerEdgeColor','k','LineWidth',0.5);
    end
end
scatter3(Zi(acepta_i==0,1),Zi(acepta_i==0,2),Zi(acepta_i==0,3),55,rojo,'x','LineWidth',2);
if any(acepta_i==1)
    scatter3(Zi(acepta_i==1,1),Zi(acepta_i==1,2),Zi(acepta_i==1,3),110,[1 0.8 0],'x','LineWidth',3);
end
xlabel(sprintf('PC1 (%.1f%%)',varexp(1))); ylabel(sprintf('PC2 (%.1f%%)',varexp(2))); zlabel(sprintf('PC3 (%.1f%%)',varexp(3)));
title(sprintf('K-means de huellas en 3D: ovalos de aceptacion (Mahalanobis, lambda = %.2f)', lambda));
h = [scatter3(nan,nan,nan,38,[0.5 0.5 0.5],'o','filled','MarkerEdgeColor','k'), scatter3(nan,nan,nan,38,[0.5 0.5 0.5],'s','filled','MarkerEdgeColor','k'), ...
     scatter3(nan,nan,nan,38,[0.5 0.5 0.5],'^','filled','MarkerEdgeColor','k'), scatter3(nan,nan,nan,55,rojo,'x','LineWidth',2)];
legend(h,{'entrenamiento','validacion','prueba','intrusos (rechazados)'},'Location','northeastoutside');
view(35,22); camlight; rotate3d on;
fprintf('Huellas autorizadas: %d | intrusos: %d | intrusos aceptados por la regla: %d\\n', size(Z,1), size(Zi,1), sum(acepta_i));
""")
    SALIDA.write_text("\n".join(L), encoding="utf-8")
    print("Guardado:", SALIDA, "| autorizadas", len(Z), "| intrusos", len(Zi), "| r =", round(r, 3), "| aceptadas", int(acepta_a.sum()), "| intrusos aceptados", int(acepta_i.sum()))


if __name__ == "__main__":
    main()
