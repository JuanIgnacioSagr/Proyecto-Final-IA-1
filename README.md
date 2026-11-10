# Reconocimiento biométrico: huella dactilar y comandos de voz

Trabajo Final de Inteligencia Artificial I (2026). Un agente de control de acceso que:

- **Huella dactilar:** identifica a 3 personas autorizadas con **K-means** (K = 3) y decide si aceptar o rechazar con la **distancia de Mahalanobis** sobre un **PCA** de 6 componentes. Una huella que no se parece a ninguna se rechaza como `NO AUTORIZADO`.
- **Voz:** reconoce 3 comandos (`reconocer`, `identificar`, `salir`) con **KNN** (K = 3, distancia euclídea) sobre MFCC.
- **Interfaz:** una ventana (Tkinter) integra todo: se genera o elige una huella y se manejan los pasos con la voz o con botones.

El informe completo está en [`informe/Informe_final_IA.pdf`](informe/Informe_final_IA.pdf) (fuente: [`informe/Informe_final_IA.md`](informe/Informe_final_IA.md)).

## Resultados principales

| | Resultado |
| --- | --- |
| Huellas: identificación de la identidad | 100 % en entrenamiento, validación y prueba |
| Huellas: prueba con intrusos (33 autorizadas + 25 intrusos) | 56 de 58 bien resueltas (96,6 %); 0 de 25 intrusos aceptados; 2 autorizadas rechazadas por error |
| Umbral de rechazo | λ = 1,48 → T² ≤ 13,6 (1,48 veces el percentil 95 de las autorizadas de validación) |
| Voz: KNN (K = 3, distancia euclídea) | 94,7 % en validación y 90,9 % en prueba (20 de 22 audios) |

Los datos de huellas son **sintéticos** (generados con el programa Anguli) y los audios fueron grabados por el equipo. Con tres identidades y datos sintéticos, los números describen el comportamiento del método, no un sistema listo para producción.

## Estructura del repositorio

```
.
├── informe/              Informe final (.md + .pdf) y sus imágenes; generar_pdf.py arma el PDF
├── src/                  Pipeline: preprocesado, caracterización, K-means, KNN y utilidades
├── interfaz/             Aplicación de escritorio (app.py) y el generador de huellas con Anguli
├── explicativos/         Scripts que generan los gráficos del informe (matrices, umbral, PCA, etc.)
├── flujos/               Diagramas de flujo del sistema (generar_flujos.py)
├── dataset_huellas/
│   ├── base/             Huellas maestras + impresiones (primera, débil, medio) de las 3 identidades
│   ├── intrusas/         50 intrusos (01-25 calibran el umbral, 26-50 evalúan)
│   ├── procesadas/       Huellas ya acondicionadas y binarizadas
│   ├── caracteristicas/  Vectores, dataset.npz y el modelo entrenado (modelo_kmeans.npz)
│   ├── pruebas_manuales/ Huellas subidas a mano y su registro de aciertos
│   └── interfaz/         Huellas que genera la interfaz (se crea sola; no se sube a git)
├── dataset_audios/
│   ├── crudos/           197 audios grabados (16 kHz, mono, 1,5 s)
│   ├── normalizados/     Audios ya acondicionados
│   └── caracteristicas/  MFCC, dataset.npz y el modelo KNN
└── docs/                 Enunciado y notas de teoría
```

## Instalación

Requiere un Python 3 reciente (se probó con 3.14) y Tkinter (viene con Python en Windows).

```bash
pip install -r requirements.txt
```

Para **generar huellas nuevas** hace falta además el programa **Anguli** (`Anguli.exe`), que no está en este repositorio. La ruta se configura en `interfaz/config.py` (o la interfaz te pide elegirlo la primera vez). Sin Anguli se puede usar todo lo demás: la interfaz permite **elegir una huella existente**.

## Uso

### Abrir la interfaz

```bash
python interfaz/app.py
```

### Huellas: de las imágenes al modelo

```bash
python src/preprocesar.py            # dataset_huellas/base      -> dataset_huellas/procesadas
python src/caracterizar_huellas.py   # procesadas -> vectores de 104 números (normalizados) y dataset.npz
python src/kmeans_huellas.py         # entrena K-means + PCA + Mahalanobis, calibra λ y guarda modelo y gráficos
python src/probar_huellas.py         # prueba huellas nuevas (dataset_huellas/pruebas_manuales/huellas_nuevas)
```

Para volver a generar el dataset sintético (requiere Anguli): `python src/generar_dataset_huellas.py`.

### Voz: de la grabación al modelo

```bash
python src/grabar_audios.py          # graba audios nuevos (micrófono)
python src/normalizar_audios.py      # dataset_audios/crudos -> normalizados
python src/caracterizar_audios.py    # MFCC y dataset.npz
python src/knn_audios.py             # elige K con la validación, evalúa en prueba y guarda el modelo
python src/probar_voz.py             # prueba en vivo desde el micrófono
```

### Gráficos del informe y PDF

```bash
python explicativos/intrusos_huellas.py   # matriz de confusión con intrusos
python explicativos/umbral_t2.py          # umbral en distancias T²
python explicativos/pca_correcto.py       # PCA con elipses y gráfico T² contra Q

pip install markdown-it-py mdit-py-plugins
python informe/generar_pdf.py             # convierte el informe a PDF (usa Edge o Chrome instalado)
```

## Cómo decide el sistema (resumen)

1. La huella se acondiciona (normalización, alineación, filtro de Gabor orientado, binarización) y se convierte en **104 números** (72 de orientación de crestas + 32 de energía de Gabor), normalizados con la media y el desvío del entrenamiento.
2. **K-means** (sin etiquetas, desde cero) agrupa las huellas de entrenamiento en 3 clusters y el **algoritmo húngaro** asigna cada cluster a una identidad.
3. Se ajusta un **PCA de 6 componentes** y, por identidad, una covarianza. La huella se compara con cada identidad con la distancia de Mahalanobis (T²) y se queda con la menor.
4. Se **acepta** si T² ≤ λ · límite; si no, `NO AUTORIZADO`. El límite es el percentil 95 de las autorizadas de validación y λ es el mayor valor con el que ningún intruso de calibración queda aceptado.

## Limitaciones

- Datos de huellas sintéticos (Anguli) y solo 3 identidades; las distorsiones son las nativas del generador.
- Con el nivel de distorsión **fuerte** (que no se usó para entrenar) el sistema rechaza a la mayoría de las autorizadas.
- El umbral queda pegado al intruso de calibración más cercano (criterio conservador); con pocos intrusos puede ser optimista.
- El KNN de voz siempre elige uno de los tres comandos: no rechaza palabras desconocidas.
