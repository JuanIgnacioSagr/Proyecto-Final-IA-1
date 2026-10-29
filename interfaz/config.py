"""Configuracion de la interfaz: rutas y opciones. Es el unico archivo que hay que tocar al cambiar de computadora.

- ANGULI_EXE_DEFECTO: donde esta Anguli.exe (el generador de huellas sinteticas). Vive FUERA del repositorio.
  Si la ruta no existe, la interfaz te deja elegir el .exe y guarda la eleccion en interfaz/config.json.
- CARPETA_HUELLAS: la UNICA carpeta donde quedan las huellas que genera la interfaz, sin subcarpetas y
  con el nombre de la identidad o "intruso" (id2_20261027_153012.jpg, intruso_20261027_153100.jpg).
"""
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent          # carpeta del proyecto
CARPETA_INTERFAZ = Path(__file__).resolve().parent
SRC = RAIZ / "src"                                     # modulos del proyecto (preprocesar, kmeans, knn, ...)

# --- Anguli (generador de huellas) ---
ANGULI_EXE_DEFECTO = Path(r"C:\Users\juani\Desktop\4TO\4.2\IA\Anguli-MinGw\Anguli.exe")
TIEMPO_MAX_ANGULI_S = 90          # una huella maestra nueva (intruso) tarda ~20 s; una impresion ~3 s

# --- carpetas de huellas ---
CARPETA_HUELLAS = RAIZ / "huellasConID"                # donde se guardan las huellas generadas
CARPETA_BASE = RAIZ / "huellasDactilares" / "Fingerprints" / "fp_1"   # 1.jpg, 2.jpg, 3.jpg = las 3 identidades

# --- voz ---
MICROFONO = "usb"                 # numero de dispositivo o parte del nombre; si no se encuentra se usa el de Windows
DURACION_S = 1.5                  # igual que cuando se grabo el dataset
EXIGIR_CONFIANZA_ALTA = True      # True: un comando solo se ejecuta si los 3 vecinos del KNN coinciden

# --- registro de cada intento (verdad conocida + decision del sistema) ---
REGISTRO = CARPETA_INTERFAZ / "registro_interfaz.csv"
ARCHIVO_CONFIG = CARPETA_INTERFAZ / "config.json"


def ruta_anguli():
    """Ruta de Anguli.exe: la guardada en config.json si existe, o la de por defecto."""
    try:
        datos = json.loads(ARCHIVO_CONFIG.read_text(encoding="utf-8"))
        if "anguli_exe" in datos:
            return Path(datos["anguli_exe"])
    except (OSError, ValueError):
        pass
    return ANGULI_EXE_DEFECTO


def guardar_ruta_anguli(ruta):
    """Recuerda la ruta elegida por el usuario para las proximas veces."""
    ARCHIVO_CONFIG.write_text(json.dumps({"anguli_exe": str(ruta)}, indent=2), encoding="utf-8")
