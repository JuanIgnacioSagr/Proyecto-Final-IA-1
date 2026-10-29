"""Logica de la interfaz SIN ventanas: analizar una huella, escuchar un comando de voz y registrar el intento.

Aca no se reimplementa ningun algoritmo: se llama a los modulos que ya estan en src/
    huella: preprocesar_huella -> rasgo_orientacion (72 numeros) -> K-means + umbral (identificar_huella_detalle)
    voz   : grabar -> normalizar_audio -> MFCC -> KNN con DTW (clasificar_audio_detalle)
Mantenerlo separado de la ventana permite probarlo por consola y reutilizarlo.
"""
import csv
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

import config

sys.path.insert(0, str(config.SRC))            # para poder importar los modulos de src/
import caracterizar_huellas as CH              # noqa: E402
import grabar_audios as G                      # noqa: E402
import kmeans_huellas as KH                    # noqa: E402
import knn_audios as KA                        # noqa: E402
import preprocesar as P                        # noqa: E402
import probar_voz as PV                        # noqa: E402

COMANDOS = ("reconocer", "identificar", "salir")
PREFIJO_ORIGEN = re.compile(r"(?i)^(id([123])|intruso)[_\-. ]")


def origen_desde_nombre(nombre):
    """'id2_2026...jpg' -> 'id2' ; 'intruso_...jpg' -> 'intruso' ; cualquier otro nombre -> None (se desconoce)."""
    m = PREFIJO_ORIGEN.match(nombre)
    if not m:
        return None
    return "intruso" if m.group(1).lower() == "intruso" else f"id{m.group(2)}"


def resolver_microfono():
    """Numero del microfono configurado (por numero o por parte del nombre); None = el de por defecto de Windows."""
    try:
        return PV.resolver_dispositivo(str(config.MICROFONO)) if config.MICROFONO else None
    except SystemExit:                          # resolver_dispositivo termina el programa si no lo encuentra
        return None


class Servicios:
    """Carga los dos modelos UNA vez (K-means de huellas y KNN de voz) y ofrece las operaciones de la interfaz."""

    def __init__(self):
        if not KH.RUTA_MODELO.exists():
            raise FileNotFoundError(f"Falta el modelo de huellas: {KH.RUTA_MODELO}\nGeneralo con: python src/kmeans_huellas.py")
        if not KA.RUTA_MODELO.exists():
            raise FileNotFoundError(f"Falta el modelo de voz: {KA.RUTA_MODELO}\nGeneralo con: python src/knn_audios.py")
        self.modelo_huella = np.load(KH.RUTA_MODELO, allow_pickle=False)
        self.modelo_voz = np.load(KA.RUTA_MODELO, allow_pickle=False)
        self.microfono = resolver_microfono()

    def nombre_microfono(self):
        """Texto con el microfono que se va a usar (ej. '1: Microphone (...)'); si el configurado no esta conectado, el de Windows."""
        return PV.nombre_microfono(self.microfono)

    # ------------------------------------------------------------------ huellas
    @property
    def radio(self):
        return float(self.modelo_huella["radios"][0])

    def analizar_huella(self, ruta):
        """Aplica TODO el pipeline a una imagen y devuelve lo necesario para mostrar y decidir.

        gris      imagen original en gris
        binaria   huella ya procesada (alineada, filtrada con Gabor y binarizada): "la huella con filtros"
        vector    los 72 numeros que la describen
        decision  dict de identificar_huella_detalle (resultado, distancia, radio, distancias por persona, ...)
        """
        gris = P.a_gris(Path(ruta))
        binaria = P.preprocesar_huella(gris)
        vector = CH.rasgo_orientacion(binaria)
        decision = KH.identificar_huella_detalle(vector, self.modelo_huella)
        return {"gris": gris, "binaria": binaria, "vector": vector, "decision": decision}

    # ------------------------------------------------------------------ voz
    def _clasificar(self, x, fs):
        """Clasifica una grabacion. estado: 'ok' | 'silencio' (volumen muy bajo) | 'sin_voz' (solo ruido)."""
        pico = float(np.max(np.abs(x))) if len(x) else 0.0
        if pico < G.PICO_MIN:
            return {"estado": "silencio", "pico": pico}
        r = KA.clasificar_audio_detalle(x, fs, self.modelo_voz)
        if r["info"].get("sin_voz"):
            return {"estado": "sin_voz", "pico": pico}
        return {"estado": "ok", "pico": pico, "r": r}

    def escuchar(self):
        """Graba con el microfono (igual que el dataset: 16 kHz, mono) y clasifica el comando."""
        x = G.grabar(config.DURACION_S, self.microfono)
        return self._clasificar(x, G.FS)

    def clasificar_wav(self, ruta):
        """Igual que escuchar() pero con un WAV existente (para pruebas sin microfono)."""
        x, fs = sf.read(str(ruta), dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        return self._clasificar(x, fs)

    @staticmethod
    def comando_ejecutable(resultado_voz):
        """Decide si un comando entendido se EJECUTA. Devuelve (comando | None, mensaje).

        Con EXIGIR_CONFIANZA_ALTA solo se ejecuta si los 3 vecinos del KNN coinciden; asi se evitan
        ordenes ejecutadas por error (el KNN siempre elige alguno de los 3 comandos, diga lo que diga la persona).
        """
        est = resultado_voz["estado"]
        if est == "silencio":
            return None, f"No escuché nada (volumen muy bajo, pico {resultado_voz['pico']:.3f}). Acercate al micrófono."
        if est == "sin_voz":
            return None, "No detecté voz clara (solo ruido o silencio). Repetí el comando."
        r = resultado_voz["r"]
        votos = ", ".join(f"{c} {v}" for c, v in r["votos"].items())
        if config.EXIGIR_CONFIANZA_ALTA and r["confianza"] != "alta":
            return None, f"Entendí '{r['comando']}' pero no estoy seguro (votos: {votos}). Repetí el comando."
        return r["comando"], f"Entendí: {r['comando'].upper()}  (confianza {r['confianza']}; votos: {votos})"

    # ------------------------------------------------------------------ registro
    @staticmethod
    def registrar(ruta, origen_real, filtro, via, decision):
        """Guarda una fila por huella analizada, con la verdad conocida (de donde salio la huella) y lo que decidio el sistema.

        'correcto' compara el origen real con la decision: intruso <-> NO AUTORIZADO, idN <-> identidad_N.
        Con este registro se puede calibrar el radio con huellas reales de intrusos (ver GRAFICOS/teoria).
        """
        decidido = "intruso" if decision["resultado"] == "NO AUTORIZADO" else "id" + decision["resultado"][-1]
        correcto = "" if origen_real is None else str(decidido == origen_real)
        nuevo = not config.REGISTRO.exists()
        with open(config.REGISTRO, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if nuevo:
                w.writerow(["fecha", "archivo", "origen_real", "filtro", "via", "resultado", "identidad_cercana",
                            "distancia", "radio", "correcto"])
            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), Path(ruta).name, origen_real or "", filtro, via,
                        decision["resultado"], decision["identidad_cercana"], round(decision["distancia"], 4),
                        round(decision["radio"], 4), correcto])
        return correcto
