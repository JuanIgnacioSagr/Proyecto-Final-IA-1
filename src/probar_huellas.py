"""Prueba de huellas nuevas (autorizadas o intrusas) con el K-means + umbral, con feedback para calibrar.

Que hace
--------
1. Toma las imagenes de la carpeta  huellas_nuevas/  (la llenas vos, se puede ir agregando de a poco).
2. A cada imagen NUEVA le aplica todo el pipeline:
       preprocesar_huella  ->  rasgo_orientacion (72 numeros)  ->  centroide mas cercano + umbral
   y dice si la huella fue ACEPTADA (con que identidad) o RECHAZADA ("NO AUTORIZADO").
3. Te pregunta si el resultado es correcto (tu feedback) y lo guarda en  pruebas_huellas/registro.csv.
4. Con --calibrar usa ese feedback (huellas reales de intrusos y de autorizadas) para calcular
   que radio de aceptacion equilibra los errores y sugiere como aplicarlo.

Uso
---
    python src/probar_huellas.py                 # procesa las imagenes nuevas y pide feedback
    python src/probar_huellas.py --sin-preguntar # solo informa los resultados
    python src/probar_huellas.py --revisar       # vuelve a pedir feedback de las que quedaron sin responder
    python src/probar_huellas.py --calibrar      # calcula el radio con tus feedbacks (SOLO lo sugiere, no cambia el modelo)
    python src/probar_huellas.py --aplicar 0.42  # guarda ese radio en el modelo (lo usa tambien la interfaz)
    python src/probar_huellas.py --rehacer       # vuelve a procesar TODAS (p. ej. si cambio el modelo)

Como indicar la verdad sin que pregunte: poner el nombre del archivo que EMPIECE con
    id1_  id2_  id3_   (huella de esa persona autorizada)   o   intruso_   (huella de una persona NO autorizada)
por ejemplo:  id2_rotada_01.jpg   intruso_03.png

Respuestas al feedback:  s = correcto | n = incorrecto | 1, 2, 3 = era esa identidad | i = era un intruso |
                         ENTER = saltar | q = terminar

Nunca se modifica ni se borra nada de huellas_nuevas/ ni del dataset original.
"""
import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import caracterizar_huellas as CH
import kmeans_huellas as KH
import preprocesar as P

RAIZ = Path(__file__).resolve().parent.parent
ENTRADA = RAIZ / "huellas_nuevas"
SALIDA = RAIZ / "pruebas_huellas"
EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
TAMANO_ESPERADO = (275, 400)                      # ancho x alto de las huellas del dataset
IDENTIDADES = ["identidad_1", "identidad_2", "identidad_3"]
INTRUSO = "intruso"
CAMPOS = ["archivo", "bytes", "mtime_ns", "fecha", "resultado", "identidad_cercana", "distancia", "radio",
          "d_identidad_1", "d_identidad_2", "d_identidad_3", "verdad", "correcto"]
GRILLA_RADIO = np.round(np.arange(0.10, 0.601, 0.01), 2)
MINIMO_PARA_CALIBRAR = 5                          # muestras de autorizadas y de intrusos para fiarse del resultado


# =====================================================================================
#  Registro (csv) de resultados y feedback
# =====================================================================================
def leer_registro(ruta):
    if not ruta.exists():
        return {}
    with open(ruta, newline="", encoding="utf-8") as f:
        return {r["archivo"]: r for r in csv.DictReader(f)}


def escribir_registro(ruta, filas):
    """Escribe todo el csv de una vez (en un archivo temporal que luego reemplaza al anterior)."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tmp = ruta.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS)
        w.writeheader()
        for k in sorted(filas):
            w.writerow({c: filas[k].get(c, "") for c in CAMPOS})
    tmp.replace(ruta)


def prediccion_normalizada(fila):
    """'identidad_N' si fue aceptada, 'intruso' si fue rechazada (para compararla con la verdad)."""
    return INTRUSO if fila["resultado"] == "NO AUTORIZADO" else fila["resultado"]


def fijar_verdad(fila, verdad):
    fila["verdad"] = verdad
    fila["correcto"] = "" if not verdad else str(prediccion_normalizada(fila) == verdad)


def verdad_desde_nombre(nombre):
    """id1_xxx.jpg -> identidad_1 ; intruso_xxx.png -> intruso ; otro nombre -> None."""
    m = re.match(r"(?i)(id([123])|intruso)[_\-. ]", nombre)
    if not m:
        return None
    return INTRUSO if m.group(1).lower() == "intruso" else f"identidad_{m.group(2)}"


# =====================================================================================
#  Procesamiento de una imagen
# =====================================================================================
def evaluar_imagen(ruta, modelo, carpeta_previas):
    """Aplica el pipeline a una imagen. Devuelve (fila_para_el_registro, detalle). Guarda una imagen de revision."""
    gris = P.a_gris(ruta)
    binaria = P.preprocesar_huella(gris)                 # gris -> alinear -> filtrar -> binarizar
    vector = CH.rasgo_orientacion(binaria)               # 72 numeros
    r = KH.identificar_huella_detalle(vector, modelo)    # centroide mas cercano + umbral
    st = ruta.stat()
    fila = {"archivo": ruta.name, "bytes": st.st_size, "mtime_ns": st.st_mtime_ns,
            "fecha": time.strftime("%Y-%m-%d %H:%M:%S"), "resultado": r["resultado"],
            "identidad_cercana": r["identidad_cercana"], "distancia": round(r["distancia"], 4),
            "radio": round(r["radio"], 4), "verdad": "", "correcto": ""}
    for nombre in IDENTIDADES:
        fila[f"d_{nombre}"] = round(r["distancias"].get(nombre, float("nan")), 4)
    # imagen de revision: original (izquierda) y huella ya preprocesada (derecha)
    alto = binaria.shape[0]
    orig = cv2.resize(gris, (int(gris.shape[1] * alto / gris.shape[0]), alto), interpolation=cv2.INTER_AREA)
    previa = np.hstack([orig, np.full((alto, 6), 128, np.uint8), binaria])
    previa = cv2.cvtColor(previa, cv2.COLOR_GRAY2BGR)
    color = (0, 140, 0) if r["aceptada"] else (0, 0, 200)
    cv2.putText(previa, f"{r['resultado']}  d={r['distancia']:.2f} (radio {r['radio']:.2f})", (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    carpeta_previas.mkdir(parents=True, exist_ok=True)
    P.guardar_png(carpeta_previas / (ruta.stem + ".png"), previa)
    return fila, {"detalle": r, "tamano": (gris.shape[1], gris.shape[0])}


def mostrar(fila, info):
    r, (ancho, alto) = info["detalle"], info["tamano"]
    print(f"\n[{fila['archivo']}]")
    if (ancho, alto) != TAMANO_ESPERADO:
        print(f"  [!] tamano {ancho}x{alto}; las huellas del dataset son {TAMANO_ESPERADO[0]}x{TAMANO_ESPERADO[1]} (se procesa igual).")
    if r["aceptada"]:
        print(f"  >>> ACEPTADA como {r['resultado'].upper()}")
        print(f"      distancia al centroide {r['distancia']:.3f}  <=  radio {r['radio']:.3f}")
    else:
        print(f"  >>> RECHAZADA: NO AUTORIZADA   (la mas parecida seria {r['identidad_cercana']})")
        print(f"      distancia al centroide {r['distancia']:.3f}  >  radio {r['radio']:.3f}")
    ds = ", ".join(f"{n.replace('identidad_', 'id')} {d:.3f}" for n, d in r["distancias"].items())
    aviso = "  (dudosa: queda casi a la misma distancia de dos personas)" if r["margen"] < 0.05 and r["aceptada"] else ""
    print(f"      distancia a cada persona: {ds} | margen {r['margen']:.3f}{aviso}")


# =====================================================================================
#  Feedback
# =====================================================================================
def pedir_feedback(fila):
    """Pregunta si el resultado es correcto. Devuelve la verdad ('identidad_N'/'intruso'), '' (saltar) o None (terminar)."""
    prediccion = prediccion_normalizada(fila)
    while True:
        try:
            r = input("  Es correcto? [s]i / [n]o / 1,2,3 = era esa persona / i = era intruso / ENTER = saltar / q = terminar > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None
        if r == "":
            return ""
        if r == "q":
            return None
        if r == "s":
            return prediccion
        if r in ("1", "2", "3"):
            return f"identidad_{r}"
        if r == "i":
            return INTRUSO
        if r == "n":
            # estaba mal: se pregunta que era en realidad
            opciones = "i = era intruso, o 1/2/3 = era esa persona" if prediccion != INTRUSO else "1/2/3 = era esa persona"
            try:
                r2 = input(f"    Que era en realidad? ({opciones}) > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return None
            if r2 in ("1", "2", "3"):
                return f"identidad_{r2}"
            if r2 == "i" and prediccion != INTRUSO:
                return INTRUSO
            print("    Respuesta no valida.")
            continue
        print("    Respuesta no valida.")


# =====================================================================================
#  Resumen y calibracion con los feedbacks
# =====================================================================================
def con_verdad(registro):
    return [f for f in registro.values() if f.get("verdad")]


def resumen(registro):
    """Cuantas acerto el sistema segun el feedback recibido, con los dos errores del umbral por separado."""
    filas = con_verdad(registro)
    sin = len(registro) - len(filas)
    print(f"\nResumen del feedback: {len(filas)} huellas con respuesta" + (f" ({sin} sin responder)" if sin else "") + ".")
    if not filas:
        return
    aut = [f for f in filas if f["verdad"] != INTRUSO]
    intr = [f for f in filas if f["verdad"] == INTRUSO]
    ok = sum(f["correcto"] == "True" for f in filas)
    print(f"  Aciertos: {ok} de {len(filas)} ({ok / len(filas) * 100:.0f}%)")
    if aut:
        rech = sum(prediccion_normalizada(f) == INTRUSO for f in aut)
        mal = sum(prediccion_normalizada(f) not in (INTRUSO, f["verdad"]) for f in aut)
        print(f"  Autorizadas ({len(aut)}): bien identificadas {len(aut) - rech - mal}, "
              f"rechazadas por error (falso rechazo) {rech}, aceptadas con OTRA identidad {mal}")
    if intr:
        entran = sum(prediccion_normalizada(f) != INTRUSO for f in intr)
        print(f"  Intrusas ({len(intr)}): rechazadas bien {len(intr) - entran}, "
              f"ACEPTADAS por error (falsa aceptacion) {entran}")


def calibrar(registro, radio_actual):
    """Calcula, con las huellas reales con feedback, que radio equilibra falso rechazo y falsa aceptacion."""
    filas = con_verdad(registro)
    aut = [f for f in filas if f["verdad"] != INTRUSO]
    intr = [f for f in filas if f["verdad"] == INTRUSO]
    print(f"Huellas con feedback: {len(filas)}  (autorizadas: {len(aut)}, intrusas: {len(intr)})")
    if not aut or not intr:
        print("Para calibrar hacen falta AMBOS tipos: huellas de personas autorizadas e intrusas con su respuesta.")
        print("Subi mas a huellas_nuevas/ (por ejemplo id2_01.jpg, intruso_01.jpg) y volve a correr.")
        return
    d_aut = np.array([float(f["distancia"]) for f in aut])
    cerca_aut = np.array([f["identidad_cercana"] for f in aut])
    verdad_aut = np.array([f["verdad"] for f in aut])
    d_int = np.array([float(f["distancia"]) for f in intr])
    frr = np.array([(d_aut > R).mean() for R in GRILLA_RADIO])                          # autorizadas rechazadas
    otra = np.array([((d_aut <= R) & (cerca_aut != verdad_aut)).mean() for R in GRILLA_RADIO])   # aceptadas con otra identidad
    far = np.array([(d_int <= R).mean() for R in GRILLA_RADIO])                         # intrusos aceptados
    lo, hi = min(d_aut.min(), d_int.min()) - 0.03, max(d_aut.max(), d_int.max()) + 0.03
    print("\n   radio R | autorizadas rechazadas | aceptadas con otra identidad | intrusos aceptados")
    for i, R in enumerate(GRILLA_RADIO):
        if lo <= R <= hi and int(round(R * 100)) % 2 == 0:
            marca = "  <- radio actual" if abs(R - radio_actual) < 0.005 else ""
            print(f"    {R:.2f}   |        {frr[i] * 100:5.1f}%         |           {otra[i] * 100:5.1f}%           |      {far[i] * 100:5.1f}%{marca}")
    i_act = int(np.argmin(np.abs(GRILLA_RADIO - radio_actual)))
    print(f"\n  Con el radio actual ({radio_actual:.2f}): {frr[i_act] * 100:.0f}% de autorizadas rechazadas y {far[i_act] * 100:.0f}% de intrusos aceptados.")
    if len(aut) < MINIMO_PARA_CALIBRAR or len(intr) < MINIMO_PARA_CALIBRAR:
        print(f"  [!] Hay pocas muestras (conviene al menos {MINIMO_PARA_CALIBRAR} autorizadas y {MINIMO_PARA_CALIBRAR} intrusas): "
              "tomalo como orientativo, no cambies el radio todavia.")
    i_eer = int(np.argmin(np.abs(frr + otra - far)))                                    # cruce de los dos errores
    print(f"  Radio que equilibra los dos errores: R = {GRILLA_RADIO[i_eer]:.2f} "
          f"(autorizadas rechazadas/mal identificadas {(frr + otra)[i_eer] * 100:.0f}%, intrusos aceptados {far[i_eer] * 100:.0f}%)")
    seguro = np.flatnonzero(far == 0)
    if len(seguro):
        i_s = seguro[-1]
        print(f"  Radio MAS GRANDE sin dejar entrar a ningun intruso de la muestra: R = {GRILLA_RADIO[i_s]:.2f} "
              f"(autorizadas rechazadas {frr[i_s] * 100:.0f}%)")
    print("\n  OJO: --calibrar solo SUGIERE; el modelo sigue con el radio anterior hasta que lo apliques:")
    print(f"       python src/probar_huellas.py --aplicar {GRILLA_RADIO[i_eer]:.2f}")
    print("  (despues podes ver como queda con:  python src/probar_huellas.py --rehacer --sin-preguntar)")


def aplicar_radio(radio):
    """Guarda `radio` como umbral de aceptacion en el modelo K-means (sin reentrenar): centroides y mapeo no cambian.

    --calibrar solo CALCULA y sugiere un radio; esta funcion es la que realmente lo cambia. La interfaz y
    probar_huellas usan el radio guardado en modelo_kmeans.npz, asi que el cambio rige desde la proxima ejecucion.
    """
    if not 0.05 <= radio <= 1.5:
        sys.exit(f"Radio fuera de rango razonable (0,05 a 1,5): {radio}")
    modelo = {k: v for k, v in np.load(KH.RUTA_MODELO, allow_pickle=False).items()}
    anterior = float(modelo["radios"][0])
    modelo["radios"] = np.full(len(modelo["radios"]), radio)
    modelo["radio"] = radio
    np.savez(KH.RUTA_MODELO, **modelo)
    ruta_json = KH.CARPETA / "metricas_kmeans.json"
    if ruta_json.exists():                                   # se deja anotado en las metricas
        datos = json.loads(ruta_json.read_text(encoding="utf-8"))
        datos["radio_usado"] = radio
        ruta_json.write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Radio del modelo: {anterior:.2f} -> {radio:.2f}  (guardado en {KH.RUTA_MODELO.name}).")
    print("Para ver como queda con las huellas ya probadas:  python src/probar_huellas.py --rehacer --sin-preguntar")


# =====================================================================================
#  Programa principal
# =====================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--entrada", type=Path, default=ENTRADA, help="carpeta con las huellas nuevas")
    ap.add_argument("--salida", type=Path, default=SALIDA, help="carpeta donde se guardan registro e imagenes de revision")
    ap.add_argument("--sin-preguntar", action="store_true", help="no pedir feedback, solo informar")
    ap.add_argument("--revisar", action="store_true", help="pedir feedback de las que quedaron sin responder")
    ap.add_argument("--rehacer", action="store_true", help="volver a procesar todas las imagenes (conserva el feedback)")
    ap.add_argument("--calibrar", action="store_true", help="calcular el radio con el feedback acumulado (solo lo sugiere)")
    ap.add_argument("--aplicar", type=float, metavar="R", default=None, help="guardar el radio R en el modelo (rige para la interfaz)")
    a = ap.parse_args()

    if not KH.RUTA_MODELO.exists():
        sys.exit(f"No existe el modelo {KH.RUTA_MODELO}. Generalo con: python src/kmeans_huellas.py")
    modelo = np.load(KH.RUTA_MODELO, allow_pickle=False)
    radio_actual = float(modelo["radio"]) if "radio" in modelo.files else float(modelo["radios"][0])
    ruta_registro = a.salida / "registro.csv"
    registro = leer_registro(ruta_registro)

    if a.aplicar is not None:
        aplicar_radio(a.aplicar)
        return
    if a.calibrar:
        calibrar(registro, radio_actual)
        return

    a.entrada.mkdir(parents=True, exist_ok=True)
    imagenes = sorted(p for p in a.entrada.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONES)
    if not imagenes:
        print(f"La carpeta {a.entrada} no tiene imagenes. Subi las huellas ahi (jpg, png, bmp o tif) y volve a correr.")
        print("Tip: si el nombre empieza con id1_, id2_, id3_ o intruso_ el programa ya sabe la respuesta correcta.")
        if registro:
            resumen(registro)
        return

    pendientes, saltadas = [], 0
    for p in imagenes:
        previa = registro.get(p.name)
        st = p.stat()
        cambio = previa is None or str(st.st_size) != previa["bytes"] or str(st.st_mtime_ns) != previa["mtime_ns"]
        if a.rehacer or cambio:
            pendientes.append(p)
        else:
            saltadas += 1
    print(f"Modelo K-means: radio de aceptacion {radio_actual:.2f} | imagenes en {a.entrada}: {len(imagenes)} "
          f"| nuevas: {len(pendientes)} | ya vistas: {saltadas}")

    terminar = False
    for p in pendientes:
        try:
            fila, info = evaluar_imagen(p, modelo, a.salida / "previas")
        except Exception as e:                       # imagen ilegible o dañada: se avisa y se sigue
            print(f"\n[{p.name}] no se pudo procesar ({type(e).__name__}: {e})")
            continue
        if p.name in registro and registro[p.name].get("verdad"):
            fijar_verdad(fila, registro[p.name]["verdad"])         # --rehacer conserva el feedback anterior
        mostrar(fila, info)
        verdad = verdad_desde_nombre(p.name)
        if verdad:
            fijar_verdad(fila, verdad)
            print(f"  Verdad tomada del nombre del archivo: {verdad} -> " + ("CORRECTO" if fila["correcto"] == "True" else "INCORRECTO"))
        elif not fila["verdad"] and not a.sin_preguntar:
            v = pedir_feedback(fila)
            if v is None:
                terminar = True
            elif v:
                fijar_verdad(fila, v)
                print("  Anotado: " + ("correcto." if fila["correcto"] == "True" else f"incorrecto (era {v})."))
        registro[p.name] = fila
        escribir_registro(ruta_registro, registro)                 # se guarda en cada paso por si se corta
        if terminar:
            break

    if a.revisar and not terminar and not a.sin_preguntar:
        for nombre, fila in sorted(registro.items()):
            if fila.get("verdad"):
                continue
            print(f"\n[{nombre}]  resultado guardado: {fila['resultado']} (distancia {fila['distancia']}, radio {fila['radio']})")
            v = pedir_feedback(fila)
            if v is None:
                break
            if v:
                fijar_verdad(fila, v)
        escribir_registro(ruta_registro, registro)

    resumen(registro)
    print(f"\nRegistro: {ruta_registro}\nImagenes de revision (original | preprocesada): {a.salida / 'previas'}")
    print("Cuando tengas varias huellas con feedback (autorizadas e intrusas): python src/probar_huellas.py --calibrar")


if __name__ == "__main__":
    main()
