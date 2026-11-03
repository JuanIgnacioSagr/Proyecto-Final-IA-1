"""Aplicacion final: reconocimiento de huella + comandos de voz (interfaz Tkinter).

Flujo:
  1. Elegir una persona (Identidad 1, 2, 3 o Intruso) y un filtro, y apretar "Generar huella": Anguli
     (ejecutado con subprocess) crea la huella y se muestran la imagen generada y la huella con filtros.
  2. Decir (boton "Hablar") o apretar:
        RECONOCER   -> el K-means dice si la huella es RECONOCIDA o NO (no autorizada)
        IDENTIFICAR -> (solo si fue reconocida) dice a que identidad pertenece
        SALIR       -> cierra la aplicacion
  3. Opcional: ver el grafico de clusters (donde cayo la huella, el cluster mas cercano y la distancia).

Uso:
    python interfaz/app.py
    python interfaz/app.py --prueba     # recorrido automatico de verificacion (genera capturas)

La ventana nunca se congela: Anguli y el microfono se ejecutan en hilos (threading) y avisan por una cola.
"""
import argparse
import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

import anguli
import config
import estilo
import grafico
from estilo import FUENTE, P as PAL, Boton, Insignia, Pasos, Tarjeta, etiqueta, opciones
from servicios import Servicios, origen_desde_nombre

try:                                                      # nitidez en pantallas con escala de Windows
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

ALTO_IMAGEN = 270
# tonos de la insignia de resultado (ver estilo.py)
VERDE, ROJO, AZUL, GRIS = "exito", "peligro", "info", "neutro"


def a_tk(imagen_gris_o_bgr, alto=ALTO_IMAGEN):
    """Convierte una imagen de OpenCV (numpy) en una imagen que Tkinter puede mostrar, escalada a `alto` pixeles."""
    img = imagen_gris_o_bgr
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    img = cv2.resize(img, (int(w * alto / h), alto), interpolation=cv2.INTER_AREA)
    return ImageTk.PhotoImage(Image.fromarray(img))


class App(tk.Tk):
    def __init__(self, prueba=False):
        super().__init__()
        self.title("Reconocimiento biométrico: huella y voz")
        self.geometry("1200x700")
        self.minsize(1100, 640)
        estilo.aplicar_tema(self)                         # colores oscuros, barra de titulo oscura, barra de progreso
        self.prueba = prueba
        self.cola = queue.Queue()                         # los hilos dejan aca sus resultados; la ventana los lee
        self.ocupado = False                              # True mientras corre Anguli o el microfono
        self._progreso = None                             # (hora de inicio, segundos estimados, tarea) mientras hay una barra en marcha
        # duracion tipica de cada tarea (s); se va corrigiendo con lo que tardo de verdad (promedio de las ultimas 5)
        self.duraciones = {"identidad": [3.0], "intruso": [10.0], "voz": [4.0]}

        # ---- estado del flujo ----
        self.ruta = None            # imagen de la huella actual
        self.origen_real = None     # 'id1'|'id2'|'id3'|'intruso'|None: de donde salio la huella (la verdad)
        self.filtro_txt = ""        # parametros del filtro, solo para el registro
        self.analisis = None        # resultado de analizar_huella
        self.reconocida = None      # None = todavia no se dijo RECONOCER; True/False = resultado
        self.identificada = False   # True cuando se dijo IDENTIFICAR con una huella reconocida
        self.wav_prueba = None      # solo en --prueba: WAV que reemplaza al microfono
        self._fotos = {}            # referencias a las imagenes (si no, Tkinter las "olvida")
        self.ventana_grafico = None

        try:
            self.srv = Servicios()
        except FileNotFoundError as e:
            messagebox.showerror("Faltan modelos", str(e))
            self.destroy()
            raise SystemExit(1)

        self._armar_ventana()
        self._actualizar_pasos()
        self.log(f"Listo. Radio de aceptación del K-means: R = {self.srv.radio:.2f}. "
                 f"Micrófono: {self.srv.nombre_microfono()}"
                 + (" (el configurado '" + str(config.MICROFONO) + "' no está conectado)" if self.srv.microfono is None and config.MICROFONO else "") + ".")
        self.after(100, self._leer_cola)
        self.protocol("WM_DELETE_WINDOW", self._cerrar)

    # =============================================================== construccion de la ventana
    def _armar_ventana(self):
        # ---------- encabezado: titulo e indicador de pasos ----------
        cab = tk.Frame(self, bg=PAL["fondo"])
        cab.pack(fill="x", padx=24, pady=(14, 6))
        titulo = tk.Frame(cab, bg=PAL["fondo"])
        titulo.pack(side="left")
        tk.Label(titulo, text="Reconocimiento biométrico", bg=PAL["fondo"], fg=PAL["texto"], font=(FUENTE, 20, "bold")).pack(anchor="w")
        tk.Label(titulo, text="Huella dactilar (K-means)  ·  Comandos de voz (KNN + DTW)", bg=PAL["fondo"], fg=PAL["suave"],
                 font=(FUENTE, 10)).pack(anchor="w")
        self.pasos = Pasos(cab, ["Huella", "Reconocer", "Identificar"])
        self.pasos.pack(side="right")

        # ---------- abajo: registro de eventos ----------
        pie = Tarjeta(self, "Registro de eventos")
        pie.pack(side="bottom", fill="x", padx=24, pady=(10, 14))
        self.txt_log = tk.Text(pie.cuerpo, height=4, bg=PAL["fondo"], fg=PAL["suave"], insertbackground=PAL["texto"], relief="flat",
                               bd=0, font=("Consolas", 9), padx=10, pady=8, wrap="word", state="disabled")
        self.txt_log.pack(fill="x")
        for tag, color in (("hora", "#5b6b88"), ("ok", PAL["exito"]), ("err", PAL["peligro"]), ("voz", PAL["info"]), ("res", PAL["texto"])):
            self.txt_log.tag_configure(tag, foreground=color)

        cuerpo = tk.Frame(self, bg=PAL["fondo"])
        cuerpo.pack(fill="both", expand=True, padx=24)
        cuerpo.columnconfigure(1, weight=1)
        cuerpo.rowconfigure(0, weight=1)

        # ---------- columna izquierda: 1 · Huella ----------
        izq = Tarjeta(cuerpo, "1 · Huella")
        izq.grid(row=0, column=0, sticky="ns", padx=(0, 14))
        c = izq.cuerpo
        etiqueta(c, "Persona", suave=True).pack(anchor="w")
        self.var_tipo = tk.StringVar(value="id1")
        opciones(c, self.var_tipo, [(texto, clave) for clave, texto in anguli.TIPOS.items()], columnas=2).pack(fill="x", pady=(4, 12))
        etiqueta(c, "Filtro (distorsión)", suave=True).pack(anchor="w")
        self.var_filtro = tk.StringVar(value=anguli.FILTRO_DEFECTO)
        opciones(c, self.var_filtro, [(n, n) for n in anguli.FILTROS], columnas=3).pack(fill="x", pady=(4, 14))
        self.btn_generar = Boton(c, "Generar huella (Anguli)", command=self.generar, tipo="primario", grande=True)
        self.btn_generar.pack(fill="x")
        self.btn_existente = Boton(c, "Elegir existente…", command=self.elegir_existente)
        self.btn_existente.pack(fill="x", pady=(8, 0))
        self.barra = ttk.Progressbar(c, mode="determinate", value=0, style="Moderno.Horizontal.TProgressbar")
        self.barra.pack(fill="x", pady=(16, 0))
        self.lbl_tiempo = etiqueta(c, "", suave=True, tam=9)         # "Quedan ~7 s" mientras corre la barra
        self.lbl_tiempo.pack(anchor="w", pady=(4, 0))
        self.lbl_estado_huella = etiqueta(c, "Sin huella todavía.", suave=True, wraplength=210, justify="left")
        self.lbl_estado_huella.pack(anchor="w", pady=(8, 0))
        etiqueta(c, "Un intruso es una persona nueva:\ntarda unos segundos más.", suave=True, tam=8, justify="left").pack(anchor="w", pady=(10, 0))

        # ---------- centro: las dos imagenes ----------
        centro = tk.Frame(cuerpo, bg=PAL["fondo"])
        centro.grid(row=0, column=1, sticky="nsew")
        centro.columnconfigure((0, 1), weight=1)
        centro.rowconfigure(0, weight=1)
        t1, t2 = Tarjeta(centro, "Huella generada"), Tarjeta(centro, "Huella con filtros")
        t1.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        t2.grid(row=0, column=1, sticky="nsew", padx=(7, 0))
        self.lbl_img1 = tk.Label(t1.cuerpo, bg=PAL["tarjeta"])
        self.lbl_img2 = tk.Label(t2.cuerpo, bg=PAL["tarjeta"])
        self.lbl_img1.pack(expand=True)
        self.lbl_img2.pack(expand=True)
        self.lbl_origen = etiqueta(centro, "", suave=True, bg=PAL["fondo"], wraplength=560)
        self.lbl_origen.grid(row=1, column=0, columnspan=2, pady=(8, 0))
        self._poner_imagenes(None, None)

        # ---------- columna derecha: 2 · Voz y resultado ----------
        der = Tarjeta(cuerpo, "2 · Voz y resultado")
        der.grid(row=0, column=2, sticky="ns", padx=(14, 0))
        c = der.cuerpo
        self.btn_hablar = Boton(c, "Hablar (micrófono)", command=self.hablar, tipo="primario", grande=True)
        self.btn_hablar.pack(fill="x")
        etiqueta(c, "Decí: reconocer · identificar · salir", suave=True, tam=8).pack(pady=(4, 6))
        self.lbl_voz = etiqueta(c, "—", wraplength=270, justify="left")
        self.lbl_voz.pack(anchor="w", pady=(0, 10))

        etiqueta(c, "Modo manual (respaldo)", suave=True).pack(anchor="w")
        fm = tk.Frame(c, bg=PAL["tarjeta"])
        fm.pack(fill="x", pady=(4, 0))
        self.botones_cmd = {}
        for cmd in ("reconocer", "identificar", "salir"):
            b = Boton(fm, cmd.capitalize(), command=lambda cm=cmd: self.ejecutar(cm, via="manual"),
                      tipo="peligro" if cmd == "salir" else "secundario")
            b.pack(side="left", expand=True, fill="x", padx=2)
            self.botones_cmd[cmd] = b

        tk.Frame(c, bg=PAL["borde"], height=1).pack(fill="x", pady=14)
        etiqueta(c, "Resultado", suave=True).pack(anchor="w")
        self.insignia = Insignia(c)
        self.insignia.pack(fill="x", pady=(4, 6))
        self.lbl_detalle = etiqueta(c, "", suave=True, wraplength=270, justify="left")
        self.lbl_detalle.pack(anchor="w")

        tk.Frame(c, bg=PAL["borde"], height=1).pack(fill="x", pady=14)
        self.var_grafico = tk.BooleanVar(value=True)
        tk.Checkbutton(c, text="Mostrar gráfico de clusters al identificar", variable=self.var_grafico, bg=PAL["tarjeta"],
                       fg=PAL["texto"], selectcolor=PAL["tarjeta2"], activebackground=PAL["tarjeta"], activeforeground=PAL["texto"],
                       font=(FUENTE, 9), anchor="w", justify="left", wraplength=250, bd=0, highlightthickness=0,
                       cursor="hand2").pack(anchor="w")
        self.btn_grafico = Boton(c, "Ver gráfico de clusters", command=self.ver_grafico)
        self.btn_grafico.pack(fill="x", pady=(8, 0))
        self.btn_grafico.configure(state="disabled")

    # =============================================================== utilidades de la ventana
    def log(self, texto):
        """Agrega una linea al registro de eventos, con color segun el tipo de mensaje."""
        if texto.startswith("ERROR"):
            tag = "err"
        elif texto.startswith("Voz"):
            tag = "voz"
        elif "→" in texto:
            tag = "ok" if "NO " not in texto and "SE EQUIVOCÓ" not in texto else "err"
        else:
            tag = "res"
        self.txt_log.configure(state="normal")
        self.txt_log.insert("end", time.strftime("%H:%M:%S  "), "hora")
        self.txt_log.insert("end", texto + "\n", tag)
        self.txt_log.see("end")
        self.txt_log.configure(state="disabled")

    def _poner_imagenes(self, img1, img2):
        for lbl, img, clave in ((self.lbl_img1, img1, "i1"), (self.lbl_img2, img2, "i2")):
            if img is None:                                   # recuadro vacio mientras no hay huella
                self._fotos[clave] = a_tk(np.full((448, 320, 3), (60, 37, 27), "uint8"))
            else:
                self._fotos[clave] = a_tk(img)
            lbl.configure(image=self._fotos[clave])

    def _bloquear(self, ocupado, mensaje=None, tarea=None, exito=False):
        """Bloquea/desbloquea los botones y maneja la barra de progreso con tiempo restante.

        tarea  'identidad' | 'intruso' | 'voz': al empezar, la barra se arma con lo que suele tardar esa tarea;
        exito  al terminar bien, se guarda lo que tardo de verdad para que la proxima estimacion sea mas precisa.
        """
        self.ocupado = ocupado
        estado = "disabled" if ocupado else "normal"
        for b in (self.btn_generar, self.btn_existente, self.btn_hablar, *self.botones_cmd.values()):
            b.configure(state=estado)
        if ocupado:
            lista = self.duraciones.get(tarea or "", [4.0])
            self._progreso = (time.time(), sum(lista) / len(lista), tarea)
            self._tick_progreso()
        else:
            self._terminar_progreso(exito)
        if mensaje:
            self.lbl_estado_huella.configure(text=mensaje)

    def _tick_progreso(self):
        """Cada 100 ms: avanza la barra segun el tiempo transcurrido y escribe cuanto falta (estimado)."""
        if self._progreso is None:
            return
        t0, est, _ = self._progreso
        el = time.time() - t0
        if el < est:
            pct, texto = 95 * el / est, f"Quedan ~{int(est - el) + 1} s"
        else:                                           # tarda mas de lo previsto: la barra avanza muy despacio hasta 99 %
            pct, texto = 95 + 4 * min(1.0, (el - est) / 10), "Casi listo…"
        self.barra.configure(mode="determinate", value=pct)
        self.lbl_tiempo.configure(text=f"{texto}  ({el:.0f} s de ~{est:.0f} s)")
        self.after(100, self._tick_progreso)

    def _terminar_progreso(self, exito):
        if self._progreso is not None:
            t0, _, tarea = self._progreso
            el = time.time() - t0
            self._progreso = None
            if exito and tarea in self.duraciones:
                self.duraciones[tarea] = (self.duraciones[tarea] + [el])[-5:]
                self.barra.configure(value=100)
                self.lbl_tiempo.configure(text=f"Listo en {el:.1f} s")
                self.after(900, lambda: self.barra.configure(value=0) if self._progreso is None else None)
                return
        self.barra.configure(value=0)
        self.lbl_tiempo.configure(text="")

    def _resultado(self, texto, tono, detalle=""):
        self.insignia.poner(texto, tono)
        self.lbl_detalle.configure(text=detalle)

    def _actualizar_pasos(self):
        """Indicador de pasos del encabezado: Huella -> Reconocer -> Identificar."""
        e = ["pendiente"] * 3
        if self.analisis is None:
            e[0] = "activo"
        else:
            e[0] = "ok"
            e[1] = "activo" if self.reconocida is None else ("ok" if self.reconocida else "error")
            if self.reconocida:
                e[2] = "ok" if self.identificada else "activo"
        self.pasos.actualizar(e)

    # =============================================================== 1. obtener una huella
    def generar(self):
        """Ejecuta Anguli (subprocess) en un hilo para no congelar la ventana."""
        if self.ocupado:
            return
        tipo, nombre_filtro = self.var_tipo.get(), self.var_filtro.get()
        aviso = " (persona nueva, tarda unos segundos)" if tipo == "intruso" else ""
        self._bloquear(True, f"Generando {anguli.TIPOS[tipo]} con Anguli…{aviso}", tarea="intruso" if tipo == "intruso" else "identidad")
        self.log(f"Anguli: generando {anguli.TIPOS[tipo]} · filtro '{nombre_filtro}'…")

        def trabajo():
            try:
                self.cola.put(("generada", anguli.generar_huella(tipo, nombre_filtro)))
            except anguli.AnguliError as e:
                self.cola.put(("error_anguli", str(e)))
            except Exception as e:                           # cualquier otro fallo inesperado
                self.cola.put(("error_anguli", f"{type(e).__name__}: {e}"))
        threading.Thread(target=trabajo, daemon=True).start()

    def elegir_existente(self):
        if self.ocupado:
            return
        config.CARPETA_HUELLAS.mkdir(exist_ok=True)
        ruta = filedialog.askopenfilename(title="Elegí una huella", initialdir=str(config.CARPETA_HUELLAS),
                                          filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
        if ruta:
            self.cargar_huella(Path(ruta), origen=origen_desde_nombre(Path(ruta).name), filtro_txt="(huella existente)")

    def cargar_huella(self, ruta, origen, filtro_txt):
        """Analiza la huella (preprocesar + vector + decision) y la muestra. Reinicia el flujo de comandos."""
        try:
            self.analisis = self.srv.analizar_huella(ruta)
        except Exception as e:
            messagebox.showerror("No se pudo leer la huella", f"{ruta}\n\n{type(e).__name__}: {e}")
            return
        self.ruta, self.origen_real, self.filtro_txt = Path(ruta), origen, filtro_txt
        self.reconocida, self.identificada = None, False
        self.btn_grafico.configure(state="disabled")
        self._poner_imagenes(self.analisis["gris"], self.analisis["binaria"])
        nombre_origen = anguli.TIPOS.get(origen, "desconocido (el nombre no dice quién es)")
        self.lbl_origen.configure(text=f"Archivo: {self.ruta.name}   ·   origen real: {nombre_origen}")
        self.lbl_estado_huella.configure(text="Huella lista. Decí RECONOCER.")
        self._resultado("—", GRIS, "Todavía no se dijo ningún comando.")
        self.log(f"Huella cargada: {self.ruta.name} (origen real: {nombre_origen}). Falta decir RECONOCER.")
        self._actualizar_pasos()

    # =============================================================== 2. voz
    def hablar(self):
        """Cuenta regresiva, graba 1,5 s y clasifica el comando (todo en un hilo)."""
        if self.ocupado:
            return
        self._bloquear(True, tarea="voz")
        self.lbl_voz.configure(text="Preparate…")

        def trabajo():
            try:
                if self.wav_prueba is None:
                    for n in ("3", "2", "1"):
                        self.cola.put(("cuenta", n))
                        time.sleep(0.35)
                    self.cola.put(("cuenta", "¡HABLÁ!"))
                    res = self.srv.escuchar()
                else:
                    res = self.srv.clasificar_wav(self.wav_prueba)
                self.cola.put(("voz", res))
            except Exception as e:
                self.cola.put(("error_voz", f"{type(e).__name__}: {e}"))
        threading.Thread(target=trabajo, daemon=True).start()

    def ejecutar(self, comando, via):
        """Ejecuta un comando (por voz o con los botones manuales)."""
        if self.ocupado:
            return
        self.log(f"Comando {comando.upper()} ({via}).")
        if comando == "salir":
            self._cerrar()
        elif comando == "reconocer":
            self._reconocer(via)
        elif comando == "identificar":
            self._identificar()

    def _reconocer(self, via):
        if self.analisis is None:
            self._resultado("Primero cargá una huella", ROJO, "Generá una con Anguli o elegí una existente.")
            self.log("RECONOCER: no hay huella cargada.")
            return
        d = self.analisis["decision"]
        self.reconocida = d["aceptada"]
        self.btn_grafico.configure(state="normal")
        self.identificada = False
        self._actualizar_pasos()
        verdad = self.srv.registrar(self.ruta, self.origen_real, self.filtro_txt, via, d)
        detalle = (f"Distancia al cluster más cercano: {d['distancia']:.3f}  (umbral R = {d['radio']:.2f})")
        if self.reconocida:
            self._resultado("HUELLA RECONOCIDA", VERDE, detalle + "\nDecí IDENTIFICAR para saber de quién es.")
        else:
            self._resultado("NO RECONOCIDA\n(no autorizada)", ROJO, detalle + "\nLa huella queda más lejos que el umbral.")
        veredicto = {"True": "el sistema ACERTÓ", "False": "el sistema SE EQUIVOCÓ", "": "no se conoce el origen"}[verdad]
        self.log(f"RECONOCER → {'reconocida' if self.reconocida else 'NO reconocida'} (d={d['distancia']:.3f}, R={d['radio']:.2f}); {veredicto}.")

    def _identificar(self):
        if self.analisis is None:
            self._resultado("Primero cargá una huella", ROJO, "Generá una con Anguli o elegí una existente.")
            return
        if self.reconocida is None:
            self._resultado("Primero decí RECONOCER", ROJO, "Hay que reconocer la huella antes de identificarla.")
            self.log("IDENTIFICAR: falta decir RECONOCER antes.")
            return
        if not self.reconocida:
            self._resultado("No hay identidad", ROJO, "La huella no fue reconocida, así que no se identifica.")
            self.log("IDENTIFICAR: la huella no fue reconocida, no hay identidad.")
            return
        d = self.analisis["decision"]
        n = d["resultado"].replace("identidad_", "")
        self._resultado(f"IDENTIDAD {n}", AZUL, f"La huella pertenece a la identidad {n}.\nDistancia a su cluster: {d['distancia']:.3f}.")
        self.log(f"IDENTIFICAR → Identidad {n}.")
        self.identificada = True
        self._actualizar_pasos()
        if self.var_grafico.get():
            self.ver_grafico()

    def ver_grafico(self):
        """Abre (o reemplaza) la ventana con el grafico de clusters y la huella nueva."""
        if self.analisis is None:
            return
        if self.ventana_grafico is not None and self.ventana_grafico.winfo_exists():
            self.ventana_grafico.destroy()
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        fig = grafico.construir_figura(self.srv.modelo_huella, self.analisis["vector"], self.analisis["decision"],
                                       anguli.TIPOS.get(self.origen_real))
        win = tk.Toplevel(self)
        win.title("Gráfico de clusters")
        win.geometry("940x730")
        win.configure(bg=PAL["fondo"])
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        barra = tk.Frame(win, bg=PAL["fondo"])
        barra.pack(fill="x", pady=8)
        Boton(barra, "Guardar como PNG…", command=lambda: self._guardar_figura(fig)).pack(side="left", padx=12)
        Boton(barra, "Cerrar", command=win.destroy).pack(side="right", padx=12)
        canvas.draw()
        self.ventana_grafico, self.figura_actual = win, fig
        self.log("Gráfico de clusters abierto.")

    def _guardar_figura(self, fig):
        ruta = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")], initialfile="clusters_huella.png")
        if ruta:
            fig.savefig(ruta, dpi=130)
            self.log(f"Gráfico guardado en {ruta}")

    # =============================================================== cola de eventos de los hilos
    def _leer_cola(self):
        """Se llama cada 100 ms: procesa lo que dejaron los hilos (asi la ventana nunca se bloquea)."""
        try:
            while True:
                tipo, dato = self.cola.get_nowait()
                getattr(self, f"_evento_{tipo}")(dato)
        except queue.Empty:
            pass
        self.after(100, self._leer_cola)

    def _evento_generada(self, r):
        self._bloquear(False, exito=True)
        self.log(f"Anguli generó {r['ruta'].name} en {r['segundos']:.1f} s · filtro: {anguli.describir_filtro(r['filtro'])}.")
        self.cargar_huella(r["ruta"], origen=r["tipo"], filtro_txt=anguli.describir_filtro(r["filtro"]))

    def _evento_error_anguli(self, msg):
        self._bloquear(False, "No se pudo generar la huella.")
        self.log("ERROR de Anguli: " + msg.splitlines()[0])
        if "No se encontró Anguli.exe" in msg and not self.prueba:
            if messagebox.askyesno("Anguli no encontrado", msg + "\n\n¿Querés elegir la ubicación de Anguli.exe ahora?"):
                ruta = filedialog.askopenfilename(title="Elegí Anguli.exe", filetypes=[("Anguli", "Anguli.exe"), ("Programas", "*.exe")])
                if ruta:
                    config.guardar_ruta_anguli(ruta)
                    self.log(f"Ruta de Anguli guardada: {ruta}. Probá generar de nuevo.")
        elif not self.prueba:
            messagebox.showerror("Anguli", msg)

    def _evento_cuenta(self, texto):
        self.lbl_voz.configure(text=texto)

    def _evento_voz(self, res):
        self._bloquear(False, exito=True)
        comando, mensaje = self.srv.comando_ejecutable(res)
        self.lbl_voz.configure(text=mensaje)
        self.log("Voz: " + mensaje)
        if comando:
            self.ejecutar(comando, via="voz")

    def _evento_error_voz(self, msg):
        self._bloquear(False)
        self.lbl_voz.configure(text="Error con el micrófono.")
        self.log("ERROR de voz: " + msg)
        messagebox.showerror("Micrófono", msg + "\n\nPodés usar los botones del modo manual.")

    # =============================================================== cierre
    def _cerrar(self):
        if self.prueba or messagebox.askyesno("Salir", "¿Cerrar la aplicación?"):
            self.destroy()

    # =============================================================== recorrido automatico (--prueba)
    def correr_prueba(self, carpeta_capturas, wavs, persona="id2"):
        """Recorrido automatico para verificar la interfaz sin tocar nada a mano. Toma capturas de pantalla."""
        from PIL import ImageGrab
        carpeta_capturas.mkdir(parents=True, exist_ok=True)
        self.var_grafico.set(False)                       # el grafico se abre explicitamente al final del recorrido
        self.attributes("-topmost", True)                 # al frente, para que la captura de pantalla vea esta ventana
        self.lift()
        self.focus_force()

        def captura(nombre):
            self.update()
            x, y, w, h = self.winfo_rootx(), self.winfo_rooty(), self.winfo_width(), self.winfo_height()
            ImageGrab.grab(bbox=(x, y, x + w, y + h)).save(carpeta_capturas / f"{nombre}.png")

        def esperar(cond, siguiente, limite=120):
            t0 = time.time()

            def revisar():
                if cond():
                    self.after(300, siguiente)
                elif time.time() - t0 > limite:
                    print("PRUEBA: tiempo agotado esperando"); self.destroy()
                else:
                    self.after(250, revisar)
            revisar()

        def paso_generar(tipo, siguiente):
            def f():
                self.var_tipo.set(tipo)
                n0 = self.ruta
                self.generar()
                esperar(lambda: self.ruta != n0 and not self.ocupado, siguiente)
            return f

        def final():
            captura("5_final")
            print("PRUEBA: recorrido terminado")
            self.after(500, self.destroy)

        def grafico_y_cerrar():
            self.ver_grafico()
            self.update()
            self.after(800, lambda: (self.figura_actual.savefig(carpeta_capturas / "grafico_clusters.png", dpi=110), captura("4_con_grafico"), final()))

        def voz_y_identificar():
            self.wav_prueba = wavs.get("identificar")
            self.hablar()
            esperar(lambda: not self.ocupado, lambda: (captura("3_identificada"), grafico_y_cerrar()))

        def voz_reconocer():
            self.wav_prueba = wavs.get("reconocer")
            self.hablar()
            esperar(lambda: not self.ocupado and self.reconocida is not None, lambda: (captura("2_reconocida"), voz_y_identificar()))

        def inicio():
            captura("0_inicio")
            paso_generar(persona, lambda: (captura("1_huella_generada"), voz_reconocer()))()
        self.after(1200, inicio)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prueba", action="store_true", help="recorrido automatico de verificacion con capturas de pantalla")
    ap.add_argument("--capturas", type=Path, default=config.CARPETA_INTERFAZ / "capturas_prueba")
    ap.add_argument("--wav-reconocer", type=Path, default=None, help="(solo --prueba) WAV que reemplaza al microfono para RECONOCER")
    ap.add_argument("--wav-identificar", type=Path, default=None, help="(solo --prueba) WAV que reemplaza al microfono para IDENTIFICAR")
    ap.add_argument("--persona", choices=list(anguli.TIPOS), default="id2", help="(solo --prueba) persona que se genera")
    a = ap.parse_args()
    app = App(prueba=a.prueba)
    if a.prueba:
        app.correr_prueba(a.capturas, {"reconocer": a.wav_reconocer, "identificar": a.wav_identificar}, a.persona)
    app.mainloop()


if __name__ == "__main__":
    main()
