"""Aspecto visual de la interfaz: paleta de colores (tema oscuro) y widgets propios.

Tkinter trae widgets de aspecto antiguo; aca se definen unos pocos widgets planos y modernos
(botones con efecto al pasar el mouse, tarjetas, insignias de resultado y un indicador de pasos)
para que toda la ventana tenga un aspecto coherente. No contiene logica del programa.
"""
import tkinter as tk
from tkinter import ttk

FUENTE = "Segoe UI"

# Paleta (tema oscuro, tonos azul pizarra)
P = {
    "fondo": "#0b1220",        # fondo de la ventana
    "tarjeta": "#131c31",      # fondo de las tarjetas
    "tarjeta2": "#1a2540",     # elementos dentro de una tarjeta (botones secundarios, opciones)
    "borde": "#26334f",
    "texto": "#e6edf7",
    "suave": "#8fa1bf",        # texto secundario
    "acento": "#3b82f6",       # azul principal
    "acento_hover": "#2f6fdb",
    "exito": "#22c55e",
    "peligro": "#ef4444",
    "aviso": "#f59e0b",
    "info": "#60a5fa",
    "deshabilitado": "#222d46",
}

# colores de cada tipo de boton: (fondo, texto, fondo al pasar el mouse)
_BOTONES = {
    "primario": (P["acento"], "#ffffff", P["acento_hover"]),
    "secundario": (P["tarjeta2"], P["texto"], "#243255"),
    "peligro": ("#3a1c28", "#fca5a5", "#4d2231"),
}

# insignias de resultado: (fondo, texto)
_TONOS = {
    "exito": ("#0f3a26", "#4ade80"),
    "peligro": ("#451a24", "#f87171"),
    "info": ("#0f2a4d", "#7ab8ff"),
    "neutro": ("#18233b", "#8fa1bf"),
}


def aplicar_tema(raiz):
    """Fondo de la ventana, barra de titulo oscura (Windows 10/11) y estilo de la barra de progreso."""
    raiz.configure(bg=P["fondo"])
    try:                                                     # barra de titulo oscura en Windows
        import ctypes
        raiz.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(raiz.winfo_id())
        valor = ctypes.c_int(1)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(valor), ctypes.sizeof(valor))
    except Exception:
        pass
    estilo = ttk.Style(raiz)
    estilo.theme_use("clam")
    estilo.configure("Moderno.Horizontal.TProgressbar", troughcolor=P["tarjeta2"], background=P["acento"],
                     bordercolor=P["tarjeta"], lightcolor=P["acento"], darkcolor=P["acento"], thickness=6)


class Boton(tk.Label):
    """Boton plano: cambia de color al pasar el mouse y acepta configure(state=..., text=..., command=...)."""

    def __init__(self, master, text, command=None, tipo="secundario", grande=False, **kw):
        fondo, texto, _ = _BOTONES[tipo]
        super().__init__(master, text=text, bg=fondo, fg=texto, cursor="hand2",
                         font=(FUENTE, 12 if grande else 10, "bold"),
                         padx=16, pady=11 if grande else 7, **kw)
        self._tipo, self._comando, self._estado, self._encima = tipo, command, "normal", False
        self.bind("<Enter>", lambda e: self._hover(True))
        self.bind("<Leave>", lambda e: self._hover(False))
        self.bind("<ButtonRelease-1>", self._soltar)

    def _hover(self, encima):
        self._encima = encima
        self._pintar()

    def _pintar(self):
        fondo, texto, hover = _BOTONES[self._tipo]
        if self._estado == "disabled":
            super().configure(bg=P["deshabilitado"], fg=P["suave"], cursor="arrow")
        else:
            super().configure(bg=hover if self._encima else fondo, fg=texto, cursor="hand2")

    def _soltar(self, evento):
        dentro = 0 <= evento.x <= self.winfo_width() and 0 <= evento.y <= self.winfo_height()
        if self._estado != "disabled" and dentro and self._comando:
            self._comando()

    def configure(self, cnf=None, **kw):
        if "state" in kw:
            self._estado = kw.pop("state")
            self._pintar()
        if "command" in kw:
            self._comando = kw.pop("command")
        if cnf or kw:
            super().configure(cnf, **kw)

    config = configure


class Tarjeta(tk.Frame):
    """Panel con borde fino y un titulo pequeno en mayusculas. El contenido va en `self.cuerpo`."""

    def __init__(self, master, titulo="", **kw):
        super().__init__(master, bg=P["tarjeta"], highlightbackground=P["borde"], highlightthickness=1, **kw)
        if titulo:
            tk.Label(self, text=titulo.upper(), bg=P["tarjeta"], fg=P["acento"],
                     font=(FUENTE, 9, "bold")).pack(anchor="w", padx=16, pady=(14, 8))
        self.cuerpo = tk.Frame(self, bg=P["tarjeta"])
        self.cuerpo.pack(fill="both", expand=True, padx=16, pady=(0, 16))


def etiqueta(master, texto="", suave=False, negrita=False, tam=10, **kw):
    """Label con los colores del tema (texto principal o secundario)."""
    return tk.Label(master, text=texto, bg=kw.pop("bg", P["tarjeta"]), fg=P["suave"] if suave else P["texto"],
                    font=(FUENTE, tam, "bold" if negrita else "normal"), **kw)


def opciones(master, variable, textos_valores, columnas=2):
    """Grupo de opciones tipo 'pastilla' (en lugar de los circulos de siempre). textos_valores: [(texto, valor)]."""
    marco = tk.Frame(master, bg=P["tarjeta"])
    for i, (texto, valor) in enumerate(textos_valores):
        tk.Radiobutton(marco, text=texto, value=valor, variable=variable, indicatoron=False, bd=0, relief="flat",
                       bg=P["tarjeta2"], fg=P["texto"], selectcolor=P["acento"], activebackground=P["acento_hover"],
                       activeforeground="#ffffff", font=(FUENTE, 10, "bold" if False else "normal"),
                       padx=10, pady=7, cursor="hand2", highlightthickness=0).grid(
            row=i // columnas, column=i % columnas, sticky="ew", padx=2, pady=2)
    for c in range(columnas):
        marco.columnconfigure(c, weight=1, uniform="op")
    return marco


class Insignia(tk.Label):
    """Cartel grande del resultado (RECONOCIDA / NO RECONOCIDA / IDENTIDAD n). Cambia de color segun el tono."""

    def __init__(self, master, **kw):
        super().__init__(master, text="—", font=(FUENTE, 15, "bold"), wraplength=250, justify="center",
                         height=3, **kw)
        self.poner("—", "neutro")

    def poner(self, texto, tono):
        fondo, color = _TONOS[tono]
        self.configure(text=texto, bg=fondo, fg=color)


class Pasos(tk.Canvas):
    """Indicador de pasos del flujo: Huella -> Reconocer -> Identificar. Estados: pendiente, activo, ok, error."""

    def __init__(self, master, etiquetas, ancho=430, alto=56):
        super().__init__(master, width=ancho, height=alto, bg=P["fondo"], highlightthickness=0)
        self._etiquetas, self._ancho = etiquetas, ancho
        self.actualizar(["pendiente"] * len(etiquetas))

    def actualizar(self, estados):
        self.delete("all")
        n = len(self._etiquetas)
        paso = self._ancho / n
        xs = [paso * (i + 0.5) for i in range(n)]
        for i in range(n - 1):                                          # lineas entre pasos
            color = P["exito"] if estados[i] == "ok" else P["borde"]
            self.create_line(xs[i] + 17, 18, xs[i + 1] - 17, 18, fill=color, width=2)
        for x, texto, est in zip(xs, self._etiquetas, estados):
            relleno, contorno, simbolo = {"pendiente": (P["fondo"], P["borde"], ""), "activo": (P["acento"], P["acento"], ""),
                                          "ok": (P["exito"], P["exito"], "✓"), "error": (P["peligro"], P["peligro"], "✕")}[est]
            self.create_oval(x - 13, 5, x + 13, 31, fill=relleno, outline=contorno, width=2)
            numero = str(self._etiquetas.index(texto) + 1)
            self.create_text(x, 18, text=simbolo or numero, fill="#ffffff" if est != "pendiente" else P["suave"],
                             font=(FUENTE, 10, "bold"))
            self.create_text(x, 44, text=texto, fill=P["texto"] if est != "pendiente" else P["suave"], font=(FUENTE, 9))
