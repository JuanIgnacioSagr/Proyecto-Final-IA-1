"""Convierte informe/Informe_final_IA.md en informe/Informe_final_IA.pdf (con tablas, imagenes y formulas).

    pip install markdown-it-py mdit-py-plugins
    python informe/generar_pdf.py

Como funciona: markdown-it-py pasa el .md a HTML (con las formulas $...$ y $$...$$ protegidas), se le agregan un estilo de pagina A4
y KaTeX (por internet, jsdelivr) para dibujar las formulas, y el navegador Edge o Chrome ya instalado imprime esa pagina a PDF
en modo headless. No hace falta instalar LaTeX ni pandoc.
"""
import html
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from markdown_it import MarkdownIt
from mdit_py_plugins.dollarmath import dollarmath_plugin

AQUI = Path(__file__).resolve().parent
MD = AQUI / "Informe_final_IA.md"
PDF = AQUI / "Informe_final_IA.pdf"
NAVEGADORES = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
               r"C:\Program Files\Google\Chrome\Application\chrome.exe", r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"]

CSS = """
@page { size: A4; margin: 16mm 15mm 16mm 15mm; }
body { font-family: 'Segoe UI', Calibri, Arial, sans-serif; font-size: 10.3pt; line-height: 1.45; color: #1f2937; }
h1 { font-size: 19pt; border-bottom: 2px solid #1d4ed8; padding-bottom: 3px; margin-top: 22px; page-break-after: avoid; }
h2 { font-size: 14.5pt; color: #1d4ed8; margin-top: 18px; page-break-after: avoid; }
h3 { font-size: 12pt; margin-top: 14px; page-break-after: avoid; }
h4 { font-size: 10.8pt; margin-top: 12px; page-break-after: avoid; }
table { border-collapse: collapse; margin: 8px 0; width: 100%; font-size: 9.3pt; page-break-inside: avoid; }
th, td { border: 1px solid #9ca3af; padding: 3px 6px; vertical-align: top; }
th { background: #e5edff; }
img { max-width: 100%; max-height: 215mm; display: block; margin: 8px auto; page-break-inside: avoid; }
blockquote { border-left: 3px solid #93c5fd; margin: 8px 0; padding: 2px 10px; background: #f3f7ff; }
code { background: #f1f5f9; padding: 0 3px; border-radius: 3px; font-size: 9.2pt; }
pre { background: #f1f5f9; padding: 8px; border-radius: 4px; white-space: pre-wrap; font-size: 8.8pt; page-break-inside: avoid; }
pre code { background: none; }
hr { border: 0; border-top: 1px solid #d1d5db; margin: 14px 0; }
.math.block { text-align: center; margin: 6px 0; page-break-inside: avoid; }
li { margin: 2px 0; }
"""

JS = """
window.addEventListener('load', function () {
  document.querySelectorAll('.math').forEach(function (el) {
    try { katex.render(el.textContent, el, {displayMode: el.classList.contains('block'), throwOnError: false}); } catch (e) {}
  });
  document.body.setAttribute('data-listo', '1');
});
"""


def navegador():
    for ruta in NAVEGADORES:
        if Path(ruta).exists():
            return ruta
    sys.exit("No se encontro Edge ni Chrome. Instala uno o cambia NAVEGADORES en este script.")


def main():
    md = (MarkdownIt("commonmark", {"html": True, "typographer": False}).enable("table").use(dollarmath_plugin, allow_space=True, double_inline=True))
    cuerpo = md.render(MD.read_text(encoding="utf-8"))
    pagina = (f'<!doctype html><html lang="es"><head><meta charset="utf-8"><title>Trabajo Final Inteligencia Artificial I</title>'
              f'<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css"><style>{CSS}</style>'
              f'<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script></head>'
              f'<body>{cuerpo}<script>{JS}</script></body></html>')
    # la pagina se guarda junto al .md para que las rutas relativas (imagenes/...) funcionen; se borra al terminar
    tmp = AQUI / "_informe_tmp.html"
    tmp.write_text(pagina, encoding="utf-8")
    perfil = tempfile.mkdtemp(prefix="pdf_perfil_")
    try:
        cmd = [navegador(), "--headless=new", "--disable-gpu", f"--user-data-dir={perfil}", "--no-pdf-header-footer",
               "--virtual-time-budget=25000", f"--print-to-pdf={PDF}", tmp.as_uri()]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if not PDF.exists():
            sys.exit(f"No se genero el PDF.\n{r.stderr[-600:]}")
    finally:
        tmp.unlink(missing_ok=True)
        shutil.rmtree(perfil, ignore_errors=True)
    print(f"PDF generado: {PDF} ({PDF.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
