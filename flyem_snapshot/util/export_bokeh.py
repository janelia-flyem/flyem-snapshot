import os
from pathlib import Path

from bokeh.plotting import output_file, save as bokeh_save
from bokeh.io import export_png


def export_bokeh(p, filename, title, directory=""):
    """
    Export the given bokeh plot in both html and png format.
    The two files will be placed in two different
    subdirectories named 'html' and 'png'.
    """
    path = Path(directory) / Path(filename)
    png_path = path.parent / "png" / path.with_suffix('.png').name
    _rm_f(png_path)
    os.makedirs(png_path.parent, exist_ok=True)
    export_png(p, filename=png_path)

    html_path = path.parent / "html" / path.with_suffix('.html').name
    _rm_f(html_path)
    os.makedirs(html_path.parent, exist_ok=True)
    output_file(filename=html_path, title=title)
    bokeh_save(p)


def _rm_f(path):
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
