def rm_f(path):
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def export_bokeh(p, filename, title):
    path = os.path.splitext(filename)[0]
    png_path = f"png/{path}.png"
    rm_f(png_path)
    export_png(p, filename=png_path)

    html_path = f"html/{path}.html"
    rm_f(html_path)
    output_file(filename=html_path, title=title)
    bokeh_save(p)

