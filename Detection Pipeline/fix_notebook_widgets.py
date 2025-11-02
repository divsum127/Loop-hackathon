# fix_notebook_widgets.py
import sys, nbformat, pathlib

if len(sys.argv) != 2:
    print("Usage: python fix_notebook_widgets.py <notebook.ipynb>")
    sys.exit(1)

src = pathlib.Path(sys.argv[1])
dst = src.with_suffix(".clean.ipynb")

nb = nbformat.read(src, as_version=4)

# 1) Drop broken top-level ipywidgets metadata
if "widgets" in nb.metadata:
    del nb.metadata["widgets"]

# 2) Strip widget *outputs* (and other heavy outputs)
for cell in nb.cells:
    if cell.get("cell_type") != "code":
        continue
    new_outputs = []
    for out in cell.get("outputs", []):
        # Remove widget views
        data = out.get("data", {})
        if isinstance(data, dict) and "application/vnd.jupyter.widget-view+json" in data:
            continue
        new_outputs.append(out)
    cell["outputs"] = new_outputs
    cell["execution_count"] = None  # optional: de-noise

# 3) Ensure minimal kernelspec/language_info (prevents other render warns)
ks = nb.metadata.setdefault("kernelspec", {})
ks.setdefault("name", "python3")
ks.setdefault("display_name", "Python 3")

li = nb.metadata.setdefault("language_info", {})
li.setdefault("name", "python")

nbformat.write(nb, dst)
print(f"✓ Wrote cleaned notebook → {dst}")
