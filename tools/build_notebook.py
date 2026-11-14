"""Execute notebook_src.py cell by cell and write a .ipynb with all outputs embedded."""
import sys, io, json, base64, contextlib, traceback, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src_path, out_path = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
os.chdir(os.path.dirname(os.path.abspath(src_path)))
text = open(src_path, encoding="utf-8").read()

cells = []
for chunk in text.split("# %%")[1:]:
    if chunk.startswith(" [markdown]"):
        body = chunk[len(" [markdown]"):].strip("\n")
        lines = [l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines()]
        cells.append(("markdown", "\n".join(lines).strip()))
    else:
        cells.append(("code", chunk.strip("\n")))

outputs_current = []

def display(obj):
    import pandas as pd
    flush_stdout()
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        df = obj.to_frame() if isinstance(obj, pd.Series) else obj
        outputs_current.append({"output_type": "display_data", "metadata": {},
            "data": {"text/html": df.to_html(), "text/plain": df.to_string()}})
    else:
        print(obj)

def show(*a, **k):
    for num in plt.get_fignums():
        fig = plt.figure(num); buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        flush_stdout()
        outputs_current.append({"output_type": "display_data", "metadata": {},
            "data": {"image/png": base64.b64encode(buf.getvalue()).decode(), "text/plain": "<Figure>"}})
    plt.close("all")

plt.show = show
stdout_buf = io.StringIO()

def flush_stdout():
    s = stdout_buf.getvalue()
    if s:
        outputs_current.append({"output_type": "stream", "name": "stdout", "text": s})
        stdout_buf.seek(0); stdout_buf.truncate()

ns = {"display": display, "__name__": "__main__"}
nb_cells, count, failed = [], 0, False
for kind, body in cells:
    if kind == "markdown":
        nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": body.splitlines(True)})
        continue
    count += 1; outputs_current = []
    with contextlib.redirect_stdout(stdout_buf):
        try:
            exec(compile(body, f"<cell {count}>", "exec"), ns)
        except Exception:
            failed = True
            print(traceback.format_exc())
    flush_stdout()
    nb_cells.append({"cell_type": "code", "execution_count": count, "metadata": {},
                     "outputs": outputs_current, "source": body.splitlines(True)})
    if failed:
        sys.stderr.write(f"FAILED in cell {count}\n" + json.dumps(outputs_current)[-3000:]); break

nb = {"cells": nb_cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}, "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(out_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("cells:", len(nb_cells), "failed:", failed)
