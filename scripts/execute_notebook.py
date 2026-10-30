"""Run every notebook cell with the selected Python interpreter."""

import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from ipykernel.kernelspec import install

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "results/notebook_check"
    output.mkdir(parents=True, exist_ok=True)
    prefix = output / "kernel"
    os.environ["JUPYTER_PATH"] = str(prefix / "share/jupyter") + os.pathsep + os.environ.get("JUPYTER_PATH", "")
    install(prefix=str(prefix), kernel_name="status-check", display_name="Vital-status notebook check")
    notebook = nbformat.read(ROOT / "notebooks/github_breast_cancer.ipynb", as_version=4)
    client = NotebookClient(notebook, kernel_name="status-check", timeout=600,
                            resources={"metadata": {"path": str(ROOT)}},
                            on_cell_start=lambda cell, cell_index: print(f"Cell {cell_index+1}/{len(notebook.cells)}", flush=True))
    client.execute()
    nbformat.write(notebook, output / "executed.ipynb")
    print(f"Executed all {len(notebook.cells)} notebook cells. Results: {ROOT / 'results/reproduced'}")


if __name__ == "__main__":
    main()
