from pathlib import Path
import subprocess
import sys


def test_missing_file_has_actionable_cli_error(tmp_path):
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,"-m","breast_cancer","train","--input",str(tmp_path/"missing.csv")],cwd=root,capture_output=True,text=True)
    assert result.returncode==1
    assert "Input file not found" in result.stderr
    assert "Traceback" not in result.stderr
