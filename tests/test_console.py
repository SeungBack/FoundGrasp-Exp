"""Model output routing: preserve diagnostics and restore stdout after failures."""
import sys

import pytest

from efg import console


@pytest.mark.parametrize("verbose", [False, True])
def test_model_chatter_is_logged_and_warnings_remain_visible(tmp_path, capsys, verbose):
    path = tmp_path / "run" / "model_init.log"
    with console.output_mode(verbose):
        with console.model_output(path):
            print("parameter counts: 123")
            print("Warn", end="")
            print("ing: diagnostic")
            print("stderr diagnostic", file=sys.stderr)
        console.detail("refiner details")
    output = capsys.readouterr()
    assert "Warning: diagnostic" in output.out
    assert "stderr diagnostic" in output.err
    assert ("parameter counts" in output.out) is verbose
    assert ("refiner details" in output.out) is verbose
    assert path.read_text() == "parameter counts: 123\nWarning: diagnostic\n"


def test_failed_model_load_restores_stdout_and_reports_log(tmp_path, capsys):
    before = sys.stdout
    path = tmp_path / "model_init.log"
    with pytest.raises(RuntimeError, match="load failed"):
        with console.output_mode(False), console.model_output(path):
            print("last model diagnostic")
            raise RuntimeError("load failed")
    assert sys.stdout is before
    assert str(path) in capsys.readouterr().out
    assert "last model diagnostic" in path.read_text()
