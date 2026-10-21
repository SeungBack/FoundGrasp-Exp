"""Compact terminal output, with optional diagnostics and no terminal dependency."""
from contextlib import contextmanager, redirect_stdout
import logging
from pathlib import Path
import sys
import unicodedata


_verbose = False


@contextmanager
def output_mode(verbose=False):
    global _verbose
    previous, _verbose = _verbose, verbose
    sdk_logger = logging.getLogger("origin.print")
    propagate = sdk_logger.propagate
    sdk_logger.propagate = False  # SDK has its own handler; avoid duplicate INFO lines.
    try:
        yield
    finally:
        _verbose = previous
        sdk_logger.propagate = propagate


def section(title):
    print(f"\n{'─' * 66}\n  {title}\n{'─' * 66}", flush=True)


def row(label, message):
    width = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in label)
    print(f"  {label}{' ' * max(2, 10 - width)}{message}", flush=True)


def detail(message):
    if _verbose:
        print(f"  · {message.strip()}", flush=True)


class _ModelOutput:
    """Keep initialization chatter in a file; warnings/errors remain visible."""

    def __init__(self, log, terminal):
        self.log, self.terminal = log, terminal
        self.pending = ""

    def write(self, text):
        self.log.write(text)
        if _verbose:
            self.terminal.write(text)
        else:
            self.pending += text
            while "\n" in self.pending:
                line, self.pending = self.pending.split("\n", 1)
                self._forward(line + "\n")
        return len(text)

    def _forward(self, line):
        if any(word in line.lower() for word in ("warning", "error", "경고", "오류")):
            self.terminal.write(line)

    def flush(self):
        self.log.flush()
        self.terminal.flush()

    def isatty(self):
        return False

    @property
    def encoding(self):
        return self.log.encoding

    def finish(self):
        if self.pending:
            self._forward(self.pending + "\n")
            self.pending = ""
        self.flush()


@contextmanager
def model_output(path):
    """Redirect Python stdout only; stderr (warnings/progress) stays visible."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as log:
        stream = _ModelOutput(log, sys.stdout)
        try:
            with redirect_stdout(stream):
                yield
        except BaseException:
            row("모델 오류", f"초기화 상세 로그: {path}")
            raise
        finally:
            stream.finish()
