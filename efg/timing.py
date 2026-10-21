"""Wall-clock cycle timing; partial cycles never enter completed-cycle statistics."""
import time

from .console import row


class TaktStats:
    def __init__(self):
        self.count = 0
        self.total = 0.
        self.minimum = self.maximum = None

    def add(self, seconds):
        self.count += 1
        self.total += seconds
        self.minimum = seconds if self.minimum is None else min(self.minimum, seconds)
        self.maximum = seconds if self.maximum is None else max(self.maximum, seconds)

    @property
    def average(self):
        return self.total / self.count if self.count else None

    def summary(self):
        if not self.count:
            return "완료 사이클 없음"
        return (f"평균 {self.average:.2f}s · 최단 {self.minimum:.2f}s · "
                f"최장 {self.maximum:.2f}s · 완료 {self.count}회")


class CycleTiming:
    def __init__(self):
        self.started = self.last = time.perf_counter()
        self.stage = "홈 복귀"
        self.durations = {}
        self.notes = []
        self.finished = False

    def mark(self, stage):
        if self.finished:
            return
        now = time.perf_counter()
        self.durations[self.stage] = self.durations.get(self.stage, 0.) + now - self.last
        self.stage, self.last = stage, now

    @property
    def elapsed(self):
        return self.last - self.started

    def finish(self):
        if not self.finished:
            self.mark("완료")
            self.finished = True
        return self.elapsed

    def report(self, completed=False, stats=None):
        seconds = self.finish()
        summary = f"{seconds:.2f}s"
        if completed and stats is not None:
            summary += f"  |  {stats.summary()}"
        row("택타임" if completed else "소요시간", summary)
        items = [f"{stage} {duration:.2f}s" for stage, duration in self.durations.items()
                 if duration >= .005]
        for index in range(0, len(items), 3):
            row("구간" if index == 0 else "", " · ".join(items[index:index + 3]))
        for note in self.notes:
            row("상세", note)
        return summary
