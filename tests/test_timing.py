"""Timing excludes setup and cannot continue accumulating after a cycle finishes."""
import pytest

from efg import timing


def test_retries_accumulate_and_wait_after_finish_is_excluded(monkeypatch, capsys):
    clock = [100.]  # model initialization already elapsed
    monkeypatch.setattr(timing.time, "perf_counter", lambda: clock[0])
    cycle = timing.CycleTiming()
    clock[0] += 2
    cycle.mark("추론")
    clock[0] += .4
    cycle.mark("캡처")
    clock[0] += .1
    cycle.mark("추론")
    clock[0] += .6
    assert cycle.finish() == pytest.approx(3.1)
    assert cycle.durations["추론"] == pytest.approx(1.)
    clock[0] += 200  # replenishing bin / keyboard wait must not count
    cycle.mark("추론")
    assert cycle.finish() == pytest.approx(3.1)
    cycle.report()
    output = capsys.readouterr().out
    assert "소요시간" in output and "3.10s" in output
    assert "택타임" not in output


def test_completed_statistics_and_manual_confirmation_are_explicit(monkeypatch, capsys):
    clock = [0.]
    monkeypatch.setattr(timing.time, "perf_counter", lambda: clock[0])
    cycle = timing.CycleTiming()
    stats = timing.TaktStats()
    assert stats.average is None
    assert stats.summary() == "완료 사이클 없음"
    clock[0] += 2
    cycle.mark("사용자 확인")
    clock[0] += 3
    stats.add(cycle.finish())
    stats.add(7.)
    assert stats.average == 6.
    cycle.report(completed=True, stats=stats)
    output = capsys.readouterr().out
    assert "택타임" in output and "5.00s" in output
    assert "평균 6.00s" in output and "최단 5.00s" in output and "최장 7.00s" in output
    assert "사용자 확인 3.00s" in output
    assert stats.count == 2  # reporting must not add another cycle
