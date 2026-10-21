"""카메라 단절 시 접근 차단·정리 검증 — 장비와 사용자 입력은 모두 모의."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from efg import demo
from efg.camera import CameraError


class FakeStream:
    def __init__(self):
        self.failed = False
        self.quit_requested = False
        self.start = Mock()
        self.stop = Mock()
        self.set_status = Mock()
        self.set_marks = Mock()
        self.snapshot = Mock(return_value=(
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.ones((2, 2), dtype=np.float32), np.eye(3)))

    def raise_if_failed(self):
        if self.failed:
            raise CameraError("test camera disconnected")


@pytest.fixture
def setup_pick(monkeypatch):
    robot = Mock()
    robot.T_flange_tcp = np.eye(4)
    robot.R_home_tcp = np.eye(3)
    robot.check_flange_ik.return_value = True
    gripper = Mock()
    gripper.pre_open.return_value = True
    stream = FakeStream()
    selected = SimpleNamespace(
        index=0, score=0.25, width_m=0.04, open_width_m=0.048,
        tilt_deg=0.0, p_base=np.array([0.3, 0.0, 0.1]),
        T_tcp_pre=np.eye(4), T_tcp_grasp=np.eye(4), T_tcp_lift=np.eye(4))
    selected.T_tcp_pre[2, 3] = 0.2
    selected.T_tcp_grasp[2, 3] = 0.1
    selected.T_tcp_lift[2, 3] = 0.3
    cfg = {
        "robot": {"travel_speed_mm_s": 100, "approach_speed_mm_s": 20,
                  "lift_speed_mm_s": 100, "approach_acc_mm_s2": 100},
        "drop": {"joints_deg": [0.0] * 7},
    }
    move = Mock(return_value=True)
    monkeypatch.setattr(demo, "move_tcp", move)
    return SimpleNamespace(robot=robot, gripper=gripper, stream=stream,
                           selected=selected, cfg=cfg, move=move)


def assert_only_pregrasp(p):
    p.robot.enable_motion.assert_called_once_with()
    p.gripper.pre_open.assert_called_once_with(p.selected.open_width_m)
    assert p.move.call_count == 1
    assert p.move.call_args.args[0] is p.robot
    np.testing.assert_array_equal(p.move.call_args.args[1], p.selected.T_tcp_pre)
    p.robot.arm.set_tcp_maxacc.assert_not_called()
    p.gripper.grip.assert_not_called()


def test_failed_stream_prevents_any_pick_motion(setup_pick):
    p = setup_pick
    p.stream.failed = True
    with pytest.raises(CameraError, match="disconnected"):
        demo.pick_and_place(p.robot, p.gripper, p.selected, p.cfg,
                            auto=False, stream=p.stream)
    p.robot.enable_motion.assert_not_called()
    p.gripper.pre_open.assert_not_called()
    p.move.assert_not_called()
    p.gripper.grip.assert_not_called()


def test_unreachable_recovery_aborts_before_pick(setup_pick):
    p = setup_pick
    p.robot.check_flange_ik.return_value = False
    result = demo.pick_and_place(p.robot, p.gripper, p.selected, p.cfg,
                                auto=True, stream=p.stream)
    assert result == ("abort", "상승 경유점 사전 IK")
    p.robot.enable_motion.assert_not_called()
    p.gripper.pre_open.assert_not_called()
    p.gripper.grip.assert_not_called()
    p.move.assert_not_called()


@pytest.mark.parametrize('bad_joints', [None, [0.] * 6 + [100.], [float('nan')] * 7])
def test_rechecking_selected_symmetry_blocks_pick_if_ik_or_j7_changed(setup_pick, bad_joints):
    p = setup_pick
    p.cfg['selection'] = {'ik_check_per_candidate': True, 'j7_limit_deg': [-89, 90]}
    p.selected.transfer_sign = -1
    p.robot.flange_ik_joints_deg.side_effect = [[0.] * 7, [0.] * 7, bad_joints]
    assert demo.pick_and_place(p.robot, p.gripper, p.selected, p.cfg, auto=True,
                               stream=p.stream) == ('abort', '선정 경로 사전 IK/J7 재검사')
    expected = np.diag([-1., -1., 1.])
    np.testing.assert_array_equal(p.robot.flange_ik_joints_deg.call_args.args[0][:3, :3], expected)
    p.robot.enable_motion.assert_not_called()
    p.gripper.pre_open.assert_not_called()
    p.gripper.grip.assert_not_called()


def test_disconnect_during_enable_prevents_open_and_pregrasp(setup_pick):
    p = setup_pick

    def disconnect_while_enabling():
        p.stream.failed = True

    p.robot.enable_motion.side_effect = disconnect_while_enabling
    with pytest.raises(CameraError, match="disconnected"):
        demo.pick_and_place(p.robot, p.gripper, p.selected, p.cfg,
                            auto=False, stream=p.stream)
    p.robot.enable_motion.assert_called_once_with()
    p.gripper.pre_open.assert_not_called()
    p.move.assert_not_called()
    p.gripper.grip.assert_not_called()


def test_disconnect_during_confirmation_blocks_accepted_approach(
        setup_pick, monkeypatch):
    p = setup_pick

    def disconnect_and_accept(*args, **kwargs):
        assert kwargs["stream"] is p.stream
        p.stream.failed = True
        return True

    monkeypatch.setattr(demo, "confirm", disconnect_and_accept)
    with pytest.raises(CameraError, match="disconnected"):
        demo.pick_and_place(p.robot, p.gripper, p.selected, p.cfg,
                            auto=False, stream=p.stream)
    assert_only_pregrasp(p)


def test_auto_mode_disconnect_after_open_blocks_approach(setup_pick, monkeypatch):
    p = setup_pick

    def disconnect_on_open(*args, **kwargs):
        p.stream.failed = True
        return True

    p.gripper.pre_open.side_effect = disconnect_on_open
    def approach(*args, **kwargs):
        kwargs["prepare"]()
        kwargs["check"]()
    monkeypatch.setattr(demo, "approach_blended", approach)
    read_input = Mock(side_effect=AssertionError("auto mode must not prompt"))
    monkeypatch.setattr(demo, "_read_input", read_input)
    with pytest.raises(CameraError, match="disconnected"):
        demo.pick_and_place(p.robot, p.gripper, p.selected, p.cfg,
                            auto=True, stream=p.stream)
    p.move.assert_not_called()
    p.robot.arm.set_position.assert_not_called()
    p.gripper.grip.assert_not_called()
    read_input.assert_not_called()


def test_auto_opens_before_queued_approach_and_does_not_close_on_failure(
        setup_pick, monkeypatch):
    p = setup_pick

    def approach(*args, **kwargs):
        p.gripper.pre_open.assert_not_called()
        assert kwargs["prepare"]()
        assert p.gripper.pre_open.call_args.args == (p.selected.open_width_m,)
        kwargs["check"]()
        p.gripper.grip.assert_not_called()
        return False

    monkeypatch.setattr(demo, "approach_blended", approach)
    monkeypatch.setattr(demo, "confirm", Mock(side_effect=AssertionError("prompt")))
    result = demo.pick_and_place(p.robot, p.gripper, p.selected, p.cfg,
                                auto=True, stream=p.stream)
    assert result == ("abort", "연속 접근 이동")
    p.gripper.grip.assert_not_called()
    p.move.assert_not_called()


def test_pending_input_detects_disconnect_without_user_response(monkeypatch):
    stream = FakeStream()
    stdin = Mock()

    def disconnect_while_waiting(*args):
        stream.failed = True
        return [], [], []  # no keyboard input arrived

    poll = Mock(side_effect=disconnect_while_waiting)
    monkeypatch.setattr(demo.sys, "stdin", stdin)
    monkeypatch.setattr(demo.select, "select", poll)
    with pytest.raises(CameraError, match="disconnected"):
        demo._read_input("continue? ", stream)
    poll.assert_called_once()
    stdin.readline.assert_not_called()


def test_auto_closes_only_after_approach_completes(setup_pick, monkeypatch):
    p = setup_pick
    order = []
    p.gripper.pre_open.side_effect = lambda width, **kw: order.append("open") or True

    def approach(*args, **kwargs):
        assert order == []
        assert kwargs["prepare"]()
        assert order == ["open"]
        p.gripper.grip.assert_not_called()
        order.append("arrived")
        return True

    def grip():
        assert order == ["open", "arrived"]
        order.append("close")
        return False, 0.0

    p.gripper.grip.side_effect = grip
    monkeypatch.setattr(demo, "approach_blended", approach)
    monkeypatch.setattr(demo, "confirm", Mock(side_effect=AssertionError("prompt")))
    assert demo.pick_and_place(p.robot, p.gripper, p.selected, p.cfg,
                               auto=True, stream=p.stream) == ("fail", "파지")
    assert order[:3] == ["open", "arrived", "close"]


@pytest.mark.parametrize("arrived,opened", [(True, True), (True, False), (False, True)])
def test_release_requires_completed_transfer(setup_pick, monkeypatch, arrived, opened):
    p = setup_pick
    p.selected.transfer_sign = -1
    p.gripper.grip.return_value = True, 40.
    p.gripper.release.return_value = opened
    monkeypatch.setattr(demo, "approach_blended", Mock(return_value=True))

    def transfer(*args, **kwargs):
        p.gripper.grip.assert_called_once()
        p.gripper.release.assert_not_called()
        kwargs["check"]()
        assert kwargs['transfer_sign'] == -1
        checked_pose = p.robot.check_flange_ik.call_args.args[0]
        np.testing.assert_array_equal(checked_pose[:3, :3], np.diag([-1., -1., 1.]))
        return arrived

    monkeypatch.setattr(demo, "lift_and_transfer_blended", transfer)
    result = demo.pick_and_place(p.robot, p.gripper, p.selected, p.cfg,
                                auto=True, stream=p.stream)
    if arrived:
        assert result == (("success", None) if opened else ("abort", "릴리즈 개방 실패"))
        p.gripper.release.assert_called_once()
        assert p.gripper.release.call_args.args == (p.selected.open_width_m,)
    else:
        assert result == ("abort", "상승/이송 이동")
        p.gripper.release.assert_not_called()
    p.move.assert_not_called()


@pytest.mark.parametrize("retry_mode", ["none", "empty", "selection", "exhausted",
                                        "completed", "retry_completed", "motion_abort", "dryrun"])
def test_run_cleans_up_camera_failure_without_exit_home(
        setup_pick, monkeypatch, tmp_path, capsys, retry_mode):
    from efg import gripper as gripper_module

    p = setup_pick
    clock = [0.]
    monkeypatch.setattr(demo.time, "perf_counter", lambda: clock[0])

    def advance(seconds):
        clock[0] += seconds

    cfg = p.cfg | {
        "_config_path": "mock.yaml", "scan_roi_px": [0, 0, 2, 2],
        "detector": {"name": "foundgrasp"},
        "calib": {"hand_eye": "mock.yaml"},
        "output": {"dir": str(tmp_path), "save_cycles": False},
        "retry": {"exclude_radius_m": 0.02, "max_fail_streak": 3},
        "postprocess": {"roi_depth_mask": False},
    }
    grasp = SimpleNamespace(translation=np.array([0.0, 0.0, 1.0]))
    grasps = [grasp]
    detector = Mock(last_refiner_ms=None)
    detector.detect.return_value = grasps
    detector.detect.side_effect = lambda *a: advance(1.) or grasps
    detector.describe.return_value = "fake detector"
    viz = Mock()
    stop = Mock(requested=False)
    p.robot.get_flange_pose.return_value = np.eye(4), None
    p.gripper.connect.return_value = p.gripper
    home = Mock(side_effect=lambda *a: advance(2.))
    p.stream.start.side_effect = lambda: advance(100.)  # excluded setup time

    monkeypatch.setattr(demo, "CameraStream", Mock(return_value=p.stream))
    monkeypatch.setattr(demo, "build_detector", Mock(return_value=detector))
    monkeypatch.setattr(demo, "connect_robot", Mock(return_value=p.robot))
    monkeypatch.setattr(gripper_module, "GripperCtl", Mock(return_value=p.gripper))
    monkeypatch.setattr(demo, "RerunViz", Mock(return_value=viz))
    monkeypatch.setattr(demo, "StopRequest", Mock(return_value=stop))
    monkeypatch.setattr(demo, "load_hand_eye", Mock(return_value=np.eye(4)))
    monkeypatch.setattr(demo, "load_payload_check", Mock(return_value=None))
    monkeypatch.setattr(demo, "go_home", home)
    monkeypatch.setattr(demo.time, "sleep", Mock())
    monkeypatch.setattr(demo, "resolve_intrinsics", lambda K, cfg: K)
    monkeypatch.setattr(demo, "shared_filter", Mock(return_value=(
        grasps, {"detected": 1, "width_ok": 1, "collision_free": 1})))
    monkeypatch.setattr(demo, "filter_roi", lambda gg, *args: gg)
    monkeypatch.setattr(demo, "exclude_near", lambda gg, *args: gg)
    monkeypatch.setattr(demo, "select_grasp", Mock(return_value=SimpleNamespace(
        selected=p.selected, rejected=[])))
    monkeypatch.setattr(demo, "gripper_segments_px", Mock(return_value=[]))
    monkeypatch.setattr(demo, "project_px", Mock(return_value=np.array([[0.0, 0.0]])))

    def disconnect_and_accept(*args, **kwargs):
        assert kwargs["stream"] is p.stream
        p.stream.failed = True
        return True

    monkeypatch.setattr(demo, "confirm", disconnect_and_accept)
    waits = Mock(return_value=False)
    monkeypatch.setattr(demo, "wait_user", waits)
    if retry_mode in ("empty", "exhausted", "retry_completed"):
        count = [0]
        def filter_retry(*args):
            count[0] += 1
            pool = [] if retry_mode == "exhausted" or count[0] <= 2 else grasps
            return pool, {"detected": len(pool), "width_ok": len(pool), "collision_free": len(pool)}
        monkeypatch.setattr(demo, "shared_filter", filter_retry)
    if retry_mode == "selection":
        rejected = SimpleNamespace(selected=None, rejected=[(0, "IK")], summary=lambda: "IK")
        monkeypatch.setattr(demo, "select_grasp", Mock(side_effect=[
            rejected, rejected, SimpleNamespace(selected=p.selected, rejected=[])]))
    if retry_mode in ("completed", "retry_completed", "motion_abort"):
        def pick(*args, **kwargs):
            advance(3.)
            p.stream.quit_requested = True
            return ("abort", "mock stop") if retry_mode == "motion_abort" else ("success", None)
        monkeypatch.setattr(demo, "pick_and_place", pick)
    if retry_mode == "dryrun":
        monkeypatch.setattr(demo, "_home_flange_pose", lambda cfg: (np.eye(4), np.zeros(7)))
    stats = demo.run(cfg, auto=False, dry_run=retry_mode == "dryrun")
    if retry_mode in ("completed", "retry_completed", "motion_abort", "dryrun"):
        output = capsys.readouterr().out
        if retry_mode in ("completed", "retry_completed"):
            seconds = 8 if retry_mode == "retry_completed" else 6
            assert f"평균 {seconds:.2f}s" in output
            assert "완료 1회" in output
            assert stats["success"] == 1
            # One result per cycle; final statistics repeat only the aggregate.
            assert sum(line.lstrip().startswith("결과 ") for line in output.splitlines()) == 1
        else:
            assert "소요시간" in output
            assert "완료 사이클 없음" in output
            assert "평균" not in output
            assert stats["success"] == 0
        return
    if retry_mode == "exhausted":
        output = capsys.readouterr().out
        assert "소요시간" in output and "완료 사이클 없음" in output
        assert "평균" not in output
        assert count[0] == 4
        assert p.stream.snapshot.call_count == 5  # warmup + 4 attempts
        waits.assert_called_once()
        p.gripper.grip.assert_not_called()
        assert home.call_count == 2  # initial scan and normal exit
        assert stats["cycles"] == 1
        return
    waits.assert_not_called()
    if retry_mode != "none":
        assert p.stream.snapshot.call_args.kwargs == {"fresh": True}
        assert p.stream.snapshot.call_count == 4  # warmup + 3 attempts


    assert stats == {"cycles": 1, "success": 0, "fail": 0, "empty": 0}
    output = capsys.readouterr().out
    assert "test camera disconnected" in output
    assert "소요시간" in output and "완료 사이클 없음" in output
    home.assert_called_once_with(p.robot, cfg)  # initial scan home only
    assert_only_pregrasp(p)
    stop.restore.assert_called_once_with()
    viz.close.assert_called_once_with()
    detector.close.assert_called_once_with()
    p.stream.stop.assert_called_once_with()
    p.gripper.disconnect.assert_called_once_with()
    p.robot.disconnect.assert_called_once_with()
