"""카메라 단절 시 접근 차단·정리 검증 — 장비와 사용자 입력은 모두 모의."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from efg import demo, perception
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
    robot.arm.get_state.return_value = 0, 2
    robot.arm.mode = 0
    robot.arm.version_number = [2, 7, 1]
    robot.arm.only_check_result = 0
    robot.arm.set_position.return_value = 0
    robot.arm.set_servo_angle.return_value = 0
    robot.arm.get_cmdnum.return_value = 0, 0
    robot.arm.get_err_warn_code.return_value = 0, [0, 0]
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
    robot.get_tcp_pose.return_value = selected.T_tcp_grasp, None
    robot.get_joints_deg.return_value = [0.] * 7
    robot.flange_ik_joints_deg.return_value = [0.] * 7
    cfg = {
        "robot": {"travel_speed_mm_s": 100, "approach_speed_mm_s": 20,
                  "lift_speed_mm_s": 100, "approach_acc_mm_s2": 100},
        "drop": {"joints_deg": [0.0] * 7, "joint_speed_deg_s": 120},
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


@pytest.mark.parametrize('recovery_ok', [True, False])
def test_clean_late_plan_rejection_returns_before_close_and_reselects(
        setup_pick, monkeypatch, recovery_ok):
    from efg.motion_command import PlanRejected
    p = setup_pick
    p.cfg['selection'] = {'j7_limit_deg': [-89, 90]}
    planned = SimpleNamespace(check_start=Mock(), approach_commands=('pre', 'grasp'),
                              transfer_route=object(), speed_scale=1.)
    p.selected.pick_plan = planned
    approach = Mock(return_value=True)
    transfer = Mock(side_effect=PlanRejected('native check C21; all routes unavailable'))
    recovery = Mock(return_value=recovery_ok)
    monkeypatch.setattr(demo, 'approach_blended', approach)
    monkeypatch.setattr(demo, 'prepare_transfer', transfer)
    monkeypatch.setattr(demo, 'recover_unclosed_pick', recovery)
    outcome, _ = demo.pick_and_place(p.robot, p.gripper, p.selected, p.cfg,
                                     auto=True, stream=p.stream)
    assert outcome == ('replan' if recovery_ok else 'abort')
    assert approach.call_args.kwargs['commands'] is planned.approach_commands
    assert transfer.call_args.kwargs['preferred_route'] is planned.transfer_route
    recovery.assert_called_once()
    p.gripper.grip.assert_not_called()
    p.gripper.release.assert_not_called()


def test_actual_fault_after_approach_does_not_attempt_automatic_recovery(setup_pick, monkeypatch):
    p = setup_pick
    p.cfg['selection'] = {'j7_limit_deg': [-89, 90]}
    p.selected.pick_plan = SimpleNamespace(check_start=Mock(), approach_commands=('pre', 'grasp'),
                                          transfer_route=object(), speed_scale=1.)
    monkeypatch.setattr(demo, 'approach_blended', Mock(return_value=True))
    monkeypatch.setattr(demo, 'prepare_transfer', Mock(side_effect=RuntimeError('actual C31')))
    recovery = Mock()
    monkeypatch.setattr(demo, 'recover_unclosed_pick', recovery)
    assert demo.pick_and_place(p.robot, p.gripper, p.selected, p.cfg,
                               auto=True, stream=p.stream)[0] == 'abort'
    recovery.assert_not_called()
    p.gripper.grip.assert_not_called()


@pytest.mark.parametrize('actual_j7,expected,recover', [
    (88.514187, 'replan', True), (90.067247, 'abort', False),
])
def test_measured_grasp_reserve_is_checked_before_closing(
        setup_pick, monkeypatch, actual_j7, expected, recover):
    p = setup_pick
    p.cfg['selection'] = {'j7_limit_deg': [-89., 90.], 'j7_selection_margin_deg': 10.}
    p.selected.pick_plan = SimpleNamespace(
        check_start=Mock(), approach_commands=('pre', 'grasp'),
        transfer_route=object(), speed_scale=1.)
    p.robot.get_joints_deg.return_value = [0.] * 6 + [actual_j7]
    # Keep real prepare_transfer: this verifies propagation of the planning
    # reserve from the application, not only a mocked rejection branch.
    monkeypatch.setattr(demo, 'approach_blended', Mock(return_value=True))
    recovery = Mock(return_value=True)
    monkeypatch.setattr(demo, 'recover_unclosed_pick', recovery)
    result, _ = demo.pick_and_place(p.robot, p.gripper, p.selected, p.cfg,
                                     auto=True, stream=p.stream)
    assert result == expected
    assert recovery.call_count == int(recover)
    if recover:
        assert recovery.call_args.kwargs['j7_limits'] == [-89., 90.]
    p.gripper.grip.assert_not_called()
    p.gripper.release.assert_not_called()
    p.robot.arm.set_position.assert_not_called()
    p.robot.arm.set_servo_angle.assert_not_called()
    p.robot.arm.clean_error.assert_not_called()


@pytest.mark.parametrize('failure', ['c31', 'state', 'mode', 'joint_limit',
                                    'joint_read', 'joint_nan'])
def test_preopen_monitors_moving_robot_and_never_queues_approach_on_fault(
        setup_pick, monkeypatch, failure):
    from efg.motion import approach_blended
    from efg.motion_command import approach_commands

    p = setup_pick
    p.cfg['selection'] = {'j7_limit_deg': [-89., 90.]}
    start = p.selected.T_tcp_pre.copy()
    start[2, 3] += .2
    p.selected.pick_plan = SimpleNamespace(
        check_start=Mock(), transfer_route=object(), speed_scale=1.,
        approach_commands=approach_commands(start, p.selected.T_tcp_pre,
                                            p.selected.T_tcp_grasp, p.cfg['robot']))
    p.robot.arm.get_state.return_value = 0, 1  # pre-grasp is still moving
    p.robot.arm.get_cmdnum.return_value = 0, 1
    p.robot.arm.get_servo_angle.return_value = 0, [0.] * 7
    p.robot.arm.get_c31_error_info.return_value = 0, [4, 17., -7.]
    monkeypatch.setattr(demo, 'approach_blended', approach_blended)
    prepare_transfer = Mock(side_effect=AssertionError('must stop before final approach'))
    monkeypatch.setattr(demo, 'prepare_transfer', prepare_transfer)

    def opening(width, *, check):
        assert p.robot.arm.set_position.call_count == 1
        check()  # moving state and queued pre-grasp are allowed
        if failure == 'c31':
            p.robot.arm.get_err_warn_code.return_value = 0, [31, 14]
        elif failure == 'state':
            p.robot.arm.get_state.return_value = 0, 5
        elif failure == 'mode':
            p.robot.arm.mode = 1
        elif failure == 'joint_limit':
            p.robot.arm.get_servo_angle.return_value = 0, [0.] * 6 + [91.]
        elif failure == 'joint_read':
            p.robot.arm.get_servo_angle.return_value = 9, [0.] * 7
        else:
            p.robot.arm.get_servo_angle.return_value = 0, [0.] * 6 + [float('nan')]
        check()
        raise AssertionError('fault must interrupt width adjustment')

    p.gripper.pre_open.side_effect = opening
    assert demo.pick_and_place(p.robot, p.gripper, p.selected, p.cfg, auto=True,
                               stream=p.stream) == ('abort', '연속 접근 이동')
    assert p.robot.arm.set_position.call_count == 1
    p.robot.stop_motion.assert_called_once()
    p.robot.arm.clean_error.assert_not_called()
    prepare_transfer.assert_not_called()
    p.gripper.grip.assert_not_called()


def test_no_candidate_automatic_wait_is_interruptible_without_user_input(monkeypatch):
    stream = FakeStream()
    stop = SimpleNamespace(requested=False)
    clock = [0.]
    monkeypatch.setattr(demo.time, 'monotonic', lambda: clock[0])
    def advance(seconds):
        clock[0] += seconds
        stop.requested = clock[0] >= .2
    monkeypatch.setattr(demo.time, 'sleep', advance)
    prompt = Mock(side_effect=AssertionError('automatic mode must not prompt'))
    monkeypatch.setattr(demo, 'wait_user', prompt)
    assert not demo.wait_to_rescan(stream, stop, {'retry': {'no_candidate_wait_s': 2}}, True, 'retry')
    assert clock[0] < 1
    prompt.assert_not_called()


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


@pytest.mark.parametrize('travel_acc,expected_pre_acc', [(1800., 1800.), (None, 1100.)])
def test_confirm_mode_keeps_pregrasp_acceleration_out_of_final_approach(
        setup_pick, monkeypatch, travel_acc, expected_pre_acc):
    from efg.motion_command import PlanRejected

    p = setup_pick
    p.cfg['robot'].update(travel_speed_mm_s=1000., approach_speed_mm_s=250.,
                          approach_acc_mm_s2=1100.)
    if travel_acc is not None:
        p.cfg['robot']['travel_acc_mm_s2'] = travel_acc
    monkeypatch.setattr(demo, 'confirm', Mock(return_value=True))
    # End after both approach commands, before any gripper close/transfer.
    monkeypatch.setattr(demo, 'prepare_transfer', Mock(side_effect=PlanRejected('test end')))
    assert demo.pick_and_place(p.robot, p.gripper, p.selected, p.cfg,
                               auto=False, stream=p.stream) == ('abort', '상승 경로 준비')
    assert p.move.call_count == 2
    pre_call, grasp_call = p.move.call_args_list
    assert pre_call.args[1] is p.selected.T_tcp_pre
    assert grasp_call.args[1] is p.selected.T_tcp_grasp
    assert [call.args[2] for call in p.move.call_args_list] == [1000., 250.]
    assert [call.kwargs['acc_mm_s2'] for call in p.move.call_args_list] == [expected_pre_acc, 1100.]
    p.robot.arm.set_tcp_maxacc.assert_not_called()
    p.gripper.grip.assert_not_called()


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

    def grip(**kwargs):
        assert order == ["open", "arrived"]
        order.append("close")
        return False, 0.0

    p.gripper.grip.side_effect = grip
    monkeypatch.setattr(demo, "approach_blended", approach)
    monkeypatch.setattr(demo, "confirm", Mock(side_effect=AssertionError("prompt")))
    assert demo.pick_and_place(p.robot, p.gripper, p.selected, p.cfg,
                               auto=True, stream=p.stream) == ("fail", "파지")
    assert order[:3] == ["open", "arrived", "close"]


@pytest.mark.parametrize("fault", ["collision", "state", "read", "queue", "mode"])
@pytest.mark.parametrize("during_close", [True, False])
def test_close_checks_robot_before_lifting_even_without_grasp_verdict(
        setup_pick, monkeypatch, fault, during_close):
    p = setup_pick
    monkeypatch.setattr(demo, "approach_blended", Mock(return_value=True))
    transfer = Mock()
    monkeypatch.setattr(demo, "lift_and_transfer_blended", transfer)
    p.robot.arm.get_c31_error_info.return_value = 0, [1, 2., 3.]

    def close(*, check):
        if fault == "collision":
            p.robot.arm.get_err_warn_code.return_value = 0, [31, 14]
        elif fault == "state":
            p.robot.arm.get_state.return_value = 0, 5
        elif fault == "read":
            p.robot.arm.get_err_warn_code.return_value = 9, None
        elif fault == "queue":
            p.robot.arm.get_cmdnum.return_value = 0, 1
        else:
            p.robot.arm.mode = 1
        if during_close:
            check()
        return False, 0.

    p.gripper.grip.side_effect = close
    assert demo.pick_and_place(p.robot, p.gripper, p.selected, p.cfg,
                               auto=True, stream=p.stream, grip_check=False) == (
                                   "abort", "파지 닫힘")
    transfer.assert_not_called()
    p.gripper.release.assert_not_called()
    p.robot.stop_motion.assert_called_once()
    p.gripper.stop.assert_called_once()
    p.robot.arm.clean_error.assert_not_called()


@pytest.mark.parametrize("stop_fails", [False, True])
def test_close_execution_error_is_not_an_empty_grasp(setup_pick, monkeypatch, stop_fails):
    p = setup_pick
    monkeypatch.setattr(demo, "approach_blended", Mock(return_value=True))
    transfer = Mock()
    monkeypatch.setattr(demo, "lift_and_transfer_blended", transfer)
    p.gripper.grip.side_effect = RuntimeError("close did not start")
    if stop_fails:
        p.gripper.stop.side_effect = OSError("Modbus unavailable")
    assert demo.pick_and_place(p.robot, p.gripper, p.selected, p.cfg,
                               auto=True, stream=p.stream, grip_check=False) == (
                                   "abort", "파지 닫힘")
    transfer.assert_not_called()
    p.gripper.release.assert_not_called()
    p.gripper.stop.assert_called_once()


@pytest.mark.parametrize("reason", ["camera", "stop"])
def test_close_monitors_camera_and_stop_request(setup_pick, monkeypatch, reason):
    p = setup_pick
    monkeypatch.setattr(demo, "approach_blended", Mock(return_value=True))
    transfer = Mock()
    monkeypatch.setattr(demo, "lift_and_transfer_blended", transfer)
    stop = SimpleNamespace(requested=False)

    def close(*, check):
        p.stream.failed = reason == "camera"
        stop.requested = reason == "stop"
        check()
        pytest.fail("close guard did not interrupt")

    p.gripper.grip.side_effect = close
    if reason == "camera":
        with pytest.raises(CameraError):
            demo.pick_and_place(p.robot, p.gripper, p.selected, p.cfg,
                                auto=True, stream=p.stream, stop=stop)
    else:
        assert demo.pick_and_place(p.robot, p.gripper, p.selected, p.cfg,
                                   auto=True, stream=p.stream, stop=stop) == (
                                       "abort", "파지 닫힘")
    transfer.assert_not_called()
    p.gripper.release.assert_not_called()
    p.gripper.stop.assert_called_once()


@pytest.mark.parametrize("arrived,opened,failure_stage", [
    (True, True, None), (True, False, None), (False, True, None),
    (False, True, "조인트 이송 사전 검사"),
])
def test_release_requires_completed_transfer(setup_pick, monkeypatch, arrived, opened,
                                             failure_stage):
    p = setup_pick
    p.selected.transfer_sign = -1
    p.gripper.grip.return_value = True, 40.
    p.gripper.release.return_value = opened
    monkeypatch.setattr(demo, "approach_blended", Mock(return_value=True))

    def transfer(*args, **kwargs):
        p.gripper.grip.assert_called_once()
        p.gripper.release.assert_not_called()
        kwargs["check"]()
        assert args[1].transfer_sign == -1
        checked_pose = p.robot.check_flange_ik.call_args.args[0]
        np.testing.assert_array_equal(checked_pose[:3, :3], np.diag([-1., -1., 1.]))
        if failure_stage:
            kwargs["metrics"]["failure_stage"] = failure_stage
        return arrived

    monkeypatch.setattr(demo, "lift_and_transfer_blended", transfer)
    result = demo.pick_and_place(p.robot, p.gripper, p.selected, p.cfg,
                                auto=True, stream=p.stream)
    if arrived:
        assert result == (("success", None) if opened else ("abort", "릴리즈 개방 실패"))
        p.gripper.release.assert_called_once()
        assert p.gripper.release.call_args.args == (p.selected.open_width_m,)
    else:
        assert result == ("abort", failure_stage or "상승/이송 이동")
        p.gripper.release.assert_not_called()
    p.move.assert_not_called()


@pytest.mark.parametrize("blocked", ["pre", "both_upper"])
def test_actual_grasp_state_without_return_route_never_closes(setup_pick, monkeypatch, blocked):
    p = setup_pick
    p.cfg['selection'] = {'ik_check_per_candidate': True}
    at_grasp = False

    def approach(*args, **kwargs):
        nonlocal at_grasp
        at_grasp = True
        return True

    def ik(target):
        # Selection/home checks pass, but IK differs after reaching the grasp.
        if at_grasp and (blocked == 'pre' or target[2, 3] > .3):
            return [0.] * 6 + [104.]
        return [0.] * 7

    p.robot.flange_ik_joints_deg.side_effect = ik
    monkeypatch.setattr(demo, 'approach_blended', approach)
    transfer = Mock()
    monkeypatch.setattr(demo, 'lift_and_transfer_blended', transfer)
    assert demo.pick_and_place(p.robot, p.gripper, p.selected, p.cfg,
                               auto=True, stream=p.stream) == ('abort', '상승 경로 준비')
    p.gripper.grip.assert_not_called()
    p.gripper.release.assert_not_called()
    transfer.assert_not_called()


def test_actual_state_alternate_is_prepared_before_closing_then_executed(setup_pick, monkeypatch):
    p = setup_pick
    p.cfg['selection'] = {'ik_check_per_candidate': True}
    at_grasp = False
    queries_when_closed = []

    def approach(*args, **kwargs):
        nonlocal at_grasp
        at_grasp = True
        return True

    def ik(target):
        p.gripper.grip.assert_not_called()
        if at_grasp and target[2, 3] > .3 and target[0, 0] > 0:
            return [0.] * 6 + [104.]
        return [0.] * 7

    def close(**kwargs):
        queries_when_closed.append(p.robot.flange_ik_joints_deg.call_count)
        return True, 40.

    def transfer(robot, plan, *args, **kwargs):
        assert plan.transfer_sign == -1
        np.testing.assert_array_equal(plan.pre, p.selected.T_tcp_pre)
        assert p.robot.flange_ik_joints_deg.call_count == queries_when_closed[0]
        p.gripper.grip.assert_called_once()
        return True

    p.robot.flange_ik_joints_deg.side_effect = ik
    p.gripper.grip.side_effect = close
    p.gripper.release.return_value = True
    monkeypatch.setattr(demo, 'approach_blended', approach)
    monkeypatch.setattr(demo, 'lift_and_transfer_blended', transfer)
    assert demo.pick_and_place(p.robot, p.gripper, p.selected, p.cfg,
                               auto=True, stream=p.stream) == ('success', None)


@pytest.mark.parametrize("retry_mode", ["none", "empty", "selection", "exhausted",
                                        "completed", "retry_completed", "two_completed", "motion_abort", "dryrun"])
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
    from efg.paths import import_grasp_group
    rows = np.zeros((1, 17))
    rows[:, 0] = 1.
    rows[:, 1:4] = [.04, .02, .02]
    rows[:, 4:13] = np.eye(3).reshape(9)
    rows[:, 15] = 1.
    grasps = import_grasp_group()(rows)
    detector = Mock(last_refiner_ms=None, last_inference_ms=None)
    detector.detect.return_value = grasps
    detector.detect.side_effect = lambda *a: advance(1.) or grasps
    detector.describe.return_value = "fake detector"
    viz = Mock()
    stop = Mock(requested=False)
    p.robot.get_flange_pose.return_value = np.eye(4), None
    startup_order = []
    p.gripper.connect.side_effect = lambda: startup_order.append("gripper") or p.gripper
    home = Mock(side_effect=lambda *a: startup_order.append("home") or advance(2.))
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
    monkeypatch.setattr(perception, "shared_filter", Mock(return_value=(
        grasps, {"detected": 1, "width_ok": 1, "collision_free": 1})))
    monkeypatch.setattr(perception, "filter_roi", lambda gg, *args: gg)
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
        monkeypatch.setattr(perception, "shared_filter", filter_retry)
    if retry_mode == "selection":
        rejected = SimpleNamespace(selected=None, rejected=[(0, "IK")], summary=lambda: "IK")
        monkeypatch.setattr(demo, "select_grasp", Mock(side_effect=[
            rejected, rejected, SimpleNamespace(selected=p.selected, rejected=[])]))
    if retry_mode in ("completed", "retry_completed", "two_completed", "motion_abort"):
        completed = [0]
        def pick(*args, **kwargs):
            advance(3.)
            completed[0] += 1
            p.stream.quit_requested = retry_mode != "two_completed" or completed[0] == 2
            return ("abort", "mock stop") if retry_mode == "motion_abort" else ("success", None)
        monkeypatch.setattr(demo, "pick_and_place", pick)
    if retry_mode == "dryrun":
        monkeypatch.setattr(demo, "_home_flange_pose", lambda cfg: (np.eye(4), np.zeros(7)))
    stats = demo.run(cfg, auto=False, dry_run=retry_mode == "dryrun")
    if retry_mode == "dryrun":
        assert startup_order == []
        demo.connect_robot.assert_not_called()
        gripper_module.GripperCtl.assert_not_called()
    else:
        assert startup_order[:2] == ["home", "gripper"]
    if retry_mode in ("completed", "retry_completed", "two_completed", "motion_abort", "dryrun"):
        output = capsys.readouterr().out
        if retry_mode in ("completed", "retry_completed", "two_completed"):
            seconds = {"retry_completed": 6, "completed": 4, "two_completed": 5}[retry_mode]
            count = 2 if retry_mode == "two_completed" else 1
            assert f"평균 {seconds:.2f}s" in output
            assert f"완료 {count}회" in output
            assert stats["success"] == count
            assert home.call_count == count + 1  # startup, later cycles, controlled exit
            # One result per cycle; final statistics repeat only the aggregate.
            assert sum(line.lstrip().startswith("결과 ") for line in output.splitlines()) == count
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
        assert home.call_count == 2  # startup and normal exit
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
    home.assert_called_once_with(p.robot, cfg)  # startup home only; first scan reuses it
    assert_only_pregrasp(p)
    stop.restore.assert_called_once_with()
    viz.close.assert_called_once_with()
    detector.close.assert_called_once_with()
    p.stream.stop.assert_called_once_with()
    p.gripper.disconnect.assert_called_once_with()
    p.robot.disconnect.assert_called_once_with()


@pytest.mark.parametrize("failure", ["home", "camera", "stop", "gripper"])
def test_startup_failure_never_calibrates_before_home_or_homes_on_exit(
        setup_pick, monkeypatch, tmp_path, failure):
    from efg import gripper as gripper_module

    p = setup_pick
    cfg = p.cfg | {
        "_config_path": "mock.yaml", "scan_roi_px": [0, 0, 2, 2],
        "detector": {"name": "foundgrasp", "warmup_runs": 0},
        "calib": {"hand_eye": "mock.yaml"},
        "output": {"dir": str(tmp_path)},
    }
    detector = Mock()
    viz = Mock()
    stop = Mock(requested=False)
    constructor = Mock(return_value=p.gripper)
    p.gripper.connect.side_effect = RuntimeError("gripper startup failed")

    def home(*args):
        constructor.assert_not_called()
        if failure == "home":
            raise RuntimeError("home failed")
        if failure == "camera":
            p.stream.failed = True
        if failure == "stop":
            stop.requested = True

    home = Mock(side_effect=home)
    monkeypatch.setattr(demo, "CameraStream", Mock(return_value=p.stream))
    monkeypatch.setattr(demo, "build_detector", Mock(return_value=detector))
    monkeypatch.setattr(demo, "connect_robot", Mock(return_value=p.robot))
    monkeypatch.setattr(gripper_module, "GripperCtl", constructor)
    monkeypatch.setattr(demo, "RerunViz", Mock(return_value=viz))
    monkeypatch.setattr(demo, "StopRequest", Mock(return_value=stop))
    monkeypatch.setattr(demo, "load_hand_eye", Mock(return_value=np.eye(4)))
    monkeypatch.setattr(demo, "load_payload_check", Mock(return_value=None))
    monkeypatch.setattr(demo, "go_home", home)

    if failure in ("home", "gripper"):
        with pytest.raises(RuntimeError, match="failed"):
            demo.run(cfg)
    else:
        assert demo.run(cfg)["cycles"] == 0
    home.assert_called_once_with(p.robot, cfg)
    if failure == "gripper":
        constructor.assert_called_once_with(cfg)
        p.gripper.disconnect.assert_called_once_with()
    else:
        constructor.assert_not_called()
        p.gripper.connect.assert_not_called()
        p.gripper.disconnect.assert_not_called()
    viz.start_robot_stream.assert_not_called()
    stop.restore.assert_called_once_with()
    viz.close.assert_called_once_with()
    detector.close.assert_called_once_with()
    p.stream.stop.assert_called_once_with()
    p.robot.disconnect.assert_called_once_with()
