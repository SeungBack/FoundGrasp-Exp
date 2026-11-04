"""Stopped Cartesian/joint handoff with a simulated controller only."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from efg import motion
from efg.camera import CameraError
from efg.transfer_plan import prepare_transfer


@pytest.fixture
def scene(monkeypatch):
    clock = [0.]
    monkeypatch.setattr(motion.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(motion.time, 'perf_counter', lambda: clock[0])
    monkeypatch.setattr(motion.time, 'sleep', lambda dt: clock.__setitem__(0, clock[0] + dt))
    start = np.eye(4)
    start[:3, 3] = [.4, -.2, .1]
    pre, lift = start.copy(), start.copy()
    pre[2, 3], lift[2, 3] = .2, .3
    s = SimpleNamespace(clock=clock, writes=[], events=[], responses=[], hook=None,
                        start=start, pre=pre, lift=lift, upper=None,
                        upper_joints=np.full(7, 5.), arrival_at=0.)
    robot = s.robot = Mock()
    robot.T_flange_tcp = np.eye(4)
    robot.R_home_tcp = np.eye(3)
    robot.flange_ik_joints_deg.return_value = [0.] * 7
    robot.get_joints_deg.side_effect = lambda: (
        s.upper_joints.copy() if len(actual(s)) >= 2 else np.zeros(7))
    robot.get_tcp_pose.side_effect = lambda: (
        (s.upper if s.clock[0] >= s.arrival_at else s.pre)
        if len(actual(s)) >= 2 else s.start, None)
    arm = robot.arm
    arm.version_number = [2, 7, 1]
    arm.only_check_result = 0
    arm.mode = 0
    arm.get_state.return_value = 0, 2
    arm.get_cmdnum.return_value = 0, 0
    arm.get_err_warn_code.return_value = 0, [0, 0]
    arm.get_servo_angle.return_value = 0, [10.] * 7
    arm.get_position.return_value = 0, [400., -200., 360., 0., 0., 0.]
    robot.stop_motion.side_effect = lambda: s.events.append('stop')

    def write(kind, *args, **kwargs):
        s.writes.append((kind, args, kwargs))
        if kwargs['only_check_type']:
            code, result = s.responses.pop(0) if s.responses else (0, 0)
            arm.only_check_result = result
        else:
            code = 0
        if s.hook:
            s.hook(kind, kwargs)
        return code

    arm.set_position.side_effect = lambda *a, **kw: write('cartesian', *a, **kw)
    arm.set_servo_angle.side_effect = lambda *a, **kw: write('joint', *a, **kw)
    s.cfg = dict(lift_speed_mm_s=600, retreat_speed_mm_s=600,
                 lift_acc_mm_s2=700, reorient_speed_mm_s=600,
                 reorient_acc_mm_s2=700, reorient_angular_speed_deg_s=180,
                 retreat_blend_radius_mm=10)
    s.drop = dict(joints_deg=[10.] * 7, joint_speed_deg_s=120,
                  joint_acc_deg_s2=200, transfer_blend_radius_mm=0,
                  transfer_clearance_margin_mm=10, via_height_m=.35,
                  transfer_confirm_s=.04, arrival_confirm_s=.04, transfer_timeout_s=.3)
    return s


def actual(s):
    return [write for write in s.writes if write[2]['only_check_type'] == 0]


def prepare(s):
    plan = prepare_transfer(s.robot, s.pre, s.lift, s.cfg, s.drop, j7_limits=(-89., 90.))
    s.upper = plan.upper
    return plan


def execute(s, plan, **kwargs):
    return motion.lift_and_transfer_blended(s.robot, plan, s.cfg, s.drop, **kwargs)


@pytest.mark.parametrize('overspeed', [False, True])
def test_stopped_plan_is_checked_and_final_joint_rechecked_from_actual_upper(scene, overspeed):
    s = scene
    if overspeed:
        s.responses = [(1, 24)]
    plan = prepare(s)
    assert not plan.continuous
    assert plan.speed_scale == (.8 if overspeed else 1.)
    assert [call[2]['only_check_type'] for call in s.writes[-3:]] == [1, 3, 2]
    accepted = s.writes[-3:]
    s.writes.clear()
    s.arrival_at = .06
    checked_starts = []

    def observe(kind, kwargs):
        if kind == 'joint':
            assert s.clock[0] >= s.arrival_at + s.drop['transfer_confirm_s'] - 1e-9
            if kwargs['only_check_type']:
                assert len(actual(s)) == 2
                checked_starts.append(s.robot.get_joints_deg())

    s.hook = observe
    stages, metrics = [], {}
    assert execute(s, plan, on_stage=stages.append, metrics=metrics)
    assert [(k, kw['only_check_type']) for k, _, kw in s.writes] == [
        ('cartesian', 0), ('cartesian', 0), ('joint', 1), ('joint', 0)]
    np.testing.assert_array_equal(checked_starts, [np.full(7, 5.)])
    for checked, executed in zip(accepted, actual(s)):
        assert checked[:2] == executed[:2]
        assert {k: v for k, v in checked[2].items() if k != 'only_check_type'} == {
            k: v for k, v in executed[2].items() if k != 'only_check_type'}
    assert s.writes[-1][2]['radius'] == -1
    assert stages == ['Preparing joint transfer', 'Joint transfer']
    assert metrics['joint_speed_scale'] == 1.
    assert metrics['joint_speed_deg_s'] == plan.commands[-1].speed
    assert 'failure_stage' not in metrics
    s.robot.arm.set_only_check_type.assert_not_called()


def test_no_joint_check_or_command_before_upper_arrival(scene):
    s = scene
    plan = prepare(s)
    s.writes.clear()
    s.arrival_at = 1.
    assert not execute(s, plan)
    assert [call[0] for call in s.writes] == ['cartesian', 'cartesian']
    s.robot.stop_motion.assert_called_once()


def test_multi_waypoint_route_checks_joint_phase_after_all_cartesian_steps(scene):
    from dataclasses import replace
    from efg.se3 import se3_to_xarm_pose

    s = scene
    plan = prepare(s)
    middle = (plan.pre + plan.upper) / 2
    intermediate = replace(plan.commands[0], target=tuple(se3_to_xarm_pose(middle)),
                           radius=5., label='상승 중간 경유점')
    plan = replace(plan, commands=(plan.commands[0], intermediate, *plan.commands[1:]),
                   nominal_commands=(plan.nominal_commands[0], intermediate,
                                     *plan.nominal_commands[1:]))
    s.writes.clear()
    s.robot.get_joints_deg.side_effect = lambda: (
        s.upper_joints.copy() if len(actual(s)) >= 3 else np.zeros(7))
    s.robot.get_tcp_pose.side_effect = lambda: (
        s.upper if len(actual(s)) >= 3 else s.start, None)

    def observe(kind, kwargs):
        if kind == 'joint' and kwargs['only_check_type']:
            assert len(actual(s)) == 3
            assert s.clock[0] >= s.drop['transfer_confirm_s']

    s.hook = observe
    assert execute(s, plan)
    assert [(kind, kw['only_check_type']) for kind, _, kw in s.writes] == [
        ('cartesian', 0), ('cartesian', 0), ('cartesian', 0), ('joint', 1), ('joint', 0)]
    for command, (kind, args, kwargs) in zip(plan.commands, actual(s)):
        assert kind == command.kind
        if kind == 'cartesian':
            assert args == command.target
        assert kwargs['speed'] == command.speed
        assert kwargs['mvacc'] == command.acceleration
        assert kwargs['radius'] == command.radius


@pytest.mark.parametrize('outside', [-89.5, 90.5])
def test_measured_j7_crossing_during_cartesian_rise_stops_before_upper(
        scene, capsys, outside):
    s = scene
    plan = prepare(s)
    s.writes.clear()
    s.arrival_at = .2
    s.robot.arm.get_state.side_effect = lambda: (
        0, 1 if len(actual(s)) == 2 and s.clock[0] < s.arrival_at else 2)
    s.robot.arm.get_servo_angle.side_effect = lambda **kw: (
        0, [0.] * 6 + [outside if s.clock[0] >= .06 else 0.])
    assert not execute(s, plan)
    assert .06 <= s.clock[0] < s.arrival_at
    assert [kind for kind, _, _ in s.writes] == ['cartesian', 'cartesian']
    s.robot.stop_motion.assert_called_once()
    s.robot.arm.clean_error.assert_not_called()
    output = capsys.readouterr().out
    assert f'상승 중 실측 J7 {outside:.2f}° 제한 [-89, 90]° 초과' in output


@pytest.mark.parametrize('reply', [
    (9, [0.] * 7), (0, None), (0, [0.] * 6),
    (0, [0.] * 6 + [float('nan')]), (0, ['bad'] * 7),
])
def test_unreadable_measured_joints_during_rise_block_joint_transfer(scene, capsys, reply):
    s = scene
    plan = prepare(s)
    s.writes.clear()
    s.arrival_at = .2
    s.robot.arm.get_servo_angle.return_value = reply
    assert not execute(s, plan)
    assert s.clock[0] < s.arrival_at
    assert [kind for kind, _, _ in s.writes] == ['cartesian', 'cartesian']
    s.robot.stop_motion.assert_called_once()
    assert '상승 중 실제 관절 판독 오류' in capsys.readouterr().out


@pytest.mark.parametrize('boundary', [-89., 90.])
def test_valid_j7_boundary_samples_can_finish_cartesian_rise(scene, boundary):
    s = scene
    plan = prepare(s)
    s.writes.clear()
    s.arrival_at = .06
    s.robot.arm.get_servo_angle.side_effect = lambda **kw: (
        0, [0.] * 6 + [boundary] if len(actual(s)) < 3 else [10.] * 7)
    assert execute(s, plan)
    assert [kind for kind, _, _ in actual(s)] == ['cartesian', 'cartesian', 'joint']
    s.robot.stop_motion.assert_not_called()


@pytest.mark.parametrize('failure', ['rejected', 'j7', 'state', 'drift', 'camera'])
def test_actual_upper_failure_blocks_joint_execution(scene, failure):
    s = scene
    plan = prepare(s)
    s.writes.clear()
    if failure == 'rejected':
        s.responses = [(1, 24)] * 4  # Bounded retries exhaust without any physical joint move.
    elif failure == 'j7':
        s.upper_joints[-1] = 91.

    def changed(kind, kwargs):
        if kind == 'joint' and kwargs['only_check_type']:
            if failure == 'state':
                s.robot.arm.get_state.return_value = 0, 4
            elif failure == 'drift':
                s.upper_joints[0] += 1.

    s.hook = changed

    def check():
        if failure == 'camera' and len(actual(s)) == 2:
            raise CameraError('camera failed at upper handoff')

    if failure == 'camera':
        with pytest.raises(CameraError):
            execute(s, plan, check=check)
    else:
        assert not execute(s, plan, check=check)
    assert len(actual(s)) == 2
    assert all(call[0] == 'cartesian' for call in actual(s))
    assert sum(k == 'joint' for k, _, _ in s.writes) <= (4 if failure == 'rejected' else 1)
    s.robot.stop_motion.assert_called_once()
    s.robot.arm.clean_error.assert_not_called()


@pytest.mark.parametrize('initial_overspeed', [False, True])
def test_upper_overspeed_scales_only_unsent_joint_command(scene, initial_overspeed):
    s = scene
    if initial_overspeed:
        s.responses = [(1, 24)]
    plan = prepare(s)
    original = tuple(plan.commands)
    s.writes.clear()
    s.responses = [(1, 24)]
    stages, metrics = [], {}

    def slow_check(kind, kwargs):
        if kind == 'joint' and kwargs['only_check_type']:
            assert len(actual(s)) == 2
            assert stages == ['Preparing joint transfer']
            s.clock[0] += .01

    s.hook = slow_check
    assert execute(s, plan, on_stage=stages.append, metrics=metrics)
    assert [(kind, kw['only_check_type']) for kind, _, kw in s.writes] == [
        ('cartesian', 0), ('cartesian', 0), ('joint', 1), ('joint', 1), ('joint', 0)]
    assert plan.commands == original
    for call, command in zip(actual(s)[:2], original[:2]):
        assert call[2]['speed'] == command.speed
        assert call[2]['mvacc'] == command.acceleration
        assert call[2]['radius'] == command.radius
    first, accepted, executed = s.writes[-3:]
    assert first[2]['speed'] == original[-1].speed
    assert accepted[2]['speed'] == pytest.approx(original[-1].speed * .8)
    assert accepted[2]['mvacc'] == pytest.approx(original[-1].acceleration * .64)
    assert accepted[2]['angle'] == executed[2]['angle'] == list(original[-1].target)
    assert {k: v for k, v in accepted[2].items() if k != 'only_check_type'} == {
        k: v for k, v in executed[2].items() if k != 'only_check_type'}
    assert stages == ['Preparing joint transfer', 'Joint transfer']
    assert metrics['joint_preflight_s'] == pytest.approx(.02)
    assert metrics['joint_speed_scale'] == .8
    assert metrics['joint_speed_deg_s'] == pytest.approx(original[-1].speed * .8)
    assert metrics['joint_acc_deg_s2'] == pytest.approx(original[-1].acceleration * .64)
    s.robot.stop_motion.assert_not_called()


def test_upper_retiming_budget_has_no_physical_retry_or_stale_c24_diagnostic(scene):
    s = scene
    plan = prepare(s)
    s.writes.clear()
    s.responses = [(1, 24)] * 4
    stages, metrics = [], {}
    assert not execute(s, plan, on_stage=stages.append, metrics=metrics)
    checks = [kw for kind, _, kw in s.writes if kind == 'joint']
    assert [kw['speed'] for kw in checks] == pytest.approx([120, 96, 72, 60])
    assert [kw['mvacc'] for kw in checks] == pytest.approx([200, 128, 72, 50])
    assert [kw['only_check_type'] for kw in checks] == [1] * 4
    assert len(actual(s)) == 2
    assert stages == ['Preparing joint transfer']
    assert metrics['failure_stage'] == '조인트 이송 사전 검사'
    s.robot.arm.get_c24_error_info.assert_not_called()
    s.robot.arm.clean_error.assert_not_called()


@pytest.mark.parametrize('upper_overspeed', [False, True])
def test_upper_rechecks_original_joint_limits_without_compounding_preview_reduction(
        scene, upper_overspeed):
    s = scene
    # Only the preview's joint phase fails. Its accepted 96 deg/s must not
    # become the ceiling for the later check at the actual stopped upper pose.
    s.responses = [(0, 0), (0, 0), (1, 24)]
    plan = prepare(s)
    assert plan.commands[-1].speed == pytest.approx(96.)
    assert plan.nominal_commands[-1].speed == 120.
    accepted_cartesian = plan.commands[:-1]
    s.writes.clear()
    s.responses = [(1, 24)] if upper_overspeed else []
    s.drop['joint_speed_deg_s'] = 999.  # Immutable original snapshot is authoritative.
    s.drop['joint_acc_deg_s2'] = 9999.
    metrics = {}
    assert execute(s, plan, metrics=metrics)
    joint_checks = [kw for kind, _, kw in s.writes if kind == 'joint' and kw['only_check_type']]
    assert [kw['speed'] for kw in joint_checks] == ([120., 96.] if upper_overspeed else [120.])
    assert metrics['joint_speed_deg_s'] == (96. if upper_overspeed else 120.)
    assert metrics['joint_acc_deg_s2'] == pytest.approx(128. if upper_overspeed else 200.)
    for call, command in zip(actual(s)[:-1], accepted_cartesian):
        assert call[2]['speed'] == command.speed
        assert call[2]['mvacc'] == command.acceleration
    assert actual(s)[-1][2]['speed'] == joint_checks[-1]['speed']
    assert actual(s)[-1][2]['mvacc'] == joint_checks[-1]['mvacc']
    assert plan.commands[-1].speed == pytest.approx(96.)


def test_continuous_transfer_executes_accepted_limits_without_resetting_nominal(scene):
    s = scene
    s.drop['transfer_blend_radius_mm'] = 10.
    s.responses = [(1, 24)]
    plan = prepare(s)
    assert plan.continuous
    assert plan.commands[-1].speed < plan.nominal_commands[-1].speed
    s.writes.clear()
    assert execute(s, plan)
    assert len(s.writes) == len(plan.commands)
    for call, command in zip(s.writes, plan.commands):
        assert call[2]['only_check_type'] == 0
        assert call[2]['speed'] == command.speed
        assert call[2]['mvacc'] == command.acceleration


@pytest.mark.parametrize('change', ['state', 'fault', 'queue', 'drift', 'camera'])
def test_post_check_stage_callback_cannot_invalidate_joint_start_before_submit(scene, change):
    s = scene
    plan = prepare(s)
    s.writes.clear()
    invalidated = [False]

    def stage(name):
        if name != 'Joint transfer':
            return
        invalidated[0] = True
        if change == 'state':
            s.robot.arm.get_state.return_value = 0, 4
        elif change == 'fault':
            s.robot.arm.get_err_warn_code.return_value = 0, [31, 0]
        elif change == 'queue':
            s.robot.arm.get_cmdnum.return_value = 0, 1
        elif change == 'drift':
            s.upper_joints[0] += 1.

    def check():
        if invalidated[0] and change == 'camera':
            raise CameraError('camera failed after accepted joint check')

    if change == 'camera':
        with pytest.raises(CameraError):
            execute(s, plan, on_stage=stage, check=check)
    else:
        assert not execute(s, plan, on_stage=stage, check=check)
    assert invalidated[0]
    assert [kind for kind, _, _ in actual(s)] == ['cartesian', 'cartesian']
    assert sum(kind == 'joint' for kind, _, _ in s.writes) == 1  # Native check only.
    s.robot.stop_motion.assert_called_once()
    s.robot.arm.clean_error.assert_not_called()


@pytest.mark.parametrize('code,result', [(1, 21), (1, 23), (1, 31), (9, 24), (0, 24)])
def test_upper_non_speed_failure_never_retries(scene, code, result):
    s = scene
    plan = prepare(s)
    s.writes.clear()
    s.responses = [(code, result)]
    stages = []
    assert not execute(s, plan, on_stage=stages.append)
    assert len(actual(s)) == 2
    assert sum(kind == 'joint' for kind, _, _ in s.writes) == 1
    assert stages == ['Preparing joint transfer']
    s.robot.arm.get_c24_error_info.assert_not_called()
    s.robot.arm.get_c31_error_info.assert_not_called()


@pytest.mark.parametrize('change', ['state', 'fault', 'queue', 'mode', 'drift', 'camera'])
def test_upper_guard_change_after_overspeed_blocks_another_attempt(scene, change):
    s = scene
    plan = prepare(s)
    s.writes.clear()
    s.responses = [(1, 24)]
    rejected = [False]

    def invalidate(kind, kwargs):
        if kind != 'joint' or not kwargs['only_check_type']:
            return
        rejected[0] = True
        if change == 'state':
            s.robot.arm.get_state.return_value = 0, 4
        elif change == 'fault':
            s.robot.arm.get_err_warn_code.return_value = 0, [31, 0]
        elif change == 'queue':
            s.robot.arm.get_cmdnum.return_value = 0, 1
        elif change == 'mode':
            s.robot.arm.mode = 1
        elif change == 'drift':
            s.upper_joints[0] += 1.

    def check():
        if change == 'camera' and rejected[0]:
            raise CameraError('camera failed between stationary joint checks')

    s.hook = invalidate
    if change == 'camera':
        with pytest.raises(CameraError):
            execute(s, plan, check=check)
    else:
        assert not execute(s, plan, check=check)
    assert len(actual(s)) == 2
    assert sum(kind == 'joint' for kind, _, _ in s.writes) == 1
    s.robot.stop_motion.assert_called_once()


@pytest.mark.parametrize('fault_after', [2, 3])
@pytest.mark.parametrize('diagnostic_fails', [False, True])
def test_native_c24_diagnostic_precedes_stop_and_never_masks_failure(
        scene, capsys, fault_after, diagnostic_fails):
    s = scene
    plan = prepare(s)
    s.writes.clear()
    s.robot.arm.get_state.side_effect = lambda: (
        0, 4 if len(actual(s)) >= fault_after else 2)
    s.robot.arm.get_err_warn_code.side_effect = lambda: (
        0, [24, 14] if len(actual(s)) >= fault_after else [0, 0])

    def diagnostic(**kwargs):
        s.events.append('diagnostic')
        assert kwargs == {'is_radian': False}
        if diagnostic_fails:
            raise RuntimeError('diagnostic disconnected')
        return 0, [3, 234.5]

    s.robot.arm.get_c24_error_info.side_effect = diagnostic
    assert not execute(s, plan)
    assert len(actual(s)) == fault_after
    assert s.events == ['diagnostic', 'stop']
    assert sum(kind == 'joint' and bool(kw['only_check_type'])
               for kind, _, kw in s.writes) == int(fault_after == 3)
    s.robot.arm.get_c24_error_info.assert_called_once_with(is_radian=False)
    output = capsys.readouterr().out
    assert ('상승 정지 확인 실패' if fault_after == 2 else '이송 중 로봇 오류/정지') in output
    assert '24' in output
    if diagnostic_fails:
        assert 'diagnostic disconnected' in output
    else:
        assert 'servo_id=3' in output and '234.500°/s' in output


@pytest.mark.parametrize('fault_after', [2, 3])
@pytest.mark.parametrize('reply', ['ok', 'raises', 'error_code', 'short', 'none', 'nan'])
def test_native_c31_diagnostic_precedes_stop_and_never_masks_failure(
        scene, capsys, fault_after, reply):
    s = scene
    plan = prepare(s)
    s.writes.clear()
    s.robot.arm.get_state.side_effect = lambda: (
        0, 5 if len(actual(s)) >= fault_after else 2)
    s.robot.arm.get_err_warn_code.side_effect = lambda: (
        0, [31, 14] if len(actual(s)) >= fault_after else [0, 0])

    def diagnostic():
        s.events.append('diagnostic')
        if reply == 'raises':
            raise RuntimeError('diagnostic disconnected')
        return {
            'ok': (0, [2, 4.5, 8.25]),
            'error_code': (1, [2, 4.5, 8.25]),
            'short': (0, [2]),
            'none': (0, None),
            'nan': (0, [2, float('nan'), 8.25]),
        }[reply]

    s.robot.arm.get_c31_error_info.side_effect = diagnostic
    assert not execute(s, plan)
    assert len(actual(s)) == fault_after
    assert s.events == ['diagnostic', 'stop']
    s.robot.arm.get_c31_error_info.assert_called_once_with()
    s.robot.arm.get_c24_error_info.assert_not_called()
    s.robot.arm.clean_error.assert_not_called()
    output = capsys.readouterr().out
    assert ('상승 정지 확인 실패' if fault_after == 2 else '이송 중 로봇 오류/정지') in output
    assert 'C31: 로봇 관절 전류 기반 충돌 감지' in output
    if reply == 'ok':
        assert 'servo_id=2' in output
        assert 'theoretical_tau=4.500' in output and 'actual_tau=8.250' in output
        assert '(SDK 원시값)' in output
    else:
        assert 'C31 상세 조회 실패' in output
