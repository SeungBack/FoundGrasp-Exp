"""Main-loop replanning with simulated perception and robot I/O only."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from efg import demo, gripper as gripper_module
from efg.motion_command import PlanRejected
from efg.paths import import_grasp_group


@pytest.fixture
def loop(monkeypatch, tmp_path):
    rows = np.zeros((2, 17))
    rows[:, 0] = [.9, .8]
    rows[:, 1:4] = [.04, .02, .02]
    rows[:, 4:13] = np.eye(3).reshape(9)
    rows[:, 13:16] = [[.3, 0., .1], [.45, 0., .1]]
    grasps = import_grasp_group()(rows)
    s = SimpleNamespace(now=0., frame_id=0, events=[], selections=[], detections=0,
                        empty=False, reject_selection=False, stop_after=None, grasp_count=2)
    s.cfg = {
        '_config_path': 'offline.yaml', 'scan_roi_px': [0, 0, 2, 2],
        'calib': {'hand_eye': 'offline.yaml'},
        'detector': {'name': 'foundgrasp', 'warmup_runs': 0},
        'output': {'dir': str(tmp_path), 'save_cycles': False},
        'postprocess': {'roi_depth_mask': False},
        'selection': {'j7_limit_deg': [-89., 90.]},
        'robot': {'travel_speed_mm_s': 100., 'approach_speed_mm_s': 20.,
                  'lift_speed_mm_s': 100., 'approach_acc_mm_s2': 100.},
        'drop': {'joints_deg': [0.] * 7, 'joint_speed_deg_s': 120.},
        'retry': {'exclude_radius_m': .02, 'max_fail_streak': 3,
                  'max_inference_retries': 1, 'no_candidate_wait_s': .2},
    }
    s.stream = Mock(quit_requested=False)
    s.stop = Mock(requested=False)
    s.robot = Mock(T_flange_tcp=np.eye(4), R_home_tcp=np.eye(3))
    s.robot.arm.get_state.return_value = 0, 2
    s.robot.arm.get_cmdnum.return_value = 0, 0
    s.robot.arm.get_err_warn_code.return_value = 0, [0, 0]
    s.robot.arm.get_servo_angle.return_value = 0, [0.] * 7
    s.robot.arm.mode = 0
    s.robot.get_flange_pose.return_value = np.eye(4), None
    s.gripper = Mock()
    s.gripper.pre_open.return_value = True
    s.gripper.grip.return_value = True, 20.
    s.viz = Mock()
    s.detector = Mock(last_refiner_ms=None)
    s.detector.describe.return_value = 'offline detector'

    def advance(seconds):
        s.now += seconds

    def snapshot(**kwargs):
        s.frame_id += 1
        s.events.append(('capture', s.frame_id, kwargs.get('fresh', False)))
        return (np.full((2, 2, 3), s.frame_id, dtype=np.uint8),
                np.ones((2, 2), dtype=np.float32), np.eye(3))

    def detect(detector, rgb, *args):
        s.detections += 1
        s.events.append(('detect', int(rgb[0, 0, 0])))
        if s.stop_after == s.detections:
            s.stop.requested = True
        pool = grasps[:0 if s.empty else s.grasp_count]
        return SimpleNamespace(grasps=pool, detect_ms=1., post_ms=0., inference_ms=None,
                               counts=dict(detected=len(pool), score_ok=len(pool),
                                           width_ok=len(pool), collision_free=len(pool), roi=len(pool)))

    def select(pool, T_base_cam, cfg, robot=None, candidate_prepare=None):
        s.selections.append(pool.translations.copy())
        if s.reject_selection or not len(pool):
            return SimpleNamespace(selected=None, rejected=[(0, 'native C21')],
                                   summary=lambda: 'all routes rejected')
        grasp = pool[0]
        target = np.eye(4)
        target[:3, 3] = grasp.translation
        pre, lift = target.copy(), target.copy()
        pre[2, 3] += .1
        lift[2, 3] += .3
        selected = SimpleNamespace(index=0, score=grasp.score, width_m=.04,
                                   open_width_m=.04, tilt_deg=0., p_base=grasp.translation.copy(),
                                   T_tcp_pre=pre, T_tcp_grasp=target, T_tcp_lift=lift)
        assert candidate_prepare is not None
        selected = candidate_prepare(selected)
        s.events.append(('selected', selected.p_base[0]))
        return SimpleNamespace(selected=selected, rejected=[])

    def prepare_pick(robot, selected, cfg, *, check, **kwargs):
        check()
        return SimpleNamespace(check_start=Mock(), approach_commands=('pre', 'grasp'),
                               transfer_route=SimpleNamespace(name='checked route',
                                                               upper=selected.T_tcp_lift),
                               j7_range_deg=(0., 0.), speed_scale=1.)

    def approach(robot, pre, target, cfg, *, prepare, check, **kwargs):
        check()
        assert prepare()
        s.events.append(('approach', target[0, 3]))
        return True

    def release(*args, **kwargs):
        s.events.append(('release',))
        s.stream.quit_requested = True
        return True

    s.stream.snapshot.side_effect = snapshot
    s.gripper.release.side_effect = release
    s.home = Mock(side_effect=lambda *args: s.events.append(('home',)))
    s.prepare_transfer = Mock(side_effect=lambda robot, pre, lift, *args, **kwargs:
                              SimpleNamespace(transfer_sign=1, speed_scale=1., pre=pre,
                                              upper=lift, continuous=False, route_name='configured'))
    s.recovery = Mock(return_value=True)
    s.prompt = Mock(side_effect=AssertionError('automatic replanning must not prompt'))
    patches = {
        'CameraStream': Mock(return_value=s.stream), 'StopRequest': Mock(return_value=s.stop),
        'connect_robot': Mock(return_value=s.robot), 'RerunViz': Mock(return_value=s.viz),
        'build_detector': Mock(return_value=s.detector),
        'load_hand_eye': Mock(return_value=np.eye(4)), 'load_payload_check': Mock(return_value=None),
        'go_home': s.home, 'detect_frame': detect, 'select_grasp': select,
        'prepare_pick': prepare_pick, 'approach_blended': approach,
        'prepare_transfer': s.prepare_transfer, 'recover_unclosed_pick': s.recovery,
        'lift_and_transfer_blended': Mock(return_value=True),
        'gripper_segments_px': Mock(return_value=[]), 'wait_user': s.prompt,
        '_read_input': s.prompt,
    }
    for name, replacement in patches.items():
        monkeypatch.setattr(demo, name, replacement)
    monkeypatch.setattr(gripper_module, 'GripperCtl', Mock(return_value=s.gripper))
    monkeypatch.setattr(demo.time, 'monotonic', lambda: s.now)
    monkeypatch.setattr(demo.time, 'perf_counter', lambda: s.now)
    monkeypatch.setattr(demo.time, 'sleep', advance)
    return s


def test_late_rejection_returns_unclosed_then_excludes_point_and_reinfers(loop):
    s = loop
    accepted = s.prepare_transfer.side_effect
    attempts = [0]

    def prepare(*args, **kwargs):
        attempts[0] += 1
        if attempts[0] == 1:
            s.events.append(('plan_rejected',))
            raise PlanRejected('all alternative return routes rejected')
        return accepted(*args, **kwargs)

    def recover(*args, **kwargs):
        s.gripper.grip.assert_not_called()
        s.gripper.release.assert_not_called()
        kwargs['check']()
        s.events.append(('reverse_unclosed',))
        return True

    s.prepare_transfer.side_effect = prepare
    s.recovery.side_effect = recover
    stats = demo.run(s.cfg, auto=True)
    assert stats == dict(cycles=2, success=1, fail=0, empty=0)
    assert s.detections == 2
    np.testing.assert_allclose(s.selections[0][:, 0], [.3, .45])
    np.testing.assert_allclose(s.selections[1][:, 0], [.45])
    ordered = [event[0] for event in s.events if event[0] in (
        'detect', 'selected', 'approach', 'plan_rejected', 'reverse_unclosed', 'release')]
    assert ordered == ['detect', 'selected', 'approach', 'plan_rejected', 'reverse_unclosed',
                       'detect', 'selected', 'approach', 'release']
    frames = [event[1] for event in s.events if event[0] == 'detect']
    assert frames[1] > frames[0]
    captures = [event for event in s.events if event[0] == 'capture']
    assert captures[-1][2] is True  # explicit new frame after return to the scan pose
    assert s.home.call_count == 2  # startup + controlled exit; recovery already reached scan
    assert [call.args[2] for call in s.viz.log_outcome.call_args_list] == ['replan', 'success']
    s.gripper.grip.assert_called_once()
    s.recovery.assert_called_once()
    s.prompt.assert_not_called()


def test_rejected_point_expires_after_exhausted_retry_round_and_automatic_wait(loop):
    s = loop
    s.grasp_count = 1
    s.stop_after = 7  # bound the test even if a permanent blacklist prevents progress
    accepted = s.prepare_transfer.side_effect
    attempts = [0]

    def prepare(*args, **kwargs):
        attempts[0] += 1
        if attempts[0] == 1:
            raise PlanRejected('all return routes rejected at measured grasp')
        return accepted(*args, **kwargs)

    s.prepare_transfer.side_effect = prepare
    stats = demo.run(s.cfg, auto=True)
    assert stats == dict(cycles=3, success=1, fail=0, empty=0)
    assert s.detections == 4  # pick, two excluded-point attempts, pick after the wait
    assert [len(pool) for pool in s.selections] == [1, 0, 0, 1]
    assert s.selections[-1][0, 0] == pytest.approx(.3)
    assert s.now >= s.cfg['retry']['no_candidate_wait_s']
    assert [event for event in s.events if event[0] == 'capture'][-1][2] is True
    s.recovery.assert_called_once()
    s.gripper.grip.assert_called_once()
    s.prompt.assert_not_called()


def test_actual_fault_aborts_main_loop_without_reselection_or_exit_home(loop):
    s = loop

    def fault(*args, **kwargs):
        s.robot.arm.get_err_warn_code.return_value = 0, [31, 14]
        raise RuntimeError('latched controller C31')

    s.prepare_transfer.side_effect = fault
    assert demo.run(s.cfg, auto=True) == dict(cycles=1, success=0, fail=0, empty=0)
    assert s.detections == 1
    s.recovery.assert_not_called()
    s.gripper.grip.assert_not_called()
    s.gripper.release.assert_not_called()
    s.home.assert_called_once()  # initialization only; fault exit cannot move home
    s.robot.arm.clean_error.assert_not_called()
    assert s.robot.arm.get_err_warn_code.return_value == (0, [31, 14])
    s.robot.disconnect.assert_called_once()
    assert s.viz.log_outcome.call_args.args[2] == 'abort'


@pytest.mark.parametrize('exhausted', ['detection', 'selection'])
def test_automatic_exhaustion_waits_and_retries_new_frames_without_prompt(loop, exhausted):
    s = loop
    s.empty = exhausted == 'detection'
    s.reject_selection = exhausted == 'selection'
    s.stop_after = 4  # two complete rounds, each with an initial + fresh retry
    stats = demo.run(s.cfg, auto=True)
    assert stats == dict(cycles=2, success=0, fail=0, empty=0)
    frames = [event[1] for event in s.events if event[0] == 'detect']
    assert len(frames) == 4 and len(set(frames)) == 4
    assert sum(event[2] for event in s.events if event[0] == 'capture') >= 2
    assert s.now >= 2 * .3 + s.cfg['retry']['no_candidate_wait_s']
    s.prompt.assert_not_called()
    s.gripper.pre_open.assert_not_called()
    s.gripper.grip.assert_not_called()
    s.recovery.assert_not_called()
    s.prepare_transfer.assert_not_called()
