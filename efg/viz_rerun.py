"""rerun 시각화 — 실시간 뷰어 + .rrd 기록.

  3D (base 좌표계, Z-up)
    world/robot/<link>     xArm7 메쉬 (MJCF 기구학, 로봇 관절각 폴링으로 실시간 갱신)
    world/robot/tcp        그리퍼 근사 형상 (본체·핑거 박스, 개방 폭 반영)
    world/camera           스캔 시 카메라 자세 · 점군 · 그립 후보(점수 색) · 선정 그립
    world/targets          선정 그립의 pre-grasp / grasp TCP 목표 축
  2D  world/camera/image   컬러 + ROI + 후보 투영
  status / log / metrics    동작 단계, 실측 개방 폭, 검출 시간·후보 수 (성공 판정 미표시)
타임라인: elapsed(실행 후 경과 시간). 모든 기록 스레드가 같은 시간축을 쓴다.

뷰어 (rerun.viewer):
  web     rerun 웹 서버 프로세스를 띄우고(이미 있으면 재사용) 브라우저 주소를 출력.
          실행이 끝나도 서버는 남아 마지막 기록을 계속 볼 수 있다.
  native  rerun 네이티브 뷰어 실행 (디스플레이 필요)
  none    파일 기록만
"""
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlencode

import numpy as np

from .config import get
from .postprocess import backproject
from .viz import gripper_segments_3d, score_colors, display_candidate_indices

APP_ID = "exp_foundgrasp"
MAX_CANDIDATES = 25            # 3D/2D에 그릴 후보 수 (순위 상위)
POINT_STRIDE = 2                # 점군 픽셀 간격
COL_SELECTED = (65, 215, 255)



def _port_open(port):
    try:
        socket.create_connection(("127.0.0.1", port), timeout=0.3).close()
        return True
    except OSError:
        return False


def _ensure_web_viewer(grpc, web, log_path, memory_limit):
    """고정 포트의 서버 재사용. 웹/데이터 서버 중 하나만 살아 있어도 복구한다."""
    if not (1 <= grpc <= 65535 and 1 <= web <= 65535) or grpc == web:
        raise ValueError("rerun.grpc_port/web_port는 서로 다른 고정 포트(1~65535)여야 합니다")
    grpc_ready, web_ready = _port_open(grpc), _port_open(web)
    if grpc_ready and web_ready:
        return
    command = [sys.executable, "-m", "rerun"]
    if not web_ready:
        command += ["--serve-web", "--web-viewer-port", str(web)]
        if grpc_ready:
            # URI가 있으면 웹 뷰어만 제공하고 기존 gRPC 서버에 연결한다.
            command += [f"rerun+http://127.0.0.1:{grpc}/proxy"]
    else:
        command += ["--serve-grpc"]
    command += ["--port", str(grpc), "--server-memory-limit", str(memory_limit)]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True)
    for _ in range(100):
        if _port_open(grpc) and _port_open(web):
            return
        if process.poll() is not None:
            break
        time.sleep(0.1)
    if process.poll() is None:
        # python -m rerun은 별도 CLI 자식을 실행하므로 프로세스 그룹을 정리한다.
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    raise RuntimeError(f"rerun 고정 포트 서버 시작 실패 ({web}/{grpc}). 로그: {log_path}")


def _segments(gg, n, width=None):
    return [gripper_segments_3d(gg[i], width).reshape(4, 2, 3) for i in range(n)]


def _project_segments(seg, K):
    p = np.asarray(seg).reshape(-1, 3)
    if not np.isfinite(p).all() or np.any(p[:, 2] <= 0):
        return None
    uv = p @ np.asarray(K).T
    return (uv[:, :2] / uv[:, 2:3]).reshape(-1, 2, 2)


class RerunViz:
    """rerun.enabled=false 면 모든 메서드가 아무것도 하지 않는다."""

    def __init__(self, cfg, run_name, out_dir=None):
        self.rr = None
        self.cfg = cfg
        self._robot_thread = None
        self._stop = threading.Event()
        self._open_width = 0.05
        self._gripper_mesh_tcp = None
        self._has_gripper_feedback = False
        self._stage = "Initializing"
        self._width_text = "No gripper measurement"
        self._score_text = "No candidates"
        self._epoch = time.monotonic()
        self._cycle = 0
        self._scan_K = None
        self._tcp_offset = np.eye(4)
        self._tcp_offset[2, 3] = float(get(cfg, "rerun.tcp_offset_m", 0.215))
        if not get(cfg, "rerun.enabled", True):
            return
        import rerun as rr
        import rerun.blueprint as rrb

        self.rr = rr
        rr.init(APP_ID, recording_id=run_name)
        grpc = int(get(cfg, "rerun.grpc_port", 9876))
        viewer = get(cfg, "rerun.viewer", "native")
        sinks = []
        if viewer == "web":
            web = int(get(cfg, "rerun.web_port", 9090))
            log_dir = Path(out_dir or get(cfg, "output.dir", "outputs"))
            _ensure_web_viewer(grpc, web, log_dir / "rerun_server.log",
                               get(cfg, "rerun.server_memory_limit", "2GiB"))
            query = urlencode({"url": f"rerun+http://127.0.0.1:{grpc}/proxy"})
            self.viewer_url = f"http://127.0.0.1:{web}/?{query}"
            print(f"rerun 웹 뷰어 (고정 주소): {self.viewer_url}")
        elif viewer == "native":
            # rr.spawn reuses an existing listener; keep it separate from the
            # persistent web server so switching modes actually opens a window.
            grpc = int(get(cfg, "rerun.native_grpc_port", 9877))
            import rerun_cli
            executable = os.environ.get("RERUN_CLI_PATH") or str(
                Path(rerun_cli.__file__).parent / ("rerun.exe" if os.name == "nt" else "rerun"))
            rr.spawn(port=grpc, connect=False, hide_welcome_screen=True,
                     executable_path=executable)
            print("rerun 데스크톱 뷰어: 별도 창으로 실행")
        if viewer in ("web", "native"):
            sinks.append(rr.GrpcSink(f"rerun+http://127.0.0.1:{grpc}/proxy"))
        if get(cfg, "rerun.save_rrd", True) and out_dir is not None:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            self.rrd_path = Path(out_dir) / f"{run_name}.rrd"
            sinks.append(rr.FileSink(str(self.rrd_path)))
            print(f"rerun 기록: {self.rrd_path}")
        blueprint = rrb.Blueprint(rrb.Horizontal(
            rrb.Tabs(
                rrb.Spatial3DView(origin="world", name="Robot / scene"),
                rrb.Spatial3DView(origin="world/camera", name="Grasp detail",
                                  contents=["world/camera/points", "world/camera/grasps/**"])),
            rrb.Vertical(
                rrb.Spatial2DView(origin="world/camera/image", name="Scan snapshot (not live)"),
                rrb.TextDocumentView(origin="status", name="Motion / gripper / score"),
                rrb.TimeSeriesView(origin="metrics", name="metrics"),
                rrb.TextLogView(origin="log", name="log"),
                row_shares=[5, 2, 2, 1]),
            column_shares=[3, 2]),
            rrb.TimePanel(timeline="elapsed", play_state="Following", expanded=False),
            rrb.SelectionPanel(expanded=False))
        rr.set_sinks(*sinks, default_blueprint=blueprint)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        self._init_robot()

    def _stamp(self):
        # Rerun의 시간 설정은 스레드별이므로 각 기록 진입점에서 지정한다.
        self.rr.set_time("elapsed", duration=time.monotonic() - self._epoch)

    # ------------------------------------------------------------ 로봇
    def _init_robot(self):
        from .robot_model import XArm7Model

        rr = self.rr
        self.model = XArm7Model()
        for link in self.model.links:
            if link.mesh is not None:
                # GLB의 부품별 재질을 그대로 사용한다. albedo_factor를 주면
                # 제조사 모델의 흰 외장/회색 금속 구분이 덮어써진다.
                asset = (rr.Asset3D(path=str(link.mesh)) if link.mesh.suffix == ".glb"
                         else rr.Asset3D(path=str(link.mesh), albedo_factor=[0.85, 0.85, 0.88]))
                rr.log(f"world/robot/{link.name}",
                       asset,
                       static=True)
        self._log_gripper_shape()

    def _log_gripper_shape(self):
        """그리퍼 근사 (TCP 좌표계: z=접근, y=개폐). KORAS 2FPA + 플렉스 팁 대략 치수."""
        rr = self.rr
        self._stamp()
        tcp_z = float(self._tcp_offset[2, 3])
        if self._gripper_mesh_tcp != tcp_z:
            from .gripper_visual import gripper_parts
            for name, mesh, color in gripper_parts(tcp_z):
                rr.log(f"world/robot/tcp/gripper/{name}", rr.Mesh3D(
                    vertex_positions=mesh.vertices, triangle_indices=mesh.faces,
                    vertex_normals=mesh.vertex_normals, albedo_factor=[c/255 for c in color]))
            self._gripper_mesh_tcp = tcp_z
        for name, sign in (("left", 1), ("right", -1)):
            rr.log(f"world/robot/tcp/gripper/{name}", rr.Transform3D(
                translation=[0, sign*self._open_width/2, 0]))

    def _log_status(self):
        if self.rr is None:
            return
        self._stamp()
        self.rr.log("status", self.rr.TextDocument(
            f"**{self._stage}** · Cycle {self._cycle}\n\n{self._width_text}\n\n{self._score_text}",
            media_type="text/markdown"))

    def log_stage(self, stage):
        self._stage = stage
        self._log_status()


    def log_robot(self, q_rad, T_flange_tcp=None):
        if self.rr is None:
            return
        rr = self.rr
        q_rad = np.asarray(q_rad, dtype=float)
        if q_rad.shape != (7,) or not np.isfinite(q_rad).all():
            raise ValueError("로봇 관절각은 유한한 rad 값 7개여야 합니다")
        if T_flange_tcp is not None:
            self._tcp_offset = np.asarray(T_flange_tcp, dtype=float).copy()
        self._stamp()
        poses = self.model.link_poses(q_rad)
        for name, T in poses:
            rr.log(f"world/robot/{name}", rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]))
        T_tcp = poses[-1][1] @ self._tcp_offset
        rr.log("world/robot/tcp", rr.Transform3D(translation=T_tcp[:3, 3], mat3x3=T_tcp[:3, :3]))

    def set_gripper_open(self, width_m):
        if self.rr is None:
            return
        self._open_width = float(width_m)
        self._log_gripper_shape()

    def start_robot_stream(self, robot, hz=10.0, gripper=None):
        """로봇 관절각 폴링 스레드 (xArm SDK 보고 스트림의 캐시 값 — 명령 송신 없음)."""
        if self.rr is None:
            return
        if hz <= 0:
            raise ValueError("로봇 시각화 갱신 주기는 양수여야 합니다")
        if self._robot_thread is not None:
            raise RuntimeError("로봇 시각화 스트림이 이미 시작되었습니다")
        self._tcp_offset = np.asarray(robot.T_flange_tcp, dtype=float).copy()
        self._log_gripper_shape()

        self._has_gripper_feedback = gripper is not None

        def loop():
            last_error = None
            last_gripper_stamp = None
            while not self._stop.is_set():
                try:
                    if not robot.arm.connected:
                        raise RuntimeError("로봇 연결 끊김")
                    q = list(robot.arm.angles)[:7]
                    self.log_robot(q, robot.T_flange_tcp)
                    if gripper is not None:
                        reading = gripper.visual_status
                        if reading is not None and reading[1] != last_gripper_stamp:
                            width, last_gripper_stamp = reading
                            self.set_gripper_open(width)
                            self._stamp()
                            self.rr.log("metrics/gripper_width_mm", self.rr.Scalars(width*1000))
                        if reading is not None:
                            age = max(0., time.monotonic()-reading[1])
                            self._width_text = f"Measured opening: **{reading[0]*1000:.1f} mm** · {age:.1f}s ago"
                            self._log_status()
                    if last_error is not None:
                        self.log_text("로봇 자세 시각화 갱신 복구")
                    last_error = None
                except Exception as exc:
                    message = f"로봇 자세 시각화 갱신 실패: {type(exc).__name__}: {exc}"
                    if message != last_error:
                        print(message, flush=True)
                        try:
                            self.log_text(message, level="WARN")
                        except Exception:
                            pass  # 시각화 오류가 로봇 제어를 중단하지 않게 한다.
                    last_error = message
                self._stop.wait(1.0 / hz)

        self._robot_thread = threading.Thread(target=loop, daemon=True)
        self._robot_thread.start()

    # ------------------------------------------------------------ 사이클
    def set_cycle(self, n):
        if self.rr is not None:
            self._cycle = int(n)
            self._stamp()
            self.rr.log("metrics/cycle", self.rr.Scalars(float(n)))

    def log_text(self, msg, level="INFO"):
        if self.rr is not None:
            self._stamp()
            self.rr.log("log", self.rr.TextLog(msg, level=level))

    def log_scan(self, rgb, depth_m, K, T_base_cam, roi_px=None):
        if self.rr is None:
            return
        rr = self.rr
        self._stamp()
        H, W = depth_m.shape
        self._scan_K = np.asarray(K).copy()
        rr.log("world/camera", rr.Transform3D(translation=T_base_cam[:3, 3],
                                              mat3x3=T_base_cam[:3, :3]))
        rr.log("world/camera/image", rr.Pinhole(image_from_camera=K, width=W, height=H,
                                                camera_xyz=rr.ViewCoordinates.RDF,
                                                image_plane_distance=0.1))
        rr.log("world/camera/image/rgb", rr.Image(rgb))
        rr.log("world/camera/image/roi", rr.Clear(recursive=True))
        if roi_px is not None:
            x1, y1, x2, y2 = [float(v) for v in roi_px]
            rr.log("world/camera/image/roi", rr.Boxes2D(
                array=[x1, y1, x2, y2], array_format=rr.Box2DFormat.XYXY,
                colors=[0, 200, 0], labels="ROI"))
        d = depth_m[::POINT_STRIDE, ::POINT_STRIDE]
        Ks = K.copy().astype(np.float64)
        Ks[:2] /= POINT_STRIDE
        valid = d > 0
        pts = backproject(d, Ks)
        cols = rgb[::POINT_STRIDE, ::POINT_STRIDE][valid]
        rr.log("world/camera/points", rr.Points3D(pts, colors=cols, radii=0.0015))
        for p in ("world/camera/grasps", "world/camera/image/grasps", "world/targets"):
            rr.log(p, rr.Clear(recursive=True))

    def log_candidates(self, gg, K, counts=None, detect_ms=None):
        if self.rr is None:
            return
        rr = self.rr
        self._stamp()
        limit = int(get(self.cfg, "rerun.max_candidates", MAX_CANDIDATES))
        order = display_candidate_indices(gg, limit)
        shown_scores = np.asarray(gg.scores)[order]
        cols = score_colors(shown_scores)
        for rank in range(len(order)-1, -1, -1):
            i = int(order[rank])
            seg = _segments(gg[i:i+1], 1)[0]
            name = "top1" if rank == 0 else f"candidates/{rank+1:02d}"
            color = cols[rank]
            radius = rr.Radius.ui_points(4.0) if rank == 0 else .0013
            label = f"TOP 1 · {float(gg.scores[i]):.3f}" if rank == 0 else None
            rr.log(f"world/camera/grasps/{name}", rr.LineStrips3D(
                list(seg), colors=color, radii=radius))
            if rank == 0:
                anchor = seg[-1, 1] - np.asarray(gg[i].rotation_matrix)[:, 0]*.035
                rr.log("world/camera/grasps/top1/leader", rr.LineStrips3D(
                    [[seg[-1, 1], anchor]], colors=[255,255,255], radii=rr.Radius.ui_points(1.5)))
                rr.log("world/camera/grasps/top1/label", rr.Points3D(
                    [anchor], colors=[255,255,255], radii=rr.Radius.ui_points(4), labels=label))
            uv = _project_segments(seg, K)
            if uv is not None:
                if rank == 0:
                    rr.log(f"world/camera/image/grasps/{name}/halo", rr.LineStrips2D(
                        list(uv), colors=[255,255,255], radii=6.5, draw_order=40))
                    rr.log(f"world/camera/image/grasps/{name}/label", rr.Points2D(
                        [uv[-1,1]+[12,-24]], colors=[255,255,255], radii=3,
                        labels=label, draw_order=43))
                rr.log(f"world/camera/image/grasps/{name}/outline", rr.LineStrips2D(
                    list(uv), colors=[20,24,28], radii=4.5 if rank == 0 else 2.0,
                    draw_order=41 if rank == 0 else 0))
                rr.log(f"world/camera/image/grasps/{name}/color", rr.LineStrips2D(
                    list(uv), colors=color, radii=3.5 if rank == 0 else 1.2,
                    draw_order=42 if rank == 0 else 1))
        if len(gg):
            self._score_text = (f"RdYlGn: red {min(shown_scores):.3f} → yellow → green {max(shown_scores):.3f}"
                                f" · {len(order)}/{len(gg)} shown\n\nBold = top 1 · cyan = selected")
        else:
            self._score_text = "No candidates"
        self._log_status()
        for k, v in (counts or {}).items():
            rr.log(f"metrics/candidates/{k}", rr.Scalars(float(v)))
        if detect_ms is not None:
            rr.log("metrics/detect_ms", rr.Scalars(float(detect_ms)))

    def log_selected(self, gg_single, sel, K):
        if self.rr is None:
            return
        rr = self.rr
        self._stamp()
        seg = _segments(gg_single, 1, width=sel.open_width_m)[0]
        rr.log("world/camera/grasps/selected", rr.LineStrips3D(
            list(seg), colors=COL_SELECTED, radii=0.002,
            labels=f"SELECTED · {sel.score:.3f}"))
        uv = _project_segments(seg, K)
        if uv is not None:
            rr.log("world/camera/image/grasps/selected", rr.LineStrips2D(
                list(uv), colors=COL_SELECTED, radii=3.0, draw_order=20))
        for name, T in (("pre", sel.T_tcp_pre), ("grasp", sel.T_tcp_grasp)):
            rr.log(f"world/targets/{name}", rr.Transform3D(
                translation=T[:3, 3], mat3x3=T[:3, :3]))
            rr.log(f"world/targets/{name}", rr.TransformAxes3D(0.05)
                   if hasattr(rr, "TransformAxes3D") else rr.Arrows3D(
                       vectors=np.eye(3) * 0.05,
                       origins=np.zeros((3, 3)), colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]))
        if not self._has_gripper_feedback:
            self.set_gripper_open(sel.open_width_m)
            self._width_text = f"Preview opening (target): {sel.open_width_m*1000:.1f} mm"
            self._log_status()

    def log_outcome(self, gg_single, sel, outcome, reason=None):
        # 성공 판정은 신뢰도가 낮으므로 색/라벨/지표에 노출하지 않는다.
        self.log_stage("Stopped" if outcome == "abort" else "Cycle complete")

    def close(self):
        self._stop.set()
        if self._robot_thread is not None:
            self._robot_thread.join(timeout=1)
        if self.rr is not None:
            try:
                self.rr.get_data_recording().flush()
            except Exception:
                pass
