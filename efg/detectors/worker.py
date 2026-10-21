"""비교군 백엔드 — 각자의 conda 환경에서 도는 워커 프로세스를 파일 프로토콜로 호출.

비교군은 서로 다른 의존성(MinkowskiEngine, pytorch3d, AnyGrasp SDK …)이 필요해
메인 프로세스에 함께 올릴 수 없다. 논문 시뮬 벤치마크와 같은 방식을 쓴다:
  워커 = 논문 레포의 ext/<repo>/sim_worker.py (AnyGrasp는 이 레포 workers/anygrasp_worker.py)
  프로토콜 = efg/worker_protocol.py (sim/worker_protocol.py 무수정 사본)
워커는 build 시 자동으로 띄우고(모델 1회 로드), close/종료 시 내린다.

설정 (detector.<name>):
  conda_env | python   워커 인터프리터 (conda 환경 이름 또는 python 경로)
  script               워커 스크립트 경로
  args                 추가 인자 리스트 (예: [--checkpoint, <path>])
  pythonpath           워커 PYTHONPATH 앞에 넣을 경로 리스트 (예: cupoch 대체 모듈)
  startup_timeout_s / request_timeout_s
"""
import atexit
import os
import subprocess
import tempfile
import time
from pathlib import Path

from ..paths import import_grasp_group, resolve
from ..worker_protocol import WorkerClient
from .base import GraspDetector

READY_MARK = "[worker] serving"


def conda_python(env_name):
    """conda 환경 이름 → python 실행 파일 경로."""
    exe = os.environ.get("CONDA_EXE")
    base = Path(exe).resolve().parents[1] if exe else Path.home() / "miniconda3"
    if env_name == "base":
        return base / "bin" / "python"
    return base / "envs" / env_name / "bin" / "python"


class WorkerDetector(GraspDetector):
    def __init__(self, cfg, device="cuda", name="worker"):
        super().__init__(cfg, device)
        self.name = name
        py = resolve(cfg["python"]) if cfg.get("python") else conda_python(cfg["conda_env"])
        script = resolve(cfg["script"])
        for label, p in (("python", py), ("script", script)):
            if not Path(p).exists():
                raise SystemExit(f"[{name}] 워커 {label} 없음: {p}"
                                 + ("\n  envs/ 의 setup 스크립트로 환경을 만드세요"
                                    if label == "python" else ""))
        args = [str(resolve(a)) if str(a).startswith(("~", "/")) else str(a)
                for a in cfg.get("args", [])]
        self.work_dir = tempfile.mkdtemp(prefix=f"efg_{name}_")
        self.log_path = Path(self.work_dir) / "worker.log"
        env = os.environ.copy()
        extra = [str(resolve(p)) for p in cfg.get("pythonpath", [])]
        if extra:
            env["PYTHONPATH"] = os.pathsep.join(extra + [env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
        cmd = [str(py), str(script), "--work_dir", self.work_dir] + args
        print(f"[{name}] 워커 기동: {' '.join(cmd)}\n  로그: {self.log_path}")
        self._log = open(self.log_path, "w")
        self.proc = subprocess.Popen(cmd, cwd=str(Path(script).parent), env=env,
                                     stdout=self._log, stderr=subprocess.STDOUT)
        atexit.register(self.close)
        self._wait_ready(float(cfg.get("startup_timeout_s", 600)))
        self.client = WorkerClient(self.work_dir,
                                   timeout=float(cfg.get("request_timeout_s", 120)))

    def _tail(self, n=25):
        try:
            return "".join(open(self.log_path).readlines()[-n:])
        except OSError:
            return ""

    def _wait_ready(self, timeout):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise SystemExit(f"[{self.name}] 워커가 시작 중 종료됨 "
                                 f"(code {self.proc.returncode}):\n{self._tail()}")
            if READY_MARK in self._tail(200):
                print(f"[{self.name}] 워커 준비 완료 ({time.time() - t0:.1f}s)")
                return
            time.sleep(0.2)
        self.close()
        raise SystemExit(f"[{self.name}] 워커 준비 시간 초과 ({timeout:.0f}s):\n{self._tail()}")

    def describe(self):
        return f"{self.name} (worker: {Path(self.cfg['script']).name})"

    def detect(self, rgb, depth_m, K):
        GraspGroup = import_grasp_group()
        if self.proc.poll() is not None:
            raise RuntimeError(f"[{self.name}] 워커 종료됨:\n{self._tail()}")
        arr = self.client.request(rgb, depth_m, K)
        return GraspGroup(arr.reshape(-1, 17))

    def close(self):
        proc = getattr(self, "proc", None)
        if proc is None or proc.poll() is not None:
            return
        try:
            open(os.path.join(self.work_dir, "stop"), "w").close()
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        finally:
            self._log.close()
