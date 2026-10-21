# 원본: Workspaces/FoundGrasp/sim/worker_protocol.py (논문 시뮬 벤치마크) — 무수정 사본.
# 비교군 워커(sim_worker.py)와 같은 파일 프로토콜을 써야 하므로 내용을 바꾸지 말 것.
"""baseline 워커 <-> 시뮬 드라이버 파일 기반 프로토콜.

baseline들은 서로 다른 conda env(tta-grasp/hggd/cgnet/GraspSAM)에서 돌아야
하므로 in-process 연결이 불가능하다. 대신:

  드라이버(vit-grasp env):  req_{i:06d}.npz  {rgb(H,W,3 u8), depth(H,W f32 m), K(3,3)}
  워커(baseline env):       res_{i:06d}.npy  (N,17) float64 GraspGroup 배열 (camera frame)

워커는 시작 시 모델을 1회 로드하고 요청 폴더를 폴링한다 (호출마다 프로세스를
띄우면 모델 로드 10~20초 x 수백 attempt라 비실용적). 부분 파일 읽기를 막기 위해
양쪽 다 임시 이름으로 쓴 뒤 os.replace(원자적 rename)로 완성한다.

의존성은 numpy/stdlib뿐 — 어느 baseline env에서도 import 가능해야 한다.
"""
import os
import sys
import time
import glob

import inspect
import numpy as np

STOP_FILE = 'stop'


def _atomic_save_npy(path, arr):
    tmp = path + '.tmp'
    np.save(tmp, arr)              # np.save는 .npy를 덧붙이므로 실제 파일명은 tmp+'.npy'
    os.replace(tmp + '.npy', path)


def atomic_save_npz(path, **arrays):
    """path는 '.npz'로 끝나야 한다 (np.savez가 확장자를 자동으로 붙이는 것 보정).
    임시 파일명은 '_tmp_' 접두사로 만들어 워커의 req_*.npz glob에 절대 걸리지
    않게 한다 (접미사 방식 'req_X.tmp.npz'는 glob에 걸려 완성 전에 워커가
    낚아채는 레이스가 실측 발생)."""
    assert path.endswith('.npz'), path
    d, base = os.path.split(path)
    tmp = os.path.join(d, '_tmp_' + base[:-len('.npz')])
    np.savez(tmp, **arrays)                # 실제 파일: tmp + '.npz'
    os.replace(tmp + '.npz', path)


def serve(work_dir, predict_fn, poll_interval=0.05):
    """워커 메인 루프. predict_fn(rgb, depth, K) -> (N,17) float64 ndarray.

    work_dir의 req_*.npz를 오래된 순으로 처리해 res_*.npy로 응답한다.
    `stop` 파일이 생기면 종료. 예외는 res 대신 err_*.txt로 기록하고 계속 돈다
    (드라이버가 타임아웃 대신 즉시 오류를 읽을 수 있도록).
    """
    os.makedirs(work_dir, exist_ok=True)
    print(f'[worker] serving {work_dir}', flush=True)
    while True:
        if os.path.exists(os.path.join(work_dir, STOP_FILE)):
            print('[worker] stop file found, exiting', flush=True)
            return
        reqs = sorted(glob.glob(os.path.join(work_dir, 'req_*.npz')))
        if not reqs:
            time.sleep(poll_interval)
            continue
        req_path = reqs[0]
        req_id = os.path.basename(req_path)[len('req_'):-len('.npz')]
        try:
            data = np.load(req_path)
            rgb, depth, K = data['rgb'], data['depth'], data['K']
            mask = data['mask'] if 'mask' in data.files else None
            t0 = time.time()
            if mask is not None and 'mask' in inspect.signature(predict_fn).parameters:
                gg = np.asarray(predict_fn(rgb, depth, K, mask=mask), dtype=np.float64)
            else:
                gg = np.asarray(predict_fn(rgb, depth, K), dtype=np.float64)
            assert gg.ndim == 2 and gg.shape[1] == 17, f'bad shape {gg.shape}'
            _atomic_save_npy(os.path.join(work_dir, f'res_{req_id}.npy'), gg)
            # 추론 누적 GPU 메모리 방출. MinkowskiEngine(eco/sbg/gnb sparse conv)은
            # 좌표 구조별 kernel-map을 전역 캐시에 쌓아 실제 다양한 클라우드에서
            # 무한 증식 -> OOM (empty_cache로는 안 풀림). 매 요청 후 ME 좌표
            # 매니저를 비운다. ME 없는 env는 import 실패로 자동 무시.
            try:
                import MinkowskiEngine as ME
                ME.clear_global_coordinate_manager()
            except Exception:
                pass
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            print(f'[worker] req {req_id}: {len(gg)} grasps in {time.time()-t0:.2f}s', flush=True)
        except Exception:
            import traceback
            err = traceback.format_exc()
            print(err, file=sys.stderr, flush=True)
            err_path = os.path.join(work_dir, f'err_{req_id}.txt')
            with open(err_path + '.tmp', 'w') as f:
                f.write(err)
            os.replace(err_path + '.tmp', err_path)
        finally:
            os.remove(req_path)


class WorkerClient:
    """드라이버 쪽 클라이언트: 요청을 쓰고 응답을 기다린다."""

    def __init__(self, work_dir, timeout=600.0, poll_interval=0.05):
        self.work_dir = work_dir
        self.timeout = timeout
        self.poll_interval = poll_interval
        self._next_id = 0
        os.makedirs(work_dir, exist_ok=True)

    def request(self, rgb, depth, K, mask=None):
        req_id = f'{self._next_id:06d}'
        self._next_id += 1
        atomic_save_npz(os.path.join(self.work_dir, f'req_{req_id}.npz'),
                        rgb=rgb.astype(np.uint8), depth=depth.astype(np.float32),
                        K=K.astype(np.float64),
                        **({'mask': mask.astype(np.uint8)} if mask is not None else {}))
        res_path = os.path.join(self.work_dir, f'res_{req_id}.npy')
        err_path = os.path.join(self.work_dir, f'err_{req_id}.txt')
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            if os.path.exists(res_path):
                arr = np.load(res_path)
                os.remove(res_path)
                return arr
            if os.path.exists(err_path):
                err = open(err_path).read()
                os.remove(err_path)
                raise RuntimeError(f'worker error for req {req_id}:\n{err}')
            time.sleep(self.poll_interval)
        raise TimeoutError(f'worker did not answer req {req_id} within {self.timeout}s '
                           f'(워커가 {self.work_dir}를 보고 있는지 확인)')

    def stop(self):
        open(os.path.join(self.work_dir, STOP_FILE), 'w').close()
