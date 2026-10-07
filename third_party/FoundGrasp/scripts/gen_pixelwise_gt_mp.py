"""
GraspNet-1Billion 데이터셋의 여러 씬을 병렬로 처리하는 스크립트 (GPU 분산)
"""
import os
import subprocess
import multiprocessing as mp
from functools import partial
import argparse
from tqdm import tqdm

def check_scene_completed(scene_id, root, camera, num_images=256):
    """씬이 이미 처리되었는지 확인

    Args:
        scene_id: 확인할 씬 ID
        root: GraspNet 데이터셋 루트 경로
        camera: 카메라 타입
        num_images: 씬당 이미지 개수 (기본 256)

    Returns:
        bool: 마지막 파일이 존재하면 True
    """
    scene_name = f'scene_{scene_id:04d}'
    base_dir = os.path.join(root, 'pixelwise_grasp_gt_top300', camera)

    for ann_id in range(num_images):
        path = os.path.join(base_dir, f'{scene_name}_ann_{ann_id:04d}.npz')
        if not os.path.exists(path):
            return False
    return True




def process_scene(scene_id, root, camera, gpu_id, skip_existing=True):
    """단일 씬을 처리하는 함수
    
    Args:
        scene_id: 처리할 씬 ID
        root: GraspNet 데이터셋 루트 경로
        camera: 카메라 타입
        gpu_id: 사용할 GPU ID
        skip_existing: 이미 처리된 씬을 건너뛸지 여부
    
    Returns:
        tuple: (scene_id, success, error_message)
    """
    try:
        # 이미 처리된 씬인지 확인
        if skip_existing and check_scene_completed(scene_id, root, camera):
            return (scene_id, True, "Already processed (skipped)")
        
        # gen_pixelwise_gt.py 호출
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gen_pixelwise_gt_v2.py')
        cmd = [
            'python', script_path,
            '--root', root,
            '--camera', camera,
            '--scene_id', str(scene_id)
        ]
        
        # 환경 변수에 GPU ID 설정
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # 서브프로세스 실행
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 1시간 타임아웃
            env=env
        )
        
        if result.returncode == 0:
            return (scene_id, True, None)
        else:
            error_msg = result.stderr if result.stderr else "Unknown error"
            return (scene_id, False, error_msg)
            
    except subprocess.TimeoutExpired:
        return (scene_id, False, "Timeout (>1 hour)")
    except Exception as e:
        return (scene_id, False, str(e))


def worker_initializer(gpu_queue):
    """워커 프로세스 초기화 함수 - GPU ID를 할당받음"""
    global worker_gpu_id
    worker_gpu_id = gpu_queue.get()


def process_scene_with_gpu_queue(scene_id, root, camera, skip_existing):
    """GPU 큐를 사용하여 씬을 처리하는 함수"""
    return process_scene(scene_id, root, camera, worker_gpu_id, skip_existing)


def main():
    parser = argparse.ArgumentParser(description='GraspNet 씬들을 병렬로 처리 (GPU 분산)')
    parser.add_argument('--root', default='/data/Grasp/GraspNet-1Billion',
                        help='GraspNet 데이터셋 루트 경로')
    parser.add_argument('--camera', default='kinect',
                        choices=['realsense', 'kinect'],
                        help='카메라 타입')
    parser.add_argument('--start_scene', type=int, default=0,
                        help='시작 씬 ID')
    parser.add_argument('--end_scene', type=int, default=190,
                        help='종료 씬 ID (미포함, exclusive)')
    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3,4',
                        help='사용할 GPU ID 리스트 (쉼표로 구분, 예: 0,1,2,3)')
    parser.add_argument('--skip_existing', action='store_true', default=True,
                        help='이미 처리된 씬을 건너뛸지 여부 (기본: True)')
    parser.add_argument('--no_skip', action='store_false', dest='skip_existing',
                        help='이미 처리된 씬도 다시 처리')
    
    args = parser.parse_args()
    
    # GPU ID 리스트 파싱
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
    num_workers = len(gpu_ids)
    
    # 처리할 씬 ID 리스트 생성
    scene_ids = list(range(args.start_scene, args.end_scene ))
    
    print(f"=== GraspNet 씬 병렬 처리 시작 (GPU 분산) ===")
    print(f"처리할 씬 범위: {args.start_scene} ~ {args.end_scene - 1} (총 {len(scene_ids)}개)")
    print(f"카메라 타입: {args.camera}")
    print(f"사용할 GPU: {gpu_ids}")
    print(f"병렬 워커 수: {num_workers}")
    print(f"데이터셋 경로: {args.root}")
    print(f"이미 처리된 씬 건너뛰기: {args.skip_existing}")
    
    # 이미 처리된 씬 사전 체크 (optional - 진행상황 미리 확인)
    if args.skip_existing:
        print("\n이미 처리된 씬 확인 중...")
        completed = []
        for scene_id in scene_ids:
            if check_scene_completed(scene_id, args.root, args.camera):
                completed.append(scene_id)
        print(completed)
        if completed:
            print(f"이미 완료된 씬: {len(completed)}개")
            print(f"실제 처리할 씬: {len(scene_ids) - len(completed)}개")
    
    
    # GPU ID 큐 생성 - 각 워커가 순환적으로 GPU를 할당받음
    manager = mp.Manager()
    gpu_queue = manager.Queue()
    
    # GPU ID를 큐에 채움
    for gpu_id in gpu_ids:
        gpu_queue.put(gpu_id)
    
    # 부분 함수 생성
    process_func = partial(
        process_scene_with_gpu_queue,
        root=args.root,
        camera=args.camera,
        skip_existing=args.skip_existing,
    )
    
    # 멀티프로세싱 풀 생성 및 실행
    results = []
    failed_scenes = []
    skipped_scenes = []
    
    with mp.Pool(processes=num_workers, initializer=worker_initializer, initargs=(gpu_queue,)) as pool:
        pbar = tqdm(total=len(scene_ids), desc="Processing scenes")
        
        for result in pool.imap_unordered(process_func, scene_ids):
            scene_id, success, error_msg = result
            results.append(result)
            
            if not success:
                failed_scenes.append((scene_id, error_msg))
                tqdm.write(f"✗ Scene {scene_id:04d} 실패: {error_msg}")
            elif error_msg == "Already processed (skipped)":
                skipped_scenes.append(scene_id)
                tqdm.write(f"⊙ Scene {scene_id:04d} 건너뜀 (이미 처리됨)")
            else:
                tqdm.write(f"✓ Scene {scene_id:04d} 완료")
            
            pbar.update(1)
        
        pbar.close()
    
    # 결과 요약
    print("\n" + "="*60)
    print("=== 처리 완료 ===")
    print(f"총 씬 개수: {len(scene_ids)}")
    print(f"성공 (새로 처리): {len(scene_ids) - len(failed_scenes) - len(skipped_scenes)}")
    print(f"건너뜀 (이미 처리됨): {len(skipped_scenes)}")
    print(f"실패: {len(failed_scenes)}")
    
    if failed_scenes:
        print("\n실패한 씬 목록:")
        for scene_id, error_msg in sorted(failed_scenes):
            print(f"  Scene {scene_id:04d}: {error_msg}")
        
        # 실패한 씬 ID들을 파일로 저장
        failed_log_path = f"failed_scenes_{args.camera}.txt"
        with open(failed_log_path, 'w') as f:
            for scene_id, error_msg in sorted(failed_scenes):
                f.write(f"{scene_id}\t{error_msg}\n")
        print(f"\n실패 목록이 {failed_log_path}에 저장되었습니다.")
    
    print("="*60)


if __name__ == "__main__":
    main()