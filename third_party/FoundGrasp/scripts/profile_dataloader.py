"""
Data Loading Bottleneck Profiler for GraspNet1B Dataset

This script profiles each step of the data loading pipeline to identify bottlenecks:
1. RGB image loading (cv2.imread)
2. Depth image loading
3. Segmentation mask loading
4. HDF5 GT maps loading
5. Image resizing
6. Augmentation (color & geometric)
7. Tensor conversion & normalization
8. DataLoader overhead (collation, transfer)

Usage:
    python scripts/profile_dataloader.py --num_samples 100 --num_workers 8
"""

import os
os.environ.setdefault('SETUPTOOLS_USE_DISTUTILS', 'stdlib')

import sys
import time
import argparse
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import h5py
import torchvision.transforms.functional as TF
from transformers import AutoImageProcessor


@contextmanager
def timer(name, timings):
    """Context manager to measure execution time."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    timings[name].append(elapsed)

KNOWN_H5_KEYS = [
    'grasp_score', 'graspness', 'grasp_width', 'grasp_depth', 
    'valid_mask', 'approach_dir', 'gripper_dir',
]
class ProfiledGraspNetDataset(torch.utils.data.Dataset):
    """
    A profiled version of GraspNet1BDataset that measures time for each operation.
    """

    def __init__(self, root, camera='realsense', split='debug',
                 image_size=(720, 1280), backbone_name='facebook/dinov2-base',
                 augmentation=None):
        self.root = root
        self.split = split
        self.camera = camera
        self.image_size = image_size
        self.augmentation = augmentation

        # Timing storage (per-worker, will be aggregated)
        self.timings = defaultdict(list)

        # Setup scene IDs
        if split == 'train':
            self.sceneIds = list(range(100))
        elif split == 'test':
            self.sceneIds = list(range(100, 190))
        elif split == 'debug':
            self.sceneIds = [0]
        else:
            self.sceneIds = [0]  # Default to debug

        self.sceneIds = ['scene_{}'.format(str(x).zfill(4)) for x in self.sceneIds]

        # Build file paths
        self.colorpath = []
        self.depthpath = []
        self.labelpath = []
        self.pixelgtpath = []
        self.scenename = []
        self.frameid = []

        ann_range = range(0, 256)
        for x in self.sceneIds:
            for img_num in ann_range:
                self.colorpath.append(os.path.join(root, 'scenes', x, camera, 'rgb', str(img_num).zfill(4) + '.png'))
                self.depthpath.append(os.path.join(root, 'scenes', x, camera, 'depth', str(img_num).zfill(4) + '.png'))
                self.pixelgtpath.append(os.path.join(root, 'pixelwise_grasp_gt_npz', camera, x+'_ann_'+str(img_num).zfill(4)+'.npz'))
                self.labelpath.append(os.path.join(root, 'scenes', x, camera, 'label', str(img_num).zfill(4) + '.png'))
                self.scenename.append(x.strip())
                self.frameid.append(img_num)

        # Setup processor
        self.processor = AutoImageProcessor.from_pretrained(backbone_name, use_fast=True)
        self.image_mean = getattr(self.processor, "image_mean", None)
        self.image_std = getattr(self.processor, "image_std", None)

        # Setup augmentations (optional)
        self.color_tfs = None
        self.geometric_tfs = None

        if augmentation:
            import albumentations as A

            if 'color' in augmentation:
                self.color_tfs = A.Compose([
                    A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
                    A.ToGray(p=0.2),
                    A.GaussianBlur(p=0.5)
                ])

            if 'crop' in augmentation:
                self.geometric_tfs = A.Compose([
                    A.RandomSizedCrop(
                        min_max_height=(int(self.image_size[0]*0.6), self.image_size[0]),
                        w2h_ratio=self.image_size[1]/self.image_size[0],
                        size=self.image_size, p=1.0
                    )
                ], additional_targets={
                    'depth': 'mask', 'segm': 'mask', 'objectness': 'mask',
                    'graspness': 'mask', 'grasp_score': 'mask', 'grasp_width': 'mask',
                    'grasp_depth': 'mask', 'valid_mask': 'mask',
                    'approach_dir': 'mask', 'gripper_dir': 'mask',
                })

    def __len__(self):
        return len(self.colorpath)

    def __getitem__(self, index):
        timings = defaultdict(list)
        sample = {}

        # 1. Load RGB image
        with timer('1_load_rgb', timings):
            color_img = cv2.cvtColor(cv2.imread(self.colorpath[index]), cv2.COLOR_BGR2RGB)

        # 2. Load Depth image
        with timer('2_load_depth', timings):
            depth_img = cv2.imread(self.depthpath[index], cv2.IMREAD_UNCHANGED).astype(np.float16) / 1000.0

        # 3. Load Segmentation
        with timer('3_load_segm', timings):
            segm_img = cv2.imread(self.labelpath[index], cv2.IMREAD_UNCHANGED)

        # 4. Load HDF5 GT maps
        gt_maps = {}
        with timer('4_load_hdf5', timings):
            # with h5py.File(self.pixelgtpath[index], 'r', driver='core') as f:
            #     for key in KNOWN_H5_KEYS:
            #         data = f[key][()]
            #         data = data.astype(np.float16)
            #         if key == 'grasp_score':
            #             key = 'grasp_score'
            #         gt_maps[key] = data
            data = np.load(self.pixelgtpath[index])
            for key in data.files:
                arr = data[key]
                if key == 'grasp_score':
                    key = 'grasp_score'
                gt_maps[key] = arr

        # 5. Resize if needed
        original_size = depth_img.shape[:2]
        need_resize = (original_size[0] != self.image_size[0]) or (original_size[1] != self.image_size[1])

        with timer('5_resize', timings):
            if need_resize:
                color_img = cv2.resize(color_img, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_LINEAR)
                depth_img = cv2.resize(depth_img, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)
                segm_img = cv2.resize(segm_img, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)

                for key, data in gt_maps.items():
                    if data.ndim == 2:
                        gt_maps[key] = cv2.resize(data, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)
                    elif data.ndim == 3:
                        gt_maps[key] = cv2.resize(data, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)

        # 6. Color augmentation
        with timer('6_color_aug', timings):
            if self.color_tfs is not None:
                augmented = self.color_tfs(image=color_img)
                color_img = augmented['image']

        # 7. Geometric augmentation
        with timer('7_geometric_aug', timings):
            if self.geometric_tfs is not None:
                geo_aug_input = {'image': color_img, 'depth': depth_img, 'segm': segm_img}
                for key, data in gt_maps.items():
                    if data.ndim in [2, 3]:
                        geo_aug_input[key] = data
                geo_augmented = self.geometric_tfs(**geo_aug_input)
                color_img = geo_augmented['image']
                depth_img = geo_augmented['depth']
                segm_img = geo_augmented['segm']
                for key in gt_maps.keys():
                    if key in geo_augmented:
                        gt_maps[key] = geo_augmented[key]

        # 8. Convert to tensors
        with timer('8_to_tensor', timings):
            color_tensor = torch.from_numpy(color_img).permute(2, 0, 1).float() / 255.0
            sample['pixel_values'] = TF.normalize(color_tensor, mean=self.image_mean, std=self.image_std)
            sample['og_color'] = color_tensor
            sample['depth'] = torch.from_numpy(depth_img).unsqueeze(0).float()
            sample['segm'] = torch.from_numpy(segm_img).unsqueeze(0).long()

            for key, data in gt_maps.items():
                data_tensor = torch.from_numpy(np.ascontiguousarray(data))
                if data_tensor.ndim == 2:
                    data_tensor = data_tensor.unsqueeze(0)
                elif data_tensor.ndim == 3:
                    data_tensor = data_tensor.permute(2, 0, 1)
                sample[key] = data_tensor

            sample['depth_mask'] = (sample['depth'] > 0.1).float()

        # Store timing info in sample for aggregation
        sample['_timings'] = {k: v[0] for k, v in timings.items()}
        sample['scene_id'] = self.scenename[index]
        sample['im_id'] = self.frameid[index]

        return sample


def custom_collate(batch):
    """Custom collate that extracts timing info before stacking."""
    timings_list = [item.pop('_timings') for item in batch]

    # Collate the rest normally
    collated = {}
    for key in batch[0].keys():
        if key in ['scene_id', 'im_id']:
            collated[key] = [item[key] for item in batch]
        else:
            collated[key] = torch.stack([item[key] for item in batch])

    collated['_timings_batch'] = timings_list
    return collated


def profile_single_sample(dataset, index=0, num_runs=10):
    """Profile a single sample multiple times to get accurate measurements."""
    print(f"\n{'='*60}")
    print("SINGLE SAMPLE PROFILING (averaging {num_runs} runs)")
    print(f"{'='*60}")

    all_timings = defaultdict(list)

    for _ in range(num_runs):
        sample = dataset[index]
        for k, v in sample['_timings'].items():
            all_timings[k].append(v)

    # Calculate statistics
    print(f"\n{'Step':<25} {'Mean (ms)':<12} {'Std (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12} {'%':<8}")
    print("-" * 85)

    total_mean = sum(np.mean(v) for v in all_timings.values())

    results = []
    for step in sorted(all_timings.keys()):
        times = np.array(all_timings[step]) * 1000  # Convert to ms
        mean_t = np.mean(times)
        std_t = np.std(times)
        min_t = np.min(times)
        max_t = np.max(times)
        pct = (mean_t / (total_mean * 1000)) * 100
        results.append((step, mean_t, std_t, min_t, max_t, pct))
        print(f"{step:<25} {mean_t:<12.3f} {std_t:<12.3f} {min_t:<12.3f} {max_t:<12.3f} {pct:<8.1f}")

    print("-" * 85)
    print(f"{'TOTAL':<25} {total_mean*1000:<12.3f}")

    return results


def profile_dataloader(dataset, num_samples, batch_size, num_workers, prefetch_factor=2, pin_memory=True):
    """Profile the full DataLoader pipeline."""
    print(f"\n{'='*60}")
    print(f"DATALOADER PROFILING")
    print(f"  batch_size={batch_size}, num_workers={num_workers}")
    print(f"  prefetch_factor={prefetch_factor}, pin_memory={pin_memory}")
    print(f"{'='*60}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        collate_fn=custom_collate
    )

    # Warmup
    print("\nWarming up...")
    warmup_iter = iter(loader)
    for _ in range(min(3, len(loader))):
        _ = next(warmup_iter)
    del warmup_iter

    # Profile
    print(f"Profiling {num_samples} samples...")
    all_timings = defaultdict(list)
    batch_times = []
    data_transfer_times = []

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        collate_fn=custom_collate
    )

    num_batches = min(num_samples // batch_size, len(loader))

    start_total = time.perf_counter()

    for i, batch in enumerate(tqdm(loader, total=num_batches, desc="Profiling")):
        if i >= num_batches:
            break

        batch_start = time.perf_counter()

        # Aggregate timings from this batch
        for sample_timings in batch['_timings_batch']:
            for k, v in sample_timings.items():
                all_timings[k].append(v)

        # Measure GPU transfer time (if CUDA available)
        if torch.cuda.is_available():
            transfer_start = time.perf_counter()
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    _ = val.cuda(non_blocking=True)
            torch.cuda.synchronize()
            data_transfer_times.append(time.perf_counter() - transfer_start)

        batch_times.append(time.perf_counter() - batch_start)

    total_time = time.perf_counter() - start_total

    # Print results
    print(f"\n{'Step':<25} {'Mean (ms)':<12} {'Std (ms)':<12} {'Total %':<10}")
    print("-" * 65)

    total_step_time = sum(np.mean(v) for v in all_timings.values())

    for step in sorted(all_timings.keys()):
        times = np.array(all_timings[step]) * 1000
        mean_t = np.mean(times)
        std_t = np.std(times)
        pct = (np.mean(all_timings[step]) / total_step_time) * 100
        print(f"{step:<25} {mean_t:<12.3f} {std_t:<12.3f} {pct:<10.1f}")

    print("-" * 65)
    print(f"{'Per-sample total':<25} {total_step_time*1000:<12.3f}")

    if data_transfer_times:
        print(f"\n{'GPU Transfer (per batch)':<25} {np.mean(data_transfer_times)*1000:<12.3f} {np.std(data_transfer_times)*1000:<12.3f}")

    print(f"\n{'='*60}")
    print("THROUGHPUT STATISTICS")
    print(f"{'='*60}")
    print(f"Total batches processed: {num_batches}")
    print(f"Total samples processed: {num_batches * batch_size}")
    print(f"Total wall time: {total_time:.2f}s")
    print(f"Throughput: {(num_batches * batch_size) / total_time:.2f} samples/sec")
    print(f"Avg batch time: {np.mean(batch_times)*1000:.2f}ms")
    print(f"Effective samples/sec (theoretical): {1000 / (total_step_time * 1000):.2f}")

    return all_timings


def profile_worker_scaling(dataset, num_samples, batch_size, max_workers=16):
    """Profile how throughput scales with number of workers."""
    print(f"\n{'='*60}")
    print("WORKER SCALING ANALYSIS")
    print(f"{'='*60}")

    worker_counts = [0, 1, 2, 4, 8, 12, 16]
    worker_counts = [w for w in worker_counts if w <= max_workers]

    results = []

    for num_workers in worker_counts:
        print(f"\nTesting num_workers={num_workers}...")

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
            collate_fn=custom_collate
        )

        # Warmup
        warmup_iter = iter(loader)
        for _ in range(min(3, len(loader))):
            try:
                _ = next(warmup_iter)
            except StopIteration:
                break
        del warmup_iter

        # Measure
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
            collate_fn=custom_collate
        )

        num_batches = min(num_samples // batch_size, len(loader))

        start = time.perf_counter()
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
        elapsed = time.perf_counter() - start

        throughput = (num_batches * batch_size) / elapsed
        results.append((num_workers, throughput, elapsed))
        print(f"  Throughput: {throughput:.2f} samples/sec")

    print(f"\n{'Workers':<10} {'Throughput':<15} {'Time (s)':<12} {'Speedup':<10}")
    print("-" * 50)

    base_throughput = results[0][1] if results else 1
    for workers, throughput, elapsed in results:
        speedup = throughput / base_throughput
        print(f"{workers:<10} {throughput:<15.2f} {elapsed:<12.2f} {speedup:<10.2f}x")

    return results


def profile_io_vs_compute(dataset, index=0, num_runs=5):
    """Separate I/O time from compute time."""
    print(f"\n{'='*60}")
    print("I/O vs COMPUTE BREAKDOWN")
    print(f"{'='*60}")

    io_times = []
    compute_times = []

    for _ in range(num_runs):
        sample = dataset[index]
        timings = sample['_timings']

        # I/O operations
        io_time = sum(timings.get(k, 0) for k in ['1_load_rgb', '2_load_depth', '3_load_segm', '4_load_hdf5'])

        # Compute operations
        compute_time = sum(timings.get(k, 0) for k in ['5_resize', '6_color_aug', '7_geometric_aug', '8_to_tensor'])

        io_times.append(io_time)
        compute_times.append(compute_time)

    io_mean = np.mean(io_times) * 1000
    compute_mean = np.mean(compute_times) * 1000
    total = io_mean + compute_mean

    print(f"\n{'Category':<20} {'Time (ms)':<15} {'Percentage':<10}")
    print("-" * 50)
    print(f"{'I/O (disk read)':<20} {io_mean:<15.3f} {(io_mean/total)*100:<10.1f}%")
    print(f"{'Compute':<20} {compute_mean:<15.3f} {(compute_mean/total)*100:<10.1f}%")
    print("-" * 50)
    print(f"{'Total':<20} {total:<15.3f}")

    return io_mean, compute_mean


def main():
    parser = argparse.ArgumentParser(description='Profile GraspNet DataLoader')
    parser.add_argument('--root', type=str, default='/data/Grasp/GraspNet-1Billion',
                        help='Path to GraspNet dataset')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to profile')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for DataLoader profiling')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of DataLoader workers')
    parser.add_argument('--split', type=str, default='debug',
                        help='Dataset split (debug, train, test)')
    parser.add_argument('--augmentation', type=str, default=None,
                        help='Augmentation type (None, color, crop, color,crop)')
    parser.add_argument('--image_size', type=int, nargs=2, default=[720, 1280],
                        help='Image size (H W)')
    parser.add_argument('--profile_workers', action='store_true',
                        help='Profile worker scaling')
    parser.add_argument('--max_workers', type=int, default=16,
                        help='Maximum workers to test for scaling')

    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print("# GraspNet DataLoader Profiler")
    print(f"{'#'*60}")
    print(f"\nConfiguration:")
    print(f"  Dataset root: {args.root}")
    print(f"  Split: {args.split}")
    print(f"  Image size: {args.image_size}")
    print(f"  Augmentation: {args.augmentation}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Num workers: {args.num_workers}")

    # Create dataset
    dataset = ProfiledGraspNetDataset(
        root=args.root,
        camera='realsense',
        split=args.split,
        image_size=tuple(args.image_size),
        augmentation=args.augmentation
    )

    print(f"  Dataset size: {len(dataset)} samples")

    # 1. Single sample profiling
    profile_single_sample(dataset, index=0, num_runs=10)

    # 2. I/O vs Compute breakdown
    profile_io_vs_compute(dataset, index=0, num_runs=5)

    # 3. DataLoader profiling
    profile_dataloader(
        dataset,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # 4. Worker scaling (optional)
    if args.profile_workers:
        profile_worker_scaling(
            dataset,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            max_workers=args.max_workers
        )

    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}")
    print("""
Based on typical bottlenecks, consider:

1. If I/O is the bottleneck (>60%):
   - Use SSD storage for dataset
   - Pre-load data into RAM (if memory allows)
   - Use memory-mapped files (np.memmap)
   - Convert HDF5 to faster formats (LMDB, TFRecord)
   - Increase prefetch_factor

2. If HDF5 loading is slow:
   - Store GT maps as separate numpy files
   - Use chunked HDF5 with compression
   - Pre-compute and cache resized versions

3. If augmentation is slow:
   - Use GPU-based augmentation (kornia, DALI)
   - Reduce augmentation complexity
   - Cache augmented samples

4. If resize is slow:
   - Pre-resize images to target size
   - Use faster interpolation (cv2.INTER_AREA for downscaling)

5. General optimizations:
   - Increase num_workers (but watch memory)
   - Use pin_memory=True with CUDA
   - Use persistent_workers=True
   - Profile with larger batch sizes
""")


if __name__ == '__main__':
    main()
