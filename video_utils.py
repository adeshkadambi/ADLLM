import concurrent.futures
import enum
import os
import random
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


class SamplingStrategy(enum.StrEnum):
    UNIFORM = "uniform"
    RANDOM = "random"
    MOTION_GUIDED = "motion_guided"

    def get_frame_indices(self, total_frames: int, num_frames: int, cap=None) -> list[int]:
        """Get frame indices based on the sampling strategy."""
        if num_frames > total_frames:
            raise ValueError(f"Cannot sample {num_frames} frames from {total_frames} total frames")

        strategies: dict[SamplingStrategy, Callable] = {
            SamplingStrategy.UNIFORM: self._uniform_sampling,
            SamplingStrategy.RANDOM: self._random_sampling,
            SamplingStrategy.MOTION_GUIDED: self._motion_guided_sampling,
        }

        if self == SamplingStrategy.MOTION_GUIDED and cap is not None:
            return strategies[self](total_frames, num_frames, cap)
        return strategies[self](total_frames, num_frames)

    @staticmethod
    def _uniform_sampling(total_frames: int, num_frames: int) -> list[int]:
        if num_frames == 1:
            return [total_frames // 2]
        return [int(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)]

    @staticmethod
    def _random_sampling(total_frames: int, num_frames: int) -> list[int]:
        return sorted(random.sample(range(total_frames), num_frames))

    @staticmethod
    def _motion_guided_sampling(
        total_frames: int, num_frames: int, cap, mu: float = 0.5
    ) -> list[int]:
        """
        Implements motion-guided sampling with efficient frame reading and motion calculation.
        """
        motion_magnitudes = []
        num_frames = num_frames + 1

        # Read first frame
        ret, prev_frame = cap.read()
        if not ret:
            return SamplingStrategy.UNIFORM.get_frame_indices(total_frames, num_frames)

        # Sequential frame reading
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate frame difference using numpy operations
            diff = np.abs(frame.astype(np.int16) - prev_frame.astype(np.int16))
            magnitude = np.sum(diff)
            motion_magnitudes.append(magnitude)

            prev_frame = frame

        # Rest of the processing remains the same
        if motion_magnitudes:
            motion_magnitudes = np.array(motion_magnitudes)
            motion_magnitudes = motion_magnitudes / np.sum(motion_magnitudes)
            motion_magnitudes = np.power(motion_magnitudes, mu)
            motion_magnitudes = motion_magnitudes / np.sum(motion_magnitudes)

            cum_distribution = np.cumsum(motion_magnitudes)
            splits = np.linspace(0, 1, num_frames + 1)[1:-1]
            frame_indices = []

            for split in splits:
                frame_idx = np.searchsorted(cum_distribution, split)
                frame_indices.append(frame_idx)

            if len(frame_indices) > num_frames:
                frame_indices = frame_indices[:num_frames]

            return sorted(frame_indices)

        return SamplingStrategy.UNIFORM.get_frame_indices(total_frames, num_frames)


def _convert_frame_to_pil(frame: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR frame to PIL RGB Image."""
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def _save_frame(frame: Image.Image, output_dir: Path, frame_number: int) -> None:
    """Save a single frame to disk."""
    frame.save(output_dir / f"frame_{frame_number:05d}.jpg", "JPEG")


def _process_frame_batch(args: tuple[cv2.VideoCapture, list[int], bool, Path]) -> list[Image.Image]:
    """Process a batch of frames from the video."""
    cap, frame_indices, save_frames, output_dir = args
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        pil_frame = _convert_frame_to_pil(frame)
        frames.append(pil_frame)

        if save_frames:
            _save_frame(pil_frame, output_dir, idx)

    return frames


def extract_frames(
    path: str,
    sampling_method: SamplingStrategy,
    num_frames: int,
    save_frames: bool = False,
) -> tuple[list[Image.Image], list[int], int]:
    """
    Extract frames from a video file using specified sampling method.

    Args:
        video_path: Path to the video file
        sampling_method: Method to use for sampling frames
        num_frames: Number of frames to extract
        save_frames: Whether to save frames to disk

    Returns:
        list of PIL Image objects
        list of frame indices
        total number of frames in the video
    """
    video_path = Path(path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if num_frames > total_frames:
        raise ValueError(f"Requested frames ({num_frames}) exceeds video length ({total_frames})")

    # Get frame indices based on sampling method
    if sampling_method == SamplingStrategy.MOTION_GUIDED:
        frame_indices = sampling_method.get_frame_indices(total_frames, num_frames, cap)
    else:
        frame_indices = sampling_method.get_frame_indices(total_frames, num_frames)

    # Create output directory if saving frames
    output_dir = None
    if save_frames:
        output_dir = video_path.parent / f"{video_path.stem}_{sampling_method}"
        output_dir.mkdir(exist_ok=True)

    # Split frame indices into batches for parallel processing
    num_workers = min(os.cpu_count() or 1, len(frame_indices))
    batch_size = max(1, len(frame_indices) // num_workers)
    batches = [frame_indices[i : i + batch_size] for i in range(0, len(frame_indices), batch_size)]

    frames = []
    # Process batches in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_batch = [
            executor.submit(
                _process_frame_batch,
                (cv2.VideoCapture(str(video_path)), batch, save_frames, output_dir),
            )
            for batch in batches
        ]

        for future in concurrent.futures.as_completed(future_to_batch):
            frames.extend(future.result())

    cap.release()
    return frames, frame_indices, total_frames
