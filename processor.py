# mypy: ignore-errors

import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime

from PIL import Image

import video_utils as vu
from inference import ADLClassifier


def find_videos(directory: str) -> list:
    """Recursively find all videos in directory and its subdirectories."""
    video_extensions = {".mp4", ".avi", ".mov", ".MP4", ".AVI", ".MOV"}
    videos = []

    abs_directory = os.path.abspath(directory)

    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in video_extensions):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, abs_directory)
                videos.append((full_path, rel_path))

    return videos


class BatchProcessor:
    def __init__(
        self,
        video_dir: str,
        num_frames: int,
        output_dir: str | None = None,
        max_retry: int = 3,
        retry_delay: float = 2.0,
        resume: bool = True,
        model: str = "llama3.2-vision:latest",
    ):
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.output_dir = output_dir or os.path.join(video_dir, "results")
        self.resume = resume
        self.max_retry = max_retry
        self.retry_delay = retry_delay

        # Create timestamp for this batch
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.batch_dir = os.path.join(self.output_dir, f"batch_{self.timestamp}")
        self.images_dir = os.path.join(self.batch_dir, "grids")
        self.results_path = os.path.join(self.batch_dir, "results.json")
        self.progress_path = os.path.join(self.batch_dir, "progress.json")

        # Initialize results dictionary
        self.results = {}
        self.processed_videos = set()

        # Set up signal handling for graceful interruption
        signal.signal(signal.SIGINT, self.handle_interrupt)
        signal.signal(signal.SIGTERM, self.handle_interrupt)

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

        # Initialize classifier
        self.clf = ADLClassifier(model=model)

    def handle_interrupt(self, signum, frame):
        """Handle interruption signals by saving progress before exiting."""

        self.logger.warning("Received interruption signal. Saving progress...")
        self.save_results()
        self.logger.info("Progress saved to: %s", self.batch_dir)
        sys.exit(0)

    def ensure_dir_exists(self, path):
        """Ensure directory exists, creating it if necessary."""
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    def save_results(self):
        """Save current results and progress."""

        os.makedirs(self.batch_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)

        # Save results
        with open(self.results_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=4)

        self.logger.info("Results saved to: %s", self.results_path)

        # Save progress
        progress = {
            "processed_videos": list(self.processed_videos),
            "last_update": datetime.now().isoformat(),
            "completed": False,
        }
        with open(self.progress_path, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=4)

        self.logger.info("Progress saved to: %s", self.progress_path)

    def save_final_summary(self):
        """Save final summary and mark batch as completed."""
        # Calculate summary statistics
        adl_counts = {}
        error_count = 0
        for result in self.results.values():
            if "Activity" in result:
                adl_counts[result["Activity"]] = adl_counts.get(result["Activity"], 0) + 1
            else:
                error_count += 1

        # Save summary
        summary_path = os.path.join(self.batch_dir, "summary.txt")
        total_videos = len(self.results)

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("ADL Classification Summary\n")
            f.write("========================\n\n")
            f.write(f"Total videos processed: {total_videos}\n")
            f.write(f"Successfully processed: {total_videos - error_count}\n")
            f.write(f"Failed to process: {error_count}\n\n")
            f.write("ADL Counts:\n")
            for adl, count in sorted(adl_counts.items()):
                f.write(f"{adl}: {count} ({count/total_videos*100:.1f}%)\n")

        # Mark batch as completed
        progress = {
            "processed_videos": list(self.processed_videos),
            "last_update": datetime.now().isoformat(),
            "completed": True,
        }
        with open(self.progress_path, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=4)

    def find_latest_batch(self):
        """Find the most recent incomplete batch directory."""
        if not os.path.exists(self.output_dir):
            return None

        batch_dirs = [
            d
            for d in os.listdir(self.output_dir)
            if d.startswith("batch_") and os.path.isdir(os.path.join(self.output_dir, d))
        ]

        if not batch_dirs:
            return None

        latest_batch = max(batch_dirs)
        latest_dir = os.path.join(self.output_dir, latest_batch)

        # Check if this batch was completed
        progress_path = os.path.join(latest_dir, "progress.json")
        if os.path.exists(progress_path):
            with open(progress_path, "r", encoding="utf-8") as f:
                progress = json.load(f)
                if progress.get("completed", False):
                    return None

        return latest_dir

    def load_progress(self):
        """Load progress from the latest batch if it exists."""
        if not self.resume:
            return

        latest_batch = self.find_latest_batch()
        if latest_batch:
            results_path = os.path.join(latest_batch, "results.json")
            progress_path = os.path.join(latest_batch, "progress.json")

            if os.path.exists(results_path) and os.path.exists(progress_path):
                with open(results_path, "r", encoding="utf-8") as f:
                    self.results = json.load(f)
                with open(progress_path, "r", encoding="utf-8") as f:
                    progress = json.load(f)

                self.processed_videos = set(progress.get("processed_videos", []))
                self.batch_dir = latest_batch
                self.images_dir = os.path.join(self.batch_dir, "grids")
                self.results_path = results_path
                self.progress_path = progress_path

                self.logger.info("Resuming from batch: %s", os.path.basename(self.batch_dir))
                self.logger.info("Already processed: %d videos", len(self.processed_videos))

    def process_single_video(
        self, video_path, sampled_frames, sampled_indices, total_frames, retry_counter=0
    ):
        """Process a single video and return the results."""
        try:
            response = self.clf.predict(sampled_frames, sampled_indices, total_frames)
            return response, None

        except Exception as e:
            if retry_counter < self.max_retry:
                self.logger.warning("Error processing %s. Trying again...", video_path)
                self.logger.warning("Retrying in %s seconds...", self.retry_delay)
                time.sleep(self.retry_delay)

                return self.process_single_video(
                    video_path,
                    sampled_frames,
                    sampled_indices,
                    total_frames,
                    retry_counter + 1,
                )

            else:
                self.logger.error(
                    "Failed to process %s after %d retries: %s",
                    video_path,
                    self.max_retry,
                    str(e),
                )
                return None, str(e)

    def process_videos(self):
        """Process all videos with incremental saving."""
        # Find all videos
        video_files = [
            (full, rel)
            for full, rel in find_videos(self.video_dir)
            if rel not in self.processed_videos
        ]

        if not video_files:
            self.logger.info("No new videos found to process in %s", self.video_dir)
            return

        self.logger.info("Found %d new videos to process", len(video_files))

        # Ensure base directories exist
        os.makedirs(self.batch_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)

        processed_this_session = 0

        for video_path, rel_path in video_files:
            # Track progress as a fraction and percentage using logging
            progress_fraction = processed_this_session / len(video_files)
            progress_percent = progress_fraction * 100
            self.logger.info(
                "Progress: %d/%d (%.1f%%)",
                processed_this_session,
                len(video_files),
                progress_percent,
            )

            # Extract frames
            sampled_frames, sampled_indices, total_frames = vu.extract_frames(
                path=video_path,
                sampling_method=vu.SamplingStrategy.UNIFORM,
                num_frames=self.num_frames,
                save_frames=False,
            )

            # Get prediction
            response, error = self.process_single_video(
                video_path, sampled_frames, sampled_indices, total_frames
            )

            if error:
                self.logger.error("Error processing %s: %s", rel_path, error)
                self.results[rel_path] = {"error": error}

                self.processed_videos.add(rel_path)

            else:
                # Create grid image path preserving directory structure
                rel_dir = os.path.dirname(rel_path)
                grid_filename = f"{os.path.splitext(os.path.basename(video_path))[0]}_grid.png"
                grid_rel_path = os.path.join(rel_dir, grid_filename) if rel_dir else grid_filename
                grid_full_path = os.path.join(self.images_dir, grid_rel_path)

                # Ensure the directory for the grid image exists
                self.ensure_dir_exists(grid_full_path)

                # Save grid image
                if isinstance(response["Image_Grid"], Image.Image):
                    response["Image_Grid"].save(grid_full_path)

                # Store results
                self.results[rel_path] = {
                    "prediction": response["Activity"],
                    "alternate_predictions": response["Alternate Activities"],
                    "tags": response["Final List of Tags"],
                    "reasoning": response["Reasoning"],
                    "grid_path": os.path.join("grids", grid_rel_path),
                }

                self.processed_videos.add(rel_path)
                self.logger.info(
                    "Processed %s: %s (Tags: %s)",
                    rel_path,
                    response["Activity"],
                    response["Final List of Tags"],
                )

            self.save_results()
            processed_this_session += 1

        # Final save and summary
        self.save_final_summary()

        self.logger.info("Successfully processed: %d videos", len(self.processed_videos))
        self.logger.info("Results saved to: %s", self.batch_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Process videos for ADL classification with incremental saving",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Process videos with default settings
        python script.py /path/to/videos

        # Start fresh without resuming from previous batch
        python script.py /path/to/videos --no-resume

        # Custom frames and output directory
        python script.py /path/to/videos -n 6 -o /path/to/output
        """,
    )

    parser.add_argument(
        "video_dir",
        help="Root directory containing videos (will search all subdirectories)",
    )

    parser.add_argument(
        "-m",
        "--model",
        default="llama3.2-vision:latest",
        help="Model to use for inference (llama3.2-vision:latest (default), llama3.2-vision:90b)",
    )

    parser.add_argument(
        "-n",
        "--num-frames",
        type=int,
        default=4,
        help="Number of frames to sample from each video (default: 4)",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        help='Output directory for results (default: creates "results" in video directory)',
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start a new batch instead of resuming from the latest",
    )

    args = parser.parse_args()

    # Validate video directory
    if not os.path.isdir(args.video_dir):
        parser.error(f"Video directory does not exist: {args.video_dir}")

    print("Processing videos with the following settings:")
    print(f"Video directory: {args.video_dir}")
    print(f"Model: {args.model}")
    print(f"Number of frames: {args.num_frames}")
    print(f"Output directory: {args.output_dir}")

    # Create processor and run
    processor = BatchProcessor(
        args.video_dir,
        args.num_frames,
        args.output_dir,
        model=args.model,
        resume=not args.no_resume,
    )

    processor.load_progress()
    processor.process_videos()


if __name__ == "__main__":
    main()
