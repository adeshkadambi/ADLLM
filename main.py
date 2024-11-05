import os
import json
import argparse
from datetime import datetime

from PIL import Image
from tqdm import tqdm

import video_utils as vu
from inference import ADLClassifier


def find_videos(directory: str) -> list:
    """
    Recursively find all video files in directory and its subdirectories.

    Args:
        directory: Root directory to search

    Returns:
        List of tuples containing (full_path, relative_path) for each video
    """
    video_extensions = {".mp4", ".avi", ".mov", ".MP4", ".AVI", ".MOV"}
    videos = []

    print("Scanning for videos...")
    # Convert to absolute path for consistent relative path calculation
    abs_directory = os.path.abspath(directory)

    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in video_extensions):
                full_path = os.path.join(root, file)
                # Calculate path relative to input directory
                rel_path = os.path.relpath(full_path, abs_directory)
                videos.append((full_path, rel_path))

    print(f"Found {len(videos)} videos")
    return videos


def process_videos(video_dir: str, num_frames: int, output_dir: str = None) -> None:
    """
    Process all videos in directory and subdirectories and save their analysis results.

    Args:
        video_dir: Directory containing the videos
        num_frames: Number of frames to sample from each video
        output_dir: Directory to save results (default: creates 'results' in video_dir)
    """
    # Set up output directory
    if output_dir is None:
        output_dir = os.path.join(video_dir, "results")

    # Create timestamp for this batch
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = os.path.join(output_dir, f"batch_{timestamp}")

    # Create directories
    os.makedirs(batch_dir, exist_ok=True)
    images_dir = os.path.join(batch_dir, "grids")
    os.makedirs(images_dir, exist_ok=True)

    # Initialize classifier
    clf = ADLClassifier()

    # Get all video files recursively
    video_files = find_videos(video_dir)

    if not video_files:
        print(f"No video files found in {video_dir} or its subdirectories")
        return

    # Dictionary to store all results
    results = {}

    # Process each video with progress bar
    pbar = tqdm(video_files, desc="Processing videos", unit="video")

    for video_path, rel_path in pbar:
        try:
            # Update progress bar description
            pbar.set_description(f"Processing {os.path.basename(rel_path)}")

            # Extract frames
            sampled_frames, sampled_indices, total_frames = vu.extract_frames(
                path=video_path,
                sampling_method=vu.SamplingStrategy.UNIFORM,
                num_frames=num_frames,
                save_frames=False,
            )

            # Get prediction
            response = clf.predict(sampled_frames, sampled_indices, total_frames)

            # Preserve directory structure in output
            rel_dir = os.path.dirname(rel_path)
            if rel_dir:
                os.makedirs(os.path.join(images_dir, rel_dir), exist_ok=True)

            # Save grid image
            grid_filename = (
                f"{os.path.splitext(os.path.basename(video_path))[0]}_grid.png"
            )
            grid_path = os.path.join(images_dir, rel_dir, grid_filename)

            # Save the grid image from the response
            if isinstance(response["Image_Grid"], Image.Image):
                response["Image_Grid"].save(grid_path)

            # Store results using relative path as key
            results[rel_path] = {
                "ADL": response["ADL"],
                "Reasoning": response["Reasoning"],
                "Intermediate_Steps": response["Intermediate_Steps"],
                "grid_path": os.path.relpath(grid_path, batch_dir),
            }

        except Exception as e:
            print(f"\nError processing {rel_path}: {str(e)}")
            results[rel_path] = {"error": str(e)}
            # Update progress bar postfix with error
            pbar.set_postfix({"status": "error"})

    # Close progress bar
    pbar.close()

    # Save results to JSON with progress indication
    print("\nSaving results...")
    results_path = os.path.join(batch_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    # Create a summary file with ADL counts
    print("Generating summary...")
    adl_counts = {}
    error_count = 0
    for result in results.values():
        if "ADL" in result:
            adl_counts[result["ADL"]] = adl_counts.get(result["ADL"], 0) + 1
        else:
            error_count += 1

    summary_path = os.path.join(batch_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("ADL Classification Summary\n")
        f.write("========================\n\n")
        f.write(f"Total videos processed: {len(video_files)}\n")
        f.write(f"Successfully processed: {len(video_files) - error_count}\n")
        f.write(f"Failed to process: {error_count}\n\n")
        f.write("ADL Counts:\n")
        for adl, count in sorted(adl_counts.items()):
            f.write(f"{adl}: {count} ({count/len(video_files)*100:.1f}%)\n")

    print(f"\nProcessing complete!")
    print(f"Results saved to: {batch_dir}")
    print(
        f"Successfully processed: {len(video_files) - error_count}/{len(video_files)} videos"
    )
    print(f"Check {results_path} for detailed results")
    print(f"Check {summary_path} for ADL classification summary")


def main():
    parser = argparse.ArgumentParser(
        description="Process videos for ADL classification (searches recursively through subdirectories)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Process all videos in directory and subdirectories with default settings
        python script.py /path/to/videos
        
        # Process videos with custom frame count and output directory
        python script.py /path/to/videos -n 6 -o /path/to/output
        """,
    )

    parser.add_argument(
        "video_dir",
        help="Root directory containing videos (will search all subdirectories)",
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

    args = parser.parse_args()

    # Validate video directory
    if not os.path.isdir(args.video_dir):
        parser.error(f"Video directory does not exist: {args.video_dir}")

    # Validate output directory if provided
    if args.output_dir and not os.path.exists(args.output_dir):
        try:
            os.makedirs(args.output_dir)
        except Exception as e:
            parser.error(f"Could not create output directory: {str(e)}")

    # Validate number of frames
    if args.num_frames < 1:
        parser.error("Number of frames must be positive")

    # Process videos
    process_videos(args.video_dir, args.num_frames, args.output_dir)


if __name__ == "__main__":
    main()
