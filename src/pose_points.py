import os
import cv2
import mediapipe as mp
import numpy as np
import argparse
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Tuple

# Parse command-line arguments
parser = argparse.ArgumentParser(description='MediaPipe Landmark Extraction')
parser.add_argument('--input_path', required=True, help='Input folder containing MP4 files')
parser.add_argument('--output_path', required=True, help='Output folder for NumPy files')
parser.add_argument('--mode', required=True, choices=['extract', 'display'],
                    help="'extract' to save .npy files, 'display' to show pose of a file")
parser.add_argument('--num_workers', type=int, default=None,
                    help='Number of parallel workers (default: number of CPU cores)')
parser.add_argument('--force', action='store_true',
                    help='Force re-processing of videos even if output already exists')
args = parser.parse_args()

mp_holistic = mp.solutions.holistic


def extract_landmarks(results):
    """
    Extract landmarks from MediaPipe holistic results.
    Returns a numpy array of shape (543, 4) containing x, y, z, visibility for:
    - Face: 468 landmarks
    - Pose: 33 landmarks
    - Left hand: 21 landmarks
    - Right hand: 21 landmarks
    """
    def landmarks_list(landmarks, expected_len):
        if landmarks:
            return [[lm.x, lm.y, lm.z, getattr(lm, 'visibility', 1.0)] for lm in landmarks.landmark]
        else:
            return [[np.nan, np.nan, np.nan, np.nan]] * expected_len

    pose = landmarks_list(results.pose_landmarks, 33)
    left_hand = landmarks_list(results.left_hand_landmarks, 21)
    right_hand = landmarks_list(results.right_hand_landmarks, 21)
    face = landmarks_list(results.face_landmarks, 468)
    return np.concatenate([face, pose, left_hand, right_hand], axis=0)  # (543,4)


def draw_landmarks_on_image(image, results):
    """Draw MediaPipe landmarks on an image for visualization."""
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    return image


def process_video_display(input_path):
    """Display video with landmarks overlaid in real-time."""
    cap = cv2.VideoCapture(input_path)
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=False,
        min_detection_confidence=.5,
        min_tracking_confidence=.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            output_frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            output_frame = draw_landmarks_on_image(output_frame, results)
            cv2.imshow('Landmarks', output_frame)
            if cv2.waitKey(5) & 0xFF == 27:  # Press ESC to exit
                break
    cap.release()
    cv2.destroyAllWindows()


def process_video_extract_landmarks(input_path, output_npy_path):
    """
    Extract landmarks from a single video and save as numpy file.
    This function is called by worker processes in parallel.
    """
    cap = cv2.VideoCapture(input_path)
    all_frames = []
    
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=False,
        min_detection_confidence=.5,
        min_tracking_confidence=.5) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB for MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            frame_landmarks = extract_landmarks(results)
            all_frames.append(frame_landmarks)
    
    cap.release()
    
    # Stack all frames into a single numpy array
    all_frames_np = np.stack(all_frames)
    np.save(output_npy_path, all_frames_np)
    print(f"Saved numpy: {output_npy_path}")

if args.mode == 'display':
    process_video_display(args.input_path)
elif args.mode == 'extract':
    input_folder = args.input_path
    output_folder = args.output_path
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        assert filename.endswith("_color.mp4") or filename.endswith("_depth.mp4")
        if filename.endswith("_depth.mp4"): continue
        if filename.lower().endswith('.mp4'):
            input_path = os.path.join(input_folder, filename)
            base_filename = os.path.splitext(filename)[0]
            output_npy_path = os.path.join(output_folder, f"{base_filename}_landmarks.npy")
            print(f"Processing: {filename}")
            if not os.path.exists(output_npy_path):
                process_video_extract_landmarks(input_path, output_npy_path)
                print(f"Completed: {filename}")
    print("All processing complete.")
    print(f"✓ Saved: {output_npy_path}")
    return output_npy_path


def process_single_video(video_info: Tuple[str, str, bool]) -> str:
    """
    Wrapper function for processing a single video in parallel.
    Takes a tuple of (input_path, output_path, force_reprocess) and processes the video.
    This function is mapped across multiple workers.
    """
    input_path, output_npy_path, force_reprocess = video_info
    
    try:
        # Check if output already exists and skip unless force flag is set
        if os.path.exists(output_npy_path) and not force_reprocess:
            print(f"⊘ Skipping (already exists): {os.path.basename(input_path)}")
            return output_npy_path
        
        # If forcing reprocess, indicate that we're overwriting
        if os.path.exists(output_npy_path) and force_reprocess:
            print(f"↻ Re-processing (force flag set): {input_path}")
        else:
            print(f"→ Processing: {input_path}")
        
        result = process_video_extract_landmarks(input_path, output_npy_path)
        return result
    except Exception as e:
        print(f"✗ Error processing {input_path}: {str(e)}")
        return None


def find_all_videos(input_folder: str) -> List[str]:
    """
    Find all MP4 files in the input folder.
    
    Args:
        input_folder: Path to folder containing MP4 files
    
    Returns:
        List of full paths to MP4 files
    """
    input_path = Path(input_folder)
    
    if not input_path.exists():
        print(f"✗ Error: Input folder does not exist: {input_folder}")
        return []
    
    if not input_path.is_dir():
        print(f"✗ Error: Input path is not a directory: {input_folder}")
        return []
    
    # Find all MP4 files (case-insensitive)
    mp4_files = list(input_path.glob('*.mp4')) + list(input_path.glob('*.MP4'))
    
    # Convert to strings and sort
    video_paths = sorted([str(f) for f in mp4_files])
    
    return video_paths


def create_video_task_list(video_paths: List[str], 
                           output_folder: str,
                           force_reprocess: bool = False) -> Tuple[List[Tuple[str, str, bool]], int]:
    """
    Create a list of (input_path, output_path, force_flag) tuples for processing.
    
    Args:
        video_paths: List of video file paths
        output_folder: Output directory for .npy files
        force_reprocess: Whether to force reprocessing of existing files
    
    Returns:
        Tuple of (task_list, already_processed_count)
    """
    # Create output folder if it doesn't exist
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    tasks = []
    already_processed = 0
    
    for video_path in video_paths:
        # Get the video filename without extension
        video_file = Path(video_path)
        base_filename = video_file.stem
        
        # Create output .npy filename (matching the video name)
        output_npy_path = output_path / f"{base_filename}.npy"
        
        # Check if output already exists
        if output_npy_path.exists() and not force_reprocess:
            already_processed += 1
        
        tasks.append((video_path, str(output_npy_path), force_reprocess))
    
    return tasks, already_processed


if __name__ == '__main__':
    if args.mode == 'display':
        # Display mode: show a single video with landmarks
        # In display mode, input_path should be a video file
        if not os.path.isfile(args.input_path):
            print(f"✗ Error: For display mode, input_path must be a video file")
            exit(1)
        
        process_video_display(args.input_path)
        
    elif args.mode == 'extract':
        # Extract mode: process all MP4 files in the input folder
        
        # Step 1: Find all MP4 files in input folder
        print("=" * 60)
        print("Step 1: Scanning for MP4 files...")
        print("=" * 60)
        print(f"Input folder: {args.input_path}")
        
        video_paths = find_all_videos(args.input_path)
        
        if not video_paths:
            print("✗ No MP4 files found in the input folder!")
            exit(1)
        
        print(f"✓ Found {len(video_paths)} MP4 files")
        
        # Step 2: Create output directory and task list
        print("\n" + "=" * 60)
        print("Step 2: Preparing output directory...")
        print("=" * 60)
        print(f"Output folder: {args.output_path}")
        
        tasks, already_processed = create_video_task_list(
            video_paths, 
            args.output_path, 
            args.force
        )
        
        # Report on existing outputs
        if already_processed > 0 and not args.force:
            print(f"\n✓ {already_processed} videos already have pose points extracted")
            print(f"→ {len(tasks) - already_processed} videos need processing")
            print(f"\n  (Use --force flag to re-process all videos)")
        elif args.force:
            print(f"\n⚠ Force flag set: will re-process all {len(tasks)} videos")
        else:
            print(f"\n→ All {len(tasks)} videos need processing")
        
        # Show example mapping
        if tasks:
            print(f"\nExample mapping:")
            example_input, example_output, _ = tasks[0]
            print(f"  Input:  {example_input}")
            print(f"  Output: {example_output}")
        
        # Exit early if nothing needs processing
        if already_processed == len(tasks) and not args.force:
            print("\n" + "=" * 60)
            print("All videos already processed! Nothing to do.")
            print("=" * 60)
            print("Use --force flag to re-process existing files.")
            exit(0)
        
        # Step 3: Determine number of workers
        num_workers = args.num_workers if args.num_workers else cpu_count()
        print(f"\n" + "=" * 60)
        print(f"Step 3: Processing with {num_workers} parallel workers...")
        print("=" * 60)
        
        # Step 4: Process videos in parallel
        with Pool(processes=num_workers) as pool:
            # Map the processing function across all tasks
            results = pool.imap_unordered(process_single_video, tasks)
            
            # Track completion
            completed = []
            for i, result in enumerate(results, 1):
                if result:
                    completed.append(result)
                print(f"Progress: {i}/{len(tasks)} videos processed")
        
        # Step 5: Summary
        print("\n" + "=" * 60)
        print("Processing Complete!")
        print("=" * 60)
        print(f"Total videos: {len(tasks)}")
        if not args.force:
            print(f"Already existed (skipped): {already_processed}")
            print(f"Newly processed: {len(completed) - already_processed}")
        else:
            print(f"Successfully processed: {len(completed)}")
        print(f"Failed: {len(tasks) - len(completed)}")
        print(f"\nOutput folder: {args.output_path}")
