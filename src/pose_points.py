import os
import cv2
import mediapipe as mp
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='MediaPipe Landmark Extraction')
parser.add_argument('--input_path', required=True, help='Input file or folder path')
parser.add_argument('--output_path', required=False, help='Output folder path for NumPy files')
parser.add_argument('--mode', required=True, choices=['extract', 'display'],
                    help="'extract' to save .npy for a folder, 'display' to show pose of a file")
args = parser.parse_args()

mp_holistic = mp.solutions.holistic

def extract_landmarks(results):
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
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    return image

def process_video_display(input_path):
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
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()
    cv2.destroyAllWindows()

def process_video_extract_landmarks(input_path, output_npy_path):
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
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            frame_landmarks = extract_landmarks(results)
            all_frames.append(frame_landmarks)
    cap.release()
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
        if filename.lower().endswith('.mp4'):
            input_path = os.path.join(input_folder, filename)
            base_filename = os.path.splitext(filename)[0]
            output_npy_path = os.path.join(output_folder, f"{base_filename}_landmarks.npy")
            print(f"Processing: {filename}")
            if not os.path.exists(output_npy_path):
                process_video_extract_landmarks(input_path, output_npy_path)
                print(f"Completed: {filename}")
    print("All processing complete.")
