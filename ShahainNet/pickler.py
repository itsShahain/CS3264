#this creates the dataset including relevant keypoint LK velocities
#ideas prototyped with the help of Gemini
import cv2
import tensorflow_hub as hub # the magic line that fixes all binary issues ;)
import mediapipe as mp
import numpy as np
import os
import glob
import pickle
from tqdm import tqdm
import traceback

mp_pose = mp.solutions.pose

class MediaPipePoseEstimator:
    def __init__(self, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        try:
            self.pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=model_complexity,
                enable_segmentation=False,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence)
        except Exception as e:
            print(f"Error initializing MediaPipe Pose: {e}")
            raise

    def estimate_pose_4_features(self, image):
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = self.pose.process(image_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                keypoints_4_np = np.array(
                    [[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks],
                    dtype=np.float32
                )
                if keypoints_4_np.shape == (33, 4):
                    return keypoints_4_np
                else:
                    #unexpected landmark shape
                    return None
            else:
                return None 
        except Exception as e:
            #mediapipe error
            return None

    def close(self):
        if hasattr(self, 'pose'):
            self.pose.close()
            print("MediaPipe Pose resources released.")


def derive_label_from_filename(video_path):
    filename = os.path.basename(video_path)
    parts = filename.lower().replace('_', '-').split('-')
    if len(parts) >= 1:
        label_str = parts[0]
        if label_str == "fall": return 1
        elif label_str == "adl": return 0
        else:
            print(f"Warning: Unknown label prefix '{parts[0]}' in filename: {filename}. Defaulting to 0 (ADL). Consider adding specific handling if needed.")
            return 0
    else:
        raise ValueError(f"Filename '{filename}' format error. Cannot derive label.")


# --- MODIFIED sample_frames to also return FPS ---
def sample_frames(video_path, num_frames_to_sample):
    frames = []
    fps = 0.0
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], fps

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames <= 0:
        print(f"Warning: Video {video_path} reported 0 frames.")
        cap.release()
        return [], fps
    if total_frames <= num_frames_to_sample:
        frame_indices = np.arange(total_frames)
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)

    read_success_count = 0
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
            read_success_count += 1

    cap.release()

    if read_success_count == 0 and total_frames > 0:
         print(f"Error: Failed to read ANY frames from {video_path} despite it reporting {total_frames} total frames.")
         return [], fps

    if len(frames) != len(frame_indices):
         print(f"Warning: Read {len(frames)} frames, expected {len(frame_indices)} based on sampling from {video_path}")

    return frames, fps

def create_pose_data_pickle_7_features(video_dir, output_pickle_file, num_frames=500, pose_estimator_config=None):
    """

    Pickle Structure per video:
    {
        'video_path': str,
        'label': int,
        'frames_data': [
            {'keypoints': np.array(33, 4), 'velocities': np.array(33, 3)}, # Frame 0
            {'keypoints': np.array(33, 4), 'velocities': np.array(33, 3)}, # Frame 1
            ...
        ]
    }

    """
    if not os.path.isdir(video_dir):
        return False

    # Find video files
    video_paths = sorted(glob.glob(os.path.join(video_dir, '*.mp4'))) + \
                  sorted(glob.glob(os.path.join(video_dir, '*.avi'))) + \
                  sorted(glob.glob(os.path.join(video_dir, '*.mov')))
    if not video_paths:
        return False
    print(f"Found {len(video_paths)} potential video files.")

    if pose_estimator_config is None:
        pose_estimator_config = {'model_complexity': 1} 

    pose_estimator = MediaPipePoseEstimator(**pose_estimator_config)
    all_processed_data = []
    skipped_videos = 0
    videos_with_no_pose = 0

    print(f"Starting pre-processing for {len(video_paths)} videos...")
    for video_path in tqdm(video_paths, desc="Processing Videos"):
        try:
            label = derive_label_from_filename(video_path)
        except ValueError as e:
            print(f"Skipping video {os.path.basename(video_path)}: {e}")
            skipped_videos += 1
            continue

        sampled_frames, fps = sample_frames(video_path, num_frames)
        if not sampled_frames:
            print(f"Skipping video {os.path.basename(video_path)}: No frames could be sampled.")
            skipped_videos += 1
            continue

        if fps > 0:
            delta_t = 1.0 / fps
            using_fps_for_velocity = True
        else:
            delta_t = 1.0
            using_fps_for_velocity = False

        video_frames_data_list = []
        prev_keypoints_4_np = None

        for frame in sampled_frames:
            current_keypoints_4_np = pose_estimator.estimate_pose_4_features(frame)

            if isinstance(current_keypoints_4_np, np.ndarray) and current_keypoints_4_np.shape == (33, 4):
                if prev_keypoints_4_np is not None:
                    delta_coords = current_keypoints_4_np[:, :3] - prev_keypoints_4_np[:, :3]
                    velocities_np = delta_coords / delta_t
                else:
                    velocities_np = np.zeros((33, 3), dtype=np.float32)
                frame_dict = {
                    'keypoints': current_keypoints_4_np,
                    'velocities': velocities_np
                }
                video_frames_data_list.append(frame_dict)

                prev_keypoints_4_np = current_keypoints_4_np

        if video_frames_data_list:
            all_processed_data.append({
                'video_path': video_path,
                'label': label,
                'frames_data': video_frames_data_list
            })

            if len(all_processed_data) == 1:
                print(f"Velocity Calculation Note: Using {'FPS ({:.2f})'.format(fps) if using_fps_for_velocity else 'Frame Index Difference'} for delta_t.")
        else:

            videos_with_no_pose += 1
            skipped_videos += 1 

    pose_estimator.close()

    print(f"\nPre-processing complete")
    try:
        output_dir = os.path.dirname(output_pickle_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        with open(output_pickle_file, 'wb') as f:
            pickle.dump(all_processed_data, f)
        print("Data saved successfully.")
        return True
    except Exception as e:
        print(f"Error saving data to pickle file: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":

    VIDEO_SOURCE_DIR = 'val' 
    NUM_FRAMES_TO_SAMPLE = 50     
    OUTPUT_PICKLE_NAME = f'{os.path.basename(VIDEO_SOURCE_DIR)}_ur_fall_{NUM_FRAMES_TO_SAMPLE}_frames_7_features.pkl' 
    OUTPUT_DIR = 'preprocessed_data'
    OUTPUT_PICKLE_PATH = os.path.join(OUTPUT_DIR, OUTPUT_PICKLE_NAME)

    MP_CONFIG = {
        'model_complexity': 1,          
        'min_detection_confidence': 0.5,
        'min_tracking_confidence': 0.5
    }

    print("--- Starting Pose Data Pickle Creation (7 Features: XYZ+Vis + Vxyz) ---")
    print(f"Video Source Directory: '{VIDEO_SOURCE_DIR}'")
    print(f"Frames to Sample per Video: {NUM_FRAMES_TO_SAMPLE}")
    print(f"Output Pickle Path: '{OUTPUT_PICKLE_PATH}'")
    print(f"MediaPipe Config: {MP_CONFIG}")
    print("-" * 60)


    success = create_pose_data_pickle_7_features(
        video_dir=VIDEO_SOURCE_DIR,
        output_pickle_file=OUTPUT_PICKLE_PATH,
        num_frames=NUM_FRAMES_TO_SAMPLE,
        pose_estimator_config=MP_CONFIG
    )

    print("-" * 60)
    if success:
        print(f"--- Pickle file created successfully at '{OUTPUT_PICKLE_PATH}' ---")
    else:
        print("--- Pickle file creation failed ---")
