import argparse
import os
from tqdm import tqdm
import json
from moviepy.editor import VideoFileClip
from json_to_csv import json2csv

def main(folder, out_file):
    video_lengths = {}
    for video_file in tqdm(os.listdir(folder)):
        if video_file.endswith("_depth.mp4"): continue
        video_path = os.path.join(folder, video_file)
        duration = get_duration(video_path)
        assert video_file not in video_lengths
        video_lengths[video_file] = duration
    avg_len = sum(video_lengths.values()) / len(video_lengths)
    video_lengths["avg"] = avg_len
    with open(out_file, "w") as outf:
        outf.write(json.dumps(video_lengths, ensure_ascii=False, indent=2))
    json2csv(out_file)

def get_duration(video_file):
    clip = VideoFileClip(video_file)
    duration = clip.duration
    clip.close()
    return duration


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder")
    parser.add_argument("--out")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(args.folder, args.out)