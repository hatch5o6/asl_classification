import argparse
import json
import csv

def json2csv(json_file):
    assert json_file.endswith(".json")
    with open(json_file):
        data = json.load(json_file)
    csv_file = json_file[:-5] + ".csv"
    with open(csv_file, "w", newline='') as outf:
        writer = csv.writer(outf)
        for video_file, length in data.items():
            writer.writerow[video_file, length]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--json_file", help=".json file")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    json2csv(args.json_file)