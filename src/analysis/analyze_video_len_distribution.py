import argparse
import json
import statistics

def main(json_file):
    assert json_file.endswith(".json")
    with open(json_file) as inf:
        data = json.load(inf)

    data_list = [(video, length) for video, length in data.items() if video != "avg"]
    lengths = [l for v, l in data_list]
    median_len = statistics.median(lengths)
    min_len = min(lengths)
    max_len = max(lengths)
    st_dev = statistics.stdev(lengths)
    data["median"] = median_len
    data["min"] = min_len
    data["max_len"] = max_len
    data["st_dev"] = st_dev

    bucket_0 = []
    bucket_0b = []
    bucket_1 = []
    bucket_1b = []
    bucket_2 = []
    bucket_2b = []
    bucket_3 = []
    bucket_3b = []
    bucket_4 = []
    bucket_4b = []
    bucket_5 = []

    for l in lengths:
        if l < 0.5:
            bucket_0.append(l)
        elif l >= 0.5 and l < 1:
            bucket_0b.append(l)
        elif l >= 1 and l < 1.5:
            bucket_1.append(l)
        elif l >= 1.5 and l < 2:
            bucket_1b.append(l)
        elif l >= 2 and l < 2.5:
            bucket_2.append(l)
        elif l >= 2.5 and l < 3:
            bucket_2b.append(l)
        elif l >= 3 and l < 3.5:
            bucket_3.append(l)
        elif l >= 3.5 and l < 4:
            bucket_3b.append(l)
        elif l >= 4 and l < 4.5:
            bucket_4.append(l)
        elif l >= 4.5 and l < 5:
            bucket_4b.append(l)
        elif l >= 5:
            bucket_5.append(l)
    
    data["0.0-0.5"] = [len(bucket_0), len(bucket_0) / len(lengths)]
    data["0.5-1.0"] = [len(bucket_0b), len(bucket_0b) / len(lengths)]

    data["1.0-1.5"] = [len(bucket_1), len(bucket_1) / len(lengths)]
    data["1.5-2.0"] = [len(bucket_1b), len(bucket_1b) / len(lengths)]

    data["2.0-2.5"] = [len(bucket_2), len(bucket_2) / len(lengths)]
    data["2.5-3.0"] = [len(bucket_2b), len(bucket_2b) / len(lengths)]

    data["3.0-3.5"] = [len(bucket_3), len(bucket_3) / len(lengths)]
    data["3.5-4.0"] = [len(bucket_3b), len(bucket_3b) / len(lengths)]

    data["4.0-4.5"] = [len(bucket_4), len(bucket_4) / len(lengths)]
    data["4.5-5.0"] = [len(bucket_4b), len(bucket_4b) / len(lengths)]

    data["5.0+"] = [len(bucket_5), len(bucket_5) / len(lengths)]
    
    out_file = json_file[:-5] + ".metrics.json"
    with open(out_file, "w") as outf:
        outf.write(json.dumps(data, ensure_ascii=False, indent=2))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(args.json_file)