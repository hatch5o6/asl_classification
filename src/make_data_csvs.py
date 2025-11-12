import argparse
import os
import csv

def make_csvs(
    d_path,
    labels_f,
    out_f
):
    assert out_f.endswith(".csv")
    data = {}
    assert d_path.endswith("/test") or d_path.endswith("/train") or d_path.endswith("/val")
    skel_dir_path = d_path + "_skel"
    GET_SKEL = os.path.exists(skel_dir_path)

    for f in os.listdir(d_path):
        assert any([
            f.endswith("_color.mp4"),
            f.endswith("_depth.mp4")
        ])
        signer = f.split("_")[0]
        sample = f.split("_")[1]
        if (signer, sample) not in data:
            data[(signer, sample)] = {"color": None, "depth": None, "skel": None}

        f_path = os.path.join(d_path, f)
        if f.endswith("_color.mp4"):
            assert data[(signer, sample)]["color"] is None
            data[(signer, sample)]["color"] = f_path

            assert data[(signer, sample)]["skel"] is None
            if GET_SKEL:
                skel_path = os.path.join(skel_dir_path, f[:-4] + "_landmarks.npy")
                assert os.path.exists(skel_path)
                data[(signer, sample)]["skel"] = skel_path

        elif f.endswith("_depth.mp4"):
            assert data[(signer, sample)]["depth"] is None
            data[(signer, sample)]["depth"] = f_path
        # elif f.endswith("_color.skeleton.npy"):
        #     assert data[(signer, sample)]["skel"] is None
        #     data[(signer, sample)]["skel"] = f_path
    
    labels = read_labels(labels_f)

    with open(out_f, 'w', newline='') as outf:
        writer = csv.writer(outf)
        writer.writerow(["rgb_path", "depth_path", "skel_path", "label"])
        for (signer, sample), video_files in data.items():
            label = labels[(signer, sample)]
            writer.writerow([video_files["color"], video_files["depth"], video_files["skel"], label])

def read_labels(f):
    labels = {}
    with open(f, newline='') as inf:
        rows = [tuple(r) for r in csv.reader(inf)]
    for signer_sample, label in rows:
        signer, sample = tuple(signer_sample.split("_"))
        assert (signer,sample) not in labels
        labels[(signer,sample)] = int(label)
    return labels


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", help="directory of data files")
    parser.add_argument("-l", "--labels", help="path to labels")
    parser.add_argument("--out", help="path to .csv file")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    make_csvs(
        d_path=args.dir, 
        labels_f=args.labels, 
        out_f=args.out)