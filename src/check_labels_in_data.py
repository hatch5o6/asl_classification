import argparse
import csv

def main(train_csv, class_id_csv):
    train_csv_labels = read_labels_from_train_csv(train_csv)
    print("TRAIN CSV LABELS", len(train_csv_labels), sorted(list(train_csv_labels))[:10])

    class_id_labels = read_labels_from_class_id_csv(class_id_csv)
    print("CLASS ID CSV LABELS", len(class_id_labels), sorted(list(class_id_labels))[:10])

    assert train_csv_labels == class_id_labels
    print("TRAIN CSV LABELS == CLASS ID LABELS :)")

def read_labels_from_class_id_csv(f):
    with open(f, newline='') as inf:
        rows = [tuple(r) for r in csv.reader(inf)]
    header = rows[0]
    assert header == ("ClassId", "TR", "EN")
    data = rows[1:]
    labels = set()
    for class_id, tr_word, en_word in data:
        class_id = int(class_id)
        assert class_id not in labels
        labels.add(class_id)
    return labels

def read_labels_from_train_csv(f):
    with open(f, newline='') as inf:
        rows = [tuple(r) for r in csv.reader(inf)]
    header = rows[0]
    assert header == ("rgb_path", "depth_path", "skel_path", "label")
    data = rows[1:]
    labels = set()
    for rgb_path, depth_path, skel_path, label in data:
        labels.add(int(label))
    return labels

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="/home/hatch5o6/CS650R/asl/data/train.csv")
    parser.add_argument("--class_id_csv", default="/home/hatch5o6/groups/grp_asl_classification/nobackup/archive/AUTSL/class_ids/SignList_ClassId_TR_EN.csv")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(args.train_csv, args.class_id_csv)

