import os
import argparse

def main(folders):
    ground_set = set(os.listdir(folders[0]))
    print(f"G: {folders[0]}")
    for f in folders:
        print("--------------------------")
        f_set = set(os.listdir(f))
        g_minus_f = ground_set.difference(f_set)
        f_minus_g = f_set.difference(ground_set)
        print(f"F: {f}:")
        print(f"\tG - F = {len(g_minus_f)}")
        print(f"\tF - G = {len(f_minus_g)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folders", help="comma-delimited")
    args = parser.parse_args()
    folders = [f.strip() for f in args.folders.split(",")]
    main(folders)
