"""Preprocess SemEval-2018 Task 3 data into standard TSV format."""
import csv, os, argparse, re
from src.data.preprocessing import preprocess_tweet

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default="data/semeval/splits")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    for split in ["train", "test"]:
        inpath = os.path.join(args.input_dir, f"SemEval2018-T3_{split}.txt")
        outpath = os.path.join(args.output_dir, f"{split}.tsv")
        with open(inpath, encoding="utf-8") as fin, open(outpath, "w", encoding="utf-8", newline="") as fout:
            writer = csv.writer(fout, delimiter="\t")
            writer.writerow(["text", "label"])
            reader = csv.reader(fin, delimiter="\t")
            next(reader)  # header
            for row in reader:
                if len(row) >= 3:
                    text = preprocess_tweet(row[2])
                    label = int(row[1])
                    writer.writerow([text, label])

if __name__ == "__main__":
    main()
