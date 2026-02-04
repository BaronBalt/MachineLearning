from pathlib import Path

from sklearn.datasets import load_iris


def main():
    iris = load_iris(as_frame=True)
    df = iris.frame

    # Rename target to match pipeline expectation
    df = df.rename(columns={"target": "target"})

    output_dir = Path("../data")
    output_dir.mkdir(exist_ok=True)

    df.to_csv(output_dir / "train.csv", index=False)
    print("Generated data/train.csv")

if __name__ == "__main__":
    main()
