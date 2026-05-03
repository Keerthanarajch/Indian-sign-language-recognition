from src.dataset_builder import build_dataset
from src.train.train_single import train_single
from src.train.train_double import train_double

def main():
    print(" Building dataset...")
    X_single, y_single, X_double, y_double = build_dataset()

    print("\nTraining single-hand model...")
    train_single(X_single, y_single)

    print("\nTraining double-hand model...")
    train_double(X_double, y_double)

    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()