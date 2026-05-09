import os

from ml.train import generate_synthetic_data


def main():
    os.makedirs("data/processed", exist_ok=True)

    df = generate_synthetic_data(n_rows=1000, random_state=42)

    output_path = "data/processed/synthetic_rendimento.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Dataset sintético salvo em: {output_path}")
    print(f"Linhas: {df.shape[0]}")
    print(f"Colunas: {df.shape[1]}")
    print(df.head())


if __name__ == "__main__":
    main()