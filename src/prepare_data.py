from pathlib import Path
import pandas as pd

RAW_DATA = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
PROCESSED_DATA = Path("data/processed/telco.csv")


def main():
    print("Loading raw dataset...")
    df = pd.read_csv(RAW_DATA)

    print(f"Original shape: {df.shape}")

    # Convert TotalCharges to numeric.
    # Empty strings become NaN.
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip(),
        errors="coerce"
    )

    # Validate missing TotalCharges.
    missing = df["TotalCharges"].isna()

    if missing.any():
        invalid_rows = df.loc[missing]

        # Missing accumulated charges are only valid
        # for customers with zero tenure.
        if not (invalid_rows["tenure"] == 0).all():
            raise ValueError(
                "Missing TotalCharges found for customers "
                "whose tenure is not zero."
            )

        print(f"Replacing {missing.sum()} empty TotalCharges with 0.")

        df.loc[missing, "TotalCharges"] = 0.0

    # Final validation.
    if df.isnull().any().any():
        raise ValueError("Dataset still contains missing values.")

    if df["TotalCharges"].dtype not in ["float64", "int64"]:
        raise TypeError("TotalCharges was not converted correctly.")

    PROCESSED_DATA.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(PROCESSED_DATA, index=False)

    print("\n========== PREPARATION SUMMARY ==========")
    print(f"Original records: {len(df)}")
    print(f"Processed records: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"TotalCharges dtype: {df['TotalCharges'].dtype}")
    print(f"Output: {PROCESSED_DATA}")


if __name__ == "__main__":
    main()
