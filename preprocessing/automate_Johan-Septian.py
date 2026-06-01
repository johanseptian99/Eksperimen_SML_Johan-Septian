import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

def preprocess_data(
    input_path,
    output_folder="dataset_preprocessing",
    test_size=0.2,
    random_state=42
):
    """
    Preprocessing dataset secara otomatis.

    Output:
    - train.csv
    - test.csv
    """

    os.makedirs(output_folder, exist_ok=True)

    print(f"Folder output: {output_folder}")

    df = pd.read_csv(input_path)

    print("Dataset berhasil dimuat")
    print("Shape awal:", df.shape)

    df["Acidity"] = pd.to_numeric(
        df["Acidity"],
        errors="coerce"
    )

    df.dropna(inplace=True)


    df.drop(
        columns=["A_id"],
        inplace=True
    )

    df.drop_duplicates(inplace=True)

    encoder = LabelEncoder()

    df["Quality"] = encoder.fit_transform(
        df["Quality"]
    )

    X = df.drop("Quality", axis=1)
    y = df["Quality"]


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print("Train shape:", X_train.shape)
    print("Test shape :", X_test.shape)

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    train_df = pd.DataFrame(
        X_train_scaled,
        columns=X.columns
    )

    train_df["Quality"] = (
        y_train.reset_index(drop=True)
    )

    test_df = pd.DataFrame(
        X_test_scaled,
        columns=X.columns
    )

    test_df["Quality"] = (
        y_test.reset_index(drop=True)
    )

    train_csv = os.path.join(
        output_folder,
        "train.csv"
    )

    test_csv = os.path.join(
        output_folder,
        "test.csv"
    )

    train_df.to_csv(
        train_csv,
        index=False
    )

    test_df.to_csv(
        test_csv,
        index=False
    )

    print("\nPreprocessing selesai")

    return {
        "train_df": train_df,
        "test_df": test_df,
    }


if __name__ == "__main__":

    preprocess_data(
        input_path="../dataset_raw/apple_quality.csv",
        output_folder="dataset_preprocessing"
    )