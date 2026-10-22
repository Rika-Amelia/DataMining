from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


IDENTIFIER_COLUMNS = {"ticket_id"}
DATETIME_COLUMNS = ["opened_at", "resolved_at"]


def find_dataset() -> Path:
    folder = Path(__file__).resolve().parent
    candidates = [
        folder / "support_tickets.csv",
        folder / "support-tickets.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    expected = ", ".join(path.name for path in candidates)
    raise FileNotFoundError(f"Dataset tidak ditemukan. Nama file yang dicari: {expected}")


def main() -> None:
    file_path = find_dataset()
    data = pd.read_csv(file_path)
    if data.empty:
        raise ValueError(f"Dataset tidak memiliki baris data: {file_path}")

    missing_required = (IDENTIFIER_COLUMNS | set(DATETIME_COLUMNS)) - set(data.columns)
    if missing_required:
        raise ValueError(
            "Kolom wajib tidak ditemukan: " + ", ".join(sorted(missing_required))
        )

    for column in DATETIME_COLUMNS:
        data[column] = pd.to_datetime(data[column], errors="raise")

    categorical_columns = sorted(
        column
        for column, dtype in data.dtypes.items()
        if column not in DATETIME_COLUMNS
        and (
            column in IDENTIFIER_COLUMNS
            or pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
            or pd.api.types.is_bool_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
        )
    )
    numerical_columns = sorted(
        column
        for column, dtype in data.dtypes.items()
        if column not in IDENTIFIER_COLUMNS
        and column not in DATETIME_COLUMNS
        and pd.api.types.is_numeric_dtype(dtype)
        and not pd.api.types.is_bool_dtype(dtype)
    )
    sorting_columns = categorical_columns + DATETIME_COLUMNS + numerical_columns

    print(f"Dataset: {file_path.name}")
    print(f"Ukuran dataset: {data.shape[0]} baris x {data.shape[1]} kolom")
    print(f"\nKolom kategorikal ({len(categorical_columns)}):")
    if categorical_columns:
        for column in categorical_columns:
            print(f"  - {column}: {data[column].dtype} (kategori/label)")
    else:
        print("(tidak ada)")
    print(f"\nKolom numerikal ({len(numerical_columns)}):")
    if numerical_columns:
        for column in numerical_columns:
            print(f"  - {column}: {data[column].dtype} (angka untuk perhitungan)")
    else:
        print("(tidak ada)")
    print(f"\nKolom tanggal/waktu ({len(DATETIME_COLUMNS)}):")
    for column in DATETIME_COLUMNS:
        print(f"  - {column}: {data[column].dtype}")

    sorted_data = data.sort_values(
        by=sorting_columns, kind="stable", na_position="last"
    )
    print("\n10 baris pertama setelah sorting kategori, tanggal/waktu, numerik:")
    print(sorted_data.head(10).to_string(index=False))

    duplicate_mask = data.duplicated(keep=False)
    duplicate_rows = data.loc[duplicate_mask].sort_values(
        by=sorting_columns, kind="stable", na_position="last"
    )
    print(
        "\nDuplikat: "
        f"{int(data.duplicated().sum())} baris duplikat tambahan; "
        f"{int(duplicate_mask.sum())} baris jika semua kemunculan dihitung."
    )
    if duplicate_rows.empty:
        print("Tidak ditemukan baris duplikat.")
    else:
        print(duplicate_rows.head(20).to_string(index=False))

    null_mask = data.isna().any(axis=1)
    null_rows = data.loc[null_mask].sort_values(
        by=sorting_columns, kind="stable", na_position="last"
    )
    null_cell_count = int(data.isna().to_numpy().sum())
    print(f"\nNull: {null_cell_count} sel; {len(null_rows)} baris memiliki null.")
    print("Jumlah null per kolom:")
    print(data.isna().sum().to_string())
    if not null_rows.empty:
        print("Maksimal 20 baris null pertama setelah sorting:")
        print(null_rows.head(20).to_string(index=False))
    else:
        print("Tidak ditemukan baris null.")

    if numerical_columns:
        zero_mask = data[numerical_columns].to_numpy() == 0
        zero_row_mask = zero_mask.any(axis=1)
        zero_rows = data.loc[zero_row_mask].sort_values(
            by=sorting_columns, kind="stable", na_position="last"
        )
        zero_cell_count = int(np.count_nonzero(zero_mask))
        zero_counts = (data[numerical_columns] == 0).sum()
    else:
        zero_row_mask = np.zeros(len(data), dtype=bool)
        zero_rows = data.iloc[0:0]
        zero_cell_count = 0
        zero_counts = pd.Series(dtype="int64")

    print(f"\nZero pada kolom numerikal: {zero_cell_count} sel.")
    print("Jumlah zero per kolom numerikal:")
    print(zero_counts.to_string() if numerical_columns else "(tidak ada)")
    print(f"Baris dengan setidaknya satu zero: {len(zero_rows)}")
    if not zero_rows.empty:
        print(zero_rows.head(20).to_string(index=False))

    model_data = data.copy()
    model_categorical_columns = list(categorical_columns)
    for column in DATETIME_COLUMNS:
        epoch_column = f"{column}_epoch"
        model_data[epoch_column] = (
            model_data[column].astype("int64") / 1_000_000_000
        ).where(model_data[column].notna(), np.nan)
        model_categorical_columns.append(f"{column}_missing")
        model_data[f"{column}_missing"] = model_data[column].isna().astype(str)
    model_numerical_columns = numerical_columns + [
        f"{column}_epoch" for column in DATETIME_COLUMNS
    ]
    for column in model_categorical_columns:
        model_data[column] = model_data[column].astype(object)

    transformers = []
    if model_categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value="Tidak diketahui",
                    ),
                ),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(
            ("categorical", categorical_pipeline, model_categorical_columns)
        )
    if model_numerical_columns:
        numerical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(
            ("numerical", numerical_pipeline, model_numerical_columns)
        )

    preprocessor = ColumnTransformer(transformers=transformers)
    prepared_data = preprocessor.fit_transform(model_data)
    print(
        "\nUkuran hasil preprocessing scikit-learn: "
        f"{prepared_data.shape}"
    )

    quality_counts = {
        "Duplikat\n(baris)": int(duplicate_mask.sum()),
        "Null\n(baris)": int(null_mask.sum()),
        "Zero\n(baris)": int(np.count_nonzero(zero_row_mask)),
    }
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(
        ["Kategorikal", "Numerikal", "Tanggal/waktu"],
        [
            len(categorical_columns),
            len(numerical_columns),
            len(DATETIME_COLUMNS),
        ],
        color=["#4C78A8", "#F58518", "#72B7B2"],
    )
    axes[0].set_title("Jumlah kolom berdasarkan jenis")
    axes[0].set_ylabel("Jumlah kolom")

    axes[1].bar(
        list(quality_counts),
        list(quality_counts.values()),
        color=["#E45756", "#72B7B2", "#54A24B"],
    )
    axes[1].set_title("Baris dengan masalah data")
    axes[1].set_ylabel("Jumlah baris")
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()