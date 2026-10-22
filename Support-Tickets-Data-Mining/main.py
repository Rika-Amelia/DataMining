"""Eksplorasi data dan klasifikasi prioritas tiket bantuan."""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Support Tickets.csv"
OUTPUT_DIR = BASE_DIR / "output"
RANDOM_STATE = 42
TEST_SIZE = 0.2

TARGET_CANDIDATES = (
    "ticket_priority",
    "priority",
    "ticket_status",
    "status",
)
DATE_NAME_HINTS = ("date", "time", "created", "opened", "resolved", "closed")
LEAKAGE_NAME_HINTS = (
    "resolved",
    "resolution",
    "closed",
    "csat",
    "satisfaction",
    "reopened",
    "first_response",
    "response_time",
    "reply",
)


def print_section(title: str) -> None:
    """Cetak judul bagian agar hasil terminal mudah dibaca."""
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def normalized_name(column: Any) -> str:
    """Normalisasi nama kolom untuk pencocokan nama target."""
    return re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")


def find_target_column(columns: pd.Index) -> Any | None:
    """Cari target prioritas terlebih dahulu, lalu target status."""
    lookup = {normalized_name(column): column for column in columns}
    for candidate in TARGET_CANDIDATES:
        if candidate in lookup:
            return lookup[candidate]
    return None


def infer_datetime_columns(frame: pd.DataFrame) -> list[str]:
    """Parse kolom object yang namanya mengindikasikan tanggal atau waktu."""
    datetime_columns: list[str] = []
    for column in frame.columns:
        if not any(hint in normalized_name(column) for hint in DATE_NAME_HINTS):
            continue
        if not (
            pd.api.types.is_object_dtype(frame[column].dtype)
            or pd.api.types.is_string_dtype(frame[column].dtype)
        ):
            continue

        parsed = pd.to_datetime(frame[column], errors="coerce")
        non_null_count = int(frame[column].notna().sum())
        if non_null_count and int(parsed.notna().sum()) / non_null_count >= 0.8:
            frame[column] = parsed
            datetime_columns.append(column)
    return datetime_columns


def identify_column_types(
    frame: pd.DataFrame, datetime_columns: list[str]
) -> tuple[list[str], list[str]]:
    """Temukan kolom kategorikal dan numerikal secara otomatis."""
    categorical_columns = [
        column
        for column in frame.select_dtypes(
            include=["object", "string", "category", "bool"]
        ).columns
        if column not in datetime_columns
    ]
    numerical_columns = [
        column
        for column in frame.select_dtypes(include=["number"]).columns
        if not pd.api.types.is_bool_dtype(frame[column].dtype)
    ]
    return categorical_columns, numerical_columns


def create_eda_plots(
    frame: pd.DataFrame,
    categorical_columns: list[str],
    numerical_columns: list[str],
    target_column: Any | None,
) -> list[Path]:
    """Buat dan simpan grafik distribusi yang sesuai dengan kolom dataset."""
    sns.set_theme(style="whitegrid")
    saved_paths: list[Path] = []

    if target_column is not None:
        figure, axis = plt.subplots(figsize=(8, 5))
        order = frame[target_column].value_counts().index
        sns.countplot(data=frame, x=target_column, order=order, ax=axis)
        axis.set_title(f"Distribusi target: {target_column}")
        axis.set_xlabel(str(target_column))
        axis.set_ylabel("Jumlah data")
        axis.tick_params(axis="x", rotation=30)
        figure.tight_layout()
        path = OUTPUT_DIR / "distribusi_target.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        saved_paths.append(path)

    plotted_categoricals = [
        column for column in categorical_columns if column != target_column
    ][:5]
    for index, column in enumerate(plotted_categoricals, start=1):
        counts = frame[column].astype("string").fillna("Missing").value_counts()
        counts = counts.head(15).sort_values()
        figure, axis = plt.subplots(figsize=(9, 5))
        sns.barplot(x=counts.values, y=counts.index, ax=axis, color="#4C78A8")
        axis.set_title(f"Distribusi kategori: {column} (maks. 15 kategori)")
        axis.set_xlabel("Jumlah data")
        axis.set_ylabel(str(column))
        figure.tight_layout()
        path = OUTPUT_DIR / f"distribusi_kategorikal_{index}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        saved_paths.append(path)

    for index, column in enumerate(numerical_columns[:5], start=1):
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            continue
        figure, axis = plt.subplots(figsize=(8, 5))
        sns.histplot(values, kde=True, ax=axis, color="#F58518")
        axis.set_title(f"Distribusi numerik: {column}")
        axis.set_xlabel(str(column))
        axis.set_ylabel("Frekuensi")
        figure.tight_layout()
        path = OUTPUT_DIR / f"distribusi_numerik_{index}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        saved_paths.append(path)

    return saved_paths


def detect_identifier_columns(columns: list[str]) -> set[str]:
    """Tandai kolom identitas berdasarkan pola nama umum."""
    identifiers: set[str] = set()
    for column in columns:
        name = normalized_name(column)
        if (
            str(column).strip() == "#"
            or name in {"id", "ticket_id", "case_id"}
            or name.endswith("_id")
        ):
            identifiers.add(column)
    return identifiers


def build_features(
    frame: pd.DataFrame, target_column: Any, datetime_columns: list[str]
) -> tuple[pd.DataFrame, set[str], list[str]]:
    """Pisahkan target, keluarkan ID/kebocoran, dan ekstrak waktu pembukaan."""
    excluded_columns = {target_column}
    excluded_columns.update(detect_identifier_columns(list(frame.columns)))
    leakage_columns = {
        column
        for column in frame.columns
        if column != target_column
        and any(
            hint in normalized_name(column)
            for hint in LEAKAGE_NAME_HINTS
        )
    }
    excluded_columns.update(leakage_columns)

    features = frame.drop(columns=list(excluded_columns), errors="ignore").copy()
    excluded_datetime_columns: list[str] = []
    for column in datetime_columns:
        if column not in features.columns:
            continue
        name = normalized_name(column)
        # Tanggal tiket dibuka dapat diketahui pada saat prioritas/status ditentukan.
        if any(hint in name for hint in ("opened", "created", "received")):
            parsed = features[column]
            features[f"{column}_year"] = parsed.dt.year
            features[f"{column}_month"] = parsed.dt.month
            features[f"{column}_dayofweek"] = parsed.dt.dayofweek
            features[f"{column}_hour"] = parsed.dt.hour
        features = features.drop(columns=column)
        excluded_datetime_columns.append(column)

    return features, excluded_columns, excluded_datetime_columns


def save_evaluation(
    y_test: pd.Series,
    y_pred: np.ndarray,
    target_column: Any,
) -> dict[str, float]:
    """Cetak dan simpan metrik, laporan klasifikasi, dan confusion matrix."""
    labels = np.unique(np.concatenate([y_test.to_numpy(), y_pred]))
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )
    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }

    print_section("EVALUASI MODEL")
    print(f"Target: {target_column}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision macro: {precision:.4f}")
    print(f"Recall macro: {recall:.4f}")
    print(f"F1-score macro: {f1:.4f}")
    print("\nClassification report (per kelas):")
    print(report)
    print(
        "Interpretasi: accuracy adalah proporsi prediksi yang benar. "
        "Precision menunjukkan ketepatan prediksi suatu kelas; recall "
        "menunjukkan data kelas yang berhasil ditemukan; F1-score "
        "merangkum precision dan recall. Nilai macro memberi bobot sama "
        "kepada setiap kelas."
    )

    matrix = confusion_matrix(y_test, y_pred, labels=labels)
    matrix_frame = pd.DataFrame(
        matrix,
        index=[f"Aktual: {label}" for label in labels],
        columns=[f"Prediksi: {label}" for label in labels],
    )
    matrix_frame.to_csv(OUTPUT_DIR / "confusion_matrix.csv", index=True)

    figure, axis = plt.subplots(figsize=(7, 6))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[str(label) for label in labels],
    )
    display.plot(ax=axis, cmap="Blues", values_format="d", colorbar=False)
    axis.set_title("Confusion Matrix - Decision Tree")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(figure)

    (OUTPUT_DIR / "laporan_evaluasi.txt").write_text(
        f"Target: {target_column}\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"Precision macro: {precision:.4f}\n"
        f"Recall macro: {recall:.4f}\n"
        f"F1-score macro: {f1:.4f}\n\n"
        f"{report}\n"
        "Confusion matrix (baris=aktual, kolom=prediksi):\n"
        f"{matrix_frame.to_string()}\n\n"
        "Interpretasi: accuracy adalah proporsi prediksi yang benar. "
        "Precision menunjukkan ketepatan prediksi suatu kelas; recall "
        "menunjukkan data kelas yang berhasil ditemukan; F1-score "
        "merangkum precision dan recall. Nilai macro memberi bobot sama "
        "kepada setiap kelas. Pada confusion matrix, nilai diagonal "
        "menunjukkan prediksi benar, sedangkan nilai di luar diagonal "
        "menunjukkan kelas yang tertukar.\n",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings(
        "ignore",
        message="Could not infer format",
        category=UserWarning,
    )

    print_section("1. DATA LOADING")
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"File CSV tidak ditemukan: {DATA_PATH}\n"
            "Letakkan 'Support Tickets.csv' di folder project."
        )

    df = pd.read_csv(DATA_PATH)
    if df.empty:
        raise ValueError("Dataset berhasil dibaca tetapi tidak memiliki baris.")

    print(f"File: {DATA_PATH.name}")
    print(f"Jumlah baris: {df.shape[0]}")
    print(f"Jumlah kolom: {df.shape[1]}")
    print("\nNama kolom:")
    print(df.columns.tolist())
    print("\nLima data pertama:")
    print(df.head().to_string(index=False))

    print_section("2. DATA UNDERSTANDING")
    datetime_columns = infer_datetime_columns(df)
    categorical_columns, numerical_columns = identify_column_types(
        df, datetime_columns
    )
    print("\nInformasi tipe data:")
    df.info()
    print("\nTipe data per kolom:")
    print(df.dtypes.to_string())
    print("\nStatistik deskriptif:")
    print(df.describe(include="all").T.to_string())
    print("\nKolom kategorikal:", categorical_columns)
    print("Kolom numerikal:", numerical_columns)
    print("Kolom tanggal/waktu:", datetime_columns)

    target_column = find_target_column(df.columns)
    if target_column is None:
        target_message = (
            "Target klasifikasi tidak tersedia. Program mencari kolom "
            "'priority'/'ticket_priority' atau 'status'/'ticket_status'. "
            "Eksplorasi dan grafik tetap dibuat; pelatihan model dilewati."
        )
        print("\n" + target_message)
    else:
        target_message = f"Target klasifikasi yang dipilih: {target_column}"
        print("\n" + target_message)

    print_section("3. MISSING VALUES, DUPLIKAT, DAN DISTRIBUSI")
    missing_by_column = df.isna().sum()
    missing_summary = pd.DataFrame(
        {
            "missing_count": missing_by_column,
            "missing_percent": (df.isna().mean() * 100).round(2),
        }
    )
    print("\nMissing values:")
    print(missing_summary.to_string())
    print(f"\nTotal missing values: {int(missing_by_column.sum())}")

    duplicate_count = int(df.duplicated().sum())
    print(f"\nJumlah baris duplikat penuh: {duplicate_count}")
    if duplicate_count:
        print("Contoh baris duplikat:")
        print(df.loc[df.duplicated(keep=False)].head(10).to_string(index=False))

    distribution_frames = []
    for column in categorical_columns:
        counts = df[column].astype("string").fillna("Missing").value_counts(
            dropna=False
        )
        distribution_frames.append(
            pd.DataFrame(
                {
                    "column": column,
                    "value": counts.index.astype(str),
                    "count": counts.to_numpy(),
                }
            )
        )
        print(f"\nDistribusi kategori '{column}':")
        print(counts.head(20).to_string())

    if distribution_frames:
        pd.concat(distribution_frames, ignore_index=True).to_csv(
            OUTPUT_DIR / "distribusi_kategorikal.csv", index=False
        )
    missing_summary.to_csv(OUTPUT_DIR / "missing_values.csv", index=True)

    print_section("4. PREPROCESSING DAN EDA")
    df_clean = df.drop_duplicates().copy()
    print(f"Ukuran data setelah hapus duplikat: {df_clean.shape}")
    print(
        "Missing kategori akan diisi dengan label 'Tidak diketahui'; "
        "missing numerik dengan median training."
    )
    print("Encoding kategorikal menggunakan OneHotEncoder di dalam pipeline.")
    chart_paths = create_eda_plots(
        df_clean, categorical_columns, numerical_columns, target_column
    )
    print("\nGrafik yang disimpan:")
    for chart_path in chart_paths:
        print(f"- {chart_path.name}")

    if numerical_columns:
        zero_counts = (df_clean[numerical_columns] == 0).sum()
        print("\nJumlah nilai zero per kolom numerikal:")
        print(zero_counts.to_string())
        print(f"Total sel bernilai zero: {int(zero_counts.sum())}")

    if target_column is None:
        print("\nModel tidak dijalankan karena kolom target tidak ditemukan.")
        (OUTPUT_DIR / "status_model.txt").write_text(
            target_message + "\n", encoding="utf-8"
        )
        return

    print_section("5. DATA MINING: DECISION TREE")
    before_target_filter = len(df_clean)
    df_model = df_clean.dropna(subset=[target_column]).copy()
    removed_target_missing = before_target_filter - len(df_model)
    print(
        f"Baris tanpa label target yang dikeluarkan: {removed_target_missing}"
    )

    target_counts = df_model[target_column].value_counts()
    number_of_classes = int(target_counts.size)
    if number_of_classes < 2:
        message = (
            f"Model tidak dapat dilatih: target '{target_column}' hanya "
            "memiliki satu kelas setelah baris tanpa target dikeluarkan."
        )
        print(message)
        (OUTPUT_DIR / "status_model.txt").write_text(message + "\n", encoding="utf-8")
        return

    X, excluded_columns, _ = build_features(
        df_model, target_column, datetime_columns
    )
    y = df_model[target_column]
    if X.shape[1] == 0:
        message = "Model tidak dapat dilatih: tidak ada fitur yang tersisa."
        print(message)
        (OUTPUT_DIR / "status_model.txt").write_text(message + "\n", encoding="utf-8")
        return

    categorical_features, numerical_features = identify_column_types(X, [])
    print(f"Target: {target_column}")
    print("Distribusi target:")
    print(target_counts.to_string())
    print("\nKolom yang dikeluarkan dari fitur untuk mencegah ID/kebocoran:")
    print(sorted(str(column) for column in excluded_columns))
    print("\nFitur kategorikal:", categorical_features)
    print("Fitur numerikal:", numerical_features)

    class_minimum = int(target_counts.min())
    test_count = int(np.ceil(TEST_SIZE * len(y)))
    train_count = len(y) - test_count
    if (
        class_minimum < 2
        or test_count < number_of_classes
        or train_count < number_of_classes
    ):
        message = (
            "Model tidak dapat dilatih dengan stratified train-test split: "
            "data per kelas atau jumlah baris tidak mencukupi. "
            f"Jumlah data={len(y)}, kelas={number_of_classes}, "
            f"minimum per kelas={class_minimum}."
        )
        print("\n" + message)
        (OUTPUT_DIR / "status_model.txt").write_text(message + "\n", encoding="utf-8")
        return

    transformers = []
    if categorical_features:
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
            ("categorical", categorical_pipeline, categorical_features)
        )
    if numerical_features:
        numerical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(
            ("numerical", numerical_pipeline, numerical_features)
        )

    preprocessor = ColumnTransformer(transformers=transformers)
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                DecisionTreeClassifier(
                    random_state=RANDOM_STATE,
                    max_depth=6,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"\nJumlah data training: {len(X_train)}")
    print(f"Jumlah data testing: {len(X_test)}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = save_evaluation(y_test, y_pred, target_column)

    prediction_results = pd.DataFrame(
        {
            "aktual": y_test.to_numpy(),
            "prediksi": y_pred,
        },
        index=y_test.index,
    ).sort_index()
    prediction_results.to_csv(OUTPUT_DIR / "hasil_prediksi.csv", index_label="row_index")

    summary = {
        "dataset": DATA_PATH.name,
        "baris_awal": len(df),
        "kolom_awal": df.shape[1],
        "baris_setelah_duplikat_dihapus": len(df_clean),
        "jumlah_duplikat_dihapus": duplicate_count,
        "jumlah_missing": int(missing_by_column.sum()),
        "jumlah_kolom_kategorikal": len(categorical_columns),
        "jumlah_kolom_numerikal": len(numerical_columns),
        "target": str(target_column),
        "jumlah_data_training": len(X_train),
        "jumlah_data_testing": len(X_test),
        **metrics,
    }
    pd.DataFrame([summary]).to_csv(OUTPUT_DIR / "ringkasan_analisis.csv", index=False)
    (OUTPUT_DIR / "status_model.txt").write_text(
        f"Model berhasil dilatih dengan target '{target_column}'.\n"
        f"Accuracy: {metrics['accuracy']:.4f}\n"
        "Gunakan laporan metrik lengkap dan confusion matrix untuk evaluasi.\n",
        encoding="utf-8",
    )
    print_section("6. PENYIMPANAN HASIL")
    print(f"Semua hasil disimpan di: {OUTPUT_DIR}")
    print("Laporan: laporan_evaluasi.txt")
    print("Prediksi: hasil_prediksi.csv")
    print("Grafik evaluasi: confusion_matrix.png")
    print("Ringkasan: ringkasan_analisis.csv")


if __name__ == "__main__":
    main()
