"""Module đọc dữ liệu, làm sạch, kiểm tra chất lượng và chuẩn bị đặc trưng."""

import os
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

REQUIRED_COLUMNS: List[str] = [
    "candidate_id", "job_role", "job_role_display", "gender", "age_group",
    "edu_level", "edu_level_name", "years_experience", "edu_relevance",
    "P_mcq", "P_desc", "P_code", "S_edu", "S_exp", "S_skill",
    "S_cv", "S_int", "CSS", "passed_hard_filter", "relevance_label",
    "w_edu", "w_exp", "w_skill", "w_mcq", "w_desc", "w_code", "W_CV", "W_INT"
]

RAW_NUMERIC_FEATURES: List[str] = [
    "edu_level", "years_experience", "edu_relevance", "P_mcq", "P_desc", "P_code"
]

RAW_CATEGORICAL_FEATURES: List[str] = ["job_role"]


def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Đọc file CSV, kiểm tra sự tồn tại của file và các cột bắt buộc,
    chuẩn hóa khoảng trắng ở cột chuỗi và loại bỏ trùng lặp theo candidate_id.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại đường dẫn: {file_path}")

    df = pd.read_csv(file_path)

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dữ liệu thiếu các cột bắt buộc: {missing_cols}")

    df_clean = df.copy()
    object_cols = df_clean.select_dtypes(include=["object"]).columns
    for col in object_cols:
        df_clean[col] = df_clean[col].apply(
            lambda x: str(x).strip() if pd.notna(x) else x
        )

    df_clean = df_clean.drop_duplicates(subset=["candidate_id"]).reset_index(drop=True)
    return df_clean


def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Tổng hợp các chỉ số chất lượng dữ liệu: số dòng, số cột, số lượng thiếu,
    trùng lặp, miền giá trị và kiểm tra sai số công thức CSS = 0.4*S_cv + 0.6*S_int.
    """
    total_rows = int(len(df))
    total_cols = int(df.shape[1])
    missing_by_col = df.isnull().sum()
    total_missing = int(missing_by_col.sum())
    duplicate_ids = int(df["candidate_id"].duplicated().sum())
    role_counts = df["job_role"].value_counts()

    css_expected = 0.4 * df["S_cv"] + 0.6 * df["S_int"]
    css_abs_error = np.abs(df["CSS"] - css_expected)
    max_css_error = float(css_abs_error.max())
    mean_css_error = float(css_abs_error.mean())

    s_cv_expected = (
        df["w_edu"] * df["S_edu"]
        + df["w_exp"] * df["S_exp"]
        + df["w_skill"] * df["S_skill"]
    )
    max_scv_error = float(np.abs(df["S_cv"] - s_cv_expected).max())

    s_int_expected = (
        df["w_mcq"] * df["P_mcq"]
        + df["w_desc"] * df["P_desc"]
        + df["w_code"] * df["P_code"]
    )
    max_sint_error = float(np.abs(df["S_int"] - s_int_expected).max())

    domain_cols = [
        "edu_level", "years_experience", "edu_relevance",
        "P_mcq", "P_desc", "P_code", "S_edu", "S_exp", "S_skill",
        "S_cv", "S_int", "CSS", "passed_hard_filter", "relevance_label"
    ]
    domain_summary = (
        df[domain_cols]
        .agg(["min", "mean", "median", "max", "std"])
        .T.reset_index()
        .rename(columns={"index": "Tên cột"})
    )

    return {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "total_missing": total_missing,
        "missing_by_col": missing_by_col[missing_by_col > 0].to_dict(),
        "duplicate_ids": duplicate_ids,
        "num_job_roles": int(df["job_role"].nunique()),
        "min_candidates_per_role": int(role_counts.min()),
        "max_candidates_per_role": int(role_counts.max()),
        "max_css_error": max_css_error,
        "mean_css_error": mean_css_error,
        "max_scv_error": max_scv_error,
        "max_sint_error": max_sint_error,
        "formula_matches": bool(max_css_error < 1e-3),
        "domain_summary": domain_summary,
    }


def prepare_classification_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Tách đặc trưng đầu vào X, nhãn nhị phân y (relevance_label >= 2) và bảng nhân khẩu học.
    """
    feature_cols = RAW_NUMERIC_FEATURES + RAW_CATEGORICAL_FEATURES
    missing = [c for c in feature_cols + ["relevance_label", "gender", "age_group"] if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột cần thiết để tạo đặc trưng: {missing}")

    X = df[feature_cols].copy()
    y = (df["relevance_label"] >= 2).astype(int)
    demographics = df[["candidate_id", "gender", "age_group", "job_role"]].copy()
    return X, y, demographics