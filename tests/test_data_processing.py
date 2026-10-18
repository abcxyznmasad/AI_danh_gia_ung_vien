"""Unit test cho module data_processing sử dụng dữ liệu nhỏ tự tạo (tương thích Windows/Linux/macOS)."""

import os
import tempfile
import pandas as pd
import pytest

from modules.data_processing import (
    load_and_clean_data,
    check_data_quality,
    prepare_classification_features,
)


def _create_sample_dataframe() -> pd.DataFrame:
    """Tạo DataFrame mẫu hợp lệ có đủ 28 cột theo đúng cấu trúc dữ liệu."""
    data = {
        "candidate_id": [" SE001 ", "SE001", "DS002 "],  # Có khoảng trắng và trùng mã SE001
        "job_role": ["Software_Engineer", "Software_Engineer", "Data_Scientist"],
        "job_role_display": ["Software Engineer", "Software Engineer", "Data Scientist"],
        "gender": ["M", "M", "F"],
        "age_group": ["22-25", "22-25", "26-30"],
        "edu_level": [2, 2, 3],
        "edu_level_name": ["BSc", "BSc", "MSc"],
        "years_experience": [2.0, 2.0, 4.5],
        "edu_relevance": [0.8, 0.8, 0.9],
        "P_mcq": [0.7, 0.7, 0.8],
        "P_desc": [0.6, 0.6, 0.85],
        "P_code": [0.75, 0.75, 0.9],
        "S_edu": [0.7, 0.7, 0.85],
        "S_exp": [0.6, 0.6, 0.9],
        "S_skill": [0.8, 0.8, 0.88],
        "S_cv": [0.72, 0.72, 0.88],
        "S_int": [0.695, 0.695, 0.865],
        "CSS": [0.4 * 0.72 + 0.6 * 0.695, 0.4 * 0.72 + 0.6 * 0.695, 0.4 * 0.88 + 0.6 * 0.865],
        "passed_hard_filter": [1, 1, 0],
        "relevance_label": [2, 2, 1],
        "w_edu": [0.2, 0.2, 0.2],
        "w_exp": [0.3, 0.3, 0.3],
        "w_skill": [0.5, 0.5, 0.5],
        "w_mcq": [0.2, 0.2, 0.2],
        "w_desc": [0.3, 0.3, 0.3],
        "w_code": [0.5, 0.5, 0.5],
        "W_CV": [0.4, 0.4, 0.4],
        "W_INT": [0.6, 0.6, 0.6],
    }
    return pd.DataFrame(data)


def _write_temp_csv(df: pd.DataFrame) -> str:
    """
    Tạo đường dẫn file tạm, đóng file descriptor ngay lập tức trước khi ghi bằng pandas
    để tránh lỗi khóa file (PermissionError) trên hệ điều hành Windows.
    """
    fd, tmp_path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    df.to_csv(tmp_path, index=False, encoding="utf-8")
    return tmp_path


def test_load_and_clean_data_strips_and_deduplicates():
    """Kiểm tra hàm đọc dữ liệu xóa khoảng trắng thừa và loại bỏ dòng trùng candidate_id."""
    df_raw = _create_sample_dataframe()
    tmp_path = _write_temp_csv(df_raw)

    try:
        df_clean = load_and_clean_data(tmp_path)
        assert len(df_clean) == 2
        assert list(df_clean["candidate_id"]) == ["SE001", "DS002"]
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_load_and_clean_data_missing_file_raises():
    """Kiểm tra báo lỗi FileNotFoundError khi đường dẫn không tồn tại."""
    with pytest.raises(FileNotFoundError):
        load_and_clean_data("duong_dan_khong_ton_tai_12345.csv")


def test_load_and_clean_data_missing_columns_raises():
    """Kiểm tra báo lỗi ValueError khi file CSV thiếu các cột bắt buộc."""
    df_bad = pd.DataFrame({"candidate_id": ["SE001"], "CSS": [0.8]})
    tmp_path = _write_temp_csv(df_bad)

    try:
        with pytest.raises(ValueError, match="Dữ liệu thiếu các cột bắt buộc"):
            load_and_clean_data(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_check_data_quality_formula_verification():
    """Kiểm tra hàm check_data_quality tính đúng sai số công thức CSS = 0.4*S_cv + 0.6*S_int."""
    df_sample = _create_sample_dataframe().iloc[[0, 2]].reset_index(drop=True)
    quality = check_data_quality(df_sample)
    assert quality["total_rows"] == 2
    assert quality["max_css_error"] < 1e-9
    assert quality["formula_matches"] is True


def test_prepare_classification_features_prevents_leakage():
    """Đảm bảo tập đặc trưng X không chứa các cột rò rỉ hoặc biến nhạy cảm gender, age_group."""
    df_sample = _create_sample_dataframe().iloc[[0, 2]].reset_index(drop=True)
    X, y, demo = prepare_classification_features(df_sample)

    forbidden_cols = [
        "CSS", "passed_hard_filter", "relevance_label",
        "S_cv", "S_int", "S_edu", "S_exp", "S_skill",
        "gender", "age_group", "candidate_id"
    ]
    for col in forbidden_cols:
        assert col not in X.columns

    assert list(y) == [1, 0]
    assert "gender" in demo.columns and "age_group" in demo.columns