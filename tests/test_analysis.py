"""Unit test cho các hàm phân tích và kiểm định thống kê trong module analysis."""

import numpy as np
import pandas as pd

from modules.analysis import (
    analyze_rq1_hard_filter,
    compute_chi2_cramers_v,
    compute_kruskal_effect_size,
    pairwise_mannwhitney_correction,
)


def test_analyze_rq1_hard_filter_rates():
    """Kiểm tra tính toán tỉ lệ ứng viên nhãn 2-3 và nhãn 3 bị bộ lọc cứng loại."""
    df_mini = pd.DataFrame({
        "job_role": ["RoleA", "RoleA", "RoleB", "RoleB", "RoleB"],
        "relevance_label": [3, 3, 2, 1, 0],
        "passed_hard_filter": [0, 1, 0, 1, 0],
    })
    res = analyze_rq1_hard_filter(df_mini)
    # Có 3 ứng viên nhãn 2-3 (hai người nhãn 3, một người nhãn 2), trong đó 2 người bị loại (passed_hard_filter=0)
    assert res["total_strong"] == 3
    assert res["rejected_strong"] == 2
    assert np.isclose(res["rejected_strong_rate"], 2 / 3 * 100)
    # Nhãn 3 có 2 người, 1 người bị loại -> 50%
    assert res["total_label3"] == 2
    assert res["rejected_label3"] == 1
    assert np.isclose(res["rejected_label3_rate"], 50.0)


def test_compute_chi2_cramers_v_bounds():
    """Kiểm tra Cramér's V nằm trong đoạn [0, 1] và gần bằng 0 khi hai nhóm có tỉ lệ đậu giống hệt nhau."""
    df_equal = pd.DataFrame({
        "gender": ["M"] * 50 + ["F"] * 50,
        "passed_hard_filter": [1, 0] * 25 + [1, 0] * 25,
    })
    res = compute_chi2_cramers_v(df_equal, "gender", "passed_hard_filter")
    assert 0.0 <= res["cramers_v"] <= 1.0
    assert np.isclose(res["cramers_v"], 0.0)
    assert np.isclose(res["p_value"], 1.0)


def test_compute_kruskal_effect_size_separated_groups():
    """Kiểm tra Kruskal-Wallis phát hiện sự khác biệt rõ rệt giữa 3 nhóm tách biệt."""
    df_sep = pd.DataFrame({
        "group": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
        "CSS": [0.2] * 20 + [0.5] * 20 + [0.9] * 20,
    })
    res = compute_kruskal_effect_size(df_sep, "group", "CSS")
    assert res["p_value"] < 0.001
    assert 0.8 <= res["epsilon_sq"] <= 1.0


def test_pairwise_mannwhitney_holm_and_bonferroni_order():
    """Kiểm tra hiệu chỉnh đa kiểm định đảm bảo p_raw <= p_holm <= p_bonferroni <= 1.0."""
    rng = np.random.default_rng(42)
    df_multi = pd.DataFrame({
        "age_group": ["G1"] * 30 + ["G2"] * 30 + ["G3"] * 30,
        "CSS": np.concatenate([
            rng.normal(0.5, 0.05, 30),
            rng.normal(0.6, 0.05, 30),
            rng.normal(0.8, 0.05, 30),
        ]),
    })
    pair_df = pairwise_mannwhitney_correction(df_multi, "age_group", "CSS")
    assert len(pair_df) == 3
    assert (pair_df["p_raw"] <= pair_df["p_holm"] + 1e-12).all()
    assert (pair_df["p_holm"] <= pair_df["p_bonferroni"] + 1e-12).all()
    assert (pair_df["p_bonferroni"] <= 1.0).all()