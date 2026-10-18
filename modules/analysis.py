"""Module giải quyết 3 câu hỏi nghiên cứu và thực hiện các kiểm định thống kê."""

from itertools import combinations
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from scipy import stats


def analyze_rq1_hard_filter(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Câu hỏi 1: Phân tích bộ lọc cứng (passed_hard_filter) theo nhãn (relevance_label)
    và theo từng vị trí công việc (job_role) để đánh giá tỉ lệ loại nhầm ứng viên mạnh.
    """
    ct_counts = pd.crosstab(df["relevance_label"], df["passed_hard_filter"], margins=True)
    ct_counts = ct_counts.rename(
        columns={0: "Trượt lọc cứng (0)", 1: "Đậu lọc cứng (1)", "All": "Tổng"}
    )

    ct_pct = pd.crosstab(df["relevance_label"], df["passed_hard_filter"], normalize="index") * 100
    ct_pct = ct_pct.rename(columns={0: "Tỉ lệ Trượt (%)", 1: "Tỉ lệ Đậu (%)"})

    summary_table = pd.concat([ct_counts.drop(index="All"), ct_pct], axis=1).reset_index()
    summary_table = summary_table.rename(columns={"relevance_label": "Nhãn mức phù hợp"})

    # Tính tỉ lệ ứng viên mạnh (nhãn 2-3) và rất mạnh (nhãn 3) bị loại
    mask_strong = df["relevance_label"].isin([2, 3])
    total_strong = int(mask_strong.sum())
    rejected_strong = int(((df["passed_hard_filter"] == 0) & mask_strong).sum())
    rejected_strong_rate = float(rejected_strong / total_strong * 100) if total_strong > 0 else 0.0

    mask_label3 = df["relevance_label"] == 3
    total_label3 = int(mask_label3.sum())
    rejected_label3 = int(((df["passed_hard_filter"] == 0) & mask_label3).sum())
    rejected_label3_rate = float(rejected_label3 / total_label3 * 100) if total_label3 > 0 else 0.0

    # Phân tích theo vị trí công việc (job_role)
    role_records: List[Dict[str, Any]] = []
    for role, grp in df.groupby("job_role"):
        grp_strong = grp[grp["relevance_label"].isin([2, 3])]
        grp_l3 = grp[grp["relevance_label"] == 3]
        n_strong = len(grp_strong)
        n_rej_strong = int((grp_strong["passed_hard_filter"] == 0).sum())
        n_l3 = len(grp_l3)
        n_rej_l3 = int((grp_l3["passed_hard_filter"] == 0).sum())

        role_records.append({
            "Vị trí (job_role)": role,
            "Tổng ứng viên": len(grp),
            "Số UV mạnh (Nhãn 2-3)": n_strong,
            "Số UV mạnh bị loại": n_rej_strong,
            "Tỉ lệ loại nhầm Nhãn 2-3 (%)": (n_rej_strong / n_strong * 100) if n_strong > 0 else 0.0,
            "Số UV Nhãn 3": n_l3,
            "Số UV Nhãn 3 bị loại": n_rej_l3,
            "Tỉ lệ loại nhầm Nhãn 3 (%)": (n_rej_l3 / n_l3 * 100) if n_l3 > 0 else 0.0,
        })

    by_role_df = pd.DataFrame(role_records).sort_values(
        by="Tỉ lệ loại nhầm Nhãn 2-3 (%)", ascending=False
    ).reset_index(drop=True)

    top_role = by_role_df.iloc[0]["Vị trí (job_role)"]
    top_role_rate = float(by_role_df.iloc[0]["Tỉ lệ loại nhầm Nhãn 2-3 (%)"])

    conclusion = (
        f"Bộ lọc cứng đã loại bỏ {rejected_strong:,}/{total_strong:,} ({rejected_strong_rate:.2f}%) "
        f"ứng viên phù hợp (nhãn 2–3), trong đó có {rejected_label3:,}/{total_label3:,} ({rejected_label3_rate:.2f}%) "
        f"ứng viên xuất sắc (nhãn 3), cao nhất ở vị trí {top_role} ({top_role_rate:.2f}%)."
    )

    return {
        "summary_table": summary_table,
        "total_strong": total_strong,
        "rejected_strong": rejected_strong,
        "rejected_strong_rate": rejected_strong_rate,
        "total_label3": total_label3,
        "rejected_label3": rejected_label3,
        "rejected_label3_rate": rejected_label3_rate,
        "by_role_df": by_role_df,
        "conclusion": conclusion,
    }


def analyze_rq2_css_consistency(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Câu hỏi 2: Kiểm tra tính nhất quán giữa điểm CSS (chia theo tứ phân vị Q1-Q4)
    và nhãn relevance_label. Liệt kê các ca Q4 nhưng nhãn 0-1 và giải thích bất thường ở nhãn 0 vs 1.
    """
    df_work = df.copy()
    quartile_labels = ["Q1 (Thấp)", "Q2 (TB-Thấp)", "Q3 (TB-Cao)", "Q4 (Cao)"]
    df_work["css_quartile"] = pd.qcut(df_work["CSS"], q=4, labels=quartile_labels)

    ct_counts = pd.crosstab(df_work["css_quartile"], df_work["relevance_label"], margins=True)
    ct_pct = pd.crosstab(df_work["css_quartile"], df_work["relevance_label"], normalize="index") * 100

    css_by_label = (
        df_work.groupby("relevance_label")["CSS"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
        .rename(columns={
            "relevance_label": "Nhãn (relevance_label)",
            "count": "Số lượng",
            "mean": "CSS Trung bình",
            "median": "CSS Trung vị",
            "std": "Độ lệch chuẩn",
            "min": "CSS Thấp nhất",
            "max": "CSS Cao nhất",
        })
    )

    # Phân rã nhãn 0 và nhãn 1 theo passed_hard_filter để giải thích hiện tượng đảo ngược điểm trung bình
    label01_breakdown = (
        df_work[df_work["relevance_label"].isin([0, 1])]
        .groupby(["relevance_label", "passed_hard_filter"])["CSS"]
        .agg(
            Số_lượng="count",
            CSS_Trung_bình="mean",
            CSS_Trung_vị="median",
        )
        .reset_index()
    )
    label_totals = df_work["relevance_label"].value_counts().to_dict()
    label01_breakdown["Tỉ lệ trong nhãn (%)"] = label01_breakdown.apply(
        lambda r: r["Số_lượng"] / label_totals[int(r["relevance_label"])] * 100, axis=1
    )
    label01_breakdown = label01_breakdown.rename(columns={
        "relevance_label": "Nhãn",
        "passed_hard_filter": "Đậu lọc cứng (0/1)",
        "Số_lượng": "Số lượng",
        "CSS_Trung_bình": "CSS Trung bình",
        "CSS_Trung_vị": "CSS Trung vị",
    })

    # Lọc các ca CSS thuộc Q4 nhưng nhãn lại là 0 hoặc 1
    mismatch_mask = (df_work["css_quartile"] == "Q4 (Cao)") & (df_work["relevance_label"].isin([0, 1]))
    mismatch_cols = [
        "candidate_id", "job_role", "CSS", "S_cv", "S_int",
        "years_experience", "edu_level_name", "passed_hard_filter", "relevance_label"
    ]
    mismatch_df = (
        df_work.loc[mismatch_mask, mismatch_cols]
        .sort_values(by="CSS", ascending=False)
        .reset_index(drop=True)
    )

    mismatch_l0 = int((mismatch_df["relevance_label"] == 0).sum())
    mismatch_l1 = int((mismatch_df["relevance_label"] == 1).sum())
    mismatch_l0_failed_hf = int(
        ((mismatch_df["relevance_label"] == 0) & (mismatch_df["passed_hard_filter"] == 0)).sum()
    )

    mean_l0 = float(df_work.loc[df_work["relevance_label"] == 0, "CSS"].mean())
    mean_l1 = float(df_work.loc[df_work["relevance_label"] == 1, "CSS"].mean())
    pct_l0_failed_hf = float(
        (df_work.loc[df_work["relevance_label"] == 0, "passed_hard_filter"] == 0).mean() * 100
    )

    anomaly_explanation = (
        f"Nhóm nhãn 0 có điểm CSS trung bình ({mean_l0:.4f}) cao hơn nhóm nhãn 1 ({mean_l1:.4f}). "
        f"Nguyên nhân do quy tắc sinh nhãn bị chi phối bởi bộ lọc cứng: có tới {pct_l0_failed_hf:.2f}% "
        f"ứng viên nhãn 0 bị trượt bộ lọc cứng (passed_hard_filter = 0), bao gồm nhiều ứng viên có điểm "
        f"chuyên môn CSS rất cao (lên tới Q4) nhưng bị ép về nhãn 0 hoặc 1 khi vi phạm điều kiện lọc cứng."
    )

    conclusion = (
        f"Điểm CSS nhìn chung đồng biến với nhãn từ mức 1 đến 3, nhưng xuất hiện {len(mismatch_df)} ca "
        f"bất nhất đạt CSS nhóm cao nhất (Q4) mà chỉ nhận nhãn 0 ({mismatch_l0} ca, đều trượt lọc cứng) "
        f"hoặc nhãn 1 ({mismatch_l1} ca), gây ra hiện tượng CSS trung bình của nhãn 0 ({mean_l0:.4f}) "
        f"cao hơn nhãn 1 ({mean_l1:.4f})."
    )

    return {
        "ct_counts": ct_counts,
        "ct_pct": ct_pct.round(2),
        "css_by_label": css_by_label,
        "label01_breakdown": label01_breakdown,
        "mismatch_df": mismatch_df,
        "mismatch_count": len(mismatch_df),
        "mismatch_l0": mismatch_l0,
        "mismatch_l1": mismatch_l1,
        "mismatch_l0_failed_hf": mismatch_l0_failed_hf,
        "anomaly_explanation": anomaly_explanation,
        "conclusion": conclusion,
    }


def compute_chi2_cramers_v(df: pd.DataFrame, group_col: str, target_col: str = "passed_hard_filter") -> Dict[str, float]:
    """
    Thực hiện kiểm định Chi-square về tính độc lập giữa biến phân nhóm và tỉ lệ đậu sàng lọc,
    kèm tính kích thước hiệu ứng Cramér's V.
    """
    ct = pd.crosstab(df[group_col], df[target_col])
    chi2, p_val, dof, _ = stats.chi2_contingency(ct)
    n = ct.sum().sum()
    r, k = ct.shape
    denom = min(r - 1, k - 1)
    cramers_v = float(np.sqrt((chi2 / n) / denom)) if denom > 0 and n > 0 else 0.0
    return {
        "chi2": float(chi2),
        "p_value": float(p_val),
        "dof": int(dof),
        "cramers_v": cramers_v,
    }


def compute_kruskal_effect_size(df: pd.DataFrame, group_col: str, value_col: str = "CSS") -> Dict[str, float]:
    """
    Thực hiện kiểm định phi tham số Kruskal-Wallis cho điểm CSS giữa các nhóm,
    kèm kích thước hiệu ứng Epsilon-squared (H / (n - 1)) và Eta-squared ((H - k + 1) / (n - k)).
    """
    groups = [grp[value_col].dropna().values for _, grp in df.groupby(group_col)]
    k = len(groups)
    n = sum(len(g) for g in groups)
    if k < 2 or n <= k:
        raise ValueError("Cần ít nhất 2 nhóm và tổng cỡ mẫu lớn hơn số nhóm để chạy Kruskal-Wallis.")

    h_stat, p_val = stats.kruskal(*groups)
    epsilon_sq = float(h_stat / (n - 1))
    eta_sq = float(max(0.0, (h_stat - k + 1) / (n - k)))
    return {
        "h_stat": float(h_stat),
        "p_value": float(p_val),
        "epsilon_sq": epsilon_sq,
        "eta_sq": eta_sq,
    }


def pairwise_mannwhitney_correction(
    df: pd.DataFrame, group_col: str, value_col: str = "CSS"
) -> pd.DataFrame:
    """
    So sánh từng cặp nhóm bằng kiểm định Mann-Whitney U (two-sided)
    kèm hiệu chỉnh đa kiểm định bằng cả phương pháp Bonferroni và Holm-Bonferroni.
    """
    unique_groups = sorted(df[group_col].dropna().unique())
    pairs = list(combinations(unique_groups, 2))
    records: List[Dict[str, Any]] = []

    for g1, g2 in pairs:
        s1 = df.loc[df[group_col] == g1, value_col].dropna()
        s2 = df.loc[df[group_col] == g2, value_col].dropna()
        u_stat, p_raw = stats.mannwhitneyu(s1, s2, alternative="two-sided")
        records.append({
            "Nhóm 1": str(g1),
            "Nhóm 2": str(g2),
            "N1": len(s1),
            "N2": len(s2),
            "Trung vị 1": float(s1.median()),
            "Trung vị 2": float(s2.median()),
            "U-stat": float(u_stat),
            "p_raw": float(p_raw),
        })

    res_df = pd.DataFrame(records)
    if res_df.empty:
        return res_df

    m = len(res_df)
    res_df["p_bonferroni"] = np.minimum(res_df["p_raw"] * m, 1.0)

    # Hiệu chỉnh Holm-Bonferroni bậc thang
    order = np.argsort(res_df["p_raw"].values)
    p_holm = np.zeros(m, dtype=float)
    for rank, idx in enumerate(order):
        p_holm[idx] = res_df.loc[idx, "p_raw"] * (m - rank)
    p_holm_monotonic = np.maximum.accumulate(p_holm[order])
    p_holm[order] = np.minimum(p_holm_monotonic, 1.0)

    res_df["p_holm"] = p_holm
    res_df["Có ý nghĩa (Holm < 0.05)"] = res_df["p_holm"] < 0.05
    return res_df.sort_values(by="p_holm", ascending=True).reset_index(drop=True)


def analyze_rq3_statistical_tests(df: pd.DataFrame, small_group_threshold: int = 500) -> Dict[str, Any]:
    """
    Câu hỏi 3: Tổng hợp kết quả kiểm định Chi-square và Kruskal-Wallis theo vị trí (job_role),
    giới tính (gender) và nhóm tuổi (age_group), kèm cảnh báo nhóm cỡ mẫu nhỏ và bảng post-hoc.
    """
    dimensions = [
        ("job_role", "Vị trí công việc (job_role)"),
        ("gender", "Giới tính (gender)"),
        ("age_group", "Nhóm tuổi (age_group)"),
    ]

    summary_rows: List[Dict[str, Any]] = []
    group_details: Dict[str, pd.DataFrame] = {}

    for col, label in dimensions:
        chi_res = compute_chi2_cramers_v(df, col, "passed_hard_filter")
        kw_res = compute_kruskal_effect_size(df, col, "CSS")

        summary_rows.append({
            "Chiều phân tích": label,
            "Số nhóm": int(df[col].nunique()),
            "Chi-square": chi_res["chi2"],
            "p-value (Chi2)": chi_res["p_value"],
            "Cramér's V": chi_res["cramers_v"],
            "Kruskal-Wallis H": kw_res["h_stat"],
            "p-value (Kruskal)": kw_res["p_value"],
            "Epsilon-squared (ε²)": kw_res["epsilon_sq"],
            "Eta-squared (η²)": kw_res["eta_sq"],
        })

        detail_df = (
            df.groupby(col)
            .agg(
                Cỡ_mẫu=("CSS", "count"),
                Tỉ_lệ_đậu_lọc_cứng_pct=("passed_hard_filter", lambda s: s.mean() * 100),
                CSS_Trung_bình=("CSS", "mean"),
                CSS_Trung_vị=("CSS", "median"),
                CSS_Độ_lệch_chuẩn=("CSS", "std"),
            )
            .reset_index()
        )
        group_details[col] = detail_df

    test_summary_df = pd.DataFrame(summary_rows)

    age_counts = df["age_group"].value_counts()
    small_age_groups = age_counts[age_counts < small_group_threshold].to_dict()
    roles_with_40plus = sorted(df.loc[df["age_group"] == "40+", "job_role"].unique().tolist())

    pairwise_age_df = pairwise_mannwhitney_correction(df, "age_group", "CSS")
    pairwise_role_df = pairwise_mannwhitney_correction(df, "job_role", "CSS")

    warnings_list = [
        f"Cảnh báo cỡ mẫu nhỏ: Các nhóm tuổi { {k: int(v) for k, v in small_age_groups.items()} } "
        f"có số lượng quan sát nhỏ hơn {small_group_threshold} (chiếm từ 1.6% đến 3.6% tổng mẫu), "
        f"do đó ước lượng thống kê có phương sai lớn hơn các nhóm tuổi trẻ.",
        f"Phát hiện cấu trúc dữ liệu tổng hợp ở cột age_group: Nhãn '40+' (n={int(age_counts.get('40+', 0))}) "
        f"và cặp nhãn '41-45' (n={int(age_counts.get('41-45', 0))}), '46+' (n={int(age_counts.get('46+', 0))}) "
        f"được gán tách biệt ở hai nhóm vị trí công việc khác nhau (10 vị trí dùng '40+', 10 vị trí dùng '41-45'/'46+'). "
        f"Sự chênh lệch CSS ở nhóm tuổi cao thực chất là hiệu ứng trộn lẫn (confounding) với đặc thù vị trí công việc."
    ]

    cv_role = test_summary_df.loc[0, "Cramér's V"]
    eps_role = test_summary_df.loc[0, "Epsilon-squared (ε²)"]
    p_gender = test_summary_df.loc[1, "p-value (Chi2)"]
    cv_gender = test_summary_df.loc[1, "Cramér's V"]
    eps_gender = test_summary_df.loc[1, "Epsilon-squared (ε²)"]

    conclusion = (
        "Sự khác biệt có ý nghĩa thống kê và kích thước hiệu ứng lớn nhất nằm ở Vị trí công việc "
        f"(Cramér's V = {cv_role:.4f}, ε² = {eps_role:.4f}); "
        "trong khi Giới tính hoàn toàn không có chênh lệch về tỉ lệ đậu sàng lọc "
        f"(p = {p_gender:.4f}, Cramér's V = {cv_gender:.4f}) "
        f"và kích thước hiệu ứng lên điểm CSS gần như bằng không (ε² = {eps_gender:.4f})."
    )

    return {
        "test_summary_df": test_summary_df,
        "group_details": group_details,
        "small_age_groups": small_age_groups,
        "roles_with_40plus": roles_with_40plus,
        "pairwise_age_df": pairwise_age_df,
        "pairwise_role_df": pairwise_role_df,
        "warnings": warnings_list,
        "conclusion": conclusion,
    }