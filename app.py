"""Giao diện chính của ứng dụng Streamlit đánh giá quy trình sàng lọc ứng viên IT."""

import os
from typing import Dict, Any
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from modules.data_processing import load_and_clean_data, check_data_quality
from modules.analysis import (
    analyze_rq1_hard_filter,
    analyze_rq2_css_consistency,
    analyze_rq3_statistical_tests,
)
from modules.modeling import (
    train_and_evaluate_all_models,
    build_shap_explainer,
    explain_single_candidate,
)
from modules.visualizations import (
    plot_css_distribution,
    plot_rq1_hard_filter_by_label,
    plot_rq1_rejected_by_role,
    plot_rq2_css_by_label_boxplot,
    plot_rq3_css_by_group,
    plot_confusion_matrix_heatmap,
    plot_logistic_coefficients,
    plot_shap_waterfall,
)

# Ghi chú phiên bản: Code tương thích với Streamlit >= 1.40.0 (sử dụng width="stretch" cho st.dataframe)
st.set_page_config(
    page_title="Đánh giá Quy trình Sàng lọc Ứng viên IT",
    layout="wide",
)

DATA_REL_PATH = os.path.join("data", "candidates_full.csv")
FALLBACK_REL_PATH = "candidates_full.csv"


def render_figure(fig: plt.Figure) -> None:
    """Hiển thị đối tượng Figure lên Streamlit và đóng ngay để giải phóng bộ nhớ."""
    st.pyplot(fig)
    plt.close(fig)


@st.cache_data
def get_clean_dataset(relative_path: str) -> pd.DataFrame:
    """Đọc và làm sạch dữ liệu ứng viên từ đường dẫn tương đối, có lưu cache."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    primary_path = os.path.join(base_dir, relative_path)
    if os.path.exists(primary_path):
        return load_and_clean_data(primary_path)
    fallback_path = os.path.join(base_dir, FALLBACK_REL_PATH)
    return load_and_clean_data(fallback_path)


@st.cache_data
def get_analysis_results(_df: pd.DataFrame) -> Dict[str, Any]:
    """Tính toán toàn bộ chỉ số chất lượng dữ liệu và 3 câu hỏi nghiên cứu."""
    return {
        "quality": check_data_quality(_df),
        "rq1": analyze_rq1_hard_filter(_df),
        "rq2": analyze_rq2_css_consistency(_df),
        "rq3": analyze_rq3_statistical_tests(_df),
    }


@st.cache_resource
def get_models_and_shap(_df: pd.DataFrame) -> Dict[str, Any]:
    """Huấn luyện các mô hình và khởi tạo bộ giải thích SHAP một lần duy nhất."""
    model_bundle = train_and_evaluate_all_models(_df)
    shap_bundle = build_shap_explainer(
        model_bundle["models"]["logistic_regression"],
        model_bundle["X_train"],
    )
    return {
        "model_bundle": model_bundle,
        "shap_bundle": shap_bundle,
    }


def main() -> None:
    """Hàm điều phối hiển thị 7 tab giao diện theo đúng cấu trúc yêu cầu."""
    st.title("Phân tích tuyển dụng IT: Đánh giá Quy trình Sàng lọc ứng viên")

    try:
        df = get_clean_dataset(DATA_REL_PATH)
        analysis_bundle = get_analysis_results(df)
        ml_bundle = get_models_and_shap(df)
    except Exception as exc:
        st.error(f"Lỗi khởi tạo dữ liệu hoặc mô hình: {exc}")
        st.stop()

    q_info = analysis_bundle["quality"]
    rq1 = analysis_bundle["rq1"]
    rq2 = analysis_bundle["rq2"]
    rq3 = analysis_bundle["rq3"]
    model_res = ml_bundle["model_bundle"]
    shap_res = ml_bundle["shap_bundle"]

    tabs = st.tabs([
        "1. Chất lượng dữ liệu",
        "2. Câu hỏi 1 (Bộ lọc cứng)",
        "3. Câu hỏi 2 (Nhất quán CSS)",
        "4. Câu hỏi 3 (Kiểm định thống kê)",
        "5. Mô hình (Machine Learning)",
        "6. Giải thích (SHAP)",
        "7. Hạn chế",
    ])

    # ---------------- TAB 1: CHẤT LƯỢNG DỮ LIỆU ----------------
    with tabs[0]:
        st.header("1. Kiểm tra Chất lượng Dữ liệu & Công thức Điểm")
        st.markdown(
            "Tab này kiểm tra tính toàn vẹn của tập dữ liệu đầu vào (số dòng, giá trị khuyết, trùng lặp, miền giá trị) "
            "và xác minh công thức toán học cấu thành điểm tổng hợp `CSS`."
        )

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Tổng số dòng", f"{q_info['total_rows']:,}")
        c2.metric("Số cột", f"{q_info['total_cols']}")
        c3.metric("Số vị trí IT", f"{q_info['num_job_roles']} ({q_info['min_candidates_per_role']}/vị trí)")
        c4.metric("Giá trị thiếu (Missing)", f"{q_info['total_missing']}")
        c5.metric("Trùng candidate_id", f"{q_info['duplicate_ids']}")

        st.subheader("Kiểm chứng công thức: CSS = 0,4 · S_cv + 0,6 · S_int")
        f1, f2, f3 = st.columns(3)
        f1.metric("Sai số tuyệt đối lớn nhất (CSS)", f"{q_info['max_css_error']:.6f}")
        f2.metric("Sai số tuyệt đối lớn nhất (S_cv)", f"{q_info['max_scv_error']:.6f}")
        f3.metric("Sai số tuyệt đối lớn nhất (S_int)", f"{q_info['max_sint_error']:.6f}")

        st.subheader("Miền giá trị các biến định lượng chính")
        st.dataframe(
            q_info["domain_summary"].style.format({
                "min": "{:.4f}", "mean": "{:.4f}", "median": "{:.4f}", "max": "{:.4f}", "std": "{:.4f}"
            }),
            width="stretch",
        )

        render_figure(plot_css_distribution(df))

        st.success(
            f"**Kết luận:** Tập dữ liệu gồm đúng {q_info['total_rows']:,} ứng viên thuộc {q_info['num_job_roles']} vị trí "
            f"(không có giá trị thiếu hay trùng mã), và điểm `CSS` hoàn toàn khớp với công thức "
            f"`0,4·S_cv + 0,6·S_int` với sai số làm tròn tối đa chỉ là `{q_info['max_css_error']:.6f}`."
        )

    # ---------------- TAB 2: CÂU HỎI 1 ----------------
    with tabs[1]:
        st.header("2. Câu hỏi nghiên cứu 1: Bộ lọc cứng có loại nhầm ứng viên mạnh không?")
        st.markdown(
            "Phân tích biến `passed_hard_filter` theo từng mức `relevance_label` (0: Không phù hợp, 1: Yếu, "
            "2: Khá phù hợp, 3: Rất phù hợp) và theo 20 vị trí công việc để đo lường tỉ lệ bỏ sót nhân tài."
        )

        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Ứng viên Nhãn 2–3 bị loại",
            f"{rq1['rejected_strong']:,} / {rq1['total_strong']:,}",
            f"{rq1['rejected_strong_rate']:.2f}%",
            delta_color="inverse",
        )
        m2.metric(
            "Ứng viên Nhãn 3 (Xuất sắc) bị loại",
            f"{rq1['rejected_label3']:,} / {rq1['total_label3']:,}",
            f"{rq1['rejected_label3_rate']:.2f}%",
            delta_color="inverse",
        )
        top_r = rq1["by_role_df"].iloc[0]
        m3.metric(
            f"Vị trí loại nhầm cao nhất ({top_r['Vị trí (job_role)']})",
            f"{int(top_r['Số UV mạnh bị loại'])} / {int(top_r['Số UV mạnh (Nhãn 2-3)'])}",
            f"{top_r['Tỉ lệ loại nhầm Nhãn 2-3 (%)']:.2f}%",
            delta_color="inverse",
        )

        col_left, col_right = st.columns([1.1, 1.2])
        with col_left:
            st.subheader("Bảng chéo Bộ lọc cứng × Nhãn")
            st.dataframe(
                rq1["summary_table"].style.format({
                    "Tỉ lệ Trượt (%)": "{:.2f}%",
                    "Tỉ lệ Đậu (%)": "{:.2f}%",
                }),
                width="stretch",
            )
            render_figure(plot_rq1_hard_filter_by_label(rq1["summary_table"]))

        with col_right:
            st.subheader("Tỉ lệ loại nhầm ứng viên mạnh theo Vị trí")
            render_figure(plot_rq1_rejected_by_role(rq1["by_role_df"]))

        st.subheader("Bảng chi tiết theo 20 Vị trí công việc")
        st.dataframe(
            rq1["by_role_df"].style.format({
                "Tỉ lệ loại nhầm Nhãn 2-3 (%)": "{:.2f}%",
                "Tỉ lệ loại nhầm Nhãn 3 (%)": "{:.2f}%",
            }),
            width="stretch",
        )

        st.info(f"**Kết luận:** {rq1['conclusion']}")

    # ---------------- TAB 3: CÂU HỎI 2 ----------------
    with tabs[2]:
        st.header("3. Câu hỏi nghiên cứu 2: Nhãn relevance_label có nhất quán với điểm CSS không?")
        st.markdown(
            "Chia điểm `CSS` thành 4 tứ phân vị (`Q1` đến `Q4`), đối chiếu với `relevance_label`, "
            "liệt kê các hồ sơ điểm cao (`Q4`) nhưng nhận nhãn thấp (`0–1`), và giải thích hiện tượng đảo ngược điểm trung bình giữa nhãn `0` và nhãn `1`."
        )

        col_q1, col_q2 = st.columns(2)
        with col_q1:
            st.subheader("Bảng Tứ phân vị CSS × Nhãn (Số lượng)")
            st.dataframe(rq2["ct_counts"], width="stretch")
        with col_q2:
            st.subheader("Bảng Tứ phân vị CSS × Nhãn (Tỉ lệ % theo hàng)")
            st.dataframe(rq2["ct_pct"].style.format("{:.2f}%"), width="stretch")

        st.subheader("Thống kê Điểm CSS theo từng Nhãn và Giải thích Bất thường (Nhãn 0 vs Nhãn 1)")
        c_stat, c_plot = st.columns([1.15, 1.15])
        with c_stat:
            st.dataframe(
                rq2["css_by_label"].style.format({
                    "CSS Trung bình": "{:.4f}",
                    "CSS Trung vị": "{:.4f}",
                    "Độ lệch chuẩn": "{:.4f}",
                    "CSS Thấp nhất": "{:.4f}",
                    "CSS Cao nhất": "{:.4f}",
                }),
                width="stretch",
            )
            st.markdown("**Phân rã Nhãn 0 và Nhãn 1 theo trạng thái Bộ lọc cứng (`passed_hard_filter`):**")
            st.dataframe(
                rq2["label01_breakdown"].style.format({
                    "CSS Trung bình": "{:.4f}",
                    "CSS Trung vị": "{:.4f}",
                    "Tỉ lệ trong nhãn (%)": "{:.2f}%",
                }),
                width="stretch",
            )
            st.warning(f"📌 **Giải thích bất thường:** {rq2['anomaly_explanation']}")

        with c_plot:
            render_figure(plot_rq2_css_by_label_boxplot(df))

        st.subheader(f"Danh sách {rq2['mismatch_count']} ca 'CSS cao (Q4) nhưng Nhãn 0–1'")
        st.dataframe(
            rq2["mismatch_df"].style.format({
                "CSS": "{:.4f}", "S_cv": "{:.4f}", "S_int": "{:.4f}", "years_experience": "{:.1f}"
            }),
            width="stretch",
        )

        st.info(f"**Kết luận:** {rq2['conclusion']}")

    # ---------------- TAB 4: CÂU HỎI 3 ----------------
    with tabs[3]:
        st.header("4. Câu hỏi nghiên cứu 3: Chênh lệch giữa các nhóm có ý nghĩa thống kê không?")
        st.markdown(
            "Sử dụng kiểm định **Chi-square** (kèm **Cramér's V**) cho tỉ lệ đậu bộ lọc cứng và kiểm định phi tham số "
            "**Kruskal-Wallis** (kèm **Epsilon-squared** $\\varepsilon^2$ và **Eta-squared** $\\eta^2$) cho điểm `CSS` theo Vị trí, Giới tính và Nhóm tuổi. "
            "Áp dụng hiệu chỉnh **Bonferroni** và **Holm** khi so sánh nhiều cặp."
        )

        for w_msg in rq3["warnings"]:
            st.warning(f"⚠️ {w_msg}")

        st.subheader("Bảng Tổng hợp Kiểm định Thống kê")
        st.dataframe(
            rq3["test_summary_df"].style.format({
                "Chi-square": "{:.4f}",
                "p-value (Chi2)": "{:.4e}",
                "Cramér's V": "{:.4f}",
                "Kruskal-Wallis H": "{:.4f}",
                "p-value (Kruskal)": "{:.4e}",
                "Epsilon-squared (ε²)": "{:.4f}",
                "Eta-squared (η²)": "{:.4f}",
            }),
            width="stretch",
        )

        col_g, col_a = st.columns(2)
        with col_g:
            st.subheader("Chi tiết theo Giới tính (gender)")
            st.dataframe(
                rq3["group_details"]["gender"].style.format({
                    "Tỉ_lệ_đậu_lọc_cứng_pct": "{:.2f}%",
                    "CSS_Trung_bình": "{:.4f}",
                    "CSS_Trung_vị": "{:.4f}",
                    "CSS_Độ_lệch_chuẩn": "{:.4f}",
                }),
                width="stretch",
            )
            render_figure(plot_rq3_css_by_group(df, "gender", "Phân phối CSS theo Giới tính"))

        with col_a:
            st.subheader("Chi tiết theo Nhóm tuổi (age_group)")
            st.dataframe(
                rq3["group_details"]["age_group"].style.format({
                    "Tỉ_lệ_đậu_lọc_cứng_pct": "{:.2f}%",
                    "CSS_Trung_bình": "{:.4f}",
                    "CSS_Trung_vị": "{:.4f}",
                    "CSS_Độ_lệch_chuẩn": "{:.4f}",
                }),
                width="stretch",
            )
            render_figure(plot_rq3_css_by_group(df, "age_group", "Phân phối CSS theo Nhóm tuổi"))

        st.subheader("So sánh từng cặp Nhóm tuổi (Mann-Whitney U kèm hiệu chỉnh Bonferroni & Holm)")
        st.dataframe(
            rq3["pairwise_age_df"].style.format({
                "Trung vị 1": "{:.4f}",
                "Trung vị 2": "{:.4f}",
                "U-stat": "{:.1f}",
                "p_raw": "{:.4e}",
                "p_bonferroni": "{:.4e}",
                "p_holm": "{:.4e}",
            }),
            width="stretch",
        )

        st.info(f"**Kết luận:** {rq3['conclusion']}")

    # ---------------- TAB 5: MÔ HÌNH (MACHINE LEARNING) ----------------
    with tabs[4]:
        st.header("5. Mô hình Dự đoán Mức phù hợp & Kiểm tra Công thức")
        st.markdown(
            "Huấn luyện mô hình phân loại nhị phân dự đoán ứng viên phù hợp (`relevance_label >= 2`) trên tập `train` (80%) "
            "và đánh giá trên tập `test` (20%, `stratify=y, random_state=42`). Mọi bước `StandardScaler` và `OneHotEncoder` "
            "đều nằm trong `Pipeline`. Không sử dụng `CSS`, `passed_hard_filter`, `S_*`, `gender`, `age_group` làm đặc trưng."
        )

        metrics_rows = []
        for m_name, m_val in model_res["metrics"].items():
            metrics_rows.append({
                "Mô hình": m_name,
                "Accuracy": m_val["accuracy"],
                "Precision": m_val["precision"],
                "Recall": m_val["recall"],
                "F1-Score": m_val["f1"],
            })
        metrics_df = pd.DataFrame(metrics_rows)

        st.subheader("5.1. Kết quả Đánh giá Mô hình Phân loại (`relevance_label >= 2`)")
        st.dataframe(
            metrics_df.style.format({
                "Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}", "F1-Score": "{:.4f}"
            }),
            width="stretch",
        )

        cm1, cm2, cm3 = st.columns(3)
        with cm1:
            render_figure(
                plot_confusion_matrix_heatmap(
                    model_res["metrics"]["Baseline (Lớp đa số)"]["confusion_matrix"],
                    "Confusion Matrix: Baseline",
                )
            )
        with cm2:
            render_figure(
                plot_confusion_matrix_heatmap(
                    model_res["metrics"]["Logistic Regression (Chính)"]["confusion_matrix"],
                    "Confusion Matrix: Logistic Regression",
                )
            )
        with cm3:
            render_figure(
                plot_confusion_matrix_heatmap(
                    model_res["metrics"]["Random Forest (So sánh)"]["confusion_matrix"],
                    "Confusion Matrix: Random Forest",
                )
            )

        st.subheader("5.2. Diễn giải Hệ số Logistic Regression & Odds Ratio")
        col_coef_plot, col_coef_text = st.columns([1.15, 1.15])
        with col_coef_plot:
            render_figure(plot_logistic_coefficients(model_res["raw_coef_df"]))
        with col_coef_text:
            st.markdown("**Diễn giải tự động tác động của các đặc trưng hồ sơ & bài test (đã chuẩn hóa):**")
            for line in model_res["interpretations"]:
                st.markdown(f"- {line}")

        with st.expander("Xem toàn bộ bảng hệ số Logistic Regression (bao gồm cả 19 biến giả job_role)"):
            st.dataframe(
                model_res["coef_df"].style.format({
                    "Hệ số (Beta)": "{:+.4f}", "Odds Ratio (exp(Beta))": "{:.4f}"
                }),
                width="stretch",
            )

        st.subheader("5.3. Kiểm tra Tính công bằng của Mô hình (Fairness Audit trên tập Test)")
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            st.markdown("**Theo Giới tính (`gender`):**")
            st.dataframe(
                model_res["fairness_tables"]["gender"].style.format({
                    "Tỉ lệ thực tế đạt (%)": "{:.2f}%",
                    "Tỉ lệ mô hình dự đoán đạt (%)": "{:.2f}%",
                    "True Positive Rate / Recall (%)": "{:.2f}%",
                    "False Positive Rate (%)": "{:.2f}%",
                }),
                width="stretch",
            )
        with f_col2:
            st.markdown("**Theo Nhóm tuổi (`age_group`):**")
            st.dataframe(
                model_res["fairness_tables"]["age_group"].style.format({
                    "Tỉ lệ thực tế đạt (%)": "{:.2f}%",
                    "Tỉ lệ mô hình dự đoán đạt (%)": "{:.2f}%",
                    "True Positive Rate / Recall (%)": "{:.2f}%",
                    "False Positive Rate (%)": "{:.2f}%",
                }),
                width="stretch",
            )

        st.subheader("5.4. Kiểm tra Công thức Điểm bằng Linear Regression (`CSS ~ S_cv + S_int`)")
        st.caption("Lưu ý: Đây là bước xác nhận lại công thức toán học của hệ thống chấm điểm, không phải phát hiện quy luật dự báo mới.")
        lc = model_res["linear_check"]
        l1, l2, l3, l4 = st.columns(4)
        l1.metric("Hệ số S_cv (Kỳ vọng 0,4)", f"{lc['coef_s_cv']:.6f}")
        l2.metric("Hệ số S_int (Kỳ vọng 0,6)", f"{lc['coef_s_int']:.6f}")
        l3.metric("R² Score trên tập Test", f"{lc['r2']:.8f}")
        l4.metric("MAE trên tập Test", f"{lc['mae']:.6f}")

        lr_acc = model_res["metrics"]["Logistic Regression (Chính)"]["accuracy"]
        base_acc = model_res["metrics"]["Baseline (Lớp đa số)"]["accuracy"]
        top_feat_1 = model_res["raw_coef_df"].iloc[0]["Đặc trưng"]
        top_feat_2 = model_res["raw_coef_df"].iloc[1]["Đặc trưng"]

        st.info(
            f"**Kết luận:** Mô hình `Logistic Regression` đạt Accuracy `{lr_acc:.4f}` (vượt qua mức `{base_acc:.4f}` của Baseline đoán lớp đa số), "
            f"trong đó hai yếu tố kỹ năng/hồ sơ có trọng số chuẩn hóa lớn nhất là `{top_feat_1}` và `{top_feat_2}`; "
            f"đồng thời `Linear Regression` xác nhận chính xác tuyệt đối công thức `CSS = {lc['coef_s_cv']:.2f}·S_cv + {lc['coef_s_int']:.2f}·S_int` ($R^2 = {lc['r2']:.6f}$)."
        )

    # ---------------- TAB 6: GIẢI THÍCH (SHAP) ----------------
    with tabs[5]:
        st.header("6. Giải thích Quyết định Mô hình Phân loại cho từng Ứng viên (SHAP)")
        st.markdown(
            "Sử dụng `shap.LinearExplainer` trực tiếp trên mô hình phân loại `Logistic Regression` đã chọn "
            "để giải thích đóng góp của từng đặc trưng (trên thang Log-Odds) vào dự đoán `relevance_label >= 2` của một ứng viên cụ thể."
        )

        default_id = str(df["candidate_id"].iloc[0])
        lookup_id = st.text_input(
            "Nhập mã ứng viên (candidate_id) cần giải thích (ví dụ: SE00001, DS00010, CYB00020):",
            value=default_id,
        ).strip()

        if lookup_id:
            try:
                exp_res = explain_single_candidate(df, lookup_id, shap_res)
                c_info = exp_res["candidate_info"]

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Mã ứng viên", f"{c_info['candidate_id']} ({c_info['job_role_display']})")
                k2.metric("Nhãn thực tế (relevance_label)", f"{c_info['relevance_label']} ({'Đạt >=2' if c_info['relevance_label'] >= 2 else 'Không đạt <2'})")
                k3.metric("Xác suất dự đoán (P(Nhãn >= 2))", f"{exp_res['pred_prob'] * 100:.2f}%")
                k4.metric("Kết luận của Mô hình", "Phù hợp (>= 2)" if exp_res["pred_class"] == 1 else "Chưa phù hợp (< 2)")

                fig = plot_shap_waterfall(exp_res["shap_explanation_single"])
                st.pyplot(fig, use_container_width=False)
                plt.close(fig)

                st.subheader("Bảng chi tiết đóng góp SHAP của từng đặc trưng")
                st.markdown(
                    f"- **Giá trị nền (Base Log-Odds):** `{exp_res['base_value']:.6f}`  \n"
                    f"- **Tổng đóng góp SHAP:** `{exp_res['total_shap']:+.6f}`  \n"
                    f"- **Log-Odds cuối cùng ($f(x)$):** `{exp_res['final_log_odds']:.6f}` "
                    f"→ **Xác suất $\\sigma(f(x))$:** `{exp_res['pred_prob']:.6f}`"
                )
                st.dataframe(
                    exp_res["contrib_df"].style.format({
                        "Giá trị sau chuẩn hóa / One-Hot": "{:.4f}",
                        "Đóng góp SHAP (Log-Odds)": "{:+.6f}",
                    }),
                    width="stretch",
                )

                top_contrib = exp_res["contrib_df"].iloc[0]
                st.info(
                    f"**Kết luận:** Đối với ứng viên `{lookup_id}`, mô hình dự đoán xác suất phù hợp đạt "
                    f"`{exp_res['pred_prob'] * 100:.2f}%`, trong đó đặc trưng tạo ra lực đẩy lớn nhất là "
                    f"`{top_contrib['Đặc trưng']}` với mức đóng góp SHAP `{top_contrib['Đóng góp SHAP (Log-Odds)']:+.4f}` log-odds."
                )
            except ValueError as err:
                st.warning(str(err))

    # ---------------- TAB 7: HẠN CHẾ ----------------
    with tabs[6]:
        st.header("7. Hạn chế của Dữ liệu và Phạm vi Kết luận")
        st.markdown(
            "Để đảm bảo tính trung thực học thuật, báo cáo này nêu rõ các giới hạn của dữ liệu và phương pháp như sau:"
        )

        st.markdown(
            f"""
            1. **Dữ liệu có bản chất sinh tổng hợp (Synthetic Data):**
               - Tập dữ liệu gồm đúng `{q_info['total_rows']:,}` ứng viên chia đều tuyệt đối `{q_info['min_candidates_per_role']}` người cho mỗi vị trí trong `{q_info['num_job_roles']}` vị trí IT, không có bất kỳ giá trị khuyết thiếu nào.
               - Điểm `CSS`, `S_cv`, `S_int` tuân thủ công thức tuyến tính chính xác tuyệt đối (sai số tối đa `{q_info['max_css_error']:.6f}` chỉ do làm tròn 4 chữ số thập phân).
               - Đặc biệt, cột `age_group` tồn tại cấu trúc phân tách nhân tạo: 10 vị trí công việc chỉ sử dụng nhãn `40+` (`n = 345`), trong khi 10 vị trí còn lại sử dụng cặp nhãn `41-45` (`n = 430`) và `46+` (`n = 195`). Do đó, **mọi chênh lệch thống kê tìm thấy trong đồ án này phản ánh quy tắc sinh dữ liệu của hệ thống giả lập, tuyệt đối không được diễn giải như sự thật của thị trường lao động IT thực tế.**

            2. **Hạn chế về cỡ mẫu ở các nhóm tuổi cao:**
               - Trong khi các nhóm tuổi từ `22-25` đến `36-40` có cỡ mẫu lớn (`1.572` đến `3.757` người/nhóm), các nhóm `41-45` (`430` người), `40+` (`345` người) và đặc biệt là `46+` (`195` người, tương đương chỉ `11` đến `26` người ở mỗi vị trí có xuất hiện) có cỡ mẫu nhỏ. Kết quả kiểm định và tỉ lệ dự đoán trên các nhóm nhỏ này có độ bất định cao hơn.

            3. **Giới hạn của Mô hình Dự đoán:**
               - Mô hình phân loại chỉ sử dụng các biến đầu vào rút gọn (`edu_level`, `years_experience`, `edu_relevance`, `P_mcq`, `P_desc`, `P_code`, `job_role`) và không có dữ liệu phi cấu trúc thực tế (như nội dung CV văn bản, lịch sử dự án GitHub hay đánh giá hành vi phỏng vấn).
               - Việc `relevance_label` trong dữ liệu gốc phụ thuộc mạnh vào cả ngưỡng điểm lẫn điều kiện lọc cứng theo từng vị trí khiến mô hình tuyến tính như `Logistic Regression` chỉ nắm bắt được xu hướng chính (`Accuracy = {model_res['metrics']['Logistic Regression (Chính)']['accuracy']:.4f}`) chứ không thay thế được quy trình thẩm định chuyên môn của con người.
            """
        )

        st.info(
            "**Kết luận:** Kết quả của đồ án có giá trị minh họa phương pháp kiểm toán quy trình sàng lọc bằng Python "
            "(phát hiện lỗi lọc cứng, kiểm tra tính nhất quán điểm–nhãn, kiểm định thống kê và giải thích mô hình), "
            "nhưng cần được kiểm chứng lại trên dữ liệu tuyển dụng thực tế trước khi áp dụng vào vận hành doanh nghiệp."
        )


if __name__ == "__main__":
    main()
