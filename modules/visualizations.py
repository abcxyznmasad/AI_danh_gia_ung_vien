"""Module chứa toàn bộ các hàm vẽ biểu đồ trực quan hóa, mọi hàm đều trả về matplotlib.figure.Figure."""

from typing import Any
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
import shap


def plot_css_distribution(df: pd.DataFrame) -> plt.Figure:
    """Vẽ biểu đồ phân phối (Histogram + KDE) của điểm CSS thực tế."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.histplot(data=df, x="CSS", kde=True, color="teal", bins=40, ax=ax)
    ax.set_title("Phân phối Điểm Candidate Suitability Score — (CSS)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Điểm CSS")
    ax.set_ylabel("Số lượng ứng viên")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
    fig.tight_layout()
    return fig


def plot_rq1_hard_filter_by_label(summary_table: pd.DataFrame) -> plt.Figure:
    """Vẽ biểu đồ cột chồng tỉ lệ Đậu / Trượt bộ lọc cứng theo từng nhãn relevance_label."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    labels = summary_table["Nhãn mức phù hợp"].astype(str)
    rej_pct = summary_table["Tỉ lệ Trượt (%)"]
    pass_pct = summary_table["Tỉ lệ Đậu (%)"]

    ax.bar(labels, pass_pct, label="Đậu lọc cứng (1)", color="#2a9d8f")
    p2 = ax.bar(labels, rej_pct, bottom=pass_pct, label="Trượt lọc cứng (0)", color="#e76f51")

    ax.set_title("Tỉ lệ Đậu / Trượt Bộ lọc cứng theo Nhãn Mức phù hợp", fontsize=13, fontweight="bold")
    ax.set_xlabel("Nhãn mức phù hợp (relevance_label)")
    ax.set_ylabel("Tỉ lệ (%)")
    ax.set_ylim(0, 108)
    ax.legend(loc="upper right")

    for rect in p2:
        h = rect.get_height()
        if h > 3:
            ax.annotate(
                f"{h:.1f}%",
                (rect.get_x() + rect.get_width() / 2.0, rect.get_y() + h / 2.0),
                ha="center", va="center", color="white", fontweight="bold", fontsize=9,
            )
    fig.tight_layout()
    return fig


def plot_rq1_rejected_by_role(by_role_df: pd.DataFrame) -> plt.Figure:
    """Vẽ biểu đồ ngang tỉ lệ ứng viên mạnh (nhãn 2-3) bị loại nhầm theo 20 vị trí công việc."""
    fig, ax = plt.subplots(figsize=(10, 6.5))
    df_sorted = by_role_df.sort_values(by="Tỉ lệ loại nhầm Nhãn 2-3 (%)", ascending=True)
    bars = ax.barh(
        df_sorted["Vị trí (job_role)"],
        df_sorted["Tỉ lệ loại nhầm Nhãn 2-3 (%)"],
        color="#e63946",
        alpha=0.85,
    )
    ax.set_title("Tỉ lệ Ứng viên Mạnh (Nhãn 2–3) bị Loại bởi Bộ lọc cứng theo Vị trí", fontsize=12, fontweight="bold")
    ax.set_xlabel("Tỉ lệ bị loại (%)")
    ax.set_ylabel("Vị trí công việc")

    max_val = float(df_sorted["Tỉ lệ loại nhầm Nhãn 2-3 (%)"].max())
    ax.set_xlim(0, max_val * 1.18 if max_val > 0 else 10.0)

    for bar in bars:
        w = bar.get_width()
        ax.annotate(
            f"{w:.1f}%",
            (w, bar.get_y() + bar.get_height() / 2.0),
            xytext=(4, 0), textcoords="offset points",
            ha="left", va="center", fontsize=8.5, fontweight="bold",
        )
    fig.tight_layout()
    return fig


def plot_rq2_css_by_label_boxplot(df: pd.DataFrame) -> plt.Figure:
    """Vẽ Boxplot phân phối điểm CSS theo từng nhãn relevance_label và trạng thái đậu lọc cứng."""
    fig, ax = plt.subplots(figsize=(9, 5))
    df_plot = df.copy()
    df_plot["Bộ lọc cứng"] = df_plot["passed_hard_filter"].map({0: "Trượt (0)", 1: "Đậu (1)"})
    sns.boxplot(
        data=df_plot,
        x="relevance_label",
        y="CSS",
        hue="Bộ lọc cứng",
        palette={"Trượt (0)": "#e76f51", "Đậu (1)": "#2a9d8f"},
        ax=ax,
    )
    q75 = float(df["CSS"].quantile(0.75))
    ax.axhline(q75, color="navy", linestyle="--", linewidth=1.2, label=f"Ngưỡng Q4 (CSS = {q75:.3f})")
    ax.set_title("Phân phối Điểm CSS theo Nhãn và Trạng thái Lọc cứng", fontsize=13, fontweight="bold")
    ax.set_xlabel("Nhãn mức phù hợp (relevance_label)")
    ax.set_ylabel("Điểm CSS")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_rq3_css_by_group(df: pd.DataFrame, group_col: str, title: str) -> plt.Figure:
    """Vẽ Boxplot điểm CSS theo biến phân nhóm (gender, age_group, hoặc job_role)."""
    fig, ax = plt.subplots(figsize=(10, 4.8))
    order = sorted(df[group_col].unique())
    sns.boxplot(data=df, x=group_col, y="CSS", order=order, palette="Set2", ax=ax)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(group_col)
    ax.set_ylabel("Điểm CSS")
    if len(order) > 8:
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def plot_confusion_matrix_heatmap(cm: np.ndarray, title: str) -> plt.Figure:
    """Vẽ ma trận nhầm lẫn (Confusion Matrix) dưới dạng Heatmap rõ ràng."""
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Dự đoán < 2 (0)", "Dự đoán >= 2 (1)"],
        yticklabels=["Thực tế < 2 (0)", "Thực tế >= 2 (1)"],
        ax=ax,
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Nhãn Dự đoán")
    ax.set_ylabel("Nhãn Thực tế")
    fig.tight_layout()
    return fig


def plot_logistic_coefficients(raw_coef_df: pd.DataFrame) -> plt.Figure:
    """
    Vẽ biểu đồ hệ số Beta của các đặc trưng hồ sơ & bài test gốc trong mô hình Logistic Regression.
    Hỗ trợ đầy đủ cả hệ số âm và hệ số dương mà không bị cắt lẹm thanh hay chữ.
    """
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    df_sorted = raw_coef_df.sort_values(by="Hệ số (Beta)", ascending=True)
    values = df_sorted["Hệ số (Beta)"].astype(float)
    colors = ["#e63946" if v < 0 else "#1d3557" for v in values]

    bars = ax.barh(df_sorted["Đặc trưng"], values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Hệ số Logistic Regression (Beta) của các Đặc trưng Gốc", fontsize=12, fontweight="bold")
    ax.set_xlabel("Hệ số Beta (sau chuẩn hóa StandardScaler)")
    ax.set_ylabel("Đặc trưng")

    min_w = float(values.min()) if not values.empty else 0.0
    max_w = float(values.max()) if not values.empty else 0.0
    span = max(max_w - min_w, max(abs(min_w), abs(max_w), 1.0))
    left_lim = min(0.0, min_w - span * 0.28) if min_w < 0 else 0.0
    right_lim = max(0.0, max_w + span * 0.28) if max_w > 0 else 0.0
    if left_lim == right_lim:
        left_lim, right_lim = -1.0, 1.0
    ax.set_xlim(left_lim, right_lim)

    for bar in bars:
        w = bar.get_width()
        offset = 5 if w >= 0 else -5
        align = "left" if w >= 0 else "right"
        ax.annotate(
            f"{w:+.3f} (OR={np.exp(w):.2f})",
            (w, bar.get_y() + bar.get_height() / 2.0),
            xytext=(offset, 0), textcoords="offset points",
            ha=align, va="center", fontsize=9, fontweight="bold",
        )
    fig.tight_layout()
    return fig


def plot_shap_waterfall(shap_explanation_single: Any) -> plt.Figure:
    """
    Vẽ biểu đồ SHAP Waterfall đúng chuẩn: đóng các figure cũ bằng plt.close('all'),
    gọi shap.plots.waterfall(..., show=False), và lấy đúng figure hiện hành bằng plt.gcf().
    """
    plt.close("all")
    shap.plots.waterfall(shap_explanation_single, max_display=7, show=False)
    fig = plt.gcf()
    fig.set_size_inches(9.5, 3.5)
    fig.tight_layout()
    return fig