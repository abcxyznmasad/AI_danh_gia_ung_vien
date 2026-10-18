"""Module huấn luyện, đánh giá mô hình Machine Learning, kiểm tra công bằng và giải thích SHAP."""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    r2_score,
    mean_absolute_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from modules.data_processing import (
    RAW_NUMERIC_FEATURES,
    RAW_CATEGORICAL_FEATURES,
    prepare_classification_features,
)

RANDOM_STATE = 42
TEST_SIZE = 0.2


def _compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Tính các độ đo phân loại nhị phân chuẩn: Accuracy, Precision, Recall, F1 và Confusion Matrix."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def train_and_evaluate_all_models(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Thực hiện:
    1. Chia train/test có stratify=y, random_state=42.
    2. Huấn luyện DummyClassifier (Baseline lớp đa số), LogisticRegression (Pipeline),
       và RandomForestClassifier (Pipeline đối sánh).
    3. Trích xuất bảng hệ số Logistic Regression kèm Odds Ratio và lời diễn giải tự động.
    4. Kiểm tra tính công bằng của dự đoán trên tập test theo gender và age_group.
    5. Kiểm tra công thức điểm CSS ~ S_cv + S_int bằng LinearRegression.
    """
    X, y, demographics = prepare_classification_features(df)

    X_train, X_test, y_train, y_test, demo_train, demo_test = train_test_split(
        X, y, demographics, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), RAW_NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), RAW_CATEGORICAL_FEATURES),
        ]
    )

    # 1. Baseline đoán theo lớp đa số
    baseline_clf = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    baseline_clf.fit(X_train, y_train)
    y_pred_base = baseline_clf.predict(X_test)
    metrics_base = _compute_classification_metrics(y_test.values, y_pred_base)

    # 2. Mô hình chính: Logistic Regression trong Pipeline
    lr_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )
    lr_pipeline.fit(X_train, y_train)
    y_pred_lr = lr_pipeline.predict(X_test)
    metrics_lr = _compute_classification_metrics(y_test.values, y_pred_lr)

    # 3. Mô hình so sánh: Random Forest Classifier trong Pipeline
    rf_preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), RAW_NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), RAW_CATEGORICAL_FEATURES),
        ]
    )
    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", rf_preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
        ]
    )
    rf_pipeline.fit(X_train, y_train)
    y_pred_rf = rf_pipeline.predict(X_test)
    metrics_rf = _compute_classification_metrics(y_test.values, y_pred_rf)

    # Trích xuất tên đặc trưng sau biến đổi và hệ số của Logistic Regression
    fitted_prep = lr_pipeline.named_steps["preprocessor"]
    cat_feature_names = list(
        fitted_prep.named_transformers_["cat"].get_feature_names_out(RAW_CATEGORICAL_FEATURES)
    )
    all_feature_names = RAW_NUMERIC_FEATURES + cat_feature_names
    lr_coefs = lr_pipeline.named_steps["classifier"].coef_[0]
    lr_intercept = float(lr_pipeline.named_steps["classifier"].intercept_[0])

    coef_df = pd.DataFrame({
        "Đặc trưng": all_feature_names,
        "Nhóm": ["Kỹ năng & Hồ sơ gốc" if f in RAW_NUMERIC_FEATURES else "Vị trí công việc (One-Hot)" for f in all_feature_names],
        "Hệ số (Beta)": lr_coefs,
        "Odds Ratio (exp(Beta))": np.exp(lr_coefs),
    })
    coef_df["Tác động"] = coef_df["Hệ số (Beta)"].apply(
        lambda b: "Tăng khả năng phù hợp" if b > 0 else "Giảm khả năng phù hợp"
    )
    coef_df = coef_df.sort_values(by="Hệ số (Beta)", key=np.abs, ascending=False).reset_index(drop=True)

    # Tạo diễn giải tự động cho các đặc trưng kỹ năng/hồ sơ gốc
    raw_coef_df = coef_df[coef_df["Đặc trưng"].isin(RAW_NUMERIC_FEATURES)].sort_values(
        by="Hệ số (Beta)", ascending=False
    ).reset_index(drop=True)

    interpretations: List[str] = []
    for _, row in raw_coef_df.iterrows():
        feat = row["Đặc trưng"]
        beta = float(row["Hệ số (Beta)"])
        odds = float(row["Odds Ratio (exp(Beta))"])
        direction = "tăng" if beta > 0 else "giảm"
        pct_change = abs(odds - 1.0) * 100
        interpretations.append(
            f"**{feat}**: Hệ số $\\beta = {beta:+.4f}$, Odds Ratio $= {odds:.4f}$ "
            f"→ Khi {feat} tăng 1 độ lệch chuẩn (giữ nguyên các yếu tố khác), "
            f"tỉ lệ cược (odds) ứng viên đạt mức phù hợp (nhãn $\\ge 2$) **{direction} {pct_change:.1f}%** (gấp {odds:.2f} lần)."
        )

    # 4. Kiểm tra tính công bằng (Fairness Audit) trên tập test theo gender và age_group
    fairness_tables: Dict[str, pd.DataFrame] = {}
    eval_df = demo_test.copy()
    eval_df["y_true"] = y_test.values
    eval_df["y_pred"] = y_pred_lr

    for group_col in ["gender", "age_group"]:
        rows = []
        for grp_val, grp_data in eval_df.groupby(group_col):
            actual_pos = grp_data["y_true"].mean() * 100
            pred_pos = grp_data["y_pred"].mean() * 100
            tpr = recall_score(grp_data["y_true"], grp_data["y_pred"], zero_division=0) * 100
            neg_mask = grp_data["y_true"] == 0
            fpr = (
                (grp_data.loc[neg_mask, "y_pred"] == 1).mean() * 100
                if neg_mask.sum() > 0 else 0.0
            )
            rows.append({
                "Nhóm": str(grp_val),
                "Cỡ mẫu Test": len(grp_data),
                "Tỉ lệ thực tế đạt (%)": actual_pos,
                "Tỉ lệ mô hình dự đoán đạt (%)": pred_pos,
                "True Positive Rate / Recall (%)": tpr,
                "False Positive Rate (%)": fpr,
            })
        fairness_tables[group_col] = pd.DataFrame(rows)

    # 5. Bài toán kiểm tra công thức: Linear Regression dự đoán CSS từ S_cv và S_int
    X_lin = df[["S_cv", "S_int"]].copy()
    y_lin = df["CSS"].copy()
    Xl_train, Xl_test, yl_train, yl_test = train_test_split(
        X_lin, y_lin, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    lin_reg = LinearRegression()
    lin_reg.fit(Xl_train, yl_train)
    yl_pred = lin_reg.predict(Xl_test)

    linear_check = {
        "coef_s_cv": float(lin_reg.coef_[0]),
        "coef_s_int": float(lin_reg.coef_[1]),
        "intercept": float(lin_reg.intercept_),
        "r2": float(r2_score(yl_test, yl_pred)),
        "mae": float(mean_absolute_error(yl_test, yl_pred)),
    }

    return {
        "models": {
            "baseline": baseline_clf,
            "logistic_regression": lr_pipeline,
            "random_forest": rf_pipeline,
            "linear_regression": lin_reg,
        },
        "metrics": {
            "Baseline (Lớp đa số)": metrics_base,
            "Logistic Regression (Chính)": metrics_lr,
            "Random Forest (So sánh)": metrics_rf,
        },
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": all_feature_names,
        "coef_df": coef_df,
        "raw_coef_df": raw_coef_df,
        "lr_intercept": lr_intercept,
        "interpretations": interpretations,
        "fairness_tables": fairness_tables,
        "linear_check": linear_check,
    }


def build_shap_explainer(lr_pipeline: Pipeline, X_train: pd.DataFrame) -> Dict[str, Any]:
    """
    Khởi tạo SHAP LinearExplainer cho mô hình phân loại Logistic Regression đã chọn.
    Trả về đối tượng explainer cùng tên các đặc trưng sau khi biến đổi qua ColumnTransformer.
    """
    preprocessor = lr_pipeline.named_steps["preprocessor"]
    classifier = lr_pipeline.named_steps["classifier"]

    X_train_transformed = preprocessor.transform(X_train)
    cat_names = list(
        preprocessor.named_transformers_["cat"].get_feature_names_out(RAW_CATEGORICAL_FEATURES)
    )
    feature_names = RAW_NUMERIC_FEATURES + cat_names

    X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names, index=X_train.index)
    explainer = shap.LinearExplainer(classifier, X_train_df)
    return {
        "explainer": explainer,
        "preprocessor": preprocessor,
        "classifier": classifier,
        "feature_names": feature_names,
    }


def explain_single_candidate(
    df: pd.DataFrame, candidate_id: str, shap_bundle: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Tính toán giá trị SHAP (trên thang log-odds) và xác suất dự đoán cho một ứng viên cụ thể
    dựa trên mô hình phân loại Logistic Regression đã huấn luyện.
    """
    matches = df[df["candidate_id"] == candidate_id]
    if matches.empty:
        raise ValueError(f"Không tìm thấy ứng viên có mã: {candidate_id}")

    row = matches.iloc[[0]]
    X_cand = row[RAW_NUMERIC_FEATURES + RAW_CATEGORICAL_FEATURES]

    preprocessor = shap_bundle["preprocessor"]
    classifier = shap_bundle["classifier"]
    explainer = shap_bundle["explainer"]
    feature_names = shap_bundle["feature_names"]

    X_cand_trans = pd.DataFrame(
        preprocessor.transform(X_cand),
        columns=feature_names,
        index=row.index,
    )

    shap_explanation = explainer(X_cand_trans)
    pred_prob = float(classifier.predict_proba(X_cand_trans)[0, 1])
    pred_class = int(classifier.predict(X_cand_trans)[0])

    base_val = float(np.squeeze(shap_explanation[0].base_values))
    shap_vals = np.squeeze(shap_explanation[0].values)
    trans_vals = np.squeeze(shap_explanation[0].data)

    contrib_df = pd.DataFrame({
        "Đặc trưng": feature_names,
        "Giá trị sau chuẩn hóa / One-Hot": trans_vals,
        "Đóng góp SHAP (Log-Odds)": shap_vals,
    })
    contrib_df["Mức tác động tuyệt đối"] = contrib_df["Đóng góp SHAP (Log-Odds)"].abs()
    contrib_df = (
        contrib_df.sort_values(by="Mức tác động tuyệt đối", ascending=False)
        .drop(columns=["Mức tác động tuyệt đối"])
        .reset_index(drop=True)
    )

    return {
        "candidate_info": row.iloc[0].to_dict(),
        "shap_explanation_single": shap_explanation[0],
        "pred_prob": pred_prob,
        "pred_class": pred_class,
        "base_value": base_val,
        "total_shap": float(np.sum(shap_vals)),
        "final_log_odds": float(base_val + np.sum(shap_vals)),
        "contrib_df": contrib_df,
    }