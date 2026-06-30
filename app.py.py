import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
)
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn.metrics import (
    accuracy_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Financial Risk Early Warning",
    page_icon="📊",
    layout="wide"
)

DATA_PATH = "financial_risk_early_warning_dataset.csv"
TARGET_SCORE = "Financial_Risk_Score"
TARGET_CATEGORY = "Risk_Category"
DROP_COLS = ["Firm_ID", TARGET_SCORE, TARGET_CATEGORY]

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)

    st.warning("Không tìm thấy file dữ liệu mặc định. Vui lòng upload file CSV.")
    uploaded_file = st.file_uploader(
        "Upload file financial_risk_early_warning_dataset.csv",
        type=["csv"]
    )

    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)

    st.stop()

# =========================
# PREPROCESSING
# =========================
def build_preprocess(X):
    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    num_cols = X.select_dtypes(exclude=["object", "category"]).columns

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocess = ColumnTransformer([
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", encoder)
        ]), cat_cols),

        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols)
    ])

    return preprocess


def build_category_rule(data):
    """Create a stable category rule from the training data.

    The original app predicted Risk_Category using a separate classifier. On small or
    imbalanced data, that classifier can collapse to one class, so the displayed
    category looks unchanged even when the score changes. This rule derives category
    from the predicted score using the average score of each category in the dataset.
    """
    grouped = (
        data[[TARGET_SCORE, TARGET_CATEGORY]]
        .dropna()
        .groupby(TARGET_CATEGORY)[TARGET_SCORE]
        .mean()
        .sort_values()
    )

    labels = grouped.index.astype(str).tolist()
    means = grouped.values.astype(float)

    if len(labels) <= 1:
        return labels, []

    cutoffs = [(means[i] + means[i + 1]) / 2 for i in range(len(means) - 1)]
    return labels, cutoffs


def category_from_score(score, labels, cutoffs):
    if not labels:
        return "Unknown"
    idx = int(np.searchsorted(cutoffs, float(score), side="right"))
    idx = max(0, min(idx, len(labels) - 1))
    return labels[idx]

# =========================
# TRAIN + COMPARE MODELS
# =========================
@st.cache_resource
def train_models(data):
    X = data.drop(columns=DROP_COLS)
    y_score = data[TARGET_SCORE]
    y_category = data[TARGET_CATEGORY]

    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X, y_score, test_size=0.2, random_state=42
    )

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_category, test_size=0.2, random_state=42, stratify=y_category
    )

    reg_models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(alpha=0.0001, max_iter=5000),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, min_samples_leaf=30, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42, n_jobs=-1),
        "Extra Trees": ExtraTreesRegressor(n_estimators=100, min_samples_leaf=5, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "KNN": KNeighborsRegressor(n_neighbors=7)
    }

    reg_results = []
    best_reg_model = None
    best_reg_name = None
    best_r2 = -999

    for name, algorithm in reg_models.items():
        pipe = Pipeline([
            ("preprocess", build_preprocess(X)),
            ("model", algorithm)
        ])
        pipe.fit(X_train_s, y_train_s)
        pred = pipe.predict(X_test_s)

        mae = mean_absolute_error(y_test_s, pred)
        rmse = np.sqrt(mean_squared_error(y_test_s, pred))
        r2 = r2_score(y_test_s, pred)

        reg_results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})

        if r2 > best_r2:
            best_r2 = r2
            best_reg_name = name
            best_reg_model = pipe

    cls_models = {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=30, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1),
        "Linear SVM": LinearSVC(class_weight="balanced", random_state=42, max_iter=3000),
        "KNN": KNeighborsClassifier(n_neighbors=7)
    }

    cls_results = []
    best_cls_model = None
    best_cls_name = None
    best_f1 = -1
    best_cls_test_pred = None

    for name, algorithm in cls_models.items():
        pipe = Pipeline([
            ("preprocess", build_preprocess(X)),
            ("model", algorithm)
        ])
        pipe.fit(X_train_c, y_train_c)
        pred = pipe.predict(X_test_c)

        acc = accuracy_score(y_test_c, pred)
        f1 = f1_score(y_test_c, pred, average="macro")

        cls_results.append({"Model": name, "Accuracy": acc, "F1_macro": f1})

        if f1 > best_f1:
            best_f1 = f1
            best_cls_name = name
            best_cls_model = pipe
            best_cls_test_pred = pred

    category_labels, category_cutoffs = build_category_rule(data)

    return {
        "X_columns": X.columns.tolist(),
        "reg_results": pd.DataFrame(reg_results).sort_values("R2", ascending=False),
        "cls_results": pd.DataFrame(cls_results).sort_values("F1_macro", ascending=False),
        "best_reg_name": best_reg_name,
        "best_reg_model": best_reg_model,
        "best_cls_name": best_cls_name,
        "best_cls_model": best_cls_model,
        "category_labels": category_labels,
        "category_cutoffs": category_cutoffs,
        "category_distribution": y_category.value_counts().rename_axis("Risk_Category").reset_index(name="Count"),
        "best_cls_pred_distribution": pd.Series(best_cls_test_pred).value_counts().rename_axis("Predicted_Category").reset_index(name="Count")
    }

# =========================
# APP
# =========================
df = load_data()
trained = train_models(df)

st.sidebar.title("📌 Financial Risk App")
st.sidebar.write("Ứng dụng dự báo 02 target:")
st.sidebar.markdown("""
- **Financial Risk Score**
- **Risk Category**
""")
st.sidebar.info(
    "Input là các biến tài chính trong bảng financial-risk-early-warning. "
    "Không nhập Firm_ID, Financial_Risk_Score và Risk_Category vì đây không phải biến đầu vào."
)

st.title("📊 Financial Risk Early Warning System")
st.write(
    "Ứng dụng ML dự báo **Financial Risk Score** và suy ra **Risk Category** "
    "từ các chỉ tiêu tài chính doanh nghiệp."
)

tab1, tab2, tab3 = st.tabs([
    "🔮 Dự báo thủ công",
    "📁 Dự báo bằng file CSV",
    "📈 So sánh mô hình"
])

# =========================
# TAB 1: MANUAL PREDICTION
# =========================
with tab1:
    st.subheader("Nhập dữ liệu đầu vào")

    X_template = df[trained["X_columns"]]
    user_input = {}

    col1, col2, col3 = st.columns(3)

    for i, col in enumerate(trained["X_columns"]):
        target_col = [col1, col2, col3][i % 3]
        s = X_template[col]

        if s.dtype == "object" or str(s.dtype) == "category":
            options = sorted(s.dropna().astype(str).unique().tolist())
            user_input[col] = target_col.selectbox(col, options, key=f"manual_{col}")
        else:
            s_num = pd.to_numeric(s, errors="coerce").dropna()

            if s_num.empty:
                user_input[col] = target_col.number_input(col, value=0.0, key=f"manual_{col}")
            else:
                min_val = float(s_num.min())
                max_val = float(s_num.max())
                mean_val = float(s_num.mean())
                step_val = float((max_val - min_val) / 100) if max_val > min_val else 1.0

                user_input[col] = target_col.number_input(
                    col,
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    step=step_val,
                    format="%.6f",
                    key=f"manual_{col}"
                )

    input_df = pd.DataFrame([user_input])

    if st.button("Dự báo rủi ro tài chính", type="primary"):
        score_pred = float(trained["best_reg_model"].predict(input_df)[0])
        category_by_score = category_from_score(
            score_pred,
            trained["category_labels"],
            trained["category_cutoffs"]
        )
        category_by_classifier = trained["best_cls_model"].predict(input_df)[0]

        st.success("Dự báo hoàn tất")

        m1, m2, m3 = st.columns(3)
        m1.metric("Financial Risk Score", f"{score_pred:.4f}")
        m2.metric("Risk Category", str(category_by_score))
        m3.metric("Classifier check", str(category_by_classifier))

        st.caption(
            "Risk Category chính được suy ra từ Financial Risk Score để tránh lỗi classifier "
            "dự báo lặp một nhãn khi dữ liệu bị lệch lớp."
        )

        st.write("Dữ liệu đầu vào:")
        st.dataframe(input_df, use_container_width=True)

# =========================
# TAB 2: BATCH PREDICTION
# =========================
with tab2:
    st.subheader("Upload file CSV để dự báo hàng loạt")

    uploaded = st.file_uploader(
        "Upload CSV có các cột đầu vào giống bảng financial-risk-early-warning",
        type=["csv"],
        key="batch_upload"
    )

    if uploaded is not None:
        new_df = pd.read_csv(uploaded)
        missing_cols = [c for c in trained["X_columns"] if c not in new_df.columns]

        if missing_cols:
            st.error(f"File thiếu các cột: {missing_cols}")
        else:
            X_new = new_df[trained["X_columns"]].copy()
            score_preds = trained["best_reg_model"].predict(X_new)

            new_df["Predicted_Financial_Risk_Score"] = score_preds
            new_df["Predicted_Risk_Category"] = [
                category_from_score(score, trained["category_labels"], trained["category_cutoffs"])
                for score in score_preds
            ]
            new_df["Classifier_Risk_Category_Check"] = trained["best_cls_model"].predict(X_new)

            st.success("Dự báo hàng loạt hoàn tất")
            st.dataframe(new_df, use_container_width=True)

            csv = new_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Tải kết quả dự báo CSV",
                data=csv,
                file_name="financial_risk_predictions.csv",
                mime="text/csv"
            )

# =========================
# TAB 3: MODEL COMPARISON
# =========================
with tab3:
    st.subheader("Mô hình tốt nhất")

    c1, c2 = st.columns(2)
    c1.metric("Best model - Financial Risk Score", trained["best_reg_name"])
    c2.metric("Best classifier - Risk Category", trained["best_cls_name"])

    st.markdown("### Bảng so sánh ML cho Financial_Risk_Score")
    st.dataframe(trained["reg_results"], use_container_width=True)
    st.bar_chart(trained["reg_results"].set_index("Model")["R2"])

    st.markdown("### Bảng so sánh classifier cho Risk_Category")
    st.dataframe(trained["cls_results"], use_container_width=True)
    st.bar_chart(trained["cls_results"].set_index("Model")["F1_macro"])

    st.markdown("### Kiểm tra phân phối nhãn")
    a, b = st.columns(2)
    with a:
        st.write("Phân phối nhãn thật trong dataset")
        st.dataframe(trained["category_distribution"], use_container_width=True)
    with b:
        st.write("Phân phối nhãn classifier dự báo trên test set")
        st.dataframe(trained["best_cls_pred_distribution"], use_container_width=True)

    st.markdown("### Rule suy ra Risk_Category từ Financial_Risk_Score")
    st.write("Thứ tự category theo điểm rủi ro trung bình tăng dần:", trained["category_labels"])
    st.write("Ngưỡng cắt score:", [round(x, 4) for x in trained["category_cutoffs"]])

# =========================
# SAVE MODELS OPTIONALLY
# =========================
joblib.dump(trained["best_reg_model"], "best_financial_risk_score_model.pkl")
joblib.dump(trained["best_cls_model"], "best_risk_category_model.pkl")
