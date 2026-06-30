import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Financial Risk Early Warning",
    page_icon="📊",
    layout="centered"
)

DATA_FILE = "financial_risk_early_warning_dataset.csv"

@st.cache_data
def load_data():
    file_path = Path(DATA_FILE)

    if not file_path.exists():
        st.error(f"Không tìm thấy file dữ liệu: {DATA_FILE}")
        st.stop()

    df = pd.read_csv(file_path)

    required_columns = [
        "Firm_ID",
        "Fiscal_Year",
        "Financial_Risk_Score",
        "Risk_Category"
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error("File CSV thiếu cột: " + ", ".join(missing_columns))
        st.stop()

    df["Firm_ID_Last4"] = (
        df["Firm_ID"]
        .astype(str)
        .str.extract(r"(\d+)")[0]
        .fillna("")
        .str.zfill(4)
        .str[-4:]
    )

    df["Fiscal_Year"] = df["Fiscal_Year"].astype(int)

    return df


df = load_data()

st.title("Financial Risk Early Warning")
st.write("Chọn **Year** và **4 số cuối Firm ID** để lấy kết quả tương ứng theo từng dòng dữ liệu.")

col1, col2 = st.columns(2)

with col1:
    selected_year = st.selectbox(
        "Year",
        sorted(df["Fiscal_Year"].unique())
    )

with col2:
    firm_options = sorted(
        df.loc[df["Fiscal_Year"] == selected_year, "Firm_ID_Last4"].unique()
    )
    selected_firm_last4 = st.selectbox(
        "4 số cuối Firm ID",
        firm_options
    )

if st.button("Predict", type="primary"):
    result = df[
        (df["Fiscal_Year"] == selected_year) &
        (df["Firm_ID_Last4"] == selected_firm_last4)
    ]

    if result.empty:
        st.error("Không tìm thấy dòng dữ liệu phù hợp.")
    else:
        st.subheader("Kết quả")

        for _, row in result.iterrows():
            st.metric(
                "Financial Risk Score",
                round(float(row["Financial_Risk_Score"]), 4)
            )
            st.success(f"Financial Category: {row['Risk_Category']}")

            st.dataframe(
                pd.DataFrame([{
                    "Firm_ID": row["Firm_ID"],
                    "Year": row["Fiscal_Year"],
                    "4 số cuối Firm ID": row["Firm_ID_Last4"],
                    "Financial Risk Score": round(float(row["Financial_Risk_Score"]), 4),
                    "Financial Category": row["Risk_Category"]
                }]),
                use_container_width=True,
                hide_index=True
            )
