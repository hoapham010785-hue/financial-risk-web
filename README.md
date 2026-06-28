# Financial Risk Early Warning Web App

Ứng dụng Streamlit dự báo 02 target:

- Financial_Risk_Score
- Risk_Category

## Cách chạy

1. Cài thư viện:

```bash
pip install -r requirements.txt
```

2. Chạy app:

```bash
streamlit run app.py
```

## Lưu ý

- Đặt file dữ liệu tên `financial_risk_early_warning_dataset.csv` cùng thư mục với `app.py`.
- Input gồm các cột tài chính, không gồm:
  - Firm_ID
  - Financial_Risk_Score
  - Risk_Category
