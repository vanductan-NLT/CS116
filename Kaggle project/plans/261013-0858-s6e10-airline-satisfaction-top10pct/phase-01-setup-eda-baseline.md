# Phase 01 – Setup + EDA + baseline (13–15/10)

## Việc user tự làm (t không làm thay được)
1. Đăng nhập/đăng ký Kaggle, bấm **Join Competition** + đồng ý luật S6E10.
2. Xác minh số điện thoại Kaggle (bắt buộc để dùng GPU/Internet trong Notebook).
3. Tạo API token (Settings → API → Create New Token) → đặt `kaggle.json` vào `%USERPROFILE%\.kaggle\` (không đưa vào repo).

## Setup
- Local: Python 3.11 venv, `pandas numpy scikit-learn lightgbm xgboost catboost optuna matplotlib seaborn kaggle`.
- `kaggle competitions download -c playground-series-s6e10 -p data/` (data/ ignore khỏi git).
- Dataset gốc (external): "Airline Passenger Satisfaction" (~129,880 dòng) — chỉ dùng để thử nghiệm/so sánh.
- Cấu trúc:
  ```
  data/            # csv (không commit)
  src/             # module dùng chung: cv_split.py, features.py, train_gbdt.py, stacking.py
  notebooks/       # 01_eda, 02_baseline, ... (chạy trên Kaggle khi cần GPU)
  oof/ test_preds/ # npy theo tên model
  submissions/
  ```

## EDA (đưa vào báo cáo – khớp bài 3 môn học)
- Shape, kiểu cột, missing (Arrival Delay có NaN ở bản gốc), phân bố target.
- Đơn biến: histogram Age/Flight Distance/delays; bar các rating 0–5.
- Đa biến: tương quan, univariate AUC từng feature (Online boarding ~0.84 cao nhất; Gender ~0.506 vô dụng).
- So sánh train vs test vs original (drift: Flight Distance, Age).
- Facts cộng đồng đã đo: không trùng tuple train/test; 91% delay = 0; Flight Distance ~3.5k giá trị lặp (như route ID).

## Baseline (khớp nội dung môn)
- Tiền xử lý: impute median delay, one-hot/ordinal categorical, StandardScaler cho LR/KNN.
- LogisticRegression, KNN (sample con vì 700k dòng), RandomForest, LightGBM mặc định → bảng CV AUC.
- Nộp 1 bài LightGBM baseline để test format submission.

## Done khi
- Bảng CV 4 baseline + 1 sub thành công (kỳ vọng LGBM ~0.958).
