# Phase 02 – Feature engineering + GBDT mạnh (16–21/10)

## Feature (thêm từng nhóm, đo CV từng bước = "feature ladder" cho báo cáo)
1. **Flight Distance như route ID**: frequency count + target encoding *trong fold* (cộng đồng: +0.0017 AUC, 0.9588→0.9605).
2. Route profile: thống kê các rating/delay theo Flight Distance (mean/std, in-fold).
3. Delay: `log1p`, hiệu arrival−departure, cờ delay>0.
4. Rating: tổng/mean/min/max các service rating, số rating =0 (đo – cộng đồng báo ít tác dụng).
5. Crosses categorical cho CatBoost (Class×Type of Travel×Customer Type…; tham khảo "39 Simple Crosses That Help CatBoost").
6. Feature môn học: KMeans cluster id + PCA components trên ratings (bài 5–6) — giữ nếu CV không giảm.
- Bỏ/so sánh: thêm dòng original data (cộng đồng: thường **giảm** CV) → thử kiểu "original-data teacher" (train trên original, dùng pred làm feature).

## Models (split cố định 5-fold seed 42; bản tốt nhất chạy thêm 10-fold)
- LightGBM, XGBoost, CatBoost (GPU trên Kaggle).
- Tuning: Optuna (TPE) ~50–100 trial/model trên 1–2 fold, rồi full CV; so sánh GridSearch nhỏ để minh hoạ bài 11–12.
- Lưu OOF/test pred mọi model.

## Done khi
- ≥3 GBDT, mỗi cái CV ≥ 0.9605; blend đơn giản CV ≥ 0.9610.

## Rủi ro
- Leak target encoding nếu không làm trong fold → luôn fit TE trên train-fold.
- Overfit Optuna theo 1 fold → xác nhận lại bằng full CV.
