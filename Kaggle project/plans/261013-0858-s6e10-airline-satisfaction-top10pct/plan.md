# Plan: Kaggle PS S6E10 – Predicting Airline Satisfaction (đồ án CS116, solo)

- Status: Draft – chờ user duyệt
- Mục tiêu: **top 10% private LB** (cố top 1%); nộp báo cáo trong 10/2026
- Cuộc thi: https://www.kaggle.com/competitions/playground-series-s6e10 — hạn **31/10/2026 23:59 UTC** (= 06:59 1/11 giờ VN)
- Metric: ROC AUC; 10 sub/ngày; chọn 2 bài final; được dùng external data public
- Compute: free — Kaggle Notebook (GPU T4/P100 ~30h/tuần, phiên 12h) + máy local cho việc nhẹ
- Research: [research-261013-0845-cs116-kaggle-competition-selection-report.md](../reports/research-261013-0845-cs116-kaggle-competition-selection-report.md)

## Mốc điểm (public LB 13/10, 1661 đội)
- #1 0.96243 · #10 0.96188 · #13→~#200 đồng 0.96187 (stack notebook public)
- 1 LightGBM tốt ≈ 0.9605–0.9607 · baseline LGBM thô ≈ 0.9578
- → Top 10% (~#170 nay, ~#300 lúc đóng) cần ≈ **≥0.9618 public** + CV ổn định để không rớt khi private shake-up

## Phases
| # | Phase | Ngày | File |
|---|---|---|---|
| 1 | Setup + EDA + baseline môn học | 13–15/10 | [phase-01](phase-01-setup-eda-baseline.md) |
| 2 | Feature engineering + GBDT mạnh | 16–21/10 | [phase-02](phase-02-feature-engineering-gbdt.md) |
| 3 | Model đa dạng + stacking + chọn final | 22–29/10 | [phase-03](phase-03-diverse-models-stacking-final.md) |
| 4 | Báo cáo viết + slide | 26–31/10 (song song) | [phase-04](phase-04-report-slides.md) |

## Nguyên tắc
- 1 split cố định cho mọi model: `StratifiedKFold(5, shuffle=True, random_state=42)` (trùng split cộng đồng → stack được OOF public). Thêm bản 10-fold cho model mạnh.
- Chỉ tin **CV/nested CV**, không tune theo public LB (public chỉ 20% test, chênh 1e-4 = nhiễu).
- Lưu OOF + test pred mỗi model (`oof/<model>.npy`, `test/<model>.npy`) → stacking.
- Phần tự làm là lõi báo cáo; OOF/notebook public nếu dùng thì ghi nguồn rõ.

## Acceptance
- [ ] Pipeline riêng đạt CV ≥ 0.9610 (single/ensemble tự làm)
- [ ] Bài final đạt public ≥ 0.9618, nested CV ≥ 0.9619
- [ ] Báo cáo phủ đủ: EDA, tiền xử lý, FE, LR/KNN/RF/XGBoost, clustering/PCA, tối ưu siêu tham số, ensemble, đánh giá
- [ ] Chọn 2 final sub trước 31/10

## Câu hỏi mở
- Ngày nộp/báo cáo chính xác? Có cần nộp code (repo/notebook) không?
- Báo cáo viết tiếng Việt hay Anh? Có template của khoa không?
