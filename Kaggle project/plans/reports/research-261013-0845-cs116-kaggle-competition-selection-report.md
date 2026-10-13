# CS116.R11 – Chọn cuộc thi Kaggle cho đồ án (13/10/2026)

## 1. Yêu cầu môn học (nguồn: courses.uit.edu.vn id=1081, đề cương PDF, slide B1–B4, sheet nhóm)
- Môn: CS116 Lập trình Python cho máy học, GV TS. Nguyễn Vinh Tiệp. 4 TC.
- Điểm: A1 quá trình 30% | A3 đồ án 30% | A4 cuối kỳ (tự luận) 40%.
- Đồ án (30%) = **60% điểm thí nghiệm + 40% điểm báo cáo viết**. Seminar báo cáo tuần 14–15.
- Hướng dẫn trong sheet nhóm:
  - PA1: cuộc thi Kaggle **đang diễn ra** (khuyến khích dạng bảng).
  - PA2: bộ dữ liệu từ công bố khoa học, hoặc cuộc thi khác (VOAI, AI Challenge…).
- "Top 10 = đạt" KHÔNG có trong tài liệu online (đề cương, slide, sheet, thông báo) → chắc GV nói miệng. Cần xác nhận: top 10 tuyệt đối hay top 10%? public hay private LB?
- Nội dung môn cần thể hiện trong báo cáo: EDA (đơn/đa biến), tiền xử lý (missing, outlier, encode, scale), feature extraction/transform/selection, clustering & PCA, LR/KNN/RF, tối ưu siêu tham số (GridSearch…), ensemble (bagging/boosting/XGBoost), đánh giá mô hình (CV, metric).

## 2. Cuộc thi Kaggle đang mở (13/10/2026)
| Cuộc thi | Loại | Đội | Hạn | Đánh giá |
|---|---|---|---|---|
| **Playground S6E10 – Predicting Airline Satisfaction** | Playground, tabular, ROC AUC | 1661 | 31/10/2026 | ✅ Hợp nhất hiện tại |
| Playground S6E11 (dự kiến) | tabular | – | ~1/11 → 30/11/2026 | ✅ Hợp nhất về thời gian |
| Titanic / Spaceship Titanic / House Prices / Store Sales | Getting Started, rolling LB | 550–10k | Không hạn | ❌ Top LB đầy bài leak/gian lận (score hoàn hảo) → top 10 bất khả thi hợp lệ; GV cũng khó coi là "đang diễn ra" |
| ARC, Gemma, RSNA, Enveda, NFL BDB, Kaggriculture, Pokémon | Featured/Research | 200–10k | – | ❌ Không phải tabular, cần GPU lớn, đối thủ chuyên nghiệp |
| Community (lớp học các trường khác) | Community | 0–25 | đa số sắp hết / invite-only | ⚠️ Top 10 dễ nhưng rủi ro GV không công nhận, đề bài nhỏ/kém |

### S6E10 chi tiết
- Mục tiêu: dự đoán xác suất `satisfaction`; metric ROC AUC; submit CSV.
- 10 sub/ngày, chọn 2 bài final; **team tối đa 3 người**; cho dùng external data public (dataset gốc "Airline Passenger Satisfaction" ~130k dòng trên Kaggle).
- Public LB: #1 = 0.96243, #10 = 0.96188, ~#13–#50+ đồng điểm 0.96187 (copy notebook public). Chênh top1–top10 chỉ 0.0006 → private LB (80% test) sẽ xáo trộn mạnh.

## 3. Khuyến nghị
1. **Tham gia S6E10 ngay** (còn 19 ngày): dựng pipeline đầy đủ (EDA → FE → LGBM/XGB/CatBoost + LR/KNN/RF baseline → Optuna/GridSearch → hill-climbing/stacking ensemble, dùng thêm dataset gốc).
2. **Tái dùng pipeline cho S6E11** (tháng 11, trọn 1 tháng, kết thúc 30/11 trước tuần 14–15) → 2 cơ hội lấy top 10; báo cáo cuộc thi có thứ hạng tốt hơn.
3. Không chọn Getting Started / Featured.

## 4. Đánh giá thực tế về "top 10"
- Top 10 tuyệt đối trong Playground ~1500–3500 đội: KHÓ, cạnh tranh với Kaggle Master/GM dùng ensemble hàng chục–trăm model; có yếu tố may khi private shake-up. Khả thi nếu đầu tư nghiêm túc + ensemble mạnh + CV đáng tin.
- Top 10% (~top 160): rất khả thi.
- Phần điểm báo cáo (40%) phụ thuộc thể hiện kiến thức môn, không chỉ thứ hạng.

## Câu hỏi chưa rõ
1. "Top 10" = top 10 tuyệt đối hay top 10%? Public hay private LB? (Có thể là top 10 trong lớp?)
2. Nhóm mấy người? (Kaggle giới hạn team 3 người; nhóm 4–5 người phải tách đội/chỉ 3 người join team.)
3. Hạn nộp báo cáo/seminar chính xác (tuần 14–15 ≈ giữa–cuối 12/2026?)
4. Máy/GPU có sẵn? (Kaggle Notebook free GPU 30h/tuần là đủ cho tabular.)
