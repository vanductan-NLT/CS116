# Nháp tin nhắn gửi thầy (CS116.R11) – gửi qua email/Moodle

Thay các chỗ [ ] trước khi gửi. Giữ ngắn; thầy chỉ cần 3 thông tin: làm gì, kết quả, cần thầy xác nhận gì.

---

**Tiêu đề:** [CS116.R11] Đăng ký đề tài đồ án – Kaggle Playground S6E10 – [Họ tên, MSSV]

Kính gửi thầy Nguyễn Vinh Tiệp,

Em là Văn Đức Tân, MSSV [MSSV], lớp CS116.R11. Em xin báo cáo đề tài đồ án cuối kỳ theo Phương án 1 (cuộc thi Kaggle đang diễn ra, dữ liệu dạng bảng):

- **Cuộc thi:** Kaggle Playground Series S6E10 – *Predicting Airline Satisfaction* (phân lớp nhị phân, độ đo ROC AUC, 01–31/10/2026), link: https://www.kaggle.com/competitions/playground-series-s6e10
- **Hình thức:** em làm **cá nhân (nhóm 1 người)**. Nếu thầy yêu cầu đồ án phải làm theo nhóm, em xin thầy cho biết để em điều chỉnh.
- **Tiến độ hiện tại (13/10/2026):** điểm public leaderboard **0,96185**, hạng **93/1695 đội (top 5,5 %)**; nested cross-validation 0,9622. Pipeline gồm EDA, tiền xử lý và feature engineering (target encoding trong fold, hồ sơ tuyến bay, dữ liệu gốc làm teacher), so sánh Logistic Regression / KNN / Random Forest / LightGBM / XGBoost / CatBoost, tối ưu siêu tham số bằng Optuna và GridSearch, PCA/KMeans, và stacking nhiều họ mô hình.
- **Mã nguồn:** https://github.com/vanductan-NLT/CS116 (thư mục `Kaggle project/`). Kết quả private leaderboard sẽ có sau khi cuộc thi đóng (01/11/2026); em sẽ bổ sung vào báo cáo.

Em xin hỏi thầy 2 điều để hoàn thiện báo cáo đúng yêu cầu:
1. Thời hạn nộp báo cáo viết và lịch trình bày đồ án (tuần 14–15) chính xác là ngày nào ạ?
2. Báo cáo có mẫu/định dạng bắt buộc không, và thầy có cần em nộp kèm mã nguồn (file .ipynb/.py) ngoài link GitHub không ạ?

Em cảm ơn thầy.

[Họ tên] – [MSSV] – [email @gm.uit.edu.vn]

---

## Ghi chú khi gửi
- Điền hạng/điểm từ bảng public lúc gửi (lệnh: `kaggle competitions leaderboard -c playground-series-s6e10 --show` hoặc xem trên web).
- Nếu gửi sau 01/11: thay dòng tiến độ bằng kết quả private (hạng cuối cùng).
- Gửi qua email `tiepnv@uit.edu.vn` (trong đề cương) hoặc kênh thầy quy định trên lớp.
