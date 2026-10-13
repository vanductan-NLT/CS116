# Báo cáo đồ án CS116 – Dự đoán mức độ hài lòng hành khách hàng không (Kaggle Playground S6E10)

- Môn: CS116.R11 – Lập trình Python cho máy học, GV: TS. Nguyễn Vinh Tiệp
- Sinh viên: [Họ tên – MSSV]
- Cuộc thi: Kaggle Playground Series S6E10 "Predicting Airline Satisfaction" (01–31/10/2026), https://www.kaggle.com/competitions/playground-series-s6e10
- Mã nguồn: https://github.com/vanductan-NLT/CS116 (thư mục `Kaggle project/`)

## 1. Giới thiệu

**Bài toán.** Phân lớp nhị phân: dự đoán xác suất hành khách hài lòng (`satisfaction`) từ 21 thuộc tính (4 phân loại, 2 số liên tục, 2 độ trễ, 13 điểm đánh giá dịch vụ 0–5). Dữ liệu tổng hợp (synthetic) được sinh từ bộ gốc *Airline Passenger Satisfaction* (129 880 dòng, công khai trên Kaggle).

| | train | test | original |
|---|---|---|---|
| Số dòng | 699 635 | 299 844 | 129 880 |
| Tỉ lệ hài lòng | 44,36 % | – | 43,45 % |

**Độ đo.** ROC AUC trên xác suất dự đoán. Bảng public tính trên 20 % test, bảng private (kết quả cuối) trên 80 % còn lại. Mỗi đội tối đa 10 lần nộp/ngày, chọn 2 bài cuối.

**Kết quả tóm tắt.** [cập nhật cuối cuộc thi] Nested CV tốt nhất 0,96222; public LB 0,96185 (hạng ≈ 88/1674, top 5 % tại 13/10/2026); private LB: …

## 2. Quy trình (ML pipeline)

```
raw csv ─► kiểm tra dữ liệu / EDA ─► tiền xử lý & feature engineering (trong fold) ─► huấn luyện K-fold
        ─► OOF + test preds mỗi model ─► đánh giá (AUC, nested CV) ─► ensemble (stacking) ─► submission
```

Nguyên tắc xuyên suốt:
1. **Một phép chia cố định** `StratifiedKFold(5, shuffle=True, random_state=42)` cho mọi mô hình → so sánh công bằng và ghép (stack) được với OOF của cộng đồng dùng cùng split.
2. Mọi phép biến đổi dùng nhãn (target encoding, teacher) chỉ được fit **trong fold huấn luyện** → không rò rỉ.
3. **Chỉ chọn mô hình theo CV**, không theo bảng public (chỉ 20 % test, nhiễu ≈ 1e-4).
4. Mỗi mô hình lưu dự đoán out-of-fold (OOF) và test → tái dùng cho stacking.

Mã nguồn: `src/config.py` (split, cột), `data_loading.py`, `feature_engineering.py`, `original_data_features.py`, `aux_task_features.py`, `model_zoo.py`, `run_cv.py` (chạy K-fold 1 cấu hình), `stack_oof.py`/`stack_meta_variants.py` (tầng 2), `import_public_pool.py`, `make_eda_figures.py`, `tune_hyperparams.py`, `probe_generator_structure.py`.

## 3. Phân tích dữ liệu (EDA)

Hình trong `docs/figures/`.

**Chất lượng dữ liệu.** Chỉ `Arrival Delay in Minutes` thiếu (292 dòng train, 130 test, 0,04 %) → điền bằng `Departure Delay`. Không có dòng trùng toàn bộ; không có tuple test xuất hiện y hệt trong train (theo 21 cột). Nhãn cân bằng vừa phải (44/56) nên không cần cân bằng lớp.

**Đơn biến** (hình 02, 03). `Age` 7–85, phân bố gần chuẩn; `Flight Distance` lệch phải (31–4983); 91 % chuyến có độ trễ 0 → hai cột trễ gần như nhị phân, dùng `log1p`. Các điểm đánh giá tập trung ở 2–4; giá trị 0 ("không áp dụng") hiếm (< 3 %).

**Đa biến / liên hệ với nhãn** (hình 03, 04, 05).
- AUC đơn biến: `Online boarding` 0,840, `Class` 0,782, `Inflight entertainment` 0,745, `Seat comfort` 0,732, `Type of Travel` 0,703, `Flight Distance` 0,696 … thấp nhất `Gender` 0,506, `Departure Delay` 0,509. Với n = 700k mọi kiểm định χ² đều p ≈ 0 nên cỡ hiệu ứng (AUC) là thước đo có nghĩa.
- Quan hệ rating–nhãn phần lớn đơn điệu tăng; riêng `Inflight wifi`, `Ease of Online booking`, `Gate location` có dạng chữ U (điểm 0 và 5 đều hài lòng cao) → mô hình tuyến tính kém, mô hình cây mạnh.
- Tương quan Spearman cao giữa nhóm rating "online" (wifi, booking, time convenient, gate: 0,5–0,7) và nhóm "trên máy bay" (seat, entertainment, cleanliness, food: 0,6–0,7); hai cột trễ r = 0,84 (Pearson) nhưng Spearman 0,47 do gần hết là 0.

**Phát hiện quan trọng: `Flight Distance` hoạt động như mã tuyến bay** (hình 06). Chỉ có 3 521 giá trị khác nhau cho 1 triệu dòng; tỉ lệ hài lòng theo từng giá trị khoảng cách phân tách rõ thành 2–3 cụm (≈ 0,3 dưới 1300 dặm, ≈ 0,65 ở 1500–2700, ≈ 0,9 trên 3000) và biến thiên mạnh giữa các giá trị kề nhau → mã hoá theo giá trị (count, target encoding, thống kê theo "tuyến") hiệu quả hơn dùng như số liên tục.

**Drift train/test/original** (hình 02). Train và test cùng phân bố (synthetic cùng generator). So với bộ gốc, `Flight Distance` và `Age` dự đoán nhãn kém hơn ở bộ gốc (AUC 0,696 → 0,649; 0,624 → 0,584) → không nên nối thêm dòng gốc vào train, mà dùng bộ gốc làm "teacher"/lookup.

**Kiểm tra rò rỉ** (`probe_generator_structure.py`): khớp 17 cột rời rạc với bộ gốc (27 % dòng) chỉ cho AUC 0,90 – thấp hơn mô hình thường; `id` không mang thông tin (AUC 0,500, tỉ lệ nhãn theo khối id dao động đúng mức nhị thức). Kết luận: không có rò rỉ, mọi điểm số phải đến từ mô hình.

## 4. Tiền xử lý và feature engineering

| Nhóm | Mô tả | Lý do |
|---|---|---|
| Cơ bản | mã hoá 4 cột phân loại thành số nguyên; điền `Arrival Delay` thiếu | mô hình cây không cần one-hot |
| `delay` | `log1p(dep+arr)`, hiệu `arr−dep`, cờ `has_delay` | 91 % bằng 0, phân bố lệch nặng |
| `fd_freq` | số dòng có cùng `Flight Distance` (train+test) | tuyến bay phổ biến vs hiếm |
| TE(`Flight Distance`) | target encoding làm mượt (Bayesian, k = 20), fit **trong fold** với inner 5-fold cho train | tránh rò rỉ nhãn |
| `route_full` | trung bình mọi cột khác theo từng giá trị `Flight Distance` + phần dư (giá trị − trung bình tuyến) + count | "hồ sơ tuyến bay" không dùng nhãn |
| `original` | (i) *teacher*: LightGBM huấn luyện chỉ trên bộ gốc, dự đoán cho train/test (AUC 0,955 trên train); (ii) lookup: tỉ lệ hài lòng và count theo giá trị của 6 cột trong bộ gốc | tri thức ngoài, không dùng nhãn cuộc thi |
| `aux` | 13 bài toán phụ: dự đoán từng rating từ các cột còn lại (XGBoost, 5-fold, không dùng nhãn) → kỳ vọng và phần dư | đo mức "bất ngờ" của mỗi rating so với hồ sơ hành khách |
| `cluster_pca` | PCA(4) + KMeans(12) trên 13 rating đã chuẩn hoá | góc nhìn không giám sát (bài 5–6) |
| Thử nhưng bỏ | thống kê rating (mean/min/std), cross 4 cột phân loại, TE tuple 17 cột | không tăng CV (bảng dưới) |

**Feature ladder** (LightGBM lr 0,03, fold 0 của split chung; mỗi dòng thêm vào dòng trước):

| Bước | AUC fold 0 | Δ |
|---|---|---|
| 21 cột thô | 0,95903 | |
| + delay | 0,95914 | +0,0001 |
| + fd_freq | 0,96004 | +0,0009 |
| + TE(Flight Distance) | 0,96074 | +0,0007 |
| + rating stats / cat cross | 0,96053 / 0,96060 | − (bỏ) |
| + route_full (thay rating stats) | 0,96039 | − riêng lẻ |
| + original (teacher + lookup) | 0,96087 | +0,0001 |
| route_full + original | 0,96101 | +0,0003 (bổ trợ nhau) |
| tham số chậm hơn (lr 0,01, 127 lá, colsample 0,3) | 0,96119 | +0,0002 |
| + aux (XGBoost: 0,96107 → 0,96110; LightGBM 5-fold: 0,96100 → 0,96114) | | +0,0001 |
| + cluster_pca (XGBoost fold 0: 0,96110 → 0,96123) | | +0,0001 |
| + TE tuple 17 cột (XGBoost fold 0) | 0,96113 vs 0,96110 | trong nhiễu |

Nhận xét: gần như toàn bộ cải thiện so với dữ liệu thô (+0,002 AUC) đến từ việc coi `Flight Distance` như mã tuyến (count + TE + hồ sơ tuyến). Các feature "thông minh" hơn chỉ cộng thêm 1e-4.

## 5. Mô hình

### 5.1 Baseline các thuật toán trong đề cương (21 cột thô, 5-fold)

| Mô hình | Tiền xử lý | CV AUC |
|---|---|---|
| Logistic Regression (C=1) | StandardScaler + one-hot | 0,9216 |
| KNN (k=101, 100k mẫu train/fold) | StandardScaler + one-hot | 0,9476 |
| Random Forest (100 cây, depth ≤ 20, leaf ≥ 10) | không | 0,9569 |
| LightGBM mặc định (lr 0,1, 500 cây, 31 lá) | không | 0,9584 |

LR kém vì quan hệ chữ U và tương tác giữa các rating; KNN khá nhưng chậm (700k dòng); mô hình cây gradient boosting vượt trội → hướng chính.

### 5.2 Gradient boosting với feature đầy đủ (5-fold, split chung)

| Tên | Mô hình | Feature | CV AUC | Ghi chú |
|---|---|---|---|---|
| lgbm_v1 | LightGBM lr 0,01, 127 lá | route_full + original + TE | 0,96100 | public LB 0,96057 |
| lgbm_v2_aux | + aux | 107 cột | 0,96114 | |
| xgb_v1 | XGBoost depth 7, lr 0,02 (GPU) | như trên | 0,96103 | 3 phút/5 fold |
| xgb_v1_k10 | như trên, 10 fold | | 0,96121 | +0,0002 nhờ nhiều dữ liệu train hơn |
| cat_v1 | CatBoost depth 8, FD và FD×Class là categorical | | 0,96088 | |
| cat_v1_k10 | 10 fold | | 0,96108 | |
| xgb_v2_tuple_k10 | + TE tuple 17 cột, 10 fold | 112 cột | 0,96119 | ≈ xgb_v1_k10 |
| xgb_v3_cluster_k10 | + PCA/KMeans, 10 fold | 113 cột | [cập nhật] | fold 0 (5-fold): 0,96123 vs 0,96110 |
| xgb_v1_pl / _plh | pseudo-label test (soft / hard, an toàn theo fold) | | 0,96105 / 0,96105 | +2e-5: trong nhiễu |
| xgb_v1_sd03 | self-distillation (nhãn mềm 0,7·y + 0,3·teacher nội bộ) | | [cập nhật] | fold 0: +4e-5 |

### 5.3 Tối ưu siêu tham số (bài 11–12)

Cùng bộ feature cuối, XGBoost (GPU), fold 0 của split chung, early stopping 200 vòng theo AUC (`tune_hyperparams.py`; chi tiết `logs/optuna_trials.csv`, `logs/grid_trials.csv`).

| Phương pháp | Không gian | Số cấu hình | Thời gian | AUC fold 0 tốt nhất |
|---|---|---|---|---|
| Cấu hình tay (tham khảo) | depth 7, lr 0,02, colsample 0,4 | 1 | – | 0,96110 |
| GridSearch | depth {5,7,9} × lr {0,02; 0,05} × colsample {0,3; 0,5} | 12 | 187 s | 0,96113 (depth 7, lr 0,02, col 0,3) |
| Optuna TPE | 8 tham số liên tục (lr, depth, min_child_weight, subsample, colsample, λ, α, γ) | 30 | 560 s | **0,96130** (depth 9, lr 0,024, mcw 29, subsample 0,94, col 0,55, λ 0,74, α 0,18) |
| Optuna best, kiểm chứng 5-fold | | | | [cập nhật] |

Nhận xét: (1) lr cao (0,05) luôn kém lr thấp dù early stopping – tối ưu theo bước nhỏ quan trọng hơn số cây; (2) GridSearch tốn 12 lần huấn luyện chỉ để xác nhận cấu hình tay vì lưới thô; TPE khai thác vùng tốt (min_child_weight lớn + cây sâu) mà lưới không có, cho +0,0002 với số lần huấn luyện gấp 2,5; (3) biên độ toàn bộ không gian chỉ ±0,0005 – siêu tham số không phải nơi tạo khác biệt lớn ở bài này, feature engineering và ensemble mới là.

### 5.4 Học không giám sát (bài 5–6)

PCA 4 thành phần trên 13 rating giải thích … % phương sai; KMeans 12 cụm tách rõ nhóm "hài lòng cao mọi mặt" / "online kém" / "trên máy bay kém". Dùng làm feature: [cập nhật] – mô hình cây đã tự học được các tương tác này nên lợi ích nhỏ, nhưng cụm giúp diễn giải (xem hình).

## 6. Ensemble (bài 13)

### 6.1 Stacking các mô hình tự huấn luyện

Tầng 2: Logistic Regression trên logit của OOF (L2, C nhỏ), đánh giá **nested CV** (meta-fold seed khác seed thành viên để không "nhìn trộm").

| Thành phần | AUC |
|---|---|
| Trung bình 5 mô hình riêng | 0,96130 |
| LR stack 5 mô hình riêng (nested) | 0,96132 |

Các GBDT của cùng một bộ feature tương quan > 0,998 nên ghép chỉ cộng +0,0002.

### 6.2 Stacking với thư viện OOF công khai

Cộng đồng cuộc thi công bố OOF + test preds của hàng trăm mô hình trên **cùng split seed 42** (GBDT, RealMLP, TabPFN‑3.5, TabICL, Kumo, Limix, TabFM…). Tải 11 nguồn (`import_public_pool.py`), kiểm tra độ dài/NaN, loại bản sao (corr > 0,999995) → 231 thành viên công khai + mô hình riêng.

| Stack | Số thành viên | Nested CV | Public LB |
|---|---|---|---|
| 4 riêng + 81 công khai | 85 | 0,96206 | 0,96159 |
| 5 riêng + 231 công khai, LR C=0,003 | 236 | 0,96221 | – |
| + family regularisation (TabPFN ×3, TFM ×1,5) | 236 | 0,96222 | 0,96185 |

Phân tích meta-learner (cùng pool, cùng meta-fold): LR theo phân khúc `Class` 0,96218, theo `Type of Travel` 0,96220, theo `Customer Type` 0,96219, LightGBM meta (7 lá, L2 = 50) 0,96204 → **LR tuyến tính là tốt nhất**; meta-learner phức tạp hơn chỉ thêm phương sai.

Trọng số lớn nhất rơi vào các mô hình nền tảng dạng bảng (TabFM 0,17, Limix 0,14, TabPFN 0,10) và RealMLP – tức là họ mô hình *khác loại* với GBDT đóng góp nhiều nhất, dù AUC đơn lẻ của chúng thấp hơn (TabFM 0,9589). Thêm GBDT cùng loại gần như không đổi (+1e-5).

Ghi nhận tác giả OOF công khai: Chris Deotte, goodpjw2008, Dariush Afshar, sachith7, megayak, kagankoral, sadamtorres, amanatar, koumeimaki, abdullahsafwan333, dmitriykolmikov, busyaprime, arhancanli12, mitudru, wangxintong111, thisray, najiama, samanyu1808, hemingweb.

## 7. Kết quả và thảo luận

**Bảng xếp hạng.** [cập nhật] Public 0,96185 ≈ hạng 88/1674 (13/10). Private: …

**CV vs LB.** Chênh nested CV − public LB ≈ 0,00037 ổn định cho mọi bài (0,96222 → 0,96185; 0,96206 → 0,96159) → public LB khó hơn train một chút nhưng thứ tự được bảo toàn; chênh lệch 1e-5 giữa các stack nằm dưới nhiễu của 60k dòng public. Vì vậy chọn bài cuối theo nested CV.

**Cái gì hiệu quả.** (1) Coi `Flight Distance` là mã tuyến: +0,0017. (2) Bộ dữ liệu gốc làm teacher thay vì nối dòng: +0,0003. (3) 10 fold: +0,0002. (4) Ghép nhiều *họ* mô hình khác nhau (GBDT + MLP + foundation model): +0,0010 so với mô hình đơn tốt nhất.

**Cái gì không hiệu quả** (đều đo trên cùng fold): thống kê rating, cross phân loại, pseudo-labeling test, self-distillation, TE tuple 17 cột, meta-learner phi tuyến/phân khúc, thêm dòng gốc vào train. Giá trị của các thí nghiệm âm: tiết kiệm thời gian và chứng minh trần của bài toán (~0,9622) là do nhiễu nhãn của generator, không phải do mô hình.

**Hạn chế.** Phần lớn thành viên stack là của cộng đồng; đóng góp riêng là pipeline feature + 6 mô hình + phân tích meta-learner. GPU 6 GB không chạy được TabPFN full-context (cần ~20 GB).

## 8. Kết luận

Đồ án áp dụng trọn quy trình máy học của môn: EDA đơn/đa biến → tiền xử lý và feature engineering có kiểm soát rò rỉ → so sánh LR/KNN/RF/GBDT → tối ưu siêu tham số → học không giám sát làm feature → stacking với nested CV. Kết quả [cập nhật] đạt top … cuộc thi.

## Tài liệu tham khảo

1. Kaggle, Playground Series S6E10 – Predicting Airline Satisfaction, 2026.
2. T. J. Mahal, Airline Passenger Satisfaction dataset (Kaggle).
3. Chris Deotte, "Tabular Foundation Models – Intro, Dataset, and Starter", discussion #746363.
4. goodpjw2008, "S6E10 | Twin Stacks + Family Reg", notebook series (Parts 1–11).
5. Koumei Maki, "Feature ladder on the shared 5-fold split", discussion #747358.
6. Dmitriy Kolmikov, "Flight Distance behaves like a route ID", discussion #747742.
7. Kratosyan, "Stacking every public OOF library", discussion #746148.
8. Jake VanderPlas, Python Data Science Handbook.
9. Caruana et al., "Ensemble selection from libraries of models", ICML 2004.
