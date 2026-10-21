# Báo cáo đồ án CS116 – Dự đoán mức độ hài lòng hành khách hàng không (Kaggle Playground S6E10)

- Môn: CS116.R11 – Lập trình Python cho máy học, GV: Nguyễn Đức Vũ
- Sinh viên: Văn Đức Tân – 24521586 (làm cá nhân; thầy xác nhận đồ án không bắt buộc làm nhóm)
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
| xgb_v3_cluster_k10 | + PCA/KMeans, 10 fold | 113 cột | 0,96121 | = xgb_v1_k10: lợi ích fold 0 (+0,00013) không lặp lại trên 10 fold |
| xgb_optuna_k10 | tham số Optuna (§5.3), 10 fold | 113 cột | **0,96127** | +0,00006 so với tham số tay |
| lgbm_v2_aux_k10 | LightGBM + aux, 10 fold | 107 cột | **0,96129** | mô hình đơn tốt nhất |
| cat_v2_cluster | CatBoost + PCA/KMeans | 115 cột | 0,96094 | +0,00006 so với cat_v1 |
| xgb_v4_bpe_k10 | + token BPE GPT-2 của chuỗi số (FD, Age, delay) + digit, 10 fold | 122 cột | 0,96125 | +0,00005 so với xgb_v1_k10 |
| xgb_v5_pl_teacher5 | + pseudo-label mềm 300k dòng test từ teacher = trung bình 4 model riêng (an toàn theo fold) | 122 cột | 0,96114 | +0,0001 so với xgb_v1; teacher đơn lẻ chỉ +0,00002 |
| lgbm_v3_pl_teacher5 | LightGBM cùng công thức | 122 cột | 0,96117 | +0,00004 so với lgbm_v2_aux |
| xgb_v1_pl / _plh | pseudo-label test (soft / hard, an toàn theo fold) | | 0,96105 / 0,96105 | +2e-5: trong nhiễu |
| xgb_v1_sd03 | self-distillation (nhãn mềm 0,7·y + 0,3·teacher inner-CV) | | 0,96107 | +4e-5: trong nhiễu |

### 5.3 Tối ưu siêu tham số (bài 11–12)

Cùng bộ feature cuối, XGBoost (GPU), fold 0 của split chung, early stopping 200 vòng theo AUC (`tune_hyperparams.py`; chi tiết `logs/optuna_trials.csv`, `logs/grid_trials.csv`).

| Phương pháp | Không gian | Số cấu hình | Thời gian | AUC fold 0 tốt nhất |
|---|---|---|---|---|
| Cấu hình tay (tham khảo) | depth 7, lr 0,02, colsample 0,4 | 1 | – | 0,96110 |
| GridSearch | depth {5,7,9} × lr {0,02; 0,05} × colsample {0,3; 0,5} | 12 | 187 s | 0,96113 (depth 7, lr 0,02, col 0,3) |
| Optuna TPE | 8 tham số liên tục (lr, depth, min_child_weight, subsample, colsample, λ, α, γ) | 30 | 560 s | **0,96130** (depth 9, lr 0,024, mcw 29, subsample 0,94, col 0,55, λ 0,74, α 0,18) |
| Optuna best, kiểm chứng 10-fold (xgb_optuna_k10) | | | | 0,96127 vs 0,96121 (tham số tay, cùng feature) |

Nhận xét: (1) lr cao (0,05) luôn kém lr thấp dù early stopping – tối ưu theo bước nhỏ quan trọng hơn số cây; (2) GridSearch tốn 12 lần huấn luyện chỉ để xác nhận cấu hình tay vì lưới thô; TPE khai thác vùng tốt (min_child_weight lớn + cây sâu) mà lưới không có, cho +0,0002 với số lần huấn luyện gấp 2,5; (3) biên độ toàn bộ không gian chỉ ±0,0005 – siêu tham số không phải nơi tạo khác biệt lớn ở bài này, feature engineering và ensemble mới là.

### 5.4 Học không giám sát (bài 5–6)

PCA 4 thành phần + KMeans 12 cụm trên 13 rating đã chuẩn hoá (fit trên train+test, không dùng nhãn) → 6 feature `pca_0..3`, `kmeans_12`, `kmeans_dist`. Kết quả: XGBoost fold 0 tăng 0,96110 → 0,96123 nhưng trên 10 fold đầy đủ 0,96121 = không đổi; CatBoost +0,00006. Kết luận: mô hình cây đã tự học được cấu trúc tương quan giữa các rating (hai nhóm "online" và "trên máy bay"), nên góc nhìn không giám sát chủ yếu có giá trị **diễn giải** (cụm "hài lòng mọi mặt", "online kém", "trên máy bay kém") hơn là tăng AUC. Bài học phương pháp: một cải thiện +0,0001 trên 1 fold chưa phải bằng chứng – phải xác nhận trên toàn bộ split.

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

Phân tích meta-learner (cùng pool, cùng meta-fold): LR theo phân khúc `Class` 0,96218, theo `Type of Travel` 0,96220, theo `Customer Type` 0,96219, LightGBM meta (7 lá, L2 = 50) 0,96204, blend tối ưu trực tiếp AUC (hàm mất mát pairwise sigmoid tính chính xác bằng histogram + FFT, trọng số ≥ 0, Σ = 1; `src/auc_blend.py`) 0,96195 → **LR tuyến tính (có trọng số âm) là tốt nhất**; meta-learner phức tạp hơn chỉ thêm phương sai, và ràng buộc không âm làm mất các trọng số âm dùng để khử sai số hệ thống của một số họ mô hình.

Trọng số lớn nhất rơi vào các mô hình nền tảng dạng bảng (TabFM 0,17, Limix 0,14, TabPFN 0,10) và RealMLP – tức là họ mô hình *khác loại* với GBDT đóng góp nhiều nhất, dù AUC đơn lẻ của chúng thấp hơn (TabFM 0,9589). Thêm GBDT cùng loại gần như không đổi (+1e-5).

Ghi nhận tác giả OOF công khai: Chris Deotte, goodpjw2008, Dariush Afshar, sachith7, megayak, kagankoral, sadamtorres, amanatar, koumeimaki, abdullahsafwan333, dmitriykolmikov, busyaprime, arhancanli12, mitudru, wangxintong111, thisray, najiama, samanyu1808, hemingweb.

## 7. Kết quả và thảo luận

**Bảng xếp hạng.** Public: `stack_pool_v2` 0,96185 (hạng ~88/1674, 13/10) → `twin_v2_dariush` 0,96187 (hạng 49–103/1825, 14/10) → `twin_v3_consensus_den30` 0,96189 (hạng 63–68/1999, 16/10; 198–207/2535 ngày 21/10 vì bảng đông thêm ~120 đội/ngày copy file public) → `public_hedge_v1` 0,96195 (**hạng 134–140/2537, top 5,3–5,5 %**, 21/10). Private: [cập nhật sau 1/11].

**Cơ chế xếp hạng public.** Public LB chỉ chấm 20 % test (~60k dòng) nên nhiễu giữa hai bài gần giống nhau là ±0,0001; 48 đội đồng điểm 0,96187 là cùng một file notebook công khai và chiếm trọn hạng 14–61, đẩy bài 0,96185 (chênh 0,00002 = nhiễu) xuống hạng ~111. Để không bị "đồng điểm chiếm chỗ", bài `twin_v2_dariush` lấy trung bình hạng của stack riêng (`stack_pool_v2`) và stack công khai của Dariush Afshar (tương quan hạng 0,99974 – hai stack chứa cùng thông tin): public 0,96187, hạng ≈ 14–61. Đây là thao tác giảm phương sai, trung tính về mặt thông tin; thứ hạng private giữa các bài trong nhóm này sẽ do nhiễu quyết định.

**Blend đồng thuận với file public (16/10).** Phân bố public ngày 16/10 (1 999 đội): 50 đội đúng 0,96190, 42 đội 0,96188–9, 12 đội ≥ 0,96191 và chỉ 4 đội ≥ 0,962 (0,96200 / 0,96227 / 0,96256 / 0,96273 – cao hơn cao nguyên 20–60 lần dải nhiễu ±1,5e-5, nhiều khả năng là khớp public LB bằng nhiều lần nộp; private sẽ phân xử). Cụm 0,96190 là một công thức public được copy lại: 0,7·stack 0,96188 + 0,3·file "passenger-context foundation ensemble" của Den Pugovkin (0,96183) – file có tương quan hạng thấp nhất với mọi stack (0,9965 với bài của em, so với 0,9999 của các stack khác). Em dựng `twin_v3 = 0,7·rank(twin_v2) + 0,3·rank(file Den)` (tương quan 0,99993 với file 0,96190 công khai): public **0,96189**. Lưu ý trung thực: file của Den không có OOF nên không kiểm chứng được bằng nested CV; mức +0,00002 public nằm trong dải nhiễu. Bài này chỉ dùng làm bài nộp cuối số 2 (hedge cho public); bài số 1 vẫn là stack chọn theo nested CV. Credit: Den Pugovkin (file), nina2025 (dataset `ps-s6e10-7`), yuvrajlamba01 (công thức 70/30). Các phân tích cộng đồng cùng ngày xác nhận trần: goodpjw2008 tốn 5 lần nộp để chứng minh chênh 0,96188/0,96187 là làm tròn; starkhushi tìm "hàng gốc" (khác 1–2 cột) trong bộ gốc cho từng dòng synthetic – nhãn hàng xóm đạt AUC 0,89 một mình nhưng thêm vào stack −0,000002; LightGBM học phần dư của stack trên 21 cột thô (nested) không tìm thêm được gì.

**Bài hedge public (21/10).** Từ 17/10, public LB tăng 0,96190 → 0,96207 hoàn toàn bằng *ngoại suy trên public split* (kozykappa `R₁₉₀ + 1,4·(R₁₉₁ − R₁₉₀)`, mayankraaz fit mặt đáp ứng bậc hai lên 44 bài đã chấm rồi "leo dốc"); goodpjw2008 kiểm chứng "nguyên liệu" đó bằng nested CV: **−935e-6, thua 5/5 fold** – tức là lợi public sẽ bị trả lại ở private. Hệ quả: ngưỡng top 10 % public trôi 0,96184 → 0,96187 trong 6 ngày vì hàng trăm đội copy các file đó. Vì thầy chấm theo cả public lẫn private và Kaggle lấy bài tốt hơn trong 2 bài được chọn, em dựng **một** bài hedge cho public: `public_hedge_v1 = 0,5·rank(twin_v3) + 0,25·rank(file hanifnoerrofiq 0,96193) + 0,25·rank(file mayankraaz surface-ascent v2 0,96196)` (`src/public_hedge_blend.py`; tương quan hạng với twin_v3 0,99966) → public **0,96195**. Bài này không có OOF, không chọn theo CV, và em kỳ vọng nó *thấp hơn* bài CV ở private; nó chỉ là bài số 2. Bài số 1 vẫn là stack nested-CV (`stack_pool_v2`). Credit: hanifnoerrofiq, mayankraaz, goodpjw2008 (dataset re-host).

**CV vs LB.** Chênh nested CV − public LB ≈ 0,00037 ổn định cho mọi bài (0,96222 → 0,96185; 0,96206 → 0,96159) → public LB khó hơn train một chút nhưng thứ tự được bảo toàn; chênh lệch 1e-5 giữa các stack nằm dưới nhiễu của 60k dòng public. Vì vậy chọn bài cuối theo nested CV.

**Cái gì hiệu quả.** (1) Coi `Flight Distance` là mã tuyến: +0,0017. (2) Bộ dữ liệu gốc làm teacher thay vì nối dòng: +0,0003. (3) 10 fold: +0,0002. (4) Ghép nhiều *họ* mô hình khác nhau (GBDT + MLP + foundation model): +0,0010 so với mô hình đơn tốt nhất.

**Cái gì không hiệu quả** (đều đo trên cùng fold): thống kê rating, cross phân loại, pseudo-labeling test, self-distillation, TE tuple 17 cột, meta-learner phi tuyến/phân khúc, thêm dòng gốc vào train. Giá trị của các thí nghiệm âm: tiết kiệm thời gian và chứng minh trần của bài toán (~0,9622) là do nhiễu nhãn của generator, không phải do mô hình.

**Thử nghiệm fine-tune TabPFN-3.5** (`src/tabpfn_finetune.py`, `kaggle_kernels/tabpfn-finetune/`). Mọi TabPFN trong pool đều chỉ dùng in-context learning; ý tưởng là cập nhật trọng số trên 560k dòng train để có một "họ" mô hình mới. Kết quả: (1) trên GPU 6 GB tại chỗ không khả thi – trạng thái huấn luyện chiếm 4,4 GB nên mọi bước inference đều tràn bộ nhớ, kể cả 2 000 dòng/bước; (2) trên Kaggle T4 (16 GB, 24 000 dòng/bước, 4 epoch, lr 1e-5, inference với context 50 000 dòng): fold 0 AUC 0,95897, fold 1 0,95797 – **thấp hơn 0,002 so với TabPFN không fine-tune dùng full context (0,9611)**; loss huấn luyện gần như không giảm (0,22–0,24), và mỗi fold mất 3 h 45 (inference 300k dòng test chiếm 2 h). Kết luận: với mô hình nền tảng dạng bảng, mất mát do rút ngắn context lớn hơn nhiều so với lợi ích của fine-tune ngắn; chi phí inference (attention trên context) mới là nút thắt, không phải huấn luyện. Thí nghiệm dừng sau 2 fold.

**TabPFN-3.5 in-context với context lớn** (`kaggle_kernels/tabpfn-infer/`, Kaggle T4, `fit_with_cache` + KV cache int8 theo công thức hạng 2 S6E9): context 400 000 dòng/fold vừa khít 12,7 GB VRAM, 28 phút cache + 27 phút dự đoán mỗi fold. AUC 5 fold 0,96127 / 0,95986 / 0,96138 / 0,96106 / 0,96063, gộp **0,96083**; tương quan 0,998 với 4 TabPFN sẵn có trong pool (0,96105–0,96118, context 560k). Thêm vào stack: 0,962222 → 0,962221. Đây là thí nghiệm "đối chứng" quan trọng: tự chạy được mô hình nền tảng ở quy mô lớn, xác nhận kết quả cộng đồng, và chứng minh pool đã bão hoà cả ở họ TabPFN.

**TabPFN + feature "nhóm tuyến bay"** (công thức Chris Deotte, discussion #749007; chạy hoàn toàn bằng API Kaggle: trọng số TabPFN-3.5 đưa lên dataset private riêng nên kernel không cần secret/license). 9 cột: trung bình theo từng giá trị `Flight Distance` của 6 cột khác, số lần xuất hiện, dự đoán teacher từ dữ liệu gốc (label-free), cộng target encoding của Flight Distance tính **trong fold** (inner-OOF cho dòng context). Sàng lọc fold 0, context 100k, 30k dòng val: thô 0,95864 → **+FE 0,95915 (+0,0005)** → +FE+LLR 0,95882 (LLR làm kém đi, trái với S6E9). Chạy 5 fold ở context 400k, 1 estimator: 0,96116 / 0,96018 / 0,96157 / 0,96109 / 0,96097, gộp **0,96097** (thô cùng cấu hình: 0,96083, +0,00015; bản của Chris với context 560k × 8 estimator: 0,96149 – phần chênh còn lại do context và số estimator, vượt khả năng T4). Tương quan 0,9975 với các TabPFN khác trong pool; thêm vào stack: 0,962222 → 0,962219 (không đổi). Kết luận: feature engineering "đúng cấu trúc" (coi Flight Distance là group id) cải thiện từng mô hình nền tảng, nhưng pool công khai đã chứa đủ các biến thể nên stack không nhận thêm thông tin.

**Học từ writeup tháng trước (S6E9).** Ba bài hạng 1–3 của S6E9 (Chris Deotte; Team Alicia; Masaya Kawamata – đang hạng 2 public S6E10) gợi ý: (a) token BPE GPT‑2 của chuỗi số (dữ liệu synthetic sinh bởi mô hình ngôn ngữ) +0,0013 cho TabPFN ở S6E9; (b) pseudo-label mềm từ teacher mạnh; (c) TabPFN full-context với KV cache int8. Kiểm chứng trên S6E10: (a) với XGBoost +0,00005 (10 fold), với TabPFN **−0,00025** (fold 0, context 100k: 0,95864 → 0,95839) – cấu trúc "mã tuyến" của Flight Distance đã được bắt bằng khai báo categorical, cột token chỉ làm loãng; (b) +0,0001 cho model đơn nhưng **0 cho stack** (0,962218 vs 0,962222) vì teacher chính là các member sẵn có – distillation không tạo thông tin mới; (c) chạy trên Kaggle T4 (xem §7). Bài học: mẹo "generator-aware" phụ thuộc vào bộ dữ liệu cụ thể; phải đo lại, không chuyển mặc nhiên.

**Mô hình ngôn ngữ "thế thân generator" (LLR).** Giả thuyết (Team Alicia, S6E9): dữ liệu synthetic do một LM sinh ra từ bộ gốc; một LM nhỏ fine-tune **chỉ trên 130k dòng gốc** (distilgpt2, 3 epoch, 63 phút trên T4, nhãn là một trường văn bản, thứ tự cột ngẫu nhiên) sẽ xấp xỉ P(nhãn | dòng) của chính generator. Với mỗi dòng của cuộc thi tính LLR = log p(dòng | "satisfied") − log p(dòng | "neutral or dissatisfied"), lấy trung bình 2 thứ tự cột (`kaggle_kernels/llm-surrogate/`, 5 giờ chấm 1 triệu dòng trên Kaggle). Kết quả: LLR **một mình đạt AUC 0,9500** trên train của cuộc thi mà không dùng nhãn thi (so với teacher LightGBM từ dữ liệu gốc: 0,9550), tương quan 0,92 với teacher và với TabPFN. Nhưng: làm feature cho XGBoost fold 0 **0,96096 vs 0,96110 (kém hơn)**; làm member stack **0,962222 → 0,962222 (không đổi)** – thông tin đã nằm trong teacher/lookup từ dữ liệu gốc. Đây là kết quả âm có giá trị: xác nhận giả thuyết "generator dựa trên bộ gốc" định lượng được, nhưng phần tín hiệu này đã được khai thác bằng cách rẻ hơn nhiều (một LightGBM trên bộ gốc).

**Hạn chế.** Phần lớn thành viên stack là của cộng đồng; đóng góp riêng là pipeline feature + 12 mô hình + phân tích meta-learner + các thí nghiệm âm có số liệu. GPU 6 GB không chạy được TabPFN full-context (cần ~17–20 GB). Bài `twin_v3` chứa 30 % một file test công khai không có OOF, nên không thuộc quy trình kiểm chứng bằng CV của đồ án.

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
