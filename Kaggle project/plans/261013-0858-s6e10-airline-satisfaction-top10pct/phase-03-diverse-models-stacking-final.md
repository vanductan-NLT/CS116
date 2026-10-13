# Phase 03 – Model đa dạng + stacking + chọn final (22–29/10)

## Bài học cộng đồng (discussion #746148, #747182)
- Thêm GBDT/NN cùng loại gần như không giúp stack (bão hoà); **loại model mới** mới giúp: TabPFN / TabICL (+~0.0001), bản 10-fold (+~0.0001).
- Seed-averaging làm OOF đẹp nhưng không lên LB.
- Check tương quan trước khi thêm member (nhiều OOF public là bản sao, corr = 1.0).

## Model đa dạng (Kaggle GPU)
- RealMLP / TabM (pytabkit), TabPFN-v2/3.5 hoặc TabICL (theo starter của Chris Deotte, discussion #746363).
- Tuỳ thời gian GPU: ưu tiên TabPFN/TabICL > RealMLP > TabM.

## Stacking
- Meta-model: LogisticRegression trên logit OOF (C nhỏ ~0.003), đánh giá **nested CV**.
- So sánh: trung bình đơn giản, hill-climbing (Caruana), LR stack.
- Bản A: chỉ model tự train. Bản B: + OOF public cùng split (ghi nguồn tác giả trong báo cáo).

## Chọn 2 final sub (29–30/10)
1. Stack nested-CV tốt nhất (thường bản B).
2. Bản "an toàn": stack chỉ model tự train, ít member, CV ổn định qua fold.
- Không chọn theo public LB khi chênh < 1e-4.

## Done khi
- Nested CV ≥ 0.9619, public ≥ 0.9618, đã chọn 2 final.
