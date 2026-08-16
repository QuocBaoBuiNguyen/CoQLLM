# Revision Roadmap — Luận văn CoQLLM

> Nguồn: Stage 3 REVIEW (5-reviewer panel) trong pipeline `/ars-full`.
> Quyết định: **MAJOR REVISION**. Cập nhật: 2026-07-14.
> Trạng thái pipeline: Stage 2.5 PASS ✅ → Stage 3 REVIEW ✅ → Stage 4 partial (6/10 đóng).

## Đã hoàn tất trong session (prose + code)

| ID | Ưu tiên | Vị trí | Đã làm |
|----|:---:|--------|--------|
| P0.2 | 🔴 | ch3 §3.3.1 | Reconcile: lợi thế Q-Former đến từ **learned queries + self-attention**, cross-attention chỉ là input-dependent aggregation. Hết mâu thuẫn với "Quan sát" §3.3.2. |
| P0.3 | 🔴 | ch2 §2.3.2 | Nêu rõ SeLLa-Rec **đã** đánh giá AUC/uAUC/ML-1M → gap đặc thù cho **Q-Former bridge**, không phải toàn dòng CF–LLM; CoQLLM vs SeLLa-Rec bổ sung nhau (Q-Former vs contrastive InfoNCE). |
| P1.1 | 🟠 | code + ch3 §3.6.1 | **Fix code**: `qformer_alignment_builder.py:283` — `rng.shuffle()` trước khi cắt `[:20000]`, loại thiên lệch thời gian. Prose §3.6.1 cập nhật tương ứng. |
| P1.2 | 🟠 | ch4 §4.1.4 | Điền: GPU **A100 80GB**; trainable params **8.65M** (LoRA 2.52M @ r=8,α=16 + Q-Former/proj/inject ~6.13M); runtime ~4–6 phút/epoch (0.65–0.84 s/it). |
| P2.1 | 🟡 | ch2 §2.3.4 | Tách 2 vai trò CoRA: *cơ chế* (CoQLLM hiện thực) vs *mô hình gốc* (baseline). |
| P2.2 | 🟡 | ch3 §3.5.2 | Qualify InstructBLIP: instruction cố định → gần fixed-query + task-context, không phải true per-sample conditioning; varied-instruction để ngỏ. |
| P1.3 | 🟠 | ch3 §3.1 Hình 3.1 | **Thay hình**: `logo.png` → `coqllm-architecture.drawio.png`, bỏ ghi chú "tạm". Build xác nhận nhúng OK. |

## Còn lại — CHỜ THÍ NGHIỆM / HÌNH VẼ

| ID | Ưu tiên | Reviewer | Vị trí | Việc cần làm | Chặn bởi |
|----|:---:|----------|--------|--------------|----------|
| **P0.1** | 🔴 Critical | 5/5 | ch4 Bảng 4.1/4.2, ch5 §5.2 | Chạy experiments → điền số liệu thực → viết phân tích kết quả. **Không điều chỉnh claim hồi tố** nếu CoQLLM không vượt CoLLM. | Kết quả train |
| **P1.4** | 🟠 Major | Devil's Advocate | ch4 §4.4 Bảng 4.2 | Thêm hàng ablation **"attention-MLP projector"** (MLP + self-attention, không Q-Former) → chứng minh đóng góp đến từ Q-Former structure chứ không chỉ nonlinearity + attention. **Đây là đòn phản biện mạnh nhất của DA (M2).** | Kết quả train |
| **P2.3** | 🟡 Moderate | R3 | ch4 §4.5.2 | Phân tích active vs sparse users (validation cho cold-start motivation). Kỳ vọng: cải thiện rõ hơn ở sparse users. | Kết quả train |
| P2.4 | 🟡 Minor | Devil's Advocate | ch4 §4.1.4 | (Tùy chọn) Justify LoRA r=8, layers [q,v] — brief note hoặc citation. | — |

## ⚠️ Lưu ý bắt buộc khi chạy experiments (P0.1)

1. **Rebuild Stage 1 dataset trước khi train.** Fix P1.1 thay đổi hành vi build (`build_qformer_dataset.py`); cap 20k cố định lúc build. Số liệu cuối trong luận văn PHẢI đến từ run có fix shuffle.
2. **P1.4 (attention-MLP baseline)** nên train cùng đợt với các ablation khác để đồng nhất điều kiện.
3. **1 seed** vẫn là hạn chế đã thừa nhận (ch4 mở đầu, ch5 §5.3). Nếu có thời gian → multi-seed mean±std.

## Các điểm phản biện cốt lõi cần chuẩn bị bảo vệ (từ Devil's Advocate)

- **C1 (đã vá prose P0.2):** "Q-Former > MLP nhờ cross-attention" — đã reframe sang learned queries + self-attention. Kết quả P1.4 sẽ là bằng chứng quyết định.
- **M1 (đã vá P0.3):** SeLLa-Rec cũng align CF-LLM + AUC/uAUC/ML-1M — đã phân biệt cơ chế.
- **Alt-explanation:** MF embedding quality có thể là bottleneck thực (uAUC plateau ~0.717, MF teacher ~0.671). Nếu uAUC không cải thiện → giải thích qua nghịch lý AUC/uAUC (§4.3) + hướng dense-CF-pretrain (§5.3).

## Đường đi tiếp của pipeline (khi có kết quả + hình)

```
Điền Bảng 4.1/4.2 + §5.2 + §4.5.2 + Hình 3.1 + hàng ablation P1.4
   ↓
Stage 3' RE-REVIEW  (verify roadmap đã đóng, residual issues)
   ↓
Stage 4.5 FINAL INTEGRITY  (100% pass, verify from scratch)
   ↓
Stage 5 FINALIZE  (LaTeX → tectonic → PDF)
```
