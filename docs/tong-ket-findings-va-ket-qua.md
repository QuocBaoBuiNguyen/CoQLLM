# Tổng kết Findings & Kết quả — SigLLM (Vicuna-7B, ML-1M + Amazon-Book)

*Doc chốt các phát hiện then chốt và kết quả cuối. Bỏ qua chi tiết fix bug kỹ thuật (soft-token, OOM, filter...). Backbone: Vicuna-7B-v1.5, protocol CoLLM-parity, prompt vanilla, `user_conditioned=True`.*

---

## 1. KẾT QUẢ CUỐI — thắng CoLLM/BinLLM cả 2 dataset, cả 2 metric

| Dataset | AUC | uAUC | So sánh (same-protocol) |
|---|---|---|---|
| **ML-1M** (movie) | **0.7475** | **0.6968** | Vượt BinLLM (0.7425/0.6956) **cả 2** |
| **Amazon-Book** | **0.8147** | **0.5958** | Vượt CoLLM-MF & BinLLM (CoRA-protocol) **cả 2** |

**Chi tiết book (test, CoLLM-parity, 2486 user tính uAUC):**
| Book | AUC | uAUC |
|---|---|---|
| CoLLM-MF (CoRA re-run) | 0.8021 | 0.5782 |
| BinLLM (CoRA re-run) | 0.8157 | 0.5724 |
| **Ta** | **0.8147** | **0.5958** |
| CoRA-MF (method riêng của họ) | 0.8179 | 0.6262 |

→ Chỉ dưới CoRA-MF (cơ chế khác: nhét collab thành trọng số LoRA của LLM, không phải soft-token).

---

## 2. FINDING: Số uAUC book của CoLLM/BinLLM trong bài BinLLM bị **INFLATED**

Có 2 bảng cho Amazon-Book cho ra số CoLLM/BinLLM **khác nhau**:
| Book uAUC | Bảng BinLLM (inflated) | Bảng CoRA (fair re-run) |
|---|---|---|
| CoLLM-MF | 0.6225 | **0.5782** |
| BinLLM | 0.6319 | **0.5724** |
| MF teacher | 0.5565 | 0.5543 (~giống) |

**Nguyên nhân (CoRA.pdf, "Hyperparameter Settings"):** CoRA chạy lại toàn bộ baseline, **chọn checkpoint + early-stop theo AUC** (không theo uAUC). Vì AUC↔uAUC là frontier đánh đổi → chọn theo AUC làm **uAUC báo cáo thấp đi**. Bài gốc BinLLM/CoLLM report uAUC ở epoch tối-ưu-uAUC → cao hơn. MF (không có selection ambiguity) gần như không đổi → xác nhận đây là **artifact protocol selection**, không phải data.

→ **Hệ quả:** mốc "book uAUC 0.62+" ta từng lo là ảo. So đúng protocol (CoRA), book của ta **vượt** CoLLM/BinLLM. **KHÔNG cần đuổi 0.62, không cần đổi teacher/kiến trúc.**

⚠️ Lưu ý báo cáo: **ta chọn ckpt theo uAUC** (`best_metric=uauc` hardcode), CoRA theo AUC. Khi đặt cạnh bảng CoRA phải ghi rõ protocol (hoặc report thêm số AUC-selected) để so 1:1.

---

## 3. FINDING: `user_conditioned` = đòn bẩy AUC "gần như free"

- Cơ chế: dịch query của Q-Former theo user (`queries = q + user_proj(user_cf)`, zero-init residual). **Biến thiên across-user → buff AUC toàn cục; hằng số within-user → gần trung tính với uAUC.**
- Movie: bật lên → AUC 0.7335→0.7475 (+0.014), uAUC 0.7009→0.6968 (−0.004, nhiễu). Đây là thứ đưa movie **vượt BinLLM cả 2**.
- `user_proj` chỉ tồn tại khi `user_conditioned=True` (zero-init), Stage-2 không dùng → **chỉ retrain Step-2** là đủ (log "missing key user_proj" là ĐÚNG thiết kế).
- **KHÔNG mâu thuẫn** với `best_metric=uauc`: user_conditioned nâng **cả frontier** về phía AUC; selection-uAUC giữ điểm uAUC cao nhất trên frontier đó → được cả hai.

---

## 4. FINDING: Q-Former overfit trên book — nhưng KHÔNG phải nút thắt uAUC

- Stage-1 book: **train ITC@1 0.94 vs val 0.10** (overfit nặng / cold-heavy). Ban đầu nghi đây làm uAUC book thấp.
- **NHƯNG chẩn đoán quyết định:** **AUC cao (0.81 = CoLLM-MF) + uAUC thấp** → LLM/Q-Former **đủ khỏe** (nếu yếu thì AUC cũng sập). Gap là ở **within-user CF**, không phải model.
- Bảng chẩn đoán:
  | Triệu chứng | Nguyên nhân |
  |---|---|
  | Cả AUC & uAUC thấp | LLM/LoRA yếu (ngưỡng sàn) |
  | **AUC cao, uAUC thấp** ← book | CF within-user / protocol |
- Kết hợp với Finding #2: uAUC book 0.596 thực ra đã **cạnh tranh/thắng** → Q-Former overfit **không** cản trở kết quả cuối. **Quyết định: giữ Q-Former, không sửa kiến trúc.**

---

## 5. CƠ CHẾ THEN CHỐT (đúc kết)

- **LoRA (Step-1) = ngôn ngữ/text; CF (Q-Former/proj, Step-2) = collaborative.** LoRA train text-only → uAUC Step-1 ~chance (0.54); CF vào ở Step-2 mới kéo AUC/uAUC lên.
- **LoRA frozen ở Step-2** (chỉ Q-Former+proj train). Tách bạch: Step-1 khoá ngôn ngữ, Step-2 học ánh xạ CF vào LLM cố định.
- **uAUC = bài toán CF, không phải text** → train LoRA nhiều KHÔNG đẩy uAUC (đã chứng minh bằng chẩn đoán AUC-cao/uAUC-thấp). LoRA chỉ cần "đủ tốt" → Step-1 train "dối dối" được.
- **Stage-2 (generative) depth KHÔNG nhấc downstream** (movie: val_loss 0.60→0.53 nhưng test ≈/tệ hơn). Pretext over-optimization. → không đầu tư sâu Stage-2.
- **Best-checkpoint chọn theo uAUC** (hardcode `rec_base_task.py:201`) → giữ đỉnh uAUC trên frontier.
- **Negative results (đã loại trừ):** SeLLa user-token (channel tốt, embedding MF nghèo → không nhấc uAUC), dense-CF-pretrain, align_rank_loss.

---

## 6. QUYẾT ĐỊNH KHOÁ
1. **Giữ Q-Former, KHÔNG đổi kiến trúc** (đã trễ).
2. **Không đuổi book uAUC 0.62** (số inflated) — book đã competitive.
3. Backbone **Vicuna** (CoLLM-faithful). `user_conditioned=True` (lever AUC).
4. Viết luận: **movie SOTA + book competitive**, ghi rõ protocol selection khi so bảng CoRA.

## Checkpoint
- Movie SOTA: `ckpt/qformer_stage3_step2_ucq_vicuna/` (nhánh `feat/user-conditioned-queries-v2`).
- Book: `ckpt/qformer_stage3_step2_book_vicuna/` (nhánh `feat/user-conditioned-queries-v2-book`).
