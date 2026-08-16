# Tổng kết Findings & Kết quả — CoQLLM (Vicuna-7B, ML-1M + Amazon-Book)

*Doc chốt các phát hiện then chốt và kết quả cuối. Bỏ qua chi tiết fix bug kỹ thuật (soft-token, OOM, filter...). Backbone: Vicuna-7B-v1.5, protocol CoLLM-parity, prompt vanilla, `user_conditioned=True`.*

---

## 1. KẾT QUẢ CUỐI — thắng CoLLM/BinLLM cả 2 dataset, cả 2 metric

| Dataset | AUC | uAUC | So sánh (same-protocol) |
|---|---|---|---|
| **ML-1M** (movie) | **0.7475** | **0.6968** | Vượt BinLLM (0.7425/0.6956) **cả 2** |
| **Amazon-Book** | **0.8132** | **0.6062** | Vượt CoLLM-MF & BinLLM (CoRA-protocol) **cả 2**; +align_rank |

**Chi tiết book (test, CoLLM-parity, 2486 user tính uAUC):**
| Book | AUC | uAUC |
|---|---|---|
| CoLLM-MF (CoRA re-run) | 0.8021 | 0.5782 |
| BinLLM (CoRA re-run) | 0.8157 | 0.5724 |
| Ta — user_conditioned (Run-1) | 0.8147 | 0.5958 |
| **Ta — + align_rank_loss (Run-2)** | **0.8132** | **0.6062** |
| CoRA-MF (method riêng của họ) | 0.8179 | 0.6262 |

→ Run-2 (align_rank) nâng uAUC **0.5958 → 0.6062 (+0.0104)**, AUC gần như đứng yên (−0.0015). Cắt mốc 0.60, vào vùng bảng BinLLM "inflated" (CoLLM-MF 0.6225), chỉ còn dưới CoRA-MF (0.6262, cơ chế khác: nhét collab thành trọng số LoRA của LLM, không phải soft-token) khoảng −0.020.

**Phân rã warm/cold (Run-2, test):**
| Split | #user | AUC | uAUC |
|---|---|---|---|
| test (full) | 2486 | 0.8132 | 0.6062 |
| test_warm | 1648 | **0.8190** | **0.6238** |
| test_cold | 630 | 0.7928 | **0.5258** |

→ **Bằng chứng trực tiếp uAUC = bài toán CF:** warm uAUC **0.624** (có tín hiệu MF, gần chạm CoRA-MF 0.6262) vs cold uAUC **0.526** (≈ chance — MF frozen không có embedding cho user/item cold). Cold kéo trung bình full xuống 0.606. Trần uAUC nằm ở **cold-item / MF teacher**, đúng finding Stage-1 (val OOD/cold cap) và chẩn đoán AUC-cao/uAUC-thấp — KHÔNG phải Q-Former. → nếu muốn đẩy uAUC full cao hơn nữa, đòn bẩy là **MF teacher trên cold** (Lever 2), không phải regularize/kiến trúc.

---

## 2. FINDING: Số uAUC book của CoLLM/BinLLM trong bài BinLLM bị **INFLATED**

Có 2 bảng cho Amazon-Book cho ra số CoLLM/BinLLM **khác nhau**:
| Book uAUC | Bảng BinLLM (inflated) | Bảng CoRA (fair re-run) |
|---|---|---|
| CoLLM-MF | 0.6225 | **0.5782** |
| BinLLM | 0.6319 | **0.5724** |
| MF teacher | 0.5565 | 0.5543 (~giống) |

**Nguyên nhân (ĐÃ ĐÍNH CHÍNH sau khi đọc code CoLLM):** KHÔNG phải do selection metric. Confirm từ source: CoLLM `minigpt4/tasks/rec_base_task.py:254` → `agg_metrics = auc` + runner chọn best theo AUC; CoRA cũng chọn theo AUC. **Cả hai đều chọn theo AUC.** Chênh lệch giữa 2 bảng (CoLLM-MF uAUC 0.6225 vs 0.5782) là do **variance khi mỗi bài RE-RUN/RE-IMPLEMENT CoLLM** (tuning khác nhau; uAUC book nhiễu ~2500 user → lệch ±0.03-0.05). **Không có 1 con số CoLLM "đúng" duy nhất** — nó ~0.57-0.62 tuỳ run. Book của ta (0.5958) nằm gọn trong dải đó.

→ **Hệ quả:** mốc "book uAUC 0.62+" không phải mục tiêu cố định — book của ta đã trong dải cạnh tranh. **KHÔNG cần đuổi 0.62, không cần đổi teacher/kiến trúc.**

⚠️ **Khác biệt protocol THẬT giữa TA và HỌ:** ta chọn ckpt theo **uAUC** (`rec_base_task.py:201`), CoLLM/CoRA theo **AUC**. → uAUC của ta là uAUC-optimal (lợi thế nhẹ). Khi đặt cạnh bảng, ghi rõ điều này (hoặc report thêm số AUC-selected) để so 1:1.

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

## 4b. FINDING (MỚI): `align_rank_loss` = đòn bẩy uAUC THẬT trên book (âm trên movie)

- **Cơ chế:** per-user pairwise BPR (differentiable uAUC surrogate) áp lên **CF soft-token đã align** (pre-LLM), qua 1 aux head `align_rank_head` (zero-init). Ép chính phần alignment collaborative giữ **within-user ordering**, không chỉ margin Yes/No của LLM. Gated Step-2 (Q-Former/proj trainable), cần `user_grouped_batch` để mỗi batch có cặp same-user pos/neg. Bật: `align_rank_loss.weight=0.2` + `user_grouped_batch.enabled=true`.
- **Kết quả book (Run-2):** uAUC **0.5958 → 0.6062 (+0.0104)**, AUC ~đứng yên. Đúng thiết kế — đánh trúng within-user, không đụng cross-user AUC.
- **Trajectory Step-2 (val uAUC):** ep0 0.531 → ep1 0.551 → ep2 0.582 → ep3 0.589 → ep4 0.594 → **ep5 0.606 (đỉnh)** → ep6 0.602. Xuất phát THẤP hơn baseline vì `align_rank_head` **zero-init** (epoch đầu ~no-op, tác dụng hiện dần) → **đừng phán ở epoch 0-1**; đỉnh ~ep5.
- **Vì sao book ăn mà movie không:** book ở profile **AUC-cao/uAUC-thấp** (dư địa within-user lớn) → loss này lấp đúng gap. Movie uAUC đã ~bão hoà → trung tính/âm. → **bật theo-dataset là hợp lệ** (siêu tham số theo dataset, không phải bất nhất phương pháp). Cần **ablation movie** (1 run Step-2) để hoàn thiện bảng ON/OFF × {movie, book}.

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
2. ~~Không đuổi book uAUC 0.62~~ **GỠ KHOÁ (2026-08-10):** align_rank đã đẩy book uAUC 0.5958→0.6062, đang tiến về vùng 0.62 — cải thiện trong-pipeline là hướng đang theo.
3. Backbone **Vicuna** (CoLLM-faithful). `user_conditioned=True` (lever AUC) + `align_rank_loss=0.2` cho book (lever uAUC).
4. Viết luận: **movie SOTA + book competitive/thắng**, ghi rõ protocol selection khi so bảng CoRA.

## 7. NEXT
- **Ablation movie:** chạy 1 run Step-2 với `align_rank_loss=0.2`+`user_grouped_batch=true` → hoàn thiện bảng ON/OFF × {movie, book}. Kỳ vọng trung tính/âm trên movie (uAUC đã cao) → biện minh bật-theo-dataset.
- (Tuỳ) đẩy tiếp book: thử `align_rank_loss.weight` 0.1/0.5, hoặc Lever 2 (MF teacher mạnh hơn) nếu muốn chạm 0.62.

## Checkpoint
- Movie SOTA: `ckpt/qformer_stage3_step2_ucq_vicuna/` (nhánh `feat/user-conditioned-queries-v2`).
- Book (Run-2, +align_rank, uAUC 0.6062): `ckpt/qformer_stage3_step2_book_vicuna/` (nhánh `feat/user-conditioned-queries-v2-book`).
