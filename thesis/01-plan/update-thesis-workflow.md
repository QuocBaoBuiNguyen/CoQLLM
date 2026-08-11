# Workflow cập nhật luận văn với kết quả SOTA + skill ARS

> **Mục đích:** doc self-contained để mở một **session mới** và cập nhật `/thesis` bằng skill **academic-research-skills (ARS)**.
> Đọc doc này TRƯỚC khi chạy bất kỳ `/ars-*` nào. Nguồn chân lý kết quả: `docs/tong-ket-findings-va-ket-qua.md`.
> Tạo: 2026-08-11. Trạng thái luận: **MAJOR REVISION**, blocker P0.1 (điền số thực) **đã được GỠ** — giờ đã có kết quả.

---

## 0. BỐI CẢNH CỐT LÕI (đọc kỹ)

### Nhánh SOTA để tham chiếu
- **`feat/user-conditioned-queries-v2`** = SOTA **ML-1M (movie)**: test **AUC 0.7475 / uAUC 0.6968** (vượt BinLLM 0.7425/0.6956 cả 2 metric).
- **`feat/user-conditioned-queries-v2-book`** = nhánh hiện tại, = v2 **+ Amazon-Book + align_rank_loss**: test book **AUC 0.8132 / uAUC 0.6062**. Đây là nhánh chứa TOÀN BỘ code SOTA cho cả 2 dataset (superset của v2).

→ **Khi viết luận, mô tả 1 kiến trúc thống nhất; số của 2 dataset lấy như trên.** Code đọc từ nhánh `feat/user-conditioned-queries-v2-book`.

### Trạng thái luận văn
- Đã qua `/ars-full` (5-reviewer panel) → **MAJOR REVISION**. Roadmap: `thesis/01-plan/revision-roadmap.md`.
- Prose đã vá 6/10 (P0.2, P0.3, P1.1, P1.2, P1.3, P2.1, P2.2). **Còn lại BỊ CHẶN bởi "kết quả train"** → giờ đã unblock:
  - **P0.1 (Critical, 5/5 reviewer):** điền số thực vào ch4 `tab:ket-qua-chinh` + ch5 §5.2.
  - **P1.4 (Major):** thêm hàng ablation "attention-MLP projector" (cần train thêm — xem mục 5).
  - **P2.3 (Moderate):** phân tích active vs sparse user → **giờ đã có = warm vs cold** (mục 3.E).
- Chương: `thesis/02-chapters/ch{1..5}-*.tex`. LaTeX template HCMUS/FIT ở `thesis/00-format/Thesis_Template`.
- ⚠️ Có **2 file ch3**: `ch3-moi-phuong-phap.tex` (bản mới, 386 dòng, đang dùng) và `ch3-phuong-phap.tex` (bản cũ). Xác nhận bản nào được `\input` trong main trước khi sửa.

---

## 1. NHỮNG THAY ĐỔI KIẾN TRÚC TRONG NHÁNH (vật liệu cho báo cáo kiến trúc)

Đây là các **đóng góp so với vanilla Q-Former** (mỗi mục kèm file + ý nghĩa). Nhóm theo vai trò.

### A. Khung kiến trúc SigLLM (cầu nối CF→LLM bằng Q-Former)
- **Pipeline 3 giai đoạn** (commit a01aa60, 8a1d7ce, 4c93515): 
  - *Stage-0 MF* → sinh embedding user/item (frozen teacher).
  - *Stage-1 biểu diễn* (`train_qformer_stage1_representation.py`): BLIP-2 ITC+ITM+ITG **+ item–item (ii) + user–item (ui) contrastive** (bổ sung riêng cho gợi ý, commit 120f8b6).
  - *Stage-2 sinh* (`train_qformer_stage2_generative.py`): tiền huấn luyện caption kiểu BLIP-2, **LLM đóng băng**.
  - *Stage-3 hai bước* (`..._step1_lora.py` / `..._step2_cie.py`): **Step-1** train LoRA trên prompt text-only (không CF); **Step-2 (CIE)** mở Q-Former+projection, **đóng băng LoRA** → tách bạch "ngôn ngữ" (LoRA) và "collaborative" (Q-Former/proj).
- **HFQFormerAdapter** (`src/sigllm/models/q_former/hf_qformer_adapter.py`, commit 14d8599, b0d61db): Q-Former InstructBLIP, text branch khởi tạo từ **bert-base-uncased**, `num_layers=4` (giảm 8→4, commit cfd6418), `num_queries=8`, `d_model=768`.
- **Tiêm soft-token** (`qformer_rec_llm.py`, commit 9fa64b3, b01c13e): token chuyên dụng **`<rec_soft_token>`** (id 32000) thay `<unk>` để tránh va chạm OOV/pad; scatter embedding CF vào chuỗi input LLM. LayerNorm ở projection để chuẩn hoá scale soft-token (ee48751, 0b5536a).

### B. ĐÓNG GÓP SOTA (các "đòn bẩy" tạo ra kết quả vượt baseline) — TRỌNG TÂM báo cáo
1. **`user_conditioned` queries** (commit 2478f3f) — *lever AUC, đưa ML-1M vượt BinLLM.*
   - Cơ chế: dịch query Q-Former theo user CF: `queries = q_base + user_proj(user_cf)`, **zero-init residual** (bắt đầu là no-op, chỉ lớn lên nếu CTR thưởng).
   - Biến thiên **across-user → nâng AUC toàn cục**; hằng số **within-user → gần trung tính uAUC**.
   - File: `hf_qformer_adapter.py` (`self.user_proj`, chỉ tạo khi bật cờ). Config: `model.qformer_config.user_conditioned=True`.
2. **`align_rank_loss` + `user_grouped_batch`** (commit 25f4b33, cd29c8e) — *lever uAUC cho book (0.5958→0.6062).*
   - Cơ chế: BPR per-user (surrogate uAUC) áp lên **CF soft-token đã align** (pre-LLM) qua aux head `align_rank_head` **zero-init**. Ép chính phần alignment giữ **within-user ordering**, không chỉ margin Yes/No.
   - Cần `user_grouped_batch` để mỗi batch có cặp same-user pos/neg. Gated Step-2. Config: `model.align_rank_loss.weight=0.2` + `run.user_grouped_batch.enabled=true`.
   - **Đặc tính then chốt cho luận:** dương trên **book** (profile AUC-cao/uAUC-thấp, còn dư địa within-user), **trung tính/âm trên movie** (uAUC đã bão hoà) → **bật theo-dataset là siêu tham số hợp lệ, không phải bất nhất phương pháp**. (Đã kiểm: weight 0.2 vs 0.3 ≈ nhau → kết quả robust, trần là cold-item chứ không phải weight.)
3. **Chọn checkpoint theo uAUC** (commit c84119d): `rec_base_task.py` `agg_metrics=uauc`. ⚠️ **Khác CoLLM/CoRA (chọn theo AUC)** — phải ghi rõ khi so bảng (hoặc report thêm số AUC-selected).
4. **Backbone Vicuna-7B-v1.5** (commit 9c080c6, a9c9e35): để so trung thực với CoLLM (LLaMA-2 free-merge tương đương v1.1). prompt_template USER/ASSISTANT.
5. **Sửa lịch LR Step-2** (commit 97c68f4): `min_lr < init_lr` (3e-6 < 3e-5) để cosine không bị đảo (trước đó min_lr top-level 8e-5 > init 3e-5 làm LR *leo* thay vì giảm).

### C. Trung thực giao thức đánh giá (cho so sánh công bằng)
- **CoLLM-parity filter** (commit 53c51c8, 80d6d13): toggle `his_title>=2` — CoLLM **giữ** hàng history<2 + zero-pad; SeLLa drop. `match_sella_history_filter`.
- **Test splits test/test_warm/test_cold** (commit 13702bb, c58d9b4, 60cdb0b): phân rã warm/cold (dùng cho P2.3).
- **eval_test.py** (commit 5e07491) + **fp32 logits trước softmax** (commit 515b6b8) + **khớp SeLLa eval** (b2bedab).

### D. Kỹ thuật/dữ liệu (đề cập ngắn ở ch4 setup, KHÔNG phải đóng góp kiến trúc)
- MF book: `weight_decay 1e-4→1e-6` (sửa collapse, embeddings→0). 
- Stage-1 `skip_eval` (Step-1 text-only → bỏ eval, nhanh 2x). Stage-1 dropout=0.2/wd=1e-2 (regularize; nhưng Stage-1 val bị cap bởi cold/OOD, không phải nút thắt).
- Fix **shuffle trước khi cắt 20k** ở builder (roadmap P1.1) — số liệu cuối PHẢI từ run có fix này.

### E. Kết quả phân rã warm/cold (bằng chứng cho câu chuyện uAUC=CF) — dùng cho P2.3 + §4.3
| Book (test) | #user | AUC | uAUC |
|---|---|---|---|
| full | 2486 | 0.8132 | 0.6062 |
| **warm** | 1648 | 0.8190 | **0.6238** |
| **cold** | 630 | 0.7928 | **0.5258** |
→ warm (có tín hiệu MF) ≈ CoRA-MF (0.6262); **cold ≈ chance (0.526)** vì MF không có embedding cho user/item cold → kéo trung bình xuống. **Trần uAUC = cold-item/MF teacher, KHÔNG phải Q-Former.** Khớp nghịch lý AUC-cao/uAUC-thấp (§4.3) và motivation cold-start.

---

## 2. SỐ LIỆU ĐỂ ĐIỀN (P0.1 — bảng kết quả chính ch4 `tab:ket-qua-chinh`)

**ML-1M (CoLLM-parity, chọn ckpt theo uAUC):**
| Model | AUC | uAUC |
|---|---|---|
| BinLLM | 0.7425 | 0.6956 |
| **SigLLM (ta)** | **0.7475** | **0.6968** |

**Amazon-Book (CoRA re-run protocol, 2486 user):**
| Model | AUC | uAUC |
|---|---|---|
| CoLLM-MF (CoRA re-run) | 0.8021 | 0.5782 |
| BinLLM (CoRA re-run) | 0.8157 | 0.5724 |
| **SigLLM — user_conditioned (Run-1)** | 0.8147 | 0.5958 |
| **SigLLM — + align_rank (Run-2)** | **0.8132** | **0.6062** |
| CoRA-MF (method của họ) | 0.8179 | 0.6262 |

⚠️ **Bắt buộc ghi chú protocol:** ta chọn ckpt theo **uAUC**, CoLLM/CoRA theo **AUC**. Số book baseline CoLLM/BinLLM có **2 bảng khác nhau** (BinLLM-paper 0.62+ "inflated" vs CoRA re-run 0.57) do re-run variance — xem `docs/tong-ket-findings-va-ket-qua.md` §2. Không đuổi con float cụ thể; định vị "vượt baseline dưới protocol có kiểm soát".

---

## 3. WORKFLOW ARS ĐỂ CẬP NHẬT LUẬN (chạy ở session mới)

Skill: **academic-paper** (11 mode). Luận đang ở giai đoạn **incorporate results vào revision** → mode chính là **`revision`**.

### Bước 1 — Nạp ngữ cảnh (không chạy skill vội)
Mở session mới, nói rõ với Claude:
- "Cập nhật luận `/thesis`, dùng kết quả từ `docs/tong-ket-findings-va-ket-qua.md` và doc này (`thesis/01-plan/update-thesis-workflow.md`)."
- "Roadmap: `thesis/01-plan/revision-roadmap.md`. Blocker P0.1 giờ đã có số."

### Bước 2 — `/ars-revision` (mode: revision) — cốt lõi
Trigger: *"revise paper", "incorporate reviewer feedback"*. Output: bản revised + point-by-point R&R.
Giao cho nó:
1. **P0.1** — điền `tab:ket-qua-chinh` (số ở mục 2) + viết phân tích §5.2. **KHÔNG chỉnh claim hồi tố** — báo cáo trung thực (ta vượt cả 2 metric ML-1M; book vượt CoLLM-MF/BinLLM CoRA-protocol).
2. **Bổ sung mô tả kiến trúc** vào ch3: thêm `user_conditioned` (§ssec:hai-duong / sec:qformer) và `align_rank_loss` (nối §rank-preserving có sẵn ở ch3-phuong-phap). Vật liệu ở **mục 1.B**.
3. **P2.3** — viết §ssec:user-group bằng phân rã **warm/cold** (mục 1.E).
4. **§4.3 nghịch lý AUC/uAUC** — củng cố bằng warm 0.624 / cold 0.526.

### Bước 3 — (Nếu train thêm) đóng P1.4
Roadmap P1.4 = hàng ablation **"attention-MLP projector"** (MLP+self-attention, KHÔNG Q-Former) để chứng minh đóng góp đến từ *cấu trúc Q-Former*, không chỉ nonlinearity. **Cần 1 lượt train** → làm ở session train, không phải session viết. Nếu chưa có, ghi rõ là hạn chế/để ngỏ.

### Bước 4 — `/ars-reviewer` (mode: re-review)
Trigger: *"check revisions", "verification review"*. Verify roadmap đã đóng, liệt kê residual issues.

### Bước 5 — `/ars-citation-check` + `/ars-format-convert`
- `/ars-citation-check`: dò lỗi trích dẫn vs `thesis/03-refs/references.bib`.
- `/ars-format-convert`: build LaTeX → PDF (tectonic), theo template HCMUS.

### (Tuỳ chọn) `/ars-abstract`
Cập nhật abstract song ngữ với số mới nếu cần.

---

## 4. MAPPING: roadmap item → giờ đóng được gì
| Roadmap | Cần gì | Trạng thái |
|---|---|---|
| **P0.1** điền bảng + §5.2 | số thực | ✅ có (mục 2) → `/ars-revision` |
| **P2.3** active vs sparse | phân tích nhóm user | ✅ có = warm/cold (mục 1.E) |
| **§4.3** nghịch lý AUC/uAUC | bằng chứng | ✅ warm 0.624 / cold 0.526 |
| **P1.4** ablation attention-MLP | train thêm 1 lượt | ⏳ chưa — session train |
| P2.4 justify LoRA r=8 | note ngắn | ⏳ tuỳ |
| Multi-seed | train nhiều seed | ⏳ hạn chế đã thừa nhận |

---

## 5. LỆNH KHỞI ĐỘNG SESSION MỚI (copy-paste)
```
Mình muốn cập nhật luận văn /thesis bằng skill ARS.
Đọc trước: thesis/01-plan/update-thesis-workflow.md (context + số liệu + workflow),
docs/tong-ket-findings-va-ket-qua.md (kết quả chi tiết),
thesis/01-plan/revision-roadmap.md (roadmap review).
Nhánh SOTA: feat/user-conditioned-queries-v2-book (book) / feat/user-conditioned-queries-v2 (movie).
Bắt đầu bằng /ars-revision để đóng P0.1 (điền bảng kết quả + §5.2) và bổ sung mô tả
user_conditioned + align_rank_loss vào ch3.
```
