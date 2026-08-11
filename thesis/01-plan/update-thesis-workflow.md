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
- ✅ Chỉ còn **1 file ch3**: `ch3-phuong-phap.tex` (đã hợp nhất, `\input` trong `main_draft.tex`). Bản trùng `ch3-moi-phuong-phap.tex` đã xoá (commit `aff0198`).

---

## 0.5. ⚠️ MISMATCH LỚN: luận cũ viết theo hướng CoRA — PHẢI REALIGN TRƯỚC KHI ĐIỀN SỐ

**Đây là việc quan trọng nhất — đọc trước khi làm bất cứ gì.** Luận hiện tại mô tả một kiến trúc **KHÁC** với nhánh SOTA thực tế. Nếu chỉ điền số vào một method sai thì luận sẽ mâu thuẫn với code/kết quả.

### Luận cũ đang viết gì (SAI so với implementation SOTA)
Mô tả SigLLM có **HAI đường song song** đưa CF vào LLM và đóng góp = "kết hợp cả hai":
1. Đường **soft-token** (Q-Former Z → projection → soft token).
2. Đường **tiêm trọng số CoRA (ΔW)** — sinh low-rank ΔW cộng vào `q_proj`/`v_proj` của LLM (Công thức `eq:cora-delta`, cổng tanh zero-init kiểu Flamingo).
- Vị trí: ch2 §`ssec:cora-rel` (2.3.4) + bảng so sánh; ch3-moi §`ssec:hai-duong` + `eq:cora-delta` + §`ssec:on-dinh`; ch3-phuong-phap §`sec:cora`; bảng ablation ch4 (3 chế độ: chỉ-soft / chỉ-ΔW / cả-hai); ch5 §future-work.
- Ký hiệu placeholder cũ `<CFTokens>` cũng **stale** (code thật dùng `<rec_soft_token>` + `<ItemIDList>`/`<TargetItemID>`).

### Nhánh SOTA thực tế làm gì (đã verify trong code)
- **CHỈ đường soft-token** (`qformer_rec_llm.py` dòng ~828-830: scatter CF embedding vào `inputs_embeds`). **KHÔNG có CoRA ΔW** — không có gate/tanh/P_down/P_up/delta trong model; config chỉ có `ablate_soft_tokens`, **không có toggle tiêm trọng số**.
- **THÊM 2 đòn bẩy mà luận cũ CHƯA nhắc:** `user_conditioned` (query-shift, lever AUC) + `align_rank_loss` (BPR per-user trên CF token, lever uAUC book). Xem mục 1.B.
- Số SOTA báo cáo (ML-1M 0.7475/0.6968, book 0.8132/0.6062) đến từ **soft-token + user_conditioned (+ align_rank cho book)**, **KHÔNG dùng CoRA ΔW**.

### Việc realign (giao cho `/ars-revision`) — ĐÃ CHỐT PHƯƠNG ÁN (a)
1. **✅ ĐÃ QUYẾT: Phương án (a) — BỎ HẲN đường CoRA ΔW khỏi phần method.** (Người dùng chốt 2026-08-11.) Mô tả đúng model thực = **chỉ soft-token bridge**. Cụ thể:
   - **ch3-moi §`ssec:hai-duong`:** bỏ "Đường tiêm trọng số", `eq:cora-delta`, và §`ssec:on-dinh` (4 kỹ thuật ổn định cho ΔW — không còn liên quan). Đổi tiêu đề mục từ "Hai đường đưa tín hiệu CF" → mô tả 1 đường soft-token.
   - **ch3-phuong-phap §`sec:cora`:** bỏ nguyên mục; sửa các tham chiếu `\ref{sec:cora}`.
   - **ch2 §`ssec:cora-rel` (2.3.4):** GIỮ CoRA nhưng chỉ như **related work / một hướng khác** (tiêm trọng số) + là **baseline CoRA-MF**. Bỏ câu "SigLLM hiện thực cơ chế CoRA như một trong hai đường".
   - **ch5:** bỏ future-work "kết hợp CoRA + soft-token"; nếu muốn có thể để CoRA ΔW như một hướng mở *chưa hiện thực*.
   - **Bảng ablation:** bỏ 3-chế-độ {soft/ΔW/cả-hai}.
   - Giữ `cite{cora2024}` (related work + baseline), bỏ vai trò "cơ chế lõi".
2. **Thêm 2 mục method mới:** `user_conditioned` + `align_rank_loss` (vật liệu mục 1.B) — đây mới là đóng góp thật của nhánh SOTA.
3. **Viết lại bảng ablation ch4** (`tab:ablation`): thay 3-chế-độ-CoRA cũ bằng ablation THẬT khớp code: `vanilla Q-Former → +user_conditioned → +align_rank` (+ hàng attention-MLP P1.4 nếu train được).
4. **Sửa ký hiệu** `<CFTokens>` → `<rec_soft_token>`/`<ItemIDList>`+`<TargetItemID>` xuyên suốt.
5. Rà `references.bib`: `cite{cora2024}` vẫn giữ (CoRA là related work + baseline CoRA-MF), nhưng bỏ vai trò "cơ chế lõi của SigLLM".

> Lưu ý lịch sử: từng có một bản "improvement bundle" tích hợp CoRA (uAUC ~0.7041). **Nhánh canonical cho luận giờ là v2/v2-book — KHÔNG CoRA** (Vicuna, CoLLM-faithful, vượt BinLLM cả 2 metric dưới protocol công bằng). Đừng trộn số của 2 hướng.

---

## 0.6. ⚠️ MISMATCH #2: bỏ InstructBLIP — reframe thành PLAIN BLIP-2 Q-Former (ĐÃ CHỐT)

**Người dùng chốt 2026-08-11: THỐNG NHẤT mô tả cầu nối là PLAIN Q-Former (BLIP-2) trong TOÀN luận — kể cả khi code thực tế chạy module InstructBLIP với instruction cố định. Bỏ mọi phần InstructBLIP. GIỮ SỐ CŨ (chưa train lại); dự định train lại bản plain Q-Former sau để code khớp narrative. (Hợp lệ vì instruction cố định ⇒ chức năng ≈ BLIP-2, không mang tín hiệu per-sample.)**

### Lý do (đã verify code)
- Q-Former ở Stage-3 CÓ nhận instruction (`qformer_rec_llm.py:673`), NHƯNG instruction là **chuỗi task CỐ ĐỊNH** (`QFORMER_ITEM_INSTRUCTIONS`, `_build_qformer_instructions`): train chọn ngẫu nhiên 1/12 paraphrase, **eval luôn dùng `[0]`**; 12 câu là **mô tả task chung**, KHÔNG chứa title/genre thật của item.
- → Đặc trưng định danh của InstructBLIP (**instruction conditioning theo mẫu**) **bị vô hiệu** (instruction hằng số ⇒ không mang tín hiệu phân biệt). Instruction cố định chỉ còn vai trò: (a) augmentation nhẹ khi train, (b) **mỏ neo phân phối** để khớp text Stage-1 (bỏ hẳn → OOD). KHÔNG phải đóng góp.
- → Mô tả "tăng cường InstructBLIP" là **overclaim** (chính là P2.2 mà Devil's Advocate flag). Cái làm query THẬT SỰ phụ thuộc input là **`user_conditioned`**, không phải instruction.

### Về việc GIỮ SỐ CŨ (quan trọng cho tính trung thực)
- Code hiện chạy = module **HF InstructBLIP Q-Former** nhưng với instruction cố định ⇒ **về mặt chức năng ≈ BLIP-2 Q-Former + task-token cố định**. Nên mô tả là "plain Q-Former" **với số cũ là chấp nhận được** (instruction không mang tín hiệu per-sample).
- ⏳ **Kế hoạch:** train lại bản **plain Q-Former** (bỏ hẳn nhánh instruction-text) để code khớp 100% với narrative. Khi có số mới → cập nhật; nếu số đổi không đáng kể thì giữ nguyên kết luận. **Chưa làm bây giờ.**

### Việc bỏ InstructBLIP (giao cho `/ars-revision`, song song mục 0.5)
1. **ch3 `sec:instructblip-aug` (§ Tăng cường Q-Former theo InstructBLIP + 3 tiểu mục co-che/thich-nghi/ket-qua): XOÁ nguyên section.** Sửa `\ref{sec:instructblip-aug}` (ch3 dòng ~14, ch2 dòng ~303).
2. **ch3 §3.3.1 (dòng ~112-138):** đổi "Q-Former theo InstructBLIP" → **"Q-Former (BLIP-2)"**; bỏ ý "self-attention với token chỉ dẫn"; **bỏ đoạn "Quan sát vai trò thay đổi giữa hai miền" + "đa dạng hóa chỉ dẫn có tác dụng"** (dòng 131-138 — đó là lập luận instruction-conditioning).
3. **ch2 `ssec:instructblip` (dòng 286-303): XOÁ** (hoặc rút còn 1 câu trong nền BLIP-2). Bỏ InstructBLIP ở bảng so sánh (dòng 236) và các câu "chưa tăng cường InstructBLIP / conditioning theo chỉ dẫn" (212, 331). **GIỮ nền BLIP-2/Q-Former.**
4. **ch1 (dòng 80, 99, 108, 128, 143, 166, 171): bỏ InstructBLIP như "hướng tăng cường".** Đóng góp query-conditioning = **`user_conditioned`**, không phải InstructBLIP.
5. **GIỮ NGUYÊN** `ssec:backbone-instruct` (ch3 § "Backbone LLM đã tinh chỉnh theo chỉ dẫn") — đây nói về **Vicuna là LLM instruction-tuned**, KHÁC hoàn toàn InstructBLIP Q-Former. Đừng đụng.
6. **references.bib:** sau khi bỏ, nếu `cite{instructblip2023}` không còn chỗ dùng → xoá entry; nếu còn giữ 1 câu nền BLIP-2 thì để lại. Chạy `/ars-citation-check` để bắt ref mồ côi.
7. **Ablation:** bỏ mọi hàng/ghi chú "varied vs fixed instruction"; nếu muốn, để 1 câu footnote "đã thử instruction biến thiên, không cải thiện".

### Method SẠCH sau cả 0.5 + 0.6
`MF teacher → cầu nối **plain Q-Former (BLIP-2)** (learned queries + cross-attention vào CF, chỉ soft-token) → **user_conditioned** (lever AUC) → **align_rank_loss** (lever uAUC book)`. Không CoRA ΔW, không InstructBLIP.

---

## 1. NHỮNG THAY ĐỔI KIẾN TRÚC TRONG NHÁNH (vật liệu cho báo cáo kiến trúc)

Đây là các **đóng góp so với vanilla Q-Former** (mỗi mục kèm file + ý nghĩa). Nhóm theo vai trò.

### A. Khung kiến trúc SigLLM (cầu nối CF→LLM bằng Q-Former)
- **Pipeline 3 giai đoạn** (commit a01aa60, 8a1d7ce, 4c93515): 
  - *Stage-0 MF* → sinh embedding user/item (frozen teacher).
  - *Stage-1 biểu diễn* (`train_qformer_stage1_representation.py`): BLIP-2 ITC+ITM+ITG **+ item–item (ii) + user–item (ui) contrastive** (bổ sung riêng cho gợi ý, commit 120f8b6).
  - *Stage-2 sinh* (`train_qformer_stage2_generative.py`): tiền huấn luyện caption kiểu BLIP-2, **LLM đóng băng**.
  - *Stage-3 hai bước* (`..._step1_lora.py` / `..._step2_cie.py`): **Step-1** train LoRA trên prompt text-only (không CF); **Step-2 (CIE)** mở Q-Former+projection, **đóng băng LoRA** → tách bạch "ngôn ngữ" (LoRA) và "collaborative" (Q-Former/proj).
- **HFQFormerAdapter** (`src/sigllm/models/q_former/hf_qformer_adapter.py`, commit 14d8599, b0d61db): cầu nối **plain Q-Former (BLIP-2)** — learned queries + cross-attention vào CF. ⚠️ Code hiện *instantiate* module HF InstructBLIP Q-Former nhưng chạy với **instruction cố định** ⇒ về chức năng = BLIP-2 Q-Former (xem mục 0.6 — luận mô tả là plain Q-Former, KHÔNG InstructBLIP). text branch khởi tạo từ **bert-base-uncased**, `num_layers=4` (giảm 8→4, commit cfd6418), `num_queries=8`, `d_model=768`.
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
Giao cho nó theo THỨ TỰ (realign method TRƯỚC, rồi mới điền số — điền số vào method sai là vô nghĩa):
1. **REALIGN METHOD — LÀM ĐẦU TIÊN (2 việc song song):**
   - **(0.5) Bỏ CoRA ΔW** (phương án (a) đã chốt): gỡ đường ΔW + `eq:cora-delta` + §on-dinh, bỏ §sec:cora, hạ CoRA về related-work/baseline (ch2), bỏ ablation 3-chế-độ; sửa `<CFTokens>`.
   - **(0.6) Bỏ InstructBLIP** (đã chốt): xoá `sec:instructblip-aug` (ch3) + `ssec:instructblip` (ch2), đổi "Q-Former theo InstructBLIP" → "plain Q-Former (BLIP-2)", bỏ InstructBLIP khỏi ch1 + bảng so sánh; GIỮ `ssec:backbone-instruct` (Vicuna). Giữ số cũ.
   - Rồi **thêm 2 mục method mới:** `user_conditioned` + `align_rank_loss` (mục 1.B) — đây mới là đóng góp thật.
2. **P0.1** — điền `tab:ket-qua-chinh` (số ở mục 2) + viết phân tích §5.2. **KHÔNG chỉnh claim hồi tố** — báo cáo trung thực (ta vượt cả 2 metric ML-1M; book vượt CoLLM-MF/BinLLM CoRA-protocol).
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
| Roadmap / việc | Cần gì | Trạng thái |
|---|---|---|
| **Realign 0.5** bỏ CoRA ΔW | sửa prose (viết) | ✅ đã chốt (a) → `/ars-revision` |
| **Realign 0.6** bỏ InstructBLIP → plain Q-Former | sửa prose (viết) | ✅ đã chốt, giữ số cũ → `/ars-revision` |
| **P0.1** điền bảng + §5.2 | số thực | ✅ có (mục 2) → `/ars-revision` |
| **P2.3** active vs sparse | phân tích nhóm user | ✅ có = warm/cold (mục 1.E) |
| **§4.3** nghịch lý AUC/uAUC | bằng chứng | ✅ warm 0.624 / cold 0.526 |
| **P1.4** ablation attention-MLP | train thêm 1 lượt | ⏳ chưa — session train |
| Train lại **plain Q-Former** (bỏ instruction-text) | train lại pipeline | ⏳ dự định — để code khớp narrative 0.6; giữ số cũ tới khi có |
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

QUAN TRỌNG — method cũ SAI ở 2 chỗ, phải realign TRƯỚC khi điền số:
- (mục 0.5) Bỏ hẳn CoRA ΔW — code chỉ dùng soft-token, không có tiêm trọng số. (đã chốt (a))
- (mục 0.6) Bỏ InstructBLIP — instruction thực tế là cố định ⇒ mô tả là plain Q-Former (BLIP-2). GIỮ SỐ CŨ.
Method đúng = MF → plain Q-Former (soft-token) → user_conditioned → align_rank.
Bắt đầu /ars-revision bằng REALIGN (0.5 + 0.6), rồi mới điền số P0.1.
```
