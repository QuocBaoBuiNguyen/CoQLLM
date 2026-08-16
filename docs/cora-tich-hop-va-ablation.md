# CoRA — Bài báo gốc, cách tích hợp trong dự án, và câu hỏi "overengineer?"

> Ghi chú phục vụ luận văn. Trả lời 3 câu hỏi: (1) CoRA gốc làm gì, (2) dự án này
> tích hợp CoRA thế nào và khác bài báo ở đâu, (3) việc dùng **đồng thời** CoRA +
> soft-token Q-Former có thừa (overengineer) không, và cách chứng minh bằng số liệu.
>
> **Nguồn:** `docs/CoRA.pdf` — *"CoRA: Collaborative Information Perception by Large
> Language Model's Weights for Recommendation"*, arXiv:2408.10645.
> **Liên quan:** `docs/cai-tien-va-ket-qua.md` (mục 2.A, 3.1), `collaborative_lora_injector.py`.

---

## 1. CoRA gốc nói gì

**Vấn đề CoRA nhắm tới** — 2 hạn chế của cách "input-space" (như CoLLM):

1. Fine-tune LLM trên dữ liệu rec làm **hỏng world-knowledge** vốn có của LLM.
2. Nhét feature CF vào **prompt** làm **rối ngữ nghĩa** prompt → LLM trả lời kém.

**Giải pháp:** thay vì căn chỉnh CF vào *input space*, CoRA căn chỉnh vào
**parameter space** — biến CF thành **trọng số tăng thêm ΔW** cộng vào weight của
LLM (cảm hứng từ VLoRA bên thị giác).

**Cơ chế:**

- CF model (MF) → `e_u, e_i` → ghép `[e_u, e_i]`.
- **Collaborative query generator** (giống Q-Former BLIP-2): `k` query học được +
  self-attention + **cross-attention** hút CF, rồi **pooling** → một vector `qc`.
- `qc` → ΔW low-rank: `Wc = R(qc·W_FC)·W_proj` với `W_proj` **zero-init** (ổn định)
  → **merge thẳng vào weight** ở **mọi** decoder block, cho **tất cả** loại weight
  `[WQ, WK, WV, WO, FFN]`.
- **Prompt chỉ-text (TALLRec-style), KHÔNG có token CF nào.** ← điểm cốt lõi.
- **Chỉ generator được train** (loss BCE); LLM **đông cứng hoàn toàn, không cả LoRA**.
- Setup ML-1M trùng khít dự án: 33.891 / 10.401 / 7.331; 839 user; 3.256 item.

---

## 2. Dự án tích hợp thế nào — và khác CoRA gốc ở đâu

Dự án **không copy CoRA y nguyên** mà **lai** nó với Q-Former. Khác biệt chính:

| Khía cạnh | CoRA gốc | Dự án này |
|---|---|---|
| Nguồn query | generator train *chỉ bằng BCE* | **Q-Former pretrain riêng** (Stage 1 contrastive + Stage 2 generative) |
| Số query vào ΔW | pool về **1 vector** `qc` | giữ **16 query** (`cf_q [B,16,768]`) → delta |
| Prompt | **chỉ-text, không token CF** | mode `both` → **có cả `<CFTokens>` soft token** |
| LoRA trên LLM | **không** | **có** (r=8, `q_proj/v_proj`) — Step 1 train text-LoRA |
| Vị trí tiêm | mọi block, mọi loại weight | chỉ **`q_proj`, `v_proj`** |
| Ổn định | `W_proj` zero-init | `P_up` zero-init **+ tanh gate zero-init + L2-normalize query**; gắn **sau** LoRA |

→ Tức **Cải tiến A** của dự án = **Q-Former (BLIP-2) + CoRA + soft-token (CoLLM)**,
gộp **3 paradigm**.

**Công thức delta trong code** (`collaborative_lora_injector.py`):

```
delta = scaling · gate · ((x @ Aᵀ) @ B)        # B = P_up(Z), zero-init
scaling = cora_alpha / num_queries              # cora_alpha=4, num_queries=16
```

trong đó `Z = cf_q` là **output của Q-Former** (không phải MF thô). Vì vậy **kể cả
mode `lora_weight` thuần, Q-Former vẫn chạy** và vẫn là bộ mã hóa CF; CoRA chỉ là
**cách đọc** output đó vào LLM.

**Config liên quan:**

```yaml
cf_injection_mode: "both"     # soft_token | lora_weight | both
cora_alpha: 4                 # scaling = cora_alpha / num_queries
cora_target_modules: ${model.lora_config.target_modules}   # q_proj, v_proj
```

---

## 3. "Overengineer?" — dùng đồng thời CoRA + soft-token có thừa không

**Trực giác lo ngại là ĐÚNG về lý thuyết**, và trùng chính lập luận của CoRA:

- CoRA tồn tại **để né input-space token** (vì nó phá ngữ nghĩa). Mode `both`
  **bơm lại** soft token → **phần nào đi ngược động cơ của CoRA**.
- Hai đường (soft token + CoRA) **ăn cùng một `cf_q`** từ Q-Former → **dư thừa
  nguồn tín hiệu**, chỉ khác chỗ đọc ra (input space vs weight space).

**Nhưng** "có thừa hay không" là câu hỏi **thực nghiệm, không phải hùng biện**.
Hiện trạng bằng chứng:

- Đã chứng minh: **CF (both) > text-only** (val AUC ~0.7589 vs ~0.7225 ở Step 1).
- **Chưa** chứng minh: `both` > `soft_token` thuần, hay `both` > `lora_weight` thuần.

→ Chừng nào chưa đo, `both` vẫn là **giả định**, và cảm giác overengineer là hợp lý.
Doc luận cũng đã tự ghi nhận lỗ hổng này ("Chưa có ablation tách riêng từng cải tiến").

### 3.1. Cách chứng minh = chạy Ablation A (config đã hỗ trợ sẵn)

Rerun **chỉ Step 2** từ cùng checkpoint Step 1, đổi `cf_injection_mode`:

```bash
# (1) both — đang chạy, prompt _mt.txt (mặc định)

# (2) soft_token thuần:
python .../train_qformer_stage3_step2_cie.py --cfg-path .../config.yaml \
    --options model.cf_injection_mode=soft_token

# (3) lora_weight thuần — PHẢI dùng prompt chỉ-text (bỏ <CFTokens>):
python .../train_qformer_stage3_step2_cie.py --cfg-path .../config.yaml \
    --options model.cf_injection_mode=lora_weight \
             run.qformer_stage3_step2.prompt_path=/content/CoQLLM/prompts/qformer_prompt_movie_text_only.txt
```

So sánh val/test **AUC + uAUC** của 3 chế độ (× `cora_alpha ∈ {2, 4, 8}`).
**Tái dùng** Stage 1/2 + Step 1 → rẻ.

### 3.2. Hai kịch bản — cả hai đều là kết quả tốt cho luận

- **Nếu `both` thắng rõ** cả hai đơn lẻ → **không phải overengineer**, mà là
  **bổ trợ**: soft token = *bằng chứng CF đã-align mà LLM ĐỌC trong prompt*;
  CoRA = *điều biến per-sample CÁCH LLM tính toán*. Hai tầng khác nhau. Báo cáo
  con số → thiết kế có chính danh.
- **Nếu `both` ≈ chế độ đơn tốt nhất** → **đúng là thừa**; rút về 1 cơ chế. Đây
  cũng là **kết luận mạnh** (Occam: mô hình gọn hơn mà ngang ngửa) → loại phần dư,
  luận sạch hơn.

### 3.3. Luận điểm bào chữa hợp lệ (nếu `both` thắng)

Khác CoLLM, soft token của dự án **đã được Q-Former pretrain để align với text**
(Stage 1/2), nên **ít phá ngữ nghĩa hơn** raw projection mà CoRA chỉ trích. Đó
chính là điều kiện để `both` **cộng hưởng** thay vì xung đột — **giả thuyết phân
biệt công trình này với CoRA gốc**. Nhưng vẫn phải để số liệu nói.

---

## 4. Q-Former (đã căn chỉnh) đi TRƯỚC CoRA — có chính danh không?

**Trực giác đặt ra:** Q-Former được pretrain (Stage 1 contrastive + Stage 2
generative) để làm **soft token có ý nghĩa, align với text**. Việc align đó hợp lý
cho **đường soft-token** (token sống trong không gian embedding của LLM, align text
giúp LLM "đọc" được). Nhưng CoRA cố tình **né input/text-space** — vậy đặt một
Q-Former *được tối ưu để align text* đứng trước CoRA có bị **lệch mục tiêu** không?

### 4.1. Trả lời ngắn: CÓ chính danh — trên 2 cơ sở vững

1. **Bản thân generator của CoRA CHÍNH LÀ một Q-Former.** Bài báo viết nguyên văn:
   *"Similar to the Q-Former in BLIP-2… we initialize k learnable query embeddings
   and input them into N decoder blocks with cross-attention."* Dùng Q-Former làm
   front-end **đúng đặc tả CoRA**. Khác biệt **duy nhất** là *cách huấn luyện* nó.
2. **Pretrain Q-Former = khởi tạo tốt hơn nhiều so với CoRA gốc.** CoRA train
   generator **từ đầu chỉ bằng BCE** trên ML-1M (33k mẫu) — tín hiệu yếu để học một
   ánh xạ CF→biểu diễn-hữu-ích. Stage 1/2 **bơm cấu trúc** (lân cận CF, nền tảng
   ngữ nghĩa) **trước** khi BCE tinh chỉnh. Đây là transfer learning / init tốt hơn
   — có lợi **ngay cả khi** readout cuối là weight-space, nhất là trên dataset bé.

### 4.2. Nhưng có một căng thẳng thật mà trực giác bạn bắt đúng

- Các mục tiêu Stage 1/2 (ITC/ITM/ITG/generative) tối ưu Q-Former để **dự đoán/truy
  hồi TEXT** — **không** cái nào tối ưu nó để **sinh weight delta tốt**. Đường CoRA
  chỉ nhận gradient ở Stage 3 (BCE). Nên biểu diễn `cf_q` **chủ yếu được nắn theo
  text-alignment**, còn readout weight-space phải "dùng tạm".
- Ở mode `both`, **một** `cf_q` phải phục vụ **cả hai** đường. Đó là **thỏa hiệp**:
  biểu diễn lý tưởng cho "token để LLM ĐỌC" và cho "delta điều biến CÁCH LLM tính"
  **chưa chắc giống nhau**.
- **Bằng chứng cụ thể trong code:** injector phải **L2-normalize query trước khi
  sinh delta** (`normalize_queries=True`) đúng vì query do alignment tạo ra có
  **norm lớn/không kiểm soát** (`mean_l2 ≈ 16` trong log), không "tự nhiên" cho việc
  sinh trọng số. Tức code **đã phải bù** cho một phần lệch không gian này.

### 4.3. Độ chính danh phụ thuộc CHẾ ĐỘ — đây là cách kết luận sạch

| Chế độ | Q-Former-align-trước-CoRA có chính danh? |
|---|---|
| `soft_token` | **Hoàn toàn** — align text đúng là việc cần làm (trực giác bạn đúng). |
| `both` | **Có** — front-end dùng chung: alignment phục vụ nửa soft-token, đồng thời là init tốt cho nửa CoRA. |
| `lora_weight` thuần | **Yếu hơn** — pretrain vẫn cho lợi ích *init*, nhưng các loss align-text (ITC/ITM/ITG) là **over-spec** cho readout thuần weight-space. Đây là chỗ nỗi lo "overengineer" cắn mạnh nhất. |

→ Logic gọn: **alignment càng chính danh khi đường soft-token còn được dùng.** Nếu
bỏ hẳn soft-token (CoRA thuần), phần *alignment-cho-text* khó biện minh — lúc đó một
generator **nhẹ hơn** (kiểu CoRA gốc, BCE-only, hoặc chỉ pretrain nhẹ) có thể đủ.

### 4.4. Ablation sắc hơn để trả lời thẳng câu này

Ngoài ablation `soft_token/lora_weight/both` (mục 3.1), thêm một trục **để đo riêng
"alignment có giúp đường CoRA không"**: giữ `cf_injection_mode=lora_weight`, đổi
**front-end**:

- CoRA + Q-Former **pretrain đầy đủ** (Stage 1+2) — hiện tại.
- CoRA + Q-Former **chỉ Stage 1** (bỏ generative), hoặc **không pretrain** (random +
  BCE, giống CoRA gốc).

Nếu *full-pretrain* > *no-pretrain* ở chế độ weight-space → **alignment có giá trị
thật cho CoRA**, không chỉ cho soft token → chính danh được củng cố bằng số liệu.
Nếu ngang nhau → với CoRA thuần, nên **đơn giản hóa front-end**.

---

## Tóm tắt một dòng

Q-Former vẫn là lõi mã hóa CF; CoRA là kênh tiêm vào weight-space. Đặt Q-Former
**đã-align** trước CoRA là **chính danh** (generator của CoRA vốn là Q-Former; pretrain
= init tốt hơn), nhưng phần *align-text* **chỉ thực sự đáng giá khi đường soft-token
còn được dùng** — nên dùng `both` cần ablation chứng minh nó thắng từng cơ chế đơn lẻ;
nếu chạy CoRA thuần thì nên cân nhắc front-end nhẹ hơn.
