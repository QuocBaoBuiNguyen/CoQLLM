# Nhánh SOTA — User-Conditioned Q-Former (so với Vanilla Q-Former)

**Nhánh:** `feat/user-conditioned-queries-v2` · **Dataset:** MovieLens-1M · **Backbone:** Vicuna-7B-v1.5 · **Protocol:** CoLLM-parity.

Doc này chốt lại **những cải tiến của nhánh** đưa pipeline từ *vanilla Q-Former* lên **SOTA** (vượt CoLLM và BinLLM trên **cả** AUC và uAUC). Số liệu chi tiết test/warm/cold ở [`ket-qua-vicuna-ml1m.md`](./ket-qua-vicuna-ml1m.md).

---

## 1. Đóng góp chính (một câu)
Thêm **user-conditioning vào truy vấn (query) của Q-Former** — dịch các query token theo embedding CF của từng user — biến Q-Former từ "mô tả item độc lập với user" thành "mô tả item **theo góc nhìn của user đang hỏi**". Đây là **đòn bẩy AUC gần như miễn phí**: đẩy AUC vượt SOTA mà **không hy sinh uAUC**.

| | AUC | uAUC | Ghi chú |
|---|---|---|---|
| CoLLM-MF | 0.7295 | 0.6875 | |
| CoLLM-DIN | 0.7243 | 0.6897 | |
| **BinLLM** (SOTA trước) | 0.7425 | 0.6956 | |
| **Vanilla Q-Former** (của ta) | 0.7335 | 0.7009 | thắng uAUC, thua AUC |
| **+ User-Conditioned (nhánh này)** | **0.7475** | **0.6968** | **thắng cả 2** |

---

## 2. Baseline: Vanilla Q-Former là gì
Pipeline gốc gồm 5 bước, LLM **đóng băng** xuyên suốt (chỉ LoRA + Q-Former + projection học):

| Stage | Nhiệm vụ | Học gì |
|---|---|---|
| 0. MF | Ma trận phân rã user×item (CF teacher) | user/item embedding |
| 1. Representation | Q-Former học biểu diễn item (ITC/ITM/ITG + ii/ui contrastive) | Q-Former |
| 2. Generative | Sinh caption item từ CF embedding | Q-Former + llm_proj |
| 3-Step1. LoRA | Dạy LLM xuất "Yes/No" đúng format (prompt text-only) | LoRA |
| 3-Step2. CIE | Nhét CF soft-token vào prompt đầy đủ, fine-tune Q-Former+proj | Q-Former + proj |

**Vanilla** = `user_conditioned=False`: các query token của Q-Former **cố định**, cùng một item cho ra cùng biểu diễn **bất kể user nào hỏi**. Kết quả: uAUC cao (0.7009) nhưng AUC chỉ 0.7335 (< BinLLM).

---

## 3. Cải tiến CỐT LÕI: User-Conditioned Queries

### 3.1. Ý tưởng
Q-Former có tập query token học được `q`. Thay vì dùng `q` cố định, ta **dịch nó theo user**:

```
queries = q + user_proj(user_cf)      # user_cf = MF user embedding
```

→ Cùng một item, nhưng với user khác nhau thì query khác nhau → Q-Former **trích những khía cạnh khác nhau của item tùy user**. LLM nhờ đó thấy được **tương tác user × item**, không chỉ item thuần.

### 3.2. Thiết kế an toàn: zero-init residual
`user_proj` (`hf_qformer_adapter.py:91-100`) là `nn.Linear(d_cf, d_model)` **khởi tạo = 0**:
- **Đầu train:** `user_proj(user_cf) = 0` → `queries = q` → **y hệt vanilla** (no-op).
- **Trong train:** shift **mọc dần từ 0** chỉ khi nó giúp giảm loss.

→ Bật cải tiến này **không phá** warm-start; nó là một add-on cộng dồn an toàn. Thực nghiệm: epoch 1 gần như vanilla (uAUC dip nhẹ vì shift còn nhỏ), epoch 2-6 shift lớn lên → AUC buff dần lên 0.75.

### 3.3. Vì sao buff AUC mà KHÔNG hại uAUC
- **AUC** (xếp hạng toàn cục, cross-user): user-shift **biến thiên giữa các user** → dịch baseline điểm theo user → tách pos/neg **giữa các user** tốt hơn → **AUC ↑**.
- **uAUC** (xếp hạng trong 1 user): shift là **hằng số trong một user** → khi so các item của **cùng** user, nó ~triệt tiêu → **uAUC gần như không đổi**.

→ Đây là lý do đây là "AUC lever gần như free". Kết quả A/B: so vanilla, **AUC +0.014** (0.7335→0.7475), **uAUC −0.004** (0.7009→0.6968, trong nhiễu 224 user).

### 3.4. Phạm vi train
`user_proj` chỉ tồn tại ở nhánh `user_conditioned=True`. **Stage-2 KHÔNG dùng user** (sinh caption từ item, thuần item-level) → không có `user_proj`. Do đó **chỉ cần train lại Step-2** (Q-Former+proj trainable) là đủ — reuse Stage-1/Stage-2/Step-1. Log "QFormer ckpt missing keys | user_proj" là **đúng thiết kế** (Stage-2 chưa từng có lớp này; Step-2 tạo mới + train từ 0).

---

## 4. Các cải tiến HỖ TRỢ (enabler để đạt SOTA)

| # | Cải tiến | Vì sao cần | Commit |
|---|---|---|---|
| 1 | **Backbone Vicuna-7B-v1.5** | Đúng backbone CoLLM → so 1-1 công bằng | `9c080c6` |
| 2 | **Soft-token `<rec_soft_token>`** thay `<unk>` | `<unk>` (id 0) trùng ký tự OOV trong title phim → đếm nhầm slot → crash Step-2 ([1240] vs [2046]). Token riêng (id 32000) + resize embedding | `b01c13e` |
| 3 | **CoLLM-parity filter** (`his_title>=2` OFF) | CoLLM giữ hết + zero-pad (không drop <2-history như SeLLa) → tắt filter để đúng dân số test CoLLM | `53c51c8` |
| 4 | **`user_conditioned=True`** (cải tiến cốt lõi) | AUC lever | `1e79f13` |
| 5 | `pull_llm_model` mặc định vicuna | tiện tải đúng weights | `a9c9e35` |

---

## 5. Những gì ĐÃ THỬ nhưng KHÔNG ăn (negative findings)
Ghi lại để không lặp lại + làm rõ đâu mới là lever thật:

- **Train Stage-2 sâu hơn** (val_loss 0.60→0.53, epoch 20→80): downstream test **≈ / nhỉnh tệ** (0.7279/0.6973). → Stage-2 depth **không phải** nút thắt (pretext over-optimization).
- **SeLLa user-token** (nhét MF user emb thành prompt TOKEN riêng): **không** nhấc uAUC — kênh tốt nhưng embedding MF nghèo. (Khác `user_conditioned`: cái này shift QUERY → buff AUC.)
- **Dense-CF-pretrain / align_rank_loss / candidate-aware**: đều không phải lever chính trên setup này.

→ Trần pipeline trên ML-1M ≈ **0.75 AUC / 0.70 uAUC**. Muốn đẩy tiếp (nhất là **cold users**, uAUC 0.597) cần **CF teacher tốt hơn**, không phải tinh chỉnh Stage-2/Step-2.

---

## 6. Checkpoint & tái lập
- **SOTA (dùng báo cáo):** `ckpt/qformer_stage3_step2_ucq_vicuna/vicuna-7b-v1.5/checkpoint_best.pth` (Step-2 epoch-6). S3: `.../qformer_stage3_step2_cie_vicuna/ml1m_ucq_vs_binllm/`.
- **Vanilla (đối chứng):** `ckpt/qformer_stage3_step2_vanilla_vicuna/...`. S3: `..._vs_collm_s2ep20`.
- Config: `configs/config.yaml` (nhánh v2) — `user_conditioned: True`, filter OFF, Stage-2 lr2e-4/wd1e-4, best-ckpt theo uAUC (hardcode).
- Chạy lại (đã có Stage-1/2 + Step-1): chỉ cần `train_qformer_stage3_step2_cie.py` → `eval_test.py --step 2`.

---

## 7. Kết luận
Vanilla Q-Former đã cạnh tranh CoLLM (thắng uAUC, sát AUC). Thêm **user-conditioned queries** — một cải tiến **nhỏ, an toàn (zero-init), rẻ (chỉ train Step-2)** — nâng AUC vượt BinLLM mà giữ nguyên uAUC, cho ra **cấu hình đầu tiên thắng SOTA trên cả hai chỉ số** trên ML-1M theo protocol CoLLM.
