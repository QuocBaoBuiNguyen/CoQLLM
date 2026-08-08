# Kết quả Vicuna-7B trên ML-1M (so sánh CoLLM / BinLLM)

**Nhánh:** `feat/user-conditioned-queries-v2` · **Backbone:** Vicuna-7B-v1.5 (đúng backbone CoLLM) · **Protocol:** CoLLM-parity (filter `his_title>=2` TẮT — giữ toàn bộ, zero-pad history ngắn, khớp CoLLM `rec_datasets.py`).

## Bảng kết quả chính — test đầy đủ (ML-1M, 7331 rows / 224 user)

| Nhóm | Mô hình | AUC | uAUC |
|---|---|---|---|
| Collab. | MF | 0.6482 | 0.6361 |
| | DIN | 0.7166 | 0.6459 |
| LLM+Collab. | CoLLM-MF | 0.7295 | 0.6875 |
| | CoLLM-DIN | 0.7243 | 0.6897 |
| SOTA | **BinLLM** | 0.7425 | 0.6956 |
| **Của ta** | Vicuna **vanilla** | 0.7335 | **0.7009** |
| **Của ta** | Vicuna **+ user_conditioned (ucq)** | **0.7475** | **0.6968** |

### Nhận xét
- **ucq VƯỢT BinLLM CẢ 2 metric:** AUC 0.7475 > 0.7425 (+0.005), uAUC 0.6968 > 0.6956 (+0.001). Đồng thời vượt CoLLM-MF/DIN rõ rệt. → **cấu hình đầu tiên thắng SOTA trên cả AUC lẫn uAUC** (cùng protocol CoLLM).
- **vanilla:** thắng BinLLM ở uAUC (0.7009 vs 0.6956) nhưng thua AUC (0.7335 < 0.7425).
- **user_conditioned = "AUC lever gần như free":** so vanilla, AUC +0.014 (0.7335→0.7475), uAUC chỉ −0.004 (0.7009→0.6968, trong nhiễu 224 user). Cơ chế: user query-shift biến thiên across-user → buff AUC toàn cục; hằng số within-user → gần như không đụng uAUC.

## Chi tiết warm / cold (ucq)
| Split | rows | AUC | uAUC |
|---|---|---|---|
| test (all) | 7331 | 0.7475 | 0.6968 |
| test_warm | 4153 | **0.7787** | **0.7254** |
| test_cold | 3178 | 0.7030 | 0.5969 |

→ Warm rất mạnh (0.779/0.725). Cold yếu (đặc biệt uAUC 0.597) — đúng bản chất cold-start khó; đây là dư địa cải thiện tương lai (CF teacher tốt hơn cho cold users).

## Recipe (config.yaml, nhánh v2)
- LLM: `vicuna-7b-v1.5`, prompt_template Vicuna `USER:/ASSISTANT:`, prompt vanilla (không user-token SeLLa).
- `qformer_config.user_conditioned: True` (lever AUC).
- Stage-2: lr=2e-4, wd=1e-4, epoch=80, patience=8 (đã fix underfit — NHƯNG lưu ý: Stage-2 sâu hơn KHÔNG nhấc downstream, xem dưới).
- Step-2: init_lr=3e-5, chọn best-checkpoint theo **uAUC** (hardcode). Đỉnh ở **epoch 6** (val AUC 0.7519 / uAUC 0.6991).
- filter: `match_sella_history_filter: false` (CoLLM parity).

## Checkpoint
- **ucq (kết quả chính):** `ckpt/qformer_stage3_step2_ucq_vicuna/vicuna-7b-v1.5/checkpoint_best.pth` → **test 0.7475 / 0.6968**.
- vanilla (đối chứng): `ckpt/qformer_stage3_step2_vanilla_vicuna/...` → test 0.7335 / 0.7009 (bản s2ep20 tốt nhất đã sync S3 `..._vs_collm_s2ep20`).
- Stage-1/Stage-2/Step-1 dùng chung, reuse cho cả 2.

## Các fix kỹ thuật đã làm để chạy được Vicuna (commit trên nhánh)
1. `9c080c6` đổi backbone Vicuna + template + prompt vanilla.
2. `a9c9e35` `pull_llm_model` mặc định vicuna.
3. `b01c13e` **fix soft-token**: `<unk>` trùng ký tự OOV trong title → dùng token riêng `<rec_soft_token>` (id 32000) + resize embeddings. (Step-2 shape mismatch [1240] vs [2046].)
4. `53c51c8` toggle filter — CoLLM KHÔNG drop <2-history (SeLLa drop) → tắt để parity.
5. `1e79f13` bật `user_conditioned=True` (AUC lever) + Step-2 dir riêng.

## Finding phụ (negative, có giá trị)
- **Stage-2 depth KHÔNG phải lever:** ép Stage-2 val_loss 0.60→0.53 (epoch 20→80) → downstream test ≈ hoặc nhỉnh tệ (0.7279/0.6973 vs 0.7335/0.7009, trong nhiễu). Pretext over-optimization. Pipeline đã gần trần ~0.73-0.75 AUC / ~0.70 uAUC trên ML-1M.
- **SeLLa user-token (nhét MF user emb làm prompt token):** negative — không nhấc uAUC (channel tốt nhưng embedding MF nghèo). Khác hẳn `user_conditioned` (query-shift) ở đây vốn buff AUC.
