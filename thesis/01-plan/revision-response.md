# Response to Reviewers — CoQLLM (vòng revision, /ars-revision)

> Ngày: 2026-08-11. Mode: `academic-paper` → `revision` (fidelity, patch-contained).
> Nguồn số liệu: `docs/tong-ket-findings-va-ket-qua.md`, `docs/ket-qua-vicuna-ml1m.md`, config nhánh SOTA
> `feat/user-conditioned-queries-v2-book`. Roadmap: `thesis/01-plan/revision-roadmap.md`.
> Nguyên tắc: **không bịa số** (Anti-Pattern #5), **không scope-creep** (chỉ chạm khối liên quan),
> **không sửa claim hồi tố** (báo cáo trung thực).
> **Build đã xác minh:** biber + pdflatex 2 pass → 0 lỗi, 0 undefined ref/cite, PDF 52 trang.

---

## 0. REALIGN METHOD KHỎI CoRA — phương án (a), làm TRƯỚC (điều kiện tiên quyết cho mọi số liệu)

**Vấn đề:** Luận cũ mô tả CoQLLM có **hai đường** đưa CF vào LLM (soft token + tiêm trọng số
ΔW kiểu CoRA), trong khi nhánh SOTA thực tế **chỉ có soft token** + `user_conditioned` +
`align_rank_loss`. Điền số của model thực vào method sai sẽ khiến luận mâu thuẫn code/kết quả.

**Đã làm (phương án (a) — bỏ hẳn ΔW):**

| Vị trí | Thay đổi |
|---|---|
| ch3 §`ssec:hai-duong` | Đổi tiêu đề → *"Đưa tín hiệu CF vào LLM qua soft token"*; bỏ đoạn "Đường tiêm trọng số", **xóa `eq:cora-delta`**; mô tả 1 đường soft token, nêu rõ khác biệt với CoRA (CoRA sinh ΔW; CoQLLM giữ LLM đông cứng). |
| ch3 §`ssec:on-dinh` | **Xóa nguyên mục** (4 kỹ thuật ổn định cho ΔW — không còn liên quan). |
| ch3 (mới) `ssec:user-conditioned` | **Thêm mục method**: query-shift `Q̃ = Q + user_proj(e_u)`, zero-init (`eq:user-cond`). Lever AUC. |
| ch3 (mới) `ssec:align-rank` | **Thêm mục method**: BPR per-user trên soft token CF đã align qua `align_rank_head` zero-init (`eq:align-rank`). Lever uAUC. |
| ch3 hình + §đặc điểm + Bước 2 + chi phí + kết chương | Bỏ "bộ tiêm trọng số"; nêu 1 đường soft token; ghi rõ Step-2 đông cứng LoRA + kích hoạt 2 lever + chọn ckpt theo uAUC. |
| ch3 ký hiệu | `<CFTokens>` → `<rec_soft_token>` + `<ItemIDList>` + `<TargetItemID>` (khớp code). |
| ch2 §`ssec:cora-rel` | Hạ CoRA về **hướng thiết kế khác + baseline** (CoRA-MF trên Amazon-Book); bỏ câu "CoQLLM hiện thực cơ chế CoRA như một trong hai đường". Bỏ cross-ref `\ref{ssec:hai-duong}`. |
| ch2 §`ssec:flamingo` | Giữ Flamingo làm nền; đổi liên hệ "áp dụng cho cả hai đường ΔW" → **nguyên tắc zero-init** dùng cho user-shift + align-rank head. |
| ch4 `tab:ablation` | Bỏ 3-chế-độ {soft/ΔW/cả-hai}; thay bằng ablation THẬT (vanilla→+user_conditioned; book +user_conditioned→+align_rank). |
| ch5 future-work | Bỏ "kết hợp CoRA + soft token trong pipeline thống nhất"; đưa ΔW về **hướng mở chưa hiện thực** (bổ sung, không phải lõi). |
| references.bib | Giữ `cite{cora2024}` (related work + baseline). **Thêm `vicuna2023`** (đang bị thiếu). |

**Lưu ý:** file chết `ch3-phuong-phap.tex` (không `\input`) vẫn còn `sec:cora`/`<CFTokens>` — **không
biên dịch nên không ảnh hưởng build**; để nguyên tránh nhầm lẫn. Nên xóa file này ở một pass dọn dẹp.

---

## 1. Realign phụ (bắt buộc để số liệu trung thực) — backbone + dataset

Số SOTA đến từ **Vicuna-7B-v1.5** trên **2 dataset**, nhưng luận cũ viết **Qwen2-7B-Instruct /
ML-1M-only**. Đã thống nhất theo nhánh SOTA (xác minh từ `configs/config*.yaml`):

- **Backbone Qwen2-Instruct + ChatML → Vicuna-7B-v1.5 + USER/ASSISTANT** (ch2/ch3/ch4/ch5).
- **Amazon-Book**: từ "future work" → **dataset thứ hai chính thức** (ch1 mục tiêu/bố cục, ch4 dataset/baseline/kết quả, ch5).
- **Config đã đồng bộ với code:** Q-Former `num_queries 16→8`, `num_layers 2→4`, MF `d 64→256`, LoRA `r=8/α=16 [q,v]`, chọn ckpt theo **uAUC**.
- Mục "Backbone Instruct vs Base" (Qwen-specific) → reframe sang Vicuna (instruction-tuned), giữ luận điểm "LoRA không phải nút thắt" (bằng chứng r=16 overfit trong config).

---

## 2. Đóng các roadmap item

### ✅ P0.1 (Critical, 5/5 reviewer) — điền bảng kết quả + phân tích §5.2
- **ch4 `tab:ket-qua-chinh` (ML-1M)**: MF 0.6482/0.6361 · CoLLM-MF 0.7295/0.6875 · CoLLM-DIN 0.7243/0.6897 · BinLLM 0.7425/0.6956 · **CoQLLM vanilla 0.7335/0.7009 · +user_conditioned 0.7475/0.6968** (vượt BinLLM cả 2).
- **ch4 `tab:ket-qua-book` (Amazon-Book, mới)**: CoLLM-MF 0.8021/0.5782 · BinLLM 0.8157/0.5724 · CoRA-MF 0.8179/0.6262 · **CoQLLM Run-1 0.8147/0.5958 · Run-2 (+align_rank) 0.8132/0.6062**.
- **ch5 §5.2** viết lại bằng số thực (4 kết luận: ML-1M vượt SOTA cả 2; book cạnh tranh; 2 lever 2 metric; trần uAUC ở MF).
- **Ghi chú protocol bắt buộc** đã thêm: CoQLLM chọn ckpt theo uAUC vs CoLLM/CoRA theo AUC; book baseline từ CoRA re-run; định vị "vượt dưới giao thức có kiểm soát", không đuổi con số tuyệt đối.
- Bỏ cột Acc/F1 (không có số thực → không bịa).
- **Không sửa claim hồi tố:** báo cáo đúng thực đo (ML-1M thắng cả 2; book thắng CoLLM-MF/BinLLM re-run, dưới CoRA-MF ~0.02 ở uAUC).

### ✅ P2.3 (Moderate, R3) — active vs sparse = warm/cold
- **ch4 §`ssec:user-group`** viết lại + bảng `tab:warm-cold` (mới) cho **cả 2 dataset**:
  ML-1M warm 0.7787/0.7254 vs cold 0.7030/0.5969; Book warm 0.8190/0.6238 vs cold 0.7928/0.5258.
- Kết luận: warm ≈ baseline mạnh; cold ≈ chance → validation cho động lực cold-start.

### ✅ §4.3 nghịch lý AUC/uAUC — củng cố bằng bằng chứng
- **ch4 §`sec:nghich-ly`** thay khung giả định ("nếu…") bằng số warm/cold: uAUC bị chặn trên bởi
  nhúng MF cold (AUC vẫn cao 0.81 → không phải Q-Former yếu). Nối tới lever MF teacher (ch5).

### ⏳ P1.4 (Major, Devil's Advocate) — ablation attention-MLP → **CHƯA train, ghi hạn chế**
- Cần 1 lượt train (session train, không phải session viết). Đã ghi minh bạch là **hạn chế/để ngỏ**
  ở ch4 §`sec:han-che` (item 2), ch4 §ablation ("các ablation chưa thực hiện"), và ch5. **Không bịa số.**

### ⏳ P2.4 (justify LoRA r=8) / Multi-seed — vẫn để ngỏ
- LoRA r=8: lý do "capacity không phải nút thắt" đã có ở ch3 §backbone (bằng chứng r=16 overfit). Multi-seed: hạn chế đã thừa nhận (ch4/ch5 item "một seed").

---

## 3. Việc CÒN LẠI (không thuộc session viết này)

1. **P1.4 attention-MLP**: chạy 1 run Step-2 (MLP+self-attn, no learned queries) → điền hàng ablation. **Đòn phản biện mạnh nhất của DA.**
2. **Ablation movie `align_rank`** + bật/tắt riêng multi-token / tách-projector / InstructBLIP dưới Vicuna → hoàn thiện bảng ablation.
3. **Trainable-param count chính xác**: con số cũ 8.65M (kèm ΔW, arch cũ 2 lớp/16 query) đã **stale** → ch4 hiện mô tả định tính + LoRA `r=8 [q,v]`; **cần đọc lại từ log của run cuối** (Vicuna, 8 query, 4 lớp) rồi điền số chính xác. ⚠️ MF `d`: đã đổi 64→**256** theo config — xác nhận lại với embedding thực tế của MF teacher.
4. **Multi-seed** mean±std nếu có thời gian.
5. **Dọn dẹp**: cân nhắc xóa file chết `ch3-phuong-phap.tex`.
6. Bước tiếp pipeline: `/ars-reviewer` (re-review) → `/ars-citation-check` → `/ars-format-convert` (build final).

---

## 4. Tính toàn vẹn / trung thực

- Mọi số trong bảng đều trích từ nguồn đã ghi (docs + config), **không ước lượng**.
- Khác biệt giao thức (uAUC-selection vs AUC-selection; book re-run variance) **ghi rõ tại bảng**, không giấu.
- Các thành phần chưa đo (attention-MLP, một số ablation) **gắn nhãn hạn chế**, không claim.
- Backbone/dataset/param cập nhật theo code thực; `vicuna2023` đã thêm vào .bib.
