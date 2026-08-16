# Dàn ý chi tiết — Luận văn thạc sĩ CoQLLM
> Sản phẩm của phiên `/ars-outline`. Nguồn: `thesis/01-plan/chapter-plan.md`.
> Tác giả: Bùi Nguyên Quốc Bảo (24C12002), HCMUS/FIT, Hệ thống thông tin.
> Đề tài: *Nâng cao hiệu quả hệ thống tư vấn dùng LLM kết hợp kỹ thuật căn chỉnh đa phương thức thông tin cộng tác.*

---

## Bảng phân bổ tổng thể

| Ch | Tiêu đề | Trang dự kiến | Từ ước lượng |
|----|---------|:---:|:---:|
| 1 | Giới thiệu | 6–8 | 2 000–2 500 |
| 2 | Các công trình liên quan | 10–14 | 3 500–4 500 |
| 3 | Phương pháp đề xuất | 14–18 | 4 500–6 000 |
| 4 | Thực nghiệm & Kết quả | 10–15 | 3 500–5 000 |
| 5 | Kết luận & Hướng phát triển | 4–6 | 1 200–1 800 |
| **Tổng nội dung** | | **44–61** | **~15 000–19 500** |

> Thêm ~10–15 trang cho: bìa, lời cam đoan, lời cảm ơn, danh mục hình/bảng/viết tắt,
> tài liệu tham khảo, phụ lục → **tổng thực tế 55–75 trang** ✅ trong chuẩn 50–100.

---

## Chương 1 — Giới thiệu (6–8 trang)

**Luận điểm cốt lõi:** Q-Former nối CF↔LLM tuy hứa hẹn nhưng chưa được đánh giá dưới
AUC/uAUC và chưa được tăng cường → luận văn lấp trống đó.

**Câu cuối chương dẫn sang Ch2:** "Để hiểu tại sao cầu nối Q-Former đáng được chọn và tăng
cường, chương tiếp theo phân tích các công trình liên quan theo bốn nhánh."

### 1.1 Bối cảnh và động lực (1–1.5 trang)
- Hệ thống gợi ý (RecSys) trong thương mại điện tử, phát trực tuyến
- CF truyền thống (MF, LightGCN): hiệu quả nhưng **sparse, cold-start, thiếu ngữ nghĩa**
- Sự trỗi dậy của LLM: hai hướng — *LLMs-as-RS* vs *LLM-enhanced*
- Luận văn chọn hướng **LLMs-as-RS**

*Evidence:* [koren2009], [lightgcn2020], khảo sát tổng quan LLM-Rec

### 1.2 Vấn đề: semantic gap CF ↔ LLM (1–1.5 trang)
- **Linchpin:** lịch sử tương tác nhiều items → vượt context length → phải đưa CF vào dạng
  **vector**, không text
- CoLLM dùng **MLP một lớp** để chiếu CF vector vào không gian LLM → đơn giản, không đủ
  nắm quan hệ phi tuyến

*Evidence:* [collm2023]

### 1.3 Khoảng trống nghiên cứu (0.5–1 trang)

> "Mặc dù Q-Former đã được dùng để nối thông tin cộng tác với LLM (ILM), hướng này vẫn
> chưa được đánh giá dưới giao thức CTR với AUC/uAUC như dòng CoLLM, và chưa được khảo
> sát khi tăng cường bằng các kỹ thuật hiện đại như InstructBLIP, biểu diễn đa token và
> CoRA. Luận văn này lấp khoảng trống đó."

*Evidence:* [ILM] thiếu báo cáo AUC/uAUC; [collm2023] giao thức CTR làm chuẩn

### 1.4 Mục tiêu và đóng góp (1 trang)
*(Đóng góp Bản A — không dùng nhãn A/B/C/F)*

1. Thích nghi và tăng cường cầu nối **Q-Former** thay lớp MLP tuyến tính của CoLLM — cùng
   các lựa chọn thiết kế (đa token, backbone Instruct, tách projector user/item) và khảo
   sát **InstructBLIP** như hướng tăng cường chính.
2. Đánh giá có hệ thống dưới giao thức CTR với **AUC/uAUC** — điều ILM chưa báo cáo —
   đối chiếu trực tiếp với CoLLM.

### 1.5 Bố cục luận văn (0.5 trang)
- Mô tả ngắn từng chương theo luồng: Ch2 → Ch3 → Ch4 → Ch5

---

## Chương 2 — Các công trình liên quan (10–14 trang)

**Luận điểm cốt lõi:** Trong các cách đưa CF vào LLM, Q-Former trích chọn đặc trưng có
kiểm soát (hơn MLP), nhưng chưa được kiểm chứng/tăng cường → đáng nghiên cứu sâu.

### 2.1 Lọc cộng tác truyền thống (2–2.5 trang)
- MF: biểu diễn user/item embedding; BPR objective; scalable
- LightGCN: graph-based, mở rộng GNN
- **Hạn chế:** sparse, cold-start, thiếu ngữ nghĩa văn bản
- *→ Dẫn:* cần kết hợp LLM để bổ sung ngữ nghĩa

*Evidence:* [koren2009], [lightgcn2020], [bpr2009]

### 2.2 LLM cho hệ thống gợi ý (3–4 trang)
- **LLMs-as-RS:** P5, BIGRec, ChatRec — gợi ý qua sinh văn bản; giải cold-start
- **LLM-enhanced:** LLM như encoder để enrichment
- **Linchpin:** history items → text quá dài → vượt context length → cần biểu diễn CF
  thành vector compact
- *→ Dẫn:* cần cơ chế "cầu nối" CF vector ↔ LLM

*Evidence:* [collm2023], P5, BIGRec, LLMRec surveys

### 2.3 Các cách nối CF ↔ LLM qua vector (4–5 trang)

| Phương pháp | Cơ chế | Điểm phản biện |
|-------------|--------|----------------|
| **CoLLM** [collm2023] | MLP một lớp chiếu CF→LLM | Tuyến tính; không phi tuyến; không selective |
| **SellaRec** | Contrastive alignment | Backbone Qwen khác → so sánh hạn chế |
| **BinLLM** | Mã hoá nhị phân → text-like | Mất mát thông tin khi binarize |
| **CoRA** [cora2024] | Tiêm CF vào trọng số LLM | Bỏ soft-token → không tận dụng Q-Former đã align |
| **ILM** | Q-Former bridge | Trích chọn có kiểm soát; **nhưng chưa báo AUC/uAUC** |

- *→ Dẫn:* Q-Former của ILM hứa hẹn nhất về cơ chế, nhưng thiếu kiểm chứng + tăng cường

*Evidence:* [collm2023], [cora2024], [ILM], [BinLLM], [SellaRec]

### 2.4 Nền tảng Q-Former: BLIP-2 và InstructBLIP (2–3 trang)
- **BLIP-2** [blip2_2023]: learned query tokens; cross-attention selective trong domain ảnh-ngôn ngữ
- **InstructBLIP** [instructblip2023]: instruction-conditioned Q-Former; queries phụ thuộc
  instruction → task-specific extraction
- **Tại sao mượn sang RecSys:** cơ chế trích chọn có kiểm soát; conditioning bằng instruction
- Flamingo [flamingo2022]: stability technique (tanh gate)

**Câu kết Chương 2 (CHỐT):**

> "Q-Former nổi lên như cầu nối hứa hẹn: nhờ cross-attention với tập truy vấn học được, nó
> trích chọn và giữ lại đặc trưng cộng tác có kiểm soát thay vì nén thô như MLP của CoLLM.
> Tuy nhiên, hướng này chưa được kiểm chứng dưới giao thức AUC/uAUC, và vẫn còn nhiều
> hướng tăng cường chưa khai thác — InstructBLIP, biểu diễn đa token, tiêm trọng số."

*Evidence:* [blip2_2023], [instructblip2023], [flamingo2022]

---

## Chương 3 — Phương pháp đề xuất (14–18 trang)

**Luận điểm cốt lõi:** Thay MLP của CoLLM bằng cầu nối Q-Former — vì MLP một lớp không
đủ căn chỉnh CF vector vào không gian LLM.

**⚠️ Ràng buộc bắt buộc:**
- KHÔNG dùng nhãn A/B/C/F hay bảng "năm đóng góp"
- CoRA = baseline so sánh ở Ch2/Ch4, KHÔNG phải đóng góp Ch3
- InstructBLIP = mục lớn để ngỏ kết quả (posture C)
- Nguồn tham khảo nội dung: `thesis/02-chapters/ch3-phuong-phap.tex` (dùng nội dung,
  bỏ khung cũ A/B/C/F)

### 3.1 Tổng quan kiến trúc (2 trang)
- Bài toán CTR: binary "Có/Không"; đánh giá AUC/uAUC
- **3 khối nối tiếp:** MF đông cứng → Q-Former bridge → LLM đông cứng + LoRA
- **Hình 3.1** — sơ đồ kiến trúc tổng quan *(hình tạm hiện có; cần vẽ chính thức)*
- Đặc điểm: chỉ Q-Former + lớp chiếu + LoRA được cập nhật; MF và LLM đông cứng

*Evidence:* đối chiếu [collm2023] MLP design

### 3.2 Khối Phân rã ma trận (1.5 trang)
- BPR objective; huấn luyện tách rời; sau đó đông cứng
- 3 loại tín hiệu: e_u, e_i, lịch sử item embeddings
- Lý do tách rời: cô lập chất lượng CF → phân tích nút thắt

*Evidence:* [bpr2009]

### 3.3 Cầu nối Q-Former — lõi phương pháp (4–5 trang)
- Kiến trúc: self-attention (queries + instruction tokens) + cross-attention (từ CF embeddings)
- Output: **Z** = cf_q ∈ ℝ^{B×Q×d_model}, Q=16 queries mặc định
- **Lý do chọn Q-Former thay MLP:** MLP tuyến tính không nắm phi tuyến; Q-Former selective
  trích chọn có kiểm soát ← trục lập luận chính
- Vai trò thay đổi so với domain ảnh: input CF nhỏ → self-attention là cơ chế chính;
  cross-attention nạp CF
- **2 đường đưa Z vào LLM:**
  - *Soft token path:* chiếu Z → embedding LLM, chèn tại `<CFTokens>`
  - *Weight injection path:* Z → delta weight; so sánh thực nghiệm tại Ch4

*Evidence:* [blip2_2023], [instructblip2023]

### 3.4 Các lựa chọn thiết kế (2 trang)
*(nêu nhẹ — không phải đóng góp chính)*
- **Biểu diễn đa token CF:** multi-token thay single-token; cải thiện nhỏ
- **Backbone Instruct + ChatML:** Qwen2-7B-Instruct thay Base; prior instruction-following
  cho "Có/Không"; thay backbone kéo theo thay định dạng + điều chỉnh tokenizer
- **Tách projector user/item:** user-factor và item-factor khác hình học → map riêng

*Evidence:* [qwen2]; ablation

### 3.5 InstructBLIP như hướng tăng cường chính (2–3 trang)
*(để ngỏ kết quả — posture C)*
- Cơ chế: instruction-conditioned queries; self-attention điều hướng trích xuất theo tác vụ
- Động lực: Q-Former cơ sở dùng fixed queries → không phân biệt task; InstructBLIP → task-specific
- Cài đặt: instruction ở lớp self-attention; instruction templates cho CTR
- **⚠️ Kết quả tổng hợp tại Chương 4; chưa claim cải thiện trong mục này**

*Evidence:* [instructblip2023]

### 3.6 Quy trình huấn luyện ba giai đoạn (2–3 trang)
- **Stage 1 — Contrastive pre-training:** ITC/ITM/ITG (item-text) + item-item + user-item
  - Bài toán cân bằng: ML-1M có ~888k user-item nhưng chỉ ~3k item-text (~0.34%)
    → lô 64 gần như không đủ 2 item-text samples → ITC/ITM/ITG hiếm khi train
  - Giải pháp: giới hạn user-item ở 20k (nâng tỉ lệ ~13%) + giảm trọng số loss
    user-item (1.0 → 0.5)
- **Stage 2 — Generative pre-training:** Q-Former + projector → soft token LLM hiểu
- **Stage 3 — CTR fine-tuning (giao thức CoLLM):**
  - Step 1: LoRA cơ sở, chỉ text, không CF → baseline
  - Step 2: Full model với CF, khởi từ Step 1 checkpoint
- **Ổn định (từ Flamingo):** tanh gate init 0; P_up không init 0; L2-norm queries; fp32
- **Hạn chế chi phí:** cao hơn MLP; báo cáo số tham số + thời gian; KHÔNG claim "không đáng kể"

*Evidence:* [collm2023], [blip2_2023], [lora2021], [flamingo2022]

---

## Chương 4 — Thực nghiệm & Kết quả (10–15 trang)

**Thiết kế đầy đủ; số điền sau khi train xong (posture C).**

### 4.1 Dữ liệu và môi trường (2–3 trang)
- **ML-1M:** ~888k interactions, ~6k users, ~3k movies; phân chia theo giao thức CoLLM
- **De-scope rõ ràng:** Amazon Books → hướng phát triển (không hứa làm ở luận văn này)
- **Baseline:**
  - **Chính:** CoLLM [collm2023] — cùng dataset, giao thức, metric
  - **Đối chiếu:** CoRA [cora2024], BinLLM, SellaRec (số báo cáo; caveat: khác backbone)
  - ILM — không báo AUC/uAUC; nêu ở phần thảo luận giao thức
- Cài đặt: GPU, batch size, lr, epoch, seed

### 4.2 Metric (0.5–1 trang)
- **Chính:** AUC (global ranking), uAUC (per-user ranking)
- **Phụ:** Accuracy, F1; Precision@K, NDCG@K
- **Semantic gap:** cosine similarity CF embedding vs LLM token embedding trước/sau alignment

### 4.3 Kết quả chính (2–3 trang)
- **Bảng 4.1** — CoQLLM vs CoLLM vs baseline theo AUC/uAUC/Acc/F1
- **[Số TBD — điền khi train xong]**
- Phân tích AUC và uAUC riêng; nếu AUC↑/uAUC≈ → giải thích phân kỳ bản chất uAUC

### 4.4 Phân tích ablation (2–3 trang)
- **Ablation chính:** có/không soft token → đo đóng góp toàn bộ CF bridge
- **Ablation lựa chọn thiết kế:** đa token vs single; Instruct vs Base; tách vs chung projector
- **Ablation InstructBLIP:** bật/tắt riêng
  - ⚠️ Nếu chưa có kết quả → ghi rõ "cần thêm thực nghiệm; là hướng phát triển"
- **Bảng 4.2** — ablation matrix

### 4.5 Phân tích bổ sung (1–2 trang)
- Cosine similarity trước/sau alignment → minh họa semantic gap thu hẹp
- Case analysis: cold-start user vs active user
- Phân tích lỗi

### 4.6 Thảo luận (1–2 trang)
- Đối chiếu ILM về giao thức
- Điểm mạnh: cơ chế selective hơn CoLLM; đánh giá hệ thống hơn ILM
- **Hạn chế thực nghiệm:** 1 dataset, 1 seed, ablation chưa tách InstructBLIP

---

## Chương 5 — Kết luận & Hướng phát triển (4–6 trang)

### 5.1 Tóm tắt và kết luận (1.5–2 trang)
- Nhắc lại research gap Ch1 → đóng góp đã thực hiện
- Đóng góp: (1) Q-Former bridge; (2) đánh giá AUC/uAUC vs CoLLM
- Trả lời câu hỏi nghiên cứu từ Ch1
- Kết quả chính [điền sau train]

### 5.2 Hạn chế (1 trang)
1. Chi phí tính toán cao hơn MLP
2. Mới chỉ ML-1M; chưa Amazon Books
3. Kết quả thực nghiệm một phần [điền]
4. Ablation chưa tách riêng InstructBLIP
5. 1 seed; chưa có confidence interval

### 5.3 Hướng phát triển (1.5–2 trang)
- Ablation InstructBLIP riêng → chốt đóng góp
- Bổ sung Amazon Books
- Nhiều seed → mean±std
- Cải thiện uAUC: dense CF pretraining, per-user loss
- Khám phá CoRA trong pipeline (weight injection + soft token song song)

---

## Evidence Map — Theo câu lập luận chính

| Câu lập luận | Loại bằng chứng | Citation key |
|---|---|---|
| CF truyền thống thiếu ngữ nghĩa | sparse/cold-start analysis | [koren2009], [lightgcn2020] |
| LLMs-as-RS bị giới hạn context length | History tokens quá dài | [collm2023] |
| MLP một lớp không đủ cho alignment | Không phi tuyến; không cross-attention | [collm2023] ablation |
| Q-Former trích chọn có kiểm soát | Cross-attention + learned queries | [blip2_2023], [instructblip2023] |
| ILM thiếu giao thức AUC/uAUC | Không báo CTR metric | [ILM] |
| InstructBLIP cải thiện instruction-following | Task-specific query conditioning | [instructblip2023] |
| CoRA bỏ soft-token → bỏ sót | Weight-only injection | [cora2024] |
| Flamingo tanh gate → ổn định | Zero-init → Δ=0 ban đầu | [flamingo2022] |
| BPR cho MF training | Pairwise ranking objective | [bpr2009] |
| LoRA parameter-efficient | Bảo toàn LLM knowledge | [lora2021] |
| Stage 1 imbalance → ITC/ITM/ITG ít train | 0.34% item-text trong lô | phân tích nội bộ ML-1M |
| InstructBLIP kết quả | TBD — posture C | [thực nghiệm TBD] |

---

## Điểm yếu cần vá khi viết (từ Bước 3 plan)

| # | Rủi ro | Chiến lược xử lý |
|---|--------|-----------------|
| 🔴 | **Novelty vs ILM** (cùng Q-Former) | Ch1/Ch2 nhấn giao thức AUC/uAUC (ILM không có); phân tích InstructBLIP; tư thế A+C |
| 🟠 | **Bằng chứng mỏng** (1 dataset, 1 seed) | Ch4 thừa nhận ở 4.6; Ch5 liệt kê hạn chế; KHÔNG claim SOTA |
| 🟡 | **Lệch đề cương** (Amazon Books hứa không làm) | Ch1.4 khung lại phạm vi; Amazon Books = hướng phát triển |

---

## Luồng chuyển đổi sang bước tiếp

```
outline.md (file này)
    ↓
/ars-full thesis/01-plan/outline.md   ← viết toàn bộ bản thảo
    hoặc
Viết trực tiếp LaTeX từng chương theo dàn ý này
    + Dùng ch3-phuong-phap.tex làm nguồn tham khảo nội dung (bỏ khung A/B/C/F)
```
