# Response to Reviewers — Luận văn SigLLM (vòng revision ARS)

> Đầu ra `academic-paper` mode `revision`. Ghi lại các thay đổi đã áp vào bản luận, ánh xạ tới revision-roadmap + 2 quyết định realign method. Cập nhật: 2026-08.
> Build xác nhận sau vòng này: `xelatex + biber`, **0 undefined reference/citation**, **46 trang**.

---

## A. Realign method cho khớp code SOTA (2 mismatch)

### A1 — Bỏ CoRA ΔW (mismatch #1, phương án a) — *đã hoàn tất từ vòng trước*
Code nhánh SOTA chỉ dùng soft-token injection, không có đường tiêm trọng số CoRA. Bản luận đã: gỡ "đường tiêm trọng số" + `eq:cora-delta` + §ổn-định-ΔW khỏi ch3; §hai-duong giờ mô tả 1 đường soft-token; CoRA hạ về related-work (ch2 §cora-rel) + baseline CoRA-MF; thêm 2 mục method thật (`user_conditioned`, `align_rank_loss`) với công thức `eq:user-cond`, `eq:align-rank`. *(Ghi nhận: đã có sẵn khi vào vòng này — xác nhận nhất quán.)*

### A2 — Bỏ InstructBLIP → plain BLIP-2 Q-Former (mismatch #2) — **vòng này**
**Lý do:** Q-Former chạy với instruction task **cố định** (`_build_qformer_instructions`, eval luôn dùng template[0]; 12 template không chứa nội dung item) → conditioning theo mẫu của InstructBLIP bị vô hiệu; mô tả "tăng cường InstructBLIP" là overclaim (P2.2 của Devil's Advocate). Thống nhất mô tả cầu nối là **plain Q-Former (BLIP-2)** toàn luận (giữ số cũ; hợp lệ vì instruction cố định ⇒ chức năng ≈ BLIP-2).

| Chương | Thay đổi |
|---|---|
| ch1 | Bỏ InstructBLIP như "hướng tăng cường" (7 chỗ); khoảng-trống #2 reframe về đòn bẩy thật (user_conditioned + align_rank); Q-Former nêu nguồn gốc chỉ từ BLIP-2 |
| ch2 | Xoá §`ssec:instructblip`; bỏ InstructBLIP khỏi bảng so sánh + câu ILM + trích dẫn kết chương; **giữ** BLIP-2 + Flamingo (zero-init nay motivate cho user_conditioned/align_rank) |
| ch3 | Xoá §`sec:instructblip-aug` (3 tiểu mục); §3.3.1 đổi "Q-Former theo InstructBLIP" → "Q-Former (BLIP-2)", bỏ "self-attention với token chỉ dẫn" + luận điểm "đa dạng hoá chỉ dẫn có tác dụng"; sửa intro + kết-chương |
| ch4 | Bỏ InstructBLIP khỏi ablation-todo + limitations |
| ch5 | Bỏ InstructBLIP khỏi tóm tắt đóng góp + limitations; future-work thay bằng ablation attention-MLP |
| references.bib | Xoá entry `instructblip2023` (mồ côi sau khi bỏ mọi `\cite`) |

**Giữ nguyên** `ssec:backbone-instruct` (Vicuna là LLM instruction-tuned — khác InstructBLIP Q-Former).

---

## B. Ánh xạ revision-roadmap (5-reviewer panel)

| ID | Reviewer | Trạng thái sau vòng |
|---|---|---|
| **P0.1** điền bảng kết quả + §5.2 | 5/5 | ✅ Đã điền `tab:ket-qua-chinh` (ML-1M) + `tab:ket-qua-book` (Amazon-Book) + phân tích ch5 §Kết-quả-đạt-được |
| **P2.3** active vs sparse user | R3 | ✅ §`ssec:user-group` — bảng `tab:warm-cold` (warm/cold cả 2 dataset) |
| **§4.3** nghịch lý AUC/uAUC | — | ✅ Củng cố bằng warm 0.62 / cold 0.53 (Amazon-Book) |
| P0.2/P0.3/P1.1/P1.2/P1.3/P2.1/P2.2 | — | ✅ Đã đóng ở các vòng trước (prose) |
| **P1.4** ablation attention-MLP | Devil's Advocate | ⏳ **CÒN LẠI — cần 1 lượt train.** Đã ghi minh bạch là hạn chế (ch4 §hạn-chế, ch5 future-work). Đòn phản biện mạnh nhất (M2) — nên chạy trước khi bảo vệ nếu có tài nguyên |
| Multi-seed | — | ⏳ Còn lại — 1 seed, đã thừa nhận |
| Đếm lại số tham số huấn luyện | — | ⏳ ch4 §cài-đặt hiện ghi định tính ("vài triệu"); nên điền số chính xác sau |

---

## C. Nguyên tắc trung thực đã giữ
- Không chỉnh claim hồi tố; báo cáo đúng số đo (ML-1M vượt BinLLM cả 2 metric; Amazon-Book vượt CoLLM-MF/BinLLM CoRA-protocol, dưới CoRA-MF).
- Ghi rõ khác biệt protocol: SigLLM chọn checkpoint theo **uAUC**, CoLLM/CoRA theo **AUC** (ghi chú tại bảng kết quả + §hạn-chế).
- Số baseline Amazon-Book lấy từ CoRA re-run; nêu rõ có 2 bảng khác nhau do re-run variance — định vị "vượt baseline dưới protocol có kiểm soát".
- Mọi ablation chưa chạy để **để ngỏ**, không điền số ước lượng.

---

## D. Bước tiếp theo (ngoài vòng revision này)
1. `/ars-reviewer` (re-review) — verify roadmap đã đóng, liệt kê residual.
2. `/ars-citation-check` — dò ref/cite (đã build sạch nhưng nên chạy chính thức).
3. **Session train:** đóng P1.4 (attention-MLP), đếm lại param, (tuỳ) multi-seed, (tuỳ) train lại plain Q-Former để code khớp narrative 100%.
4. `/ars-format-convert` — xuất PDF cuối theo template HCMUS.
