---
name: vi-academic-translate
description: >-
  Dịch/viết lại văn bản Anh→Việt theo ĐÚNG văn phong luận văn học thuật (HCMUS/FIT),
  nghe tự nhiên như người Việt viết chứ không phải "tiếng Anh ghép chữ Việt". Dùng khi:
  dịch method/abstract/chương luận văn từ bản tiếng Anh (vd output của skill ARS),
  gỡ văn dịch cứng/"dịch Tây" (calque, bị động máy móc, noun-stack, "Lưu ý về tư thế viết"),
  chuẩn hoá thuật ngữ RecSys/ML, giữ số liệu & cấu trúc học thuật.
  Triggers: dịch sang tiếng Việt học thuật, dịch luận văn, gỡ văn dịch cứng, việt hoá method,
  làm tự nhiên bản dịch, polish tiếng Việt luận văn, translate to Vietnamese academic.
metadata:
  version: 1.0.0
  type: project
---

# Dịch Anh→Việt chuẩn văn phong luận văn

Mục tiêu: tạo **tiếng Việt học thuật thật**, không phải cú pháp tiếng Anh khoác từ tiếng Việt.
Nguồn thường là bản tiếng Anh do skill ARS sinh ra; đầu ra phải khớp văn phong luận văn HCMUS/FIT.

Nguồn chân lý về format/văn phong: `thesis/00-format/` (Thesis_Template + các PDF mẫu đã duyệt).
Đọc lướt mẫu trước khi dịch khối lớn để bắt đúng giọng (ngôi, cách xưng, độ trang trọng).

---

## Pha 0 — Chốt "hợp đồng ngôn ngữ" (language contract)

Chốt trước khi dịch, nêu ngắn gọn 1 dòng rồi làm:

- **Register**: học thuật/luận văn — trang trọng, khách quan, không khẩu ngữ, không marketing.
- **Ngôi**: vô nhân xưng. Ưu tiên "nghiên cứu này / phương pháp đề xuất / mô hình / kết quả cho thấy".
  Dùng **"chúng tôi"** cho hành động của tác giả (thiết kế, đề xuất, huấn luyện) — nhất quán, KHÔNG "tôi/ta/bạn".
- **Glossary**: theo bảng bên dưới — giữ nguyên tiếng Anh vs. dịch, phải nhất quán toàn văn.
- **Bất biến**: số liệu (AUC/uAUC, %, tên nhánh, ký hiệu toán), tên riêng, tên dataset (ML-1M,
  Amazon-Book), tên mô hình (CoLLM, BinLLM, CoRA, Vicuna, Qwen), trích dẫn — KHÔNG đổi.

Chỉ hỏi lại khi lựa chọn ngôi/register làm thay đổi bản chất; còn lại chọn mặc định trên và nói 1 câu.

---

## Pha 1 — Dịch bám sát (直译)

Dịch trung thành, đủ ý, giữ thuật ngữ EN theo glossary. **Chưa** tối ưu độ trôi chảy ở pha này.
Không thêm ý không có trong bản gốc; không tự "diễn giải" mạnh hơn (possibility ≠ guarantee).

---

## Pha 2 — Soi lỗi "dịch Tây" (问题识别)

Rà bản Pha 1, đánh dấu đúng những thứ khiến câu nghe không tự nhiên. Deny-list cụ thể:

- **Noun-stack kiểu Anh** chưa vỡ thành mệnh đề Việt:
  ✗ "mô-đun sinh truy vấn điều kiện hoá theo người dùng"
  ✓ "mô-đun sinh truy vấn có điều kiện theo người dùng" / diễn thành mệnh đề khi cần.
- **Bị động máy móc / cấu trúc "it is ... that"**:
  ✗ "được thực hiện bởi", "nó được cho thấy rằng"  ✓ "do ... thực hiện", "kết quả cho thấy".
- **Calque & transition thừa rải bừa**: "Lưu ý về tư thế viết" (← *A note on writing posture*),
  "hơn nữa/trong khi đó/một cách..." lặp máy móc, "điều này" mơ hồ đầu câu.
- **"the" / mạo từ dịch cứng**: "the model" → "mô hình" (không "cái mô hình").
- **Song song máy móc & classifier thừa** ("các cái", "những sự").
- **Pronoun drift / lẫn mức trang trọng** giữa các đoạn.

Ghi ngắn các chỗ cần sửa; không cần liệt kê từng lỗi vụn.

---

## Pha 3 — Viết lại đúng văn phong luận văn (意译) + soát cuối

- Viết lại tự nhiên theo văn phong học thuật Việt: câu rõ, chủ động khi hợp lý, đập vỡ noun-stack
  thành mệnh đề, bỏ transition/filler thừa.
- **Giữ nguyên** độ chính xác thuật ngữ + số liệu + cấu trúc (heading, danh sách, công thức, bảng).
- Giữ loanword khi giới học thuật Việt vẫn dùng nguyên (embedding, fine-tuning) — đừng ép Việt hoá gượng.
- **Soát cuối (đọc-to)**: pronoun drift, lẫn formality, nhịp câu, số liệu/tên có đổi nhầm không,
  chấm câu và ngắt dòng.

---

## Glossary RecSys/ML (nhất quán toàn văn)

**Giữ nguyên tiếng Anh** (thuật ngữ chuẩn ngành, có thể chú giải lần đầu):
`AUC`, `uAUC`, `embedding`, `soft token`, `Q-Former`, `LoRA`, `fine-tuning`, `prompt`, `token`,
`backbone`, `contrastive`, `align_rank` (tên hàm/loss — giữ y nguyên), `user_conditioned` (tên
cờ/cấu hình — giữ y nguyên), `Stage-1/2/3`, tên mô hình/dataset (Vicuna, Qwen, CoLLM, BinLLM, CoRA,
ML-1M, Amazon-Book).

**Nên dịch** (kèm EN trong ngoặc ở lần đầu nếu cần):
- collaborative filtering → lọc cộng tác (collaborative filtering)
- matrix factorization / MF → phân rã ma trận (MF)
- recommender / recommendation → hệ đề xuất / khuyến nghị
- user / item → người dùng / vật phẩm (hoặc "sản phẩm" nếu ngữ cảnh sách/phim)
- query → truy vấn
- loss / loss function → hàm mất mát
- ranking → xếp hạng
- cold-start → khởi động nguội (cold-start)
- warm / cold user → người dùng đã biết / người dùng mới
- soft token → **giữ nguyên** "soft token" (không dịch "token mềm")

Gặp thuật ngữ mới ngoài bảng: mặc định giữ EN nếu là tên hàm/cờ/kiến trúc; dịch nếu là khái niệm phổ
thông có từ Việt ổn định. Ghi lựa chọn vào phần "cụm đã đổi" để tác giả xác nhận.

---

## Đầu ra

Trả **bản tiếng Việt hoàn chỉnh trước tiên**. Khi hữu ích, bổ sung ngắn:

- register + cặp ngôi đã chọn (1 dòng);
- bảng "cụm đã đổi đáng kể + lý do" (chỉ những chỗ có ý nghĩa, không giải thích từng sửa vụn);
- thuật ngữ/nghĩa còn mơ hồ cần tác giả xác nhận.

Không bịa số liệu, trích dẫn, tên. Không tự ý ghi đè file luận văn — chỉ đề xuất bản dịch, chờ duyệt
rồi mới chèn.
