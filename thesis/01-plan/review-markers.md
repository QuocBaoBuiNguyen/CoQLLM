# Quy ước review marker cho luận văn (inline `% REVIEW`)

Cách người dùng ghi chú review vào `.tex` để Claude scan & fix. Chọn 2026-08-11.

## Cú pháp

```latex
% REVIEW: <mô tả>          % việc thường
% REVIEW!: <mô tả>         % ưu tiên cao — làm trước
% REVIEW(P0.1): <mô tả>    % gắn tag roadmap (revision-roadmap.md)
% REVIEW? <câu hỏi>        % phân vân — Claude TƯ VẤN, chưa tự sửa
```

- Ghi rõ nếu đã biết muốn sửa gì; ghi ngắn vấn đề nếu để Claude tự quyết.
- Đặt marker ngay **dòng trên** hoặc **cuối dòng** chỗ cần sửa (Claude đọc context quanh đó).
- **Không** dùng số trang/số dòng làm neo (trôi khi sửa) — marker đặt tại chỗ là đủ.

## Loop xử lý (Claude)

1. Người dùng: gõ **"scan review"** (hoặc chỉ định chương).
2. Claude: `grep -rn "% REVIEW" thesis/02-chapters/` → liệt kê tất cả marker + vị trí.
3. Với mỗi marker:
   - `% REVIEW?` → chỉ trả lời/tư vấn, KHÔNG sửa, giữ marker.
   - còn lại → sửa tại chỗ, **xoá marker sau khi sửa**.
4. Ưu tiên: `% REVIEW!` trước, rồi `% REVIEW(Pxx)` theo roadmap, rồi `% REVIEW` thường.
5. Commit từng chương (hoặc theo cụm), báo cáo diff để nghiệm thu.

## Lệnh tiện

```bash
# đếm & liệt kê marker còn tồn
grep -rn "% REVIEW" thesis/02-chapters/ thesis/00-format/Thesis_Template/main_draft.tex
```
