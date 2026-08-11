#!/usr/bin/env bash
# Build luận văn SigLLM ra bản CÓ VERSION. Mỗi bản version là BẤT BIẾN.
#   ./build.sh            -> tự tăng version (v2, v3, ...) theo builds/ hiện có
#   ./build.sh "note"     -> gắn nhãn ngắn, vd main_draft_v3_2026-08-12_fix-bang.pdf
#
# Mô hình:
#   - builds/main_draft_vN_<ngày>[_<note>].pdf : KHO LƯU TRỮ, không bao giờ ghi đè.
#       => "bản draft cũ" của bạn (v1) nằm ở đây và an toàn tuyệt đối.
#   - builds/latest.pdf : symlink trỏ tới bản mới nhất (mở cái này để review).
#   - main_draft.pdf (gốc) : chỉ là bản làm việc mới nhất, được tái sinh mỗi lần build.
set -euo pipefail

cd "$(dirname "$0")"
export PATH="/Library/TeX/texbin:$PATH"

JOB="main_draft"
NOTE="${1:-}"
DATE="$(date +%Y-%m-%d)"
BUILDS="builds"
mkdir -p "$BUILDS"

# Version kế tiếp = max(vN trong builds/) + 1
LAST=$(ls "$BUILDS"/${JOB}_v*.pdf 2>/dev/null \
        | sed -E 's/.*_v([0-9]+)_.*/\1/' | sort -n | tail -1 || true)
NEXT=$(( ${LAST:-0} + 1 ))
LOG="/tmp/${JOB}_build.log"

echo "==> Build ${JOB} v${NEXT} (${DATE})"

# pass1 -> biber -> pass2 -> pass3 (giải hết ref + bibliography)
pdflatex -interaction=nonstopmode -halt-on-error "$JOB.tex" >"$LOG" 2>&1 || {
  echo "!! pdflatex pass 1 lỗi — xem $LOG"; tail -20 "$LOG"; exit 1; }
biber "$JOB" >>"$LOG" 2>&1 || { echo "!! biber lỗi — xem $LOG"; tail -20 "$LOG"; exit 1; }
pdflatex -interaction=nonstopmode -halt-on-error "$JOB.tex" >>"$LOG" 2>&1 || true
# Pass cuối ghi log RIÊNG để chỉ soi undefined của pass cuối (pass-1 luôn undefined là bình thường)
LOG_FINAL="/tmp/${JOB}_build_final.log"
pdflatex -interaction=nonstopmode -halt-on-error "$JOB.tex" >"$LOG_FINAL" 2>&1 || true

# Cảnh báo undefined ref/cite ở PASS CUỐI (không chặn build)
UNDEF=$(grep -c "undefined" "$LOG_FINAL" || true)
if [ "${UNDEF:-0}" -gt 0 ]; then
  echo "   ⚠ có $UNDEF cảnh báo 'undefined' — kiểm tra $LOG"
else
  echo "   ✓ 0 undefined ref/cite"
fi

# Tên bản lưu trữ
if [ -n "$NOTE" ]; then
  SAFE=$(echo "$NOTE" | tr ' ' '-' | tr -cd 'A-Za-z0-9-_')
  OUT="$BUILDS/${JOB}_v${NEXT}_${DATE}_${SAFE}.pdf"
else
  OUT="$BUILDS/${JOB}_v${NEXT}_${DATE}.pdf"
fi

cp "$JOB.pdf" "$OUT"
ln -sf "$(basename "$OUT")" "$BUILDS/latest.pdf"
# Đếm trang: ưu tiên pdfinfo (chính xác), fallback mdls
if command -v pdfinfo >/dev/null 2>&1; then
  PAGES=$(pdfinfo "$OUT" 2>/dev/null | awk '/^Pages:/{print $2}')
else
  PAGES=$(mdls -name kMDItemNumberOfPages -raw "$OUT" 2>/dev/null)
fi
if [ -z "$PAGES" ] || [ "$PAGES" = "(null)" ]; then PAGES="?"; fi

# Dọn file phụ ở thư mục gốc cho gọn (giữ .pdf làm việc)
rm -f "$JOB".{aux,bcf,bbl,bcf,blg,log,out,run.xml,toc,lof,lot} 2>/dev/null || true

echo "==> Xong: $OUT  (${PAGES} trang)"
echo "    Mở review: $BUILDS/latest.pdf"
echo "    Các bản version cũ trong $BUILDS/ KHÔNG bị đụng."
