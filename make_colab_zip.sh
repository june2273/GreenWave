#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  GreenWave Colab 업로드용 zip 생성
#  실행: bash make_colab_zip.sh
#  출력: ~/Desktop/GreenWave.zip (Google Drive에 업로드)
# ─────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
OUT_ZIP="$HOME/Desktop/GreenWave.zip"

echo "프로젝트 경로: $SCRIPT_DIR"
echo "출력 zip:      $OUT_ZIP"
echo ""

# 기존 zip 삭제
[ -f "$OUT_ZIP" ] && rm "$OUT_ZIP" && echo "기존 zip 삭제됨"

# zip 생성 (불필요한 파일 제외)
cd "$PARENT_DIR"
zip -r "$OUT_ZIP" "GreenWave/" \
  --exclude "GreenWave/.venv/*" \
  --exclude "GreenWave/.git/*" \
  --exclude "GreenWave/models/*" \
  --exclude "GreenWave/results/*" \
  --exclude "GreenWave/videos/*" \
  --exclude "GreenWave/__pycache__/*" \
  --exclude "GreenWave/**/__pycache__/*" \
  --exclude "GreenWave/**/*.pyc" \
  --exclude "GreenWave/**/*.DS_Store" \
  --exclude "GreenWave/.DS_Store"

SIZE=$(du -sh "$OUT_ZIP" | cut -f1)
echo ""
echo "✓ 완료: $OUT_ZIP ($SIZE)"
echo ""
echo "다음 단계:"
echo "  1. Google Drive (drive.google.com) 최상위 폴더에 GreenWave.zip 업로드"
echo "  2. Colab 에서 colab_train_greenwave.ipynb 열기"
echo "  3. Account 1: MODE = \"MAPPO\"  |  Account 2: MODE = \"CTDE\""
