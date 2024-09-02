#!/bin/bash

# 対象ディレクトリのパス
DIR="experiments/2024-09-02_botorch_schubert"

# 並列で実行
for FILE in "$DIR"/Schubert_*.py; do
  python3 "$FILE" &
done

# すべてのバックグラウンドプロセスの終了を待つ
wait

echo "All scripts have finished executing."
