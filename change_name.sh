#!/bin/bash

# 対象ディレクトリのパス
DIR="experiments/2024-09-02_botorch_griewank"

# ファイル名を変更
for FILE in "$DIR"/*; do
  BASENAME=$(basename "$FILE")
  if [[ "$BASENAME" == *"Simple"* ]]; then
    NEW_NAME="${BASENAME/Simple/Griewank}"
    mv "$FILE" "$DIR/$NEW_NAME"
    echo "Renamed: $BASENAME to $NEW_NAME"
  fi
done
