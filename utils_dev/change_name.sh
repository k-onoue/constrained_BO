#!/bin/bash

# 対象ディレクトリのパス
DIR="experiments/2024-09-03_botorch_simple"

# ファイル名を変更
for file in "$DIR"/*; do
    # ファイル名を取得
    filename=$(basename "$file")
    
    # 新しいファイル名を生成
    new_filename=$(echo "$filename" | sed -E 's/(constrained3_|unconstrained_)/\1d_/')
    
    # ファイルをリネーム
    mv "$DIR/$filename" "$DIR/$new_filename"
    
    echo "Renamed: $filename -> $new_filename"
done