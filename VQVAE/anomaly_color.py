#%%
import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
#%%
def display_patch_silhouette_recursive(base_directory):
    # ** を含むパターンを作成（recursive=True でサブディレクトリを走査）
    search_pattern = os.path.join(base_directory, "**", "patch_*_*.jpeg")
    # jpgも含まれる可能性がある場合はこちら: "patch_*_*.[jp][pe][g]"
    
    # 座標抽出用の正規表現（ファイル名部分のみを対象にする）
    pattern = re.compile(r'patch_(\d+)_(\d+)\.(?:jpeg|jpg)')
    
    coords = []
    max_y, max_x = 0, 0

    # 1. 再帰的にファイルを走査（iglob で省メモリ化）
    # recursive=True を忘れないように
    for filepath in glob.iglob(search_pattern, recursive=True):
        filename = os.path.basename(filepath)
        match = pattern.match(filename)
        if match:
            y, x = map(int, match.groups())
            coords.append((y, x))
            if y > max_y: max_y = y
            if x > max_x: max_x = x

    if not coords:
        print(f"画像が見つかりませんでした: {search_pattern}")
        return

    # 2. シルエット行列の作成 (0: 白, 1: 黒)
    layout = np.zeros((max_y + 1, max_x + 1))
    for y, x in coords:
        layout[y, x] = 1

    # 3. 表示
    plt.figure(figsize=(10, 10))
    # origin='upper' で y=0 を上に（スライドの座標系に合わせる）
    # interpolation='nearest' でパッチの1マスをくっきりさせる
    plt.imshow(layout, cmap='Greys', interpolation='nearest', origin='upper')
    
    plt.title(f"Patch Layout (Recursive)\nTotal: {len(coords)} patches")
    plt.axis('scaled') # アスペクト比を維持
    plt.tight_layout()
    plt.show()
#%%
# --- 実行 ---
target_dir = '/workspace/03-vq-patho-anomaly-detection/data/patches/Ctrl/14284'
display_patch_silhouette_recursive(target_dir)
#%%