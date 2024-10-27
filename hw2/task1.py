
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# 設置argparser來讀取命令列參數
parser = argparse.ArgumentParser(description="2D Semantic Map Construction")
parser.add_argument('--data-root', type=str, default='semantic_3d_pointcloud')
args = parser.parse_args()

data_root = args.data_root
points_path = os.path.join(data_root, 'point.npy')  #   point.npy: 語意點雲數據檔案
colors_path = os.path.join(data_root, "color0255.npy")

# 讀取點雲數據
points = np.load(points_path)
colors = np.load(colors_path)

""" 查看y座標最大最小值 """
y_coords = points[:, 1]
print("Y-coordinate min:", np.min(y_coords))
print("Y-coordinate max", np.max(y_coords))

""" 去除天花板和地板點雲(一樣根據y軸來過濾) """
valid_points = points[(points[:, 1] > -0.034) & (points[:, 1] < 0.013)]
valid_colors = colors[(points[:, 1] > -0.034) & (points[:, 1] < 0.013)]

print("Valid points shape:", valid_points.shape)
""" 取出x, z座標來做2D地圖繪製 """
x = valid_points[:, 0]
z = valid_points[:, 2]

""" 繪製散點圖 """
plt.figure(figsize=(10, 10))
plt.scatter(x, z, c=valid_colors / 255, s=1)    # 使用RGB, 並縮放到[0, 1]範圍
plt.title("2D Semantic Map after Ceiling and Floor Removal")
plt.xlabel("X Coordinate")
plt.ylabel("Z Coordinate")
plt.savefig("map.png")
plt.show()

