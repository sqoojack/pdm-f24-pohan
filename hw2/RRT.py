
import numpy as np
import matplotlib.pyplot as plt
import argparse
import random
import math
import cv2
import pandas as pd

""" 定義一個Node class 來存儲點的座標和父節點 """
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None  # 父節點, 用於存儲路徑
        
def distance(p1, p2):   # 定義歐幾里得距離
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

def RRT(start, goal, obstacle_list, map_size, step_size = 0.2, max_iter = 500):   # obstacle_list: 障礙物位置的列表
    start_node = Node(start[0], start[1])   # 初始化起始節點
    goal_node = Node(goal[0], goal[1])  # 初始化目標節點
    node_list = [start_node]    # 用於存儲樹中的所有節點
    
    """ 開始迭代尋找路徑 """
    for i in range(max_iter):
        rand_node = Node(random.uniform(0, map_size[0]), random.uniform(0, map_size[1]))     # 生成一個隨機點
        nearest_node = min(node_list, key = lambda node: distance(node, rand_node))     # 找到離隨機點最近的點
        
        theta = math.atan2(rand_node.y - nearest_node.y, rand_node.x - nearest_node.x)  # 計算從最近的節點到隨機點的擴展方向 (讓他想辦法到達目標節點)
        new_node = Node(nearest_node.x + step_size * math.cos(theta),
                        nearest_node.y + step_size * math.sin(theta))   # 生成新節點
        new_node.parent = nearest_node  # 設置新節點的parent node為最近的節點
        
        if not is_collision(new_node, obstacle_list):
            node_list.append(new_node)  # 如果沒有碰撞, 將新節點增加到樹中
        
        """ check新節點是否達到目標 """
        if distance(new_node, goal_node) < step_size:
            goal_node.parent = new_node
            node_list.append(new_node)
            break
    
    """ 回朔從目標到起點的路徑 """
    path = []
    node = goal_node
    while node is not None:
        path.append([node.x, node.y])   # 將節點的座標加到路徑中
        node = node.parent  # node指定為父節點 (下一個移動的點)
    path = np.array(path)   # 轉換為numpy陣列
    return node_list, path
            
""" 定義檢查碰撞的函數 """
def is_collision(node, obstacle_list):
    for (ox, oy, radius) in obstacle_list:
        dist = math.sqrt((node.x - ox) ** 2 + (node.y - oy) ** 2)   # 計算節點和障礙物圓心之間的距離
        if dist <= radius:  # 距離小於等於障礙物的半徑, 表示發生碰撞
            return True
    return False


""" 製作obstacle_list """
colors_path = './color_coding_semantic_segmentation_classes.xlsx'
obstacle_colors_df = pd.read_excel(colors_path)


""" 讀取地圖圖像 """
map_image = cv2.imread("map2.png")
map_image_rgb = cv2.cvtColor(map_image, cv2.COLOR_BGR2RGB)

obstacle_list = []

""" 從excel資料中提取顏色並掃描地圖影像 """
for index, row in obstacle_colors_df.iterrows():    
    color = eval(row['Color_Code (R,G,B)'])     # 提取顏色的RGB值
    mask = cv2.inRange(map_image_rgb, np.array(color), np.array(color))     # 第一個參數為進行檢測的輸入影像, 第二,三參數分別表示lower bound, upper bound
    coods = cv2.findNonZero(mask)   # 找出所有符合顏色的像素點
    
    if coods is not None:
        for pt in coods:
            x, y = pt[0][0], pt[0][1]
            obstacle_list.append((x, y, 1))     # 假設每個障礙物的半徑為1
            
print(f"Number of obstacles detected: {len(obstacle_list)}")

""" 繪製output圖 """
map_size = (0.30, 0.50)     # 分別對應x, z軸座標
start = (0.05, -0.05)
goal = (0.075, 0.2)

node_list, path = RRT(start, goal, obstacle_list, map_size)

plt.figure(figsize=(10, 10))
plt.imshow(map_image_rgb)

for node in node_list:
    if node.parent is not None:
        plt.plot([node.x, node.parent.x], [node.y, node.parent.y], color='cyan', linewidth=0.5)     # 畫線
        
""" 繪製找到的最終路徑 """
if len(path) > 0:
    plt.plot(path[:, 0], path[:, 1], color='red', linewidth=2)
    
plt.scatter(start[0], start[1], color='green', s=100, label='starting point')   # 繪製散點圖, start[0], start[1]: 起始點的x, y座標, s=100: 標記的大小
plt.scatter(goal[0], goal[1], color='blue', s=100, label='Desired goal point')
plt.legend()    # 將標籤表示出來
plt.title("RRT Path Planning Visualization")
plt.show()