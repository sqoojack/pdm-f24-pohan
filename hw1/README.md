---
title: Command Line

---


**Command Line:**

Task1:
python3 bev.py --photo 1
python3 bev.py --photo 2


Task2:

要先執行:
python3 load.py -f 1
python3 load.py -f 2
來製造depth圖像以及rgb圖像
存在data_collection/first_floor
跟data_collection/second_floor 資料夾中

再來執行 reconstruct.py
python3 reconstruct.py -f 1 -v open3d
python3 reconstruct.py -f 2 -v open3d
python3 reconstruct.py -f 1 -v my_icp
python3 reconstruct.py -f 2 -v my_icp