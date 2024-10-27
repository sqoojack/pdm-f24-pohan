import argparse
import numpy as np
import cv2
import math
import open3d as o3d
import plotly.graph_objects as go
from sklearn.neighbors import NearestNeighbors
import glob
import os
from tqdm import tqdm
import copy
from scipy.spatial import KDTree

size = 512
fov = 90 / 180 * np.pi  # fov為90度, 將其轉為弧度
f = size / (2 * np.tan(fov / 2.0))
depth_scale = 1000  # 深度縮放比例為1000, 代表將毫米轉成米
K = np.array([[f, 0, size / 2],
              [0, f, size / 2],
              [0, 0, 1]])
inverse_K = np.linalg.inv(K)   # linalg: linear algebra funcrion

def depth_image_to_point_cloud(rgb, depth):
    # TODO: Get point cloud from rgb and depth image 
    assert rgb.shape[:2] == depth.shape[:2], "RGB 和 深度圖大小不一致"     # rgb.shape[:2] 是用來取得rgb圖像的前兩個維度, 也就是圖像的高和寬, 而不需要後面的顏色資訊
    
    depth = depth.astype(np.float32) / depth_scale
    
    """ 將rgb和深度圖 轉為 Open3d圖像"""
    rgb_o3d = o3d.geometry.Image(rgb.astype(np.uint8))
    depth_o3d = o3d.geometry.Image(depth)
    
    """ 創建RGB-D圖像 """
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=rgb_o3d, depth=depth_o3d, depth_scale=1, depth_trunc=10.0, # 超過10米會被截斷
        convert_rgb_to_intensity=False   # 保持RGB顏色
    )
    
    """ 設定相機內參 """
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(width=size, height=size, fx=f, fy=f, cx=size / 2, cy = size / 2)
    
    """ 從RGB圖像生成點雲 """
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    
    # 翻轉Y軸和Z軸
    pcd.transform([[1, 0, 0, 0],
                  [0, -1, 0, 0],
                  [0, 0, -1, 0],
                  [0, 0, 0, 1]])
    return pcd

def preprocess_point_cloud(pcd, voxel_size, floor):
    # TODO: Do voxelization to reduce the number of points for less memory usage and speedup
    """ 將下採樣點雲點數 控制在一萬左右, """
    if floor == 1:
        pcd_down = pcd.voxel_down_sample(voxel_size * 0.07)
    elif floor == 2:
        pcd_down = pcd.voxel_down_sample(voxel_size * 0.045)
    
    pcd_down.estimate_normals(  # 估計法向量
        o3d.geometry.KDTreeSearchParamHybrid(radius = voxel_size * 2.0, max_nn = 30)
    )
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(     # 計算fpfh特徵
        pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius = voxel_size * 5.0, max_nn = 100)
        )
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.4
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, False, distance_threshold,  # True: 是否啟用是否啟用互換對稱匹配
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.95),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(200000, 0.999)
    )
    return result


def local_icp_algorithm(source_down, target_down, trans_init, threshold):
    # TODO: Use Open3D ICP function to implement

    result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    return result

class ICPResult:
    def __init__(self, transformation, inlier_rmse, fitness):
        self.transformation = transformation  # 最終的 4x4 變換矩陣
        self.inlier_rmse = inlier_rmse        # 內點的均方根誤差
        self.fitness = fitness                # 內點比例

def my_local_icp_algorithm(source_down, target_down, trans_init, voxel_size, max_iterations=150, tolerance=1e-8):
    transformation = trans_init.copy()
    target_points = np.asarray(target_down.points)
    source_points = np.asarray(source_down.points)
    kdtree = KDTree(target_points)

    prev_error = np.inf
    for i in range(max_iterations):
        transformed_source_points = np.dot(source_points, transformation[:3, :3].T) + transformation[:3, 3]
        
        distances, indices = kdtree.query(transformed_source_points)
        closest_target_points = target_points[indices]

        # 濾除距離過大的匹配
        valid_matches = distances < voxel_size * 1.5
        if np.sum(valid_matches) < len(distances) * 0.1:
            print(f"第 {i} 次迭代: 匹配點過少，可能配準失敗")
            break

        transformed_source_points = transformed_source_points[valid_matches]
        closest_target_points = closest_target_points[valid_matches]

        centroid_source = np.mean(transformed_source_points, axis=0)
        centroid_target = np.mean(closest_target_points, axis=0)
        
        centered_source = transformed_source_points - centroid_source
        centered_target = closest_target_points - centroid_target

        H = np.dot(centered_source.T, centered_target)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)

        t = centroid_target - np.dot(R, centroid_source)

        delta_transformation = np.identity(4)
        delta_transformation[:3, :3] = R
        delta_transformation[:3, 3] = t

        transformation = np.dot(delta_transformation, transformation)

        error = np.mean(distances[valid_matches])
        if abs(prev_error - error) < tolerance:
            break
        
        prev_error = error

    return ICPResult(transformation=transformation, inlier_rmse=error, fitness=1.0 - error / np.max(distances))



def reconstruct(args):
    # TODO: Return results

    """ 先讀取所需資料 """
    data_root = args.data_root
    rgb_dir = os.path.join(data_root, "rgb/")
    depth_dir = os.path.join(data_root, "depth/")
    pose_file = os.path.join(data_root, "GT_pose.npy")
    
    gt_cam_poses = np.load(pose_file)   # np.load: 用來從npy檔中讀取資料, 紀錄真實相機位姿
    pred_cam_pos = []   # 預測相機位姿
    
    voxel_size = args.voxel_size    # 設定voxel_size
    floor = args.floor
    """ 初始化物件 """
    result_pcd = o3d.geometry.PointCloud()  # 建立一個空的PointCloud物件
    
    prev_pcd = None     # 定義前一幀的點雲, 及fpfh
    prev_fpfh = None
    prev_transformation = np.identity(4)    # 定義前一幀的變換矩陣, 單位矩陣代表一開始會在世界座標的原點
    number_of_frames = gt_cam_poses.shape[0]    # 定義總幀數
    
    for i in tqdm(range(1, number_of_frames + 1), ncols=80):
        rgb_path = os.path.join(rgb_dir, f"{i}.png")    # 建立圖像路徑
        depth_path = os.path.join(depth_dir, f"{i}.png")
        
        rgb = np.asarray(o3d.io.read_image(rgb_path))
        depth = np.asarray(o3d.io.read_image(depth_path))
        
        """ 將RGB, 深度圖像 轉換成點雲"""
        pcd = depth_image_to_point_cloud(rgb, depth)
        
        """ 預處理點雲 """
        pcd_down, pcd_fpfh = preprocess_point_cloud(pcd, voxel_size, floor)
        # o3d.visualization.draw_geometries([pcd_down], window_name=f"Frame {i} Downsampled Point Cloud", width=800, height=600)
        
        # 初始化第一幀
        if prev_pcd is None:
            prev_pcd = pcd_down
            prev_fpfh = pcd_fpfh
            pred_cam_pos.append(prev_transformation)    # 加入單位矩陣到 預測相機位姿
            result_pcd += prev_pcd  # 將目前pcd_down 加到result_pcd中, 並結束第一幀
            continue
        
        """ 進行global_registration """
        global_result = execute_global_registration(prev_pcd, pcd_down, prev_fpfh, pcd_fpfh, voxel_size)    # 將上一幀的pcd_down轉到現在的pcd_down

        """ 執行refine registration(細緻配準) """
        if args.version == 'open3d':
            # local_result = local_icp_algorithm(prev_pcd, pcd_down, trans_init=prev_transformation, threshold=voxel_size)  # 不使用global_registration的情況下
            local_result = local_icp_algorithm(prev_pcd, pcd_down, trans_init=global_result.transformation, threshold=voxel_size)
            local_result_transformation = local_result.transformation   # 紀錄細緻配準的變換矩陣, 代表著當前幀到前一幀的變換
        if args.version == 'my_icp':
            local_result = my_local_icp_algorithm(prev_pcd, pcd_down, trans_init=global_result.transformation, voxel_size=voxel_size)
            local_result_transformation = local_result.transformation
            
        if local_result_transformation is None:
            print(f"第{i}幀配準失敗, 使用上一幀的變換矩陣 ")
            pred_cam_pos.append(prev_transformation.copy())    # 變換矩陣描述了相機在空間中的位置跟方向, 所以將他加入到pred_cam_pos裡  
            result_pcd += pcd_down
            continue
        
        """ 更新預測的相機位姿 """
        if args.version == 'open3d':
            current_transformation = prev_transformation @ np.linalg.inv(local_result_transformation)
            pred_cam_pos.append(current_transformation.copy())  # 加入到預測的相機位姿中
        elif args.version == 'my_icp':
            current_transformation = prev_transformation @ np.linalg.inv(local_result_transformation)
            pred_cam_pos.append(current_transformation.copy())
            
        """ 移除天花板 """
        points = np.asarray(pcd_down.points)
        colors = np.asarray(pcd_down.colors)
        
        if args.floor == 1:
            mask = points[:, 1] < 0.07 * 0.001  # points[:, 1]: y軸座標
        elif args.floor == 2:
            mask = points[:, 1] < 0.3 * 0.001
            
        filtered_points = points[mask]
        filtered_colors = colors[mask]
        
        tqdm.write(f"過濾前點雲點數: {len(points)}, 過濾後點雲點數: {len(filtered_points)}")
        
        if len(filtered_points) == 0:
            print(f"第{i}幀過濾後點雲為空, 跳過")
            pass
        
        pcd_down.points = o3d.utility.Vector3dVector(filtered_points)
        pcd_down.colors = o3d.utility.Vector3dVector(filtered_colors)
        
        """ 累積點雲 """
        pcd_transformed = copy.deepcopy(pcd_down)    # 將目前的轉換矩陣放到pcd_transformed中
        pcd_transformed.transform(current_transformation)
        result_pcd += pcd_transformed
        
        """ 在特定幀數查看 result_pcd 的情況 """
        # if i == 20 or i == 40:
            # o3d.visualization.draw_geometries([result_pcd])
        
        """ 更新前一幀資料 """
        prev_pcd = pcd_down
        prev_fpfh = pcd_fpfh
        prev_transformation = current_transformation.copy()
    return result_pcd, pred_cam_pos, gt_cam_poses

def compute_L2_distance(pred_cam_pos, gt_cam_poses):
    pred_positions = []
    gt_positions = []
    
    """ 取得預測相機位姿的位移部分 """
    for transformation in pred_cam_pos:
        position = transformation[:3, 3]    # 取得位移部分
        pred_positions.append(position)
    
    """ 取得真實相機位姿的位移部分 (格式: [x, y, z, rw, rx, ry, rz])"""    
    for pose in gt_cam_poses:
        position = pose[:3]     # 即x, y, z
        gt_positions.append(position)
    
    """ 轉換成NumPy陣列 """    
    pred_positions = np.array(pred_positions)
    gt_positions = np.array(gt_positions)

    assert pred_positions.shape == gt_positions.shape, "預測位姿跟真實位姿, 數量或形狀不一致"
    
    """ 計算平均L2 distance, axis = 1: 沿著矩陣的第二個維度進行計算 """
    distance = np.linalg.norm(gt_positions - pred_positions, axis=1)      # 因為gt_position跟pred_position 形狀都是(N, 3), 其中N是幀數 沒有axis = 2的部分
    mean_distance = np.mean(distance)
    return mean_distance

def create_line_set(positions, color):
    lines = [[i, i+1] for i in range(len(positions) - 1)]
    colors = [color for _ in lines]
    
    line_set = o3d.geometry.LineSet(
        points = o3d.utility.Vector3dVector(positions), lines = o3d.utility.Vector2iVector(lines))
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set  

""" 將相機位置標準化 """
def normalize_positions(positions):
    max_value = np.max(np.linalg.norm(positions, axis=1))
    if max_value > 0:
        return positions / max_value
    return positions

def visualize_result(result_pcd, pred_cam_pos, gt_cam_poses, floor, version):
    
    """ 在二樓時會發生 黑線比較高的狀況, 所以要降低一下 (但軌跡是相同的) """
    if floor == 2:
        offset_y = 2.8    
        gt_cam_poses[:, 1] -= offset_y
    
    """ 提取預測和真實相機位姿的位移部分 """
    pred_positions = np.array([transformation[:3, 3] for transformation in pred_cam_pos])
    gt_positions = np.array([pose[:3] for pose in gt_cam_poses])   # 格式:[x, y, z, rw, rx, ry, rz]
    
    # scale_factor = 1 / 1000
    scale_factor = np.mean(np.linalg.norm(pred_positions, axis=1)) / np.mean(np.linalg.norm(gt_positions, axis=1))
    gt_positions = gt_positions * scale_factor
    
    # if version == "my_icp":
        # pred_positions = pred_positions / 1.5
    
    pred_line_set = create_line_set(pred_positions, [1, 0, 0])  # 將預測軌跡用紅色表示
    gt_line_set = create_line_set(gt_positions, [0, 0, 0])  # 將真實軌跡用黑色表示
    
    o3d.visualization.draw_geometries([result_pcd, pred_line_set, gt_line_set], window_name="Demo Result", mesh_show_back_face=True)     

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--floor', type=int, default=1)
    parser.add_argument('-v', '--version', type=str, default='open3d', help='open3d or my_icp')
    parser.add_argument('--data_root', type=str)
    parser.add_argument("--voxel_size", type=float, default=0.03)
    args = parser.parse_args()

    if args.floor == 1:
        args.data_root = "data_collection/first_floor/"
    elif args.floor == 2:
        args.data_root = "data_collection/second_floor/"
    
    # TODO: Output result point cloud and estimated camera pose
    result_pcd, pred_cam_pos, gt_cam_poses = reconstruct(args)
    
    # TODO: Calculate and print L2 distance
    Mean_L2_distance = compute_L2_distance(pred_cam_pos, gt_cam_poses)    
    print("Mean L2 distance: ", Mean_L2_distance)
    
    # TODO: Visualize result
    """ 將結果可視化並儲存起來 """
    o3d.io.write_point_cloud("reconstructed_point_cloud.ply", result_pcd)
    
    estimate_positions = np.array([trans[:3, 3] for trans in pred_cam_pos])
    np.save("estimated_camera_position.npy", estimate_positions)    # 將預測的camera position 儲存起來

    visualize_result(result_pcd, pred_cam_pos, gt_cam_poses, args.floor, args.version)