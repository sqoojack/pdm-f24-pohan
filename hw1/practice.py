import numpy as np
import open3d as o3d
import argparse
import os
import copy
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

# 定義相機內參
SIZE = 512
fov = 90 / 180 * np.pi
f = SIZE / (2 * np.tan(fov / 2.0))
depth_scale = 1000
K = np.array([[f, 0, SIZE / 2], [0, f, SIZE / 2], [0, 0, 1]])
K_inverse = np.linalg.inv(K)

""" 將深度圖像從8位元轉換為浮點數, 並縮放到0-10米之間 """
def transform_depth(image):

    img = np.asarray(image, dtype=np.float32)
    depth_img = img / 255.0 * 10.0  # 假設深度圖像是8位元，深度範圍為0-10米
    depth_img = o3d.geometry.Image(depth_img)
    return depth_img

""" 將RGB和深度圖像轉換為點雲 """
def depth_image_to_point_cloud(rgb, depth):

    rgb_o3d = o3d.geometry.Image(rgb.astype(np.uint8))  # 用來提供顏色資訊
    depth_o3d = transform_depth(depth)                  # 處理深度圖像
    
    # 創建RGBD圖像
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d, depth_o3d, depth_scale=1.0, depth_trunc=10.0, convert_rgb_to_intensity=False)
    
    # 使用相機內參數
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(width=SIZE, height=SIZE, fx=f, fy=f, cx=SIZE / 2, cy=SIZE / 2)
    
    # 從RGBD圖像創建點雲
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    
    # 翻轉Y軸和Z軸
    pcd.transform([[1, 0, 0, 0],
                  [0, -1, 0, 0],
                  [0, 0, -1, 0],
                  [0, 0, 0, 1]])
    return pcd

""" 對點雲進行預處理, 包括下採樣、法向量估計和FPFH特徵計算 """
def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size * 1.1)  # 進行體素下採樣
    
    # 設定搜索半徑
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(      # 估計法向量
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    
    pcd_down.normalize_normals()  # 如果法向量長度不一致則標準化法向量

    # 設定搜索半徑並計算FPFH特徵
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

""" 使用RANSAC進行全局配準 """
def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(source_down, target_down, source_fpfh, target_fpfh,
        True, distance_threshold, o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.95),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    )
    return result

""" 使用ICP進行細緻配準 """
def local_icp_algorithm(source, target, source_fpfh, target_fpfh, voxel_size, global_transformation):
    # 確保目標點雲有法向量
    if not target.has_normals():
        target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        target.normalize_normals()
    
    distance_threshold = voxel_size * 0.4
    result = o3d.pipelines.registration.registration_icp(source, target, distance_threshold, global_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

""" 計算最佳擬合變換矩陣（旋轉和平移）"""
def best_fit_transform(A, B):
    assert len(A) == len(B)

    # 計算質心
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # 計算旋轉矩陣
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R_matrix = np.dot(Vt.T, U.T)

    # 處理反射情況
    if np.linalg.det(R_matrix) < 0:
        Vt[2,:] *= -1
        R_matrix = np.dot(Vt.T, U.T)

    # 計算平移向量
    t = centroid_B.T - np.dot(R_matrix, centroid_A.T)

    # 創建4x4變換矩陣
    T = np.identity(4)
    T[0:3, 0:3] = R_matrix
    T[0:3, 3] = t

    return T, R_matrix, t

    
""" 尋找目標點雲中每個源點的最近鄰點 """
def nearest_neighbor(src, dst):
    all_dists = cdist(src, dst, 'euclidean')
    indices = all_dists.argmin(axis=1)
    distances = all_dists[np.arange(all_dists.shape[0]), indices]
    return distances, indices

""" 使用 KD-Tree 進行高效的最近鄰搜尋 """
def nearest_neighbor_kdtree(src, dst):

    tree = cKDTree(dst)
    distances, indices = tree.query(src, k=1)
    return distances, indices

def my_local_icp_algorithm(A, B, init_pose, max_iterations, voxel_size):
    """
    優化後的 ICP 算法
    """
    tolerance = voxel_size * 0.001
    # 將點轉換為齊次座標，保持原始點的副本
    src = np.ones((4, A.shape[0]))
    dst = np.ones((4, B.shape[0]))
    src[0:3, :] = np.copy(A.T)
    dst[0:3, :] = np.copy(B.T)

    # 應用初始變換（如果有）
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # 使用 KD-Tree 進行高效的最近鄰搜尋
        distances, indices = nearest_neighbor_kdtree(src[0:3, :].T, dst[0:3, :].T)

        # 設定距離閾值過濾離群點
        threshold = np.mean(distances)
        valid = distances < threshold
        if np.sum(valid) < 3:  # 防止有效點數過少
            print(f"第 {i} 次迭代：對應點數量不足，跳出迴圈。")
            break

        # 計算最佳擬合變換
        T, _, _ = best_fit_transform(src[0:3, valid].T, dst[0:3, indices[valid]].T)

        # 更新源點雲
        src = np.dot(T, src)

        # 計算平均誤差
        mean_error = np.mean(distances)
        if abs(prev_error - mean_error) < tolerance:
            print(f"第 {i} 次迭代：收斂，跳出迴圈。")
            break
        prev_error = mean_error

    # 計算最終變換
    T, _, _ = best_fit_transform(A, src[0:3, :].T)
    return T


def align_trajectories(pred_positions, gt_positions):
    """
    使用ICP對齊預估軌跡到真實軌跡
    """
    # 創建Open3D點雲
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(pred_positions)
    
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(gt_positions)
    
    # 使用ICP進行全局對齊
    threshold = 1.0  # 根據需要調整
    trans_init = np.identity(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    
    transformation = reg_p2p.transformation
    print("對齊變換矩陣:\n", transformation)
    
    # 將變換應用到預估軌跡
    pred_positions_hom = np.hstack((pred_positions, np.ones((pred_positions.shape[0], 1))))
    pred_positions_aligned_hom = (transformation @ pred_positions_hom.T).T
    pred_positions_aligned = pred_positions_aligned_hom[:, :3]
    
    return pred_positions_aligned, transformation

def create_line_set(positions, color):
    """
    創建線集合以可視化軌跡
    """
    lines = [[i, i + 1] for i in range(len(positions) - 1)]
    colors = [color for _ in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(positions)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def reconstruct(args):
    """
    從RGB-D圖像資料和相機位置資訊中重建3D點雲, 並配準相鄰幀的點雲以生成連續的點雲模型
    """
    # 創建所需目錄
    data_root = args.data_root
    rgb_dir = os.path.join(data_root, 'rgb/')
    depth_dir = os.path.join(data_root, 'depth/')    
    pose_file = os.path.join(data_root, 'GT_pose.npy')

    # 讀取真實相機位姿（包含位置和四元數）
    gt_cam_poses = np.load(pose_file)
    
    # 參數初始化
    voxel_size = args.voxel_size  # 使用命令列參數
    threshold = voxel_size * 1.5   # 距離閥值
    
    pred_cam_pos = []   # 用來預估相機的真實位置
    trajectory = []
    
    result_pcd = o3d.geometry.PointCloud()
    prev_pcd = None     # 定義前一幀的點雲
    prev_fpfh = None    # 定義前一幀的FPFH特徵
    prev_transformation = np.identity(4)    # 前一幀的變換矩陣
    
    number_of_frames = gt_cam_poses.shape[0]    # 總幀數
    
    # 定義座標變換矩陣（翻轉Y軸和Z軸）
    coord_transform = np.array([[1, 0, 0, 0],
                                [0, -1, 0, 0],
                                [0, 0, -1, 0],
                                [0, 0, 0, 1]])
    
    # 構建並對齊真實相機位姿
    gt_cam_poses_transformed = []
    first_position = gt_cam_poses[0][:3]  # 設定第一幀位置為原點
    for pose in gt_cam_poses:
        # 提取位置 [x, y, z]
        position = pose[:3] - first_position  # 將位置偏移到原點
        # 提取四元數 [qw, qx, qy, qz]
        quaternion = pose[3:]
        # 構建旋轉矩陣
        rotation_matrix = R.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]]).as_matrix()
        # 創建4x4變換矩陣
        transformation = np.identity(4)
        transformation[:3, :3] = rotation_matrix
        transformation[:3, 3] = position
        # 應用座標變換
        transformed_pose = coord_transform @ transformation
        gt_cam_poses_transformed.append(transformed_pose)
    gt_cam_poses_transformed = np.array(gt_cam_poses_transformed)
    
    for i in tqdm(range(1, number_of_frames + 1), ncols=80):
        # 讀取RGB和深度圖像
        rgb_path = os.path.join(rgb_dir, f'{i}.png')
        depth_path = os.path.join(depth_dir, f'{i}.png')
        if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
            print(f"第 {i} 幀的RGB或深度圖像不存在, 跳過。")
            continue
        
        rgb = np.asarray(o3d.io.read_image(rgb_path))  # 讀取RGB圖像
        depth = np.asarray(o3d.io.read_image(depth_path))  # 讀取深度圖像
        
        # 將RGB和深度圖像轉換為點雲
        pcd = depth_image_to_point_cloud(rgb, depth)
        
        # 預處理點雲（下採樣、法向量估計、FPFH特徵計算）
        pcd_down, pcd_fpfh = preprocess_point_cloud(pcd, voxel_size)
        
        # 初始化第一幀
        if prev_pcd is None:
            prev_pcd = pcd_down
            prev_fpfh = pcd_fpfh
            pred_cam_pos.append(prev_transformation)
            result_pcd += pcd_down
            continue
        
        # 全局配準（RANSAC）
        result_global = execute_global_registration(pcd_down, prev_pcd, pcd_fpfh, prev_fpfh, voxel_size)
        
        # 細緻配準（ICP）
        if args.icp == "open3d":
            result_icp = local_icp_algorithm(pcd_down, prev_pcd, pcd_fpfh, prev_fpfh, voxel_size, result_global.transformation)
            result_transformation = result_icp.transformation
        elif args.icp == "our_ICP":
            # 使用自定義ICP算法
            result_transformation = my_local_icp_algorithm(np.asarray(pcd_down.points), np.asarray(prev_pcd.points), init_pose=result_global.transformation,
                max_iterations=20, voxel_size=voxel_size)
        else:
            print("未知的ICP策略, 請選擇 'open3d' 或 'own'")
            return
        
        # 檢查ICP配準是否成功
        if result_transformation is None:
            print(f"第 {i} 幀 ICP 配準失敗，使用上一幀的變換矩陣。")
            pred_cam_pos.append(prev_transformation)
            result_pcd += pcd_down
            continue
        
        # 更新變換矩陣
        current_transformation = prev_transformation @ result_transformation
        pcd_down.transform(result_transformation)
        prev_transformation = current_transformation
        
        # 移除天花板（假設Y軸為垂直向上方向，移除Y坐標高於0.1米的點）
        points = np.asarray(pcd_down.points)
        colors = np.asarray(pcd_down.colors) if pcd_down.has_colors() else None
        
        # mask = points[:, 1] < 0.98  # 根據實際場景調整閾值(1樓: 0.1)
        mask = points[:, 1] < 0.1  # 根據實際場景調整閾值(2樓: 0.3)
        points_filtered = points[mask]
        colors_filtered = colors[mask] if colors is not None else None

        # 在進度條中輸出過濾前後的點雲點數
        # tqdm.write(f"過濾前點雲點數: {len(points)}, 過濾後點雲點數: {len(points_filtered)}")

        if len(points_filtered) == 0:
            print(f"第 {i} 幀過濾後點雲為空，使用上一幀的變換矩陣。")
            pred_cam_pos.append(prev_transformation)
            result_pcd += pcd_down
            continue
        
        pcd_down.points = o3d.utility.Vector3dVector(points_filtered)
        if colors is not None:
            pcd_down.colors = o3d.utility.Vector3dVector(colors_filtered)
        
        # 累積點雲
        result_pcd += pcd_down
        pred_cam_pos.append(current_transformation)
        
        """ 每20幀就查看 result_pcd的情況 """
        # if i % 15 == 0:
        #     o3d.visualization.draw_geometries([result_pcd])
            
        # 更新前一幀資訊
        prev_pcd = pcd_down
        prev_fpfh = pcd_fpfh    
    return result_pcd, np.array(pred_cam_pos), gt_cam_poses_transformed



if __name__ == "__main__":
    # 參數解析
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--icp", help="選擇ICP策略: [open3d, our_ICP]", type=str, default="open3d")
    parser.add_argument("-v", "--voxel_size", help="體素大小", type=float, default=0.05)    # 0.05 is best
    parser.add_argument("-f", "--floor", help="選擇樓層: [1, 2]", type=int, default=1)
    parser.add_argument("--data_root", type=str, default=None, help="資料根目錄")
    args = parser.parse_args()
    
    if args.data_root is None:
        if args.floor == 1: 
            args.data_root = "data_collection/first_floor/"
        elif args.floor == 2:
            args.data_root = "data_collection/second_floor/"
        else:
            print("未知的樓層, 請選擇1或2。")
            pass
    else:
        if not os.path.exists(args.data_root):
            print(f"資料根目錄 {args.data_root} 不存在。")
            pass
    
    # 執行重建
    result_pcd, pred_cam_pos, gt_cam_poses_transformed = reconstruct(args)
    
    """Calculate and print L2 distance (計算相機預估路徑與真實路徑之間的誤差)"""
    # t[:3, 3]: 取出矩陣t中前三行的第四列, 即[tx, ty, tz], 表示相機的平移向量(位置), 並忽略其相機的旋轉
    pred_positions = np.array([t[:3, 3] for t in pred_cam_pos])     # 從預估的相機位姿中提取每一幀的平移部分(也就是相機的座標位置), 將這些位置儲存在pred_positions中
    gt_positions = np.array([t[:3, 3] for t in gt_cam_poses_transformed])      # 從真實相機位姿中提取相機的真實位置
    
    diff = pred_positions - gt_positions
    distances = np.linalg.norm(diff, axis=1)    # 對每一對差異計算L2距離(歐式距離)
    mean_distances = np.mean(distances)
    
    print("Mean L2 distance: ", mean_distances)

    pred_line_set = create_line_set(pred_positions, [1, 0, 0])  # 建立預估相機路徑的線集合, 並用紅色表示
    gt_line_set = create_line_set(gt_positions, [0, 0, 0])  # 建立真實相機路徑的線集合, 並用黑色表示
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    
    o3d.visualization.draw_geometries([result_pcd, pred_line_set, gt_line_set, coordinate_frame])


    