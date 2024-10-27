import cv2
import numpy as np

points = []     # 用來儲存user在影像上左鍵點擊的像素座標

class Projection(object):

    def __init__(self, image_path, points):
        if type(image_path) != str:     # 如果image_path是字串, 表示是影像路徑, 用cv2.imread讀取他
            self.image = image_path
        else:
            self.image = cv2.imread(image_path)     # 存在self.image中, 並提取高, 寬, 通道數
        self.height, self.width, self.channels = self.image.shape
        self.points = points    # 存放user點取的BEV座標

    def get_rotation_matrix(self, angles):
        pitch, yaw, rolls = angles
        
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(pitch), -np.sin(pitch)],
                       [0, np.sin(pitch), np.cos(pitch)]])
        
        Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                       [0, 1, 0],
                       [-np.sin(yaw), 0, np.cos(yaw)]])
        
        Rz = np.array([[np.cos(rolls), -np.sin(rolls), 0],
                       [np.sin(rolls), np.cos(rolls), 0],
                       [0, 0, 1]])
        return Rz @ Ry @ Rx
    
    
    """ theta, phi, gamma: 控制三個軸上的旋轉, dx, dy, dz 為平移參數, fov: 視角
        BEV的相機: 位置為(0, 2.5, 0), 方向為-(np.pi / 2)
        Front的相機: 位置為(0, 1, 0), 方向為(0, 0, 0), 即沒有旋轉 """
    def top_to_front(self, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0, fov=90):      
        # TODO 
        """ Camera Intrinsic parameters """
        img_size = 512  # Image resolution is 512x512
        fov_rad = np.deg2rad(fov)   # convert degree to rad(弧度)
        f = img_size / (2 * np.tan(fov_rad / 2))    # 計算焦距
        
        K = np.array([[f, 0, img_size / 2],     # 內參矩陣
                      [0, f, img_size / 2],
                      [0, 0, 1]])
        
        """ Extrinsic parameters """
        bev_position = np.array([0, 2.5, 0])
        bev_orientation = np.array([-(np.pi / 2), 0, 0])    
        front_position = np.array([0, 1, 0])
        front_orientation = np.array([0, 0, 0])
        
        R_bev = self.get_rotation_matrix(bev_orientation)   # rotation matrix for BEV camera
        
        T_bev = np.eye(4)    # 代表4x4的單位矩陣
        T_bev[:3, 3] = bev_position     # translation matrix for BEV camera

        bev_to_world = T_bev    # bev to world transformation
        bev_to_world[:3, :3] = R_bev    # [R T]
        
        R_front = self.get_rotation_matrix(front_orientation)   # rotation matrix for front camera
        
        T_front = np.eye(4) 
        T_front[:3, 3] = front_position     # translation matrix for front camera
        
        """ world to front transformation """
        world_to_front = np.linalg.inv(T_front)
        world_to_front[:3, :3] = np.linalg.inv(R_front)
        
        bev_to_front = world_to_front @ bev_to_world    # combine transformation(變換矩陣) from BEV to Front view
        
        new_pixels = []     # project BEV points to front view
        for point in self.points:
            """ convert 2D point from BEV image space to 3D space, 然後通過相機的變換矩陣, 將其投影到投影到front圖的 2D影像平面上 """
            u, v = point    
            # 2.5代表BEV相機的高度, 乘以2.5 / f 表示將影像上的像素單位轉換為真實世界中的距離
            bev_point_3D = np.array([(u - img_size / 2), (v - img_size / 2), 1]) * 2.5 / f   # 移動到影像中心, 而不是(0, 0). 
            bev_point_3D = np.append(bev_point_3D, 1)   # convert to homogenous coordinate (4x1)
            
            front_point_3D = bev_to_front @ bev_point_3D    # 將BEV的3D點 轉到front座標系中的3D點
            
            # project 3D front point onto 2D image plane (pinhole camera model)
            if front_point_3D[2] > 0:   # check if point is in front of the camera
                pixel_2D = K @ front_point_3D[:3]
                pixel_2D = pixel_2D[:2] / pixel_2D[2]   # Normalize by depth (z)
            
                new_pixels.append(pixel_2D) # add the projected point to the list
        return np.int32(new_pixels)

    """ new_pixels: 投影後的像素座標, alpha: 透明度 """
    def show_image(self, new_pixels, img_name='projection.png', color=(0, 0, 255), alpha=0.3):  
        new_image = cv2.fillPoly(   # 將所表示的區域塗滿紅色(0, 0, 255) 
            self.image.copy(), [np.array(new_pixels)], color)
        new_image = cv2.addWeighted(    # 用addweighted() 來融合原始影像和新影像
            new_image, alpha, self.image, (1 - alpha), 0)

        """ 顯示影像, 並將影像儲為檔案 """
        cv2.imshow(
            f'Top to front view projection {img_name}', new_image)
        cv2.imwrite(img_name, new_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return new_image


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        print(x, ' ', y)
        points.append([x, y])   # 將點選的像素座標加到points中
        font = cv2.FONT_HERSHEY_SIMPLEX     # 設定字型
        # cv2.putText(img, str(x) + ',' + str(y), (x+5, y+5), font, 0.5, (0, 0, 255), 1)
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)     # 創造紅點點
        cv2.imshow('image', img)    # 在圖像上顯示

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:
        print(x, ' ', y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        # cv2.putText(img, str(b) + ',' + str(g) + ',' + str(r), (x, y), font, 1, (255, 255, 0), 2)   # 顯示顏色信息
        cv2.imshow('image', img)

if __name__ == "__main__":
    pitch_ang = -90

    front_rgb = "bev_data/front1.png"
    top_rgb = "bev_data/bev1.png"

    # click the pixels on window
    img = cv2.imread(top_rgb, 1)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    projection = Projection(front_rgb, points)  # 創造projection class
    new_pixels = projection.top_to_front(theta=pitch_ang)
    projection.show_image(new_pixels)