import cv2
import numpy as np

points = []

class Projection(object):

    def __init__(self, image_path, points):
        if type(image_path) != str:
            self.image = image_path
        else:
            self.image = cv2.imread(image_path)
            
        if self.image is None:
            raise ValueError(f"無法載入圖片: {image_path}")
        
        self.height, self.width, self.channels = self.image.shape
        self.points = points

    def top_to_front(self, theta=-90, phi=0, gamma=0, dx=0, dy=-1.5, dz=0, fov=90):
        """
        Project the top view pixels to the front view pixels.
        :return: New pixels on perspective(front) view image
        """
        # 轉換為弧度
        theta_rad = np.deg2rad(theta)
        phi_rad = np.deg2rad(phi)
        gamma_rad = np.deg2rad(gamma)
        fov_rad = np.deg2rad(fov)

        # 焦距和相機內參矩陣 K
        f = 0.5 * self.width / np.tan(fov_rad / 2)
        Cx = self.width / 2
        Cy = self.height / 2
        K = np.array([[f, 0, Cx],
                    [0, f, Cy],
                    [0, 0, 1]])

        # 旋轉矩陣 Rx, Ry, Rz
        Rx = np.array([[1, 0, 0],
                    [0, np.cos(theta_rad), -np.sin(theta_rad)],
                    [0, np.sin(theta_rad), np.cos(theta_rad)]])
        
        Ry = np.array([[np.cos(phi_rad), 0, np.sin(phi_rad)],
                    [0, 1, 0],
                    [-np.sin(phi_rad), 0, np.cos(phi_rad)]])
        
        Rz = np.array([[np.cos(gamma_rad), -np.sin(gamma_rad), 0],
                    [np.sin(gamma_rad), np.cos(gamma_rad), 0],
                    [0, 0, 1]])

        # 最終旋轉矩陣
        R = Rx @ Ry @ Rz

        # 平移矩陣，dy = -1.5 來處理相機高度的偏移
        T = np.array([dx, dy, dz]).reshape(3, 1)

        new_pixels = []
        
        # 轉換每個選定點
        for point in self.points:
            u, v = point
            X = u
            Y = 0  # 將 BEV 圖像中的 Y 設為 0，因為其平面為 X-Z
            Z = v
            P_world = np.array([X, Y, Z]).reshape(3, 1)
            P_camera = R @ P_world + T  # 轉換為相機座標系
            if P_camera[2, 0] <= 0:  # 避免投影深度小於等於 0 的點
                continue
            P_image = K @ P_camera  # 投影到圖像平面
            u_new = P_image[0, 0] / P_image[2, 0]  # 正規化
            v_new = P_image[1, 0] / P_image[2, 0]
            new_pixels.append([int(u_new), int(v_new)])

        return new_pixels

        
    def show_image(self, new_pixels, img_name='projection.png', color=(0, 0, 255), alpha=0.4):
        if len(new_pixels) < 3:
            raise ValueError("多邊形至少需要三個點。")
        
        new_pixels = cv2.convexHull(np.array(new_pixels)).tolist()
        
        new_image = cv2.fillPoly(
            self.image.copy(), [np.array(new_pixels, dtype=np.int32)], color)
        
        new_image = cv2.addWeighted(
            new_image, alpha, self.image, (1 - alpha), 0)

        cv2.imshow(f'Top to Front View Projection {img_name}', new_image)
        cv2.imwrite(img_name, new_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return new_image


def click_event(event, x, y, flags, params):
    global img, points
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"左鍵點擊: ({x}, {y})")
        points.append([x, y])
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('image', img)

    elif event == cv2.EVENT_RBUTTONDOWN:
        print(f"右鍵點擊: ({x}, {y})")
        b, g, r = img[y, x]
        print(f"該點的RGB值: ({b}, {g}, {r})")
        cv2.imshow('image', img)


if __name__ == "__main__":
    pitch_ang = -90  

    front_rgb = "bev_data/front1.png"
    top_rgb = "bev_data/bev1.png"
    
    # front_rgb = "bev_data/front2.png"
    # top_rgb = "bev_data/bev2.png"

    img = cv2.imread(top_rgb, 1)
    if img is None:
        raise ValueError(f"無法載入圖片: {top_rgb}")
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) < 3:
        raise ValueError("至少選擇三個點以形成多邊形。")

    projection = Projection(front_rgb, points)
    new_pixels = projection.top_to_front(theta=pitch_ang, dz=450)
    projection.show_image(new_pixels)
