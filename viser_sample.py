import numpy as np
from PIL import Image
import cv2
import time
import viser
from scipy.spatial.transform import Rotation as R_scipy

import reconstruct


depth_prefix = '../depth_data/depth/'
color_prefix = '../frames/frame_'

depth_prefix = '../board_captures/depth_captures/20260404_122039/frame_'
color_prefix = '../board_captures/color_captures/20260404_122039/frame_'

depth_prefix = '../20260407_173846/depth_captures/frame_'
color_prefix = '../20260407_173846/color_captures/frame_'

# intrinsics
fx, fy, cx, cy = 1366.3287, 1366.3287, 957.5452, 722.60974

fx *= 0.1333333333
fy *= 0.1333333333
cx *= 0.1333333333
cy *= 0.1333333333

#fx, fy, cx, cy = 639.836731, 639.195068, 641.591858, 397.998596


fx, fy, cx, cy = 597.01702881, 597.04272461, 327.60223389, 240.29771423

intrinsic_matrix = np.eye(3)
intrinsic_matrix[0, 0] = fx
intrinsic_matrix[1, 1] = fy
intrinsic_matrix[0, 2] = cx
intrinsic_matrix[1, 2] = cy

def cloudify_frame(cloud_handle, axis_handle, box_handle, frame_no):

    depth_img = cv2.imread(depth_prefix + f'{frame_no}.png', cv2.IMREAD_UNCHANGED)
    color_img = cv2.imread(color_prefix + f'{frame_no}.png')
    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

    _, rvec, tvec = reconstruct.detect_aruco_pose(color_img, intrinsic_matrix, None, 0.05)[0]
    quaternion, position = reconstruct.quatnpos_from_vector(tvec, rvec)

    axis_handle.wxyz = quaternion
    axis_handle.position = position

    box_handle.wxyz = quaternion
    box_handle.position = position


    depth = np.asarray(depth_img) / 1000

    colors = np.asarray(color_img).reshape(-1, 3)


    H, W = depth.shape
    v, u = np.meshgrid(
        np.arange(H),
        np.arange(W),
        indexing="ij"
    )

    Z = depth.flatten()
    X = (u.flatten() - cx) * Z / fx
    Y = (v.flatten() - cy) * Z / fy
    points = np.stack([X, Y, Z], axis=1)

    cloud_handle.points = points
    cloud_handle.colors = colors



def cloudify_video(server, num_frames, fps=30):
    dt = 1.0 / fps

    cloud_handle = server.scene.add_point_cloud(
        name="depth_cloud",
        points=np.zeros((1, 3)),
        colors=np.zeros((1, 3)),
        point_size=0.001,
    )

    axis_handle = server.scene.add_frame(
        name="my_pose",
        wxyz=(1, 0, 0, 0),
        position=np.zeros(3),
    )

    box_handle = server.scene.add_box(
        name="box",
        dimensions=(.304, .304, .004), # width, height, depth
        color=(255, 0, 0),          # RGB
        position=np.zeros(3),   # x, y, z
    )

    for i in range(num_frames):
        t0 = time.time()

        frame_no = str(i).zfill(6)

        cloudify_frame(cloud_handle, axis_handle, box_handle, frame_no)

        elapsed = time.time() - t0
        sleep_time = max(0.0, dt - elapsed)
        time.sleep(sleep_time)

server = viser.ViserServer()

H = 480
W = 640

frustum = server.scene.add_camera_frustum(
                f"/iphone/frustum",
                fov=2 * np.arctan2(H / 2, fy),
                aspect=W / H,
                scale=0.1,
                #image=image,
            )

cloudify_video(server, 1200)

print("check out http://localhost:8080")

while True:
    time.sleep(1)
