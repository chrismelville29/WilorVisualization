import numpy as np
from PIL import Image
import time
import cv2
from scipy.spatial.transform import Rotation as R_scipy
from scipy.ndimage import gaussian_filter1d

import rendering_utils
import reconstruct
import viser
import yourdfpy
from viser.extras import ViserUrdf

GRASPNESS_THRESHOLD = 0.06

#intrinsics
fx, fy, cx, cy = 1366.3287, 1366.3287, 957.5452, 722.60974

fx *= 0.1333333333
fy *= 0.1333333333
cx *= 0.1333333333
cy *= 0.1333333333

fx, fy, cx, cy = 597.01702881, 597.04272461, 327.60223389, 240.29771423
intrinsic_matrix = reconstruct.matrix_from_intrix(fx, fy, cx, cy)

timestamp = '20260415_145604'

depth_prefix = '../' + timestamp + '/depth_captures/frame_'
color_prefix = '../' + timestamp + '/color_captures/frame_'

data = np.load("../" + timestamp + "_output.npz")

DEPTHIMG_HEIGHT = 192
DEPTHIMG_WIDTH = 256

COLOR_2D = (250, 250, 150) #yellow
COLOR_3D = np.array((250, 150, 250)) #pink
COLOR_CORRECTED = (150, 250, 250) #teal
COLOR_GRIPPER = (40, 40, 40)
COLOR_TRANSFORMED = (150, 150, 250) 
COLOR_LEFT = (250, 0, 0)
COLOR_RIGHT = (0, 0, 250)

URDF_PATH = "../gripper_model/robots/robotiq_arg85_description.URDF" 
urdf = yourdfpy.URDF.load(
        URDF_PATH,
        build_scene_graph=True,
        load_meshes=True,
    )




def smooth_box_quats(quats, sigma=1.0):

    quats_adj = quats.copy()

    for i in range(len(quats_adj)):
        if quats_adj[i, 3] < 0:
            quats_adj[i] = -quats_adj[i]

    smoothed_quats = gaussian_filter1d(quats_adj, sigma=sigma, axis=0)

    norms = np.linalg.norm(smoothed_quats, axis=1, keepdims=True)
    smoothed_quats /= norms

    return smoothed_quats
    


def get_point_cloud(frame_no):
    depth_frame_no = frame_no
    depth_img = Image.open(depth_prefix + str(depth_frame_no).zfill(6) + ".png")
    color_img = Image.open(color_prefix + str(frame_no).zfill(6) + ".png")

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
    return points, depth, colors


server = viser.ViserServer()


#point cloud reconstructed from rgbd data

point_cloud_handle = rendering_utils.initialize_cloud(server, "point cloud", point_size=0.001)

axes_handle = rendering_utils.initialize_axes(server, "aruco pose")

box_handle = rendering_utils.initialize_box(server, "cardboard sheet")

left_gripper_handle, left_urdf_handle = rendering_utils.initialize_gripper(server, "/left gripper", urdf)
right_gripper_handle, right_urdf_handle = rendering_utils.initialize_gripper(server, "/right gripper", urdf)
gripper_frame_handles = [left_gripper_handle, right_gripper_handle]
gripper_urdf_handles = [left_urdf_handle, right_urdf_handle]


hands_centroid = np.array((0.03,0.07,0.45))
#hands_centroid = None


box_bases = data['box_bases']
box_quats = data['box_quats']
left_bases = data['left_bases']
right_bases = data['right_bases']
right_quats = data['right_quats']
left_quats = data['left_quats']
right_gripper_grasps = data['right_gripper_grasps']
left_gripper_grasps = data['left_gripper_grasps']

frames_count = len(box_bases)

print(box_quats)

box_quats = smooth_box_quats(box_quats, sigma=1.2)

fps = 10
dt = 1.0 / fps

print("wahoo made it")

for i in range(frames_count):
    t0 = time.time()

    frame_no = str(i).zfill(6)

    points, depth, colors = get_point_cloud(i)

    rendering_utils.render_cloud(point_cloud_handle, points, colors, hands_centroid)

    rendering_utils.render_box(axes_handle, box_handle, box_quats[i], box_bases[i], hands_centroid)

    rendering_utils.render_gripper(left_gripper_handle, left_urdf_handle, left_bases[i], left_quats[i], left_gripper_grasps[i], hands_centroid)

    rendering_utils.render_gripper(right_gripper_handle, right_urdf_handle, right_bases[i], right_quats[i], right_gripper_grasps[i], hands_centroid)


    elapsed = time.time() - t0
    sleep_time = max(0.0, dt - elapsed)
    time.sleep(sleep_time)


while True:
    time.sleep(1)

