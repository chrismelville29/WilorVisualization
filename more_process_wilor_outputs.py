import numpy as np
from PIL import Image
import time
import cv2

from quaternion_utils import generate_rotation_quaternion
from quaternion_utils import generate_xyzrpy
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

npz_prefix = '../chris_hands/frame_'
depth_prefix = '../board_captures/depth_captures/20260404_122039/frame_'
color_prefix = '../board_captures/color_captures/20260404_122039/frame_'

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

initial_approach = np.array((0, 0, 1.0))
initial_lateral = np.array((-1.0, 0, 0))


npz_path = '../hand_npzs/'

def determine_hands(handednesses):
    left_index, right_index = None, None

    if len(handednesses) == 0:
        print("no hands detected")
    elif len(handednesses) > 2:
        print("too many hands in frame")
    elif len(handednesses) == 2 and handednesses[0] == handednesses[1]:
        print("two of the same side hand")
    elif len(handednesses) == 1:
        if handednesses[0] == 0:
            left_index = 0
        else:
            right_index = 0
    elif len(handednesses) == 2:
        if handednesses[0] == 0:
            left_index = 0
            right_index = 1
        else:
            left_index = 1
            right_index = 0
    return left_index, right_index

def snap_gripper(bases, gripper_directions, grasp_directions, box_quaternion, box_position):
    snapped_bases = []
    snapped_gripper_directions = []
    snapped_grasp_directions = []
    for i in range(len(bases)):
        base = bases[i]
        gripper_direction = gripper_directions[i]
        grasp_direction = grasp_directions[i]


from scipy.spatial.transform import Rotation as R

def snap_gripper(bases, gripper_directions, grasp_directions, box_quaternion, box_position):
    """
    Snap grasp + gripper directions to align with a box frame.

    bases, gripper_directions, grasp_directions: list of (3,) np arrays
    box_quaternion: (w, x, y, z)
    box_position: (3,) np array
    """

    # Convert quaternion → rotation matrix
    quat = np.array(box_quaternion)
    quat_xyzw = np.roll(quat, -1)  # (w,x,y,z) → (x,y,z,w)
    R_box = R.from_quat(quat_xyzw).as_matrix()

    # Box axes (columns of rotation matrix)
    box_axes = [R_box[:, 0], R_box[:, 1], R_box[:, 2]]

    snapped_bases = []
    snapped_gripper_directions = []
    snapped_grasp_directions = []

    for i in range(len(bases)):
        base = bases[i]
        g_dir = gripper_directions[i]

        snapped_grasp = box_axes[2]

        g_proj = g_dir - np.dot(g_dir, snapped_grasp) * snapped_grasp
        snapped_gripper = g_proj / np.linalg.norm(g_proj)

        # --- 3. Recompute base (keep geometry consistent) ---
        # assume base lies along negative grasp direction from box
        # project original offset onto new grasp direction
        plane_point = box_position.reshape(3)
        plane_normal = snapped_grasp  # already aligned with box axis

        offset = base - plane_point
        dist = np.dot(offset, plane_normal)

        snapped_base = base - dist * plane_normal

        # --- store ---
        snapped_bases.append(snapped_base)
        snapped_gripper_directions.append(snapped_gripper)
        snapped_grasp_directions.append(snapped_grasp)

    return snapped_bases, snapped_gripper_directions, snapped_grasp_directions

    
def gripperify_skeletons(skeletons_3d):
    bases = []
    gripper_directions = []
    grasp_directions = []
    graspnesses = []
    for skeleton_3d in skeletons_3d:

        wrist = skeleton_3d[0]
        thumb = skeleton_3d[2]
        fingers = (skeleton_3d[6] + skeleton_3d[10] + skeleton_3d[14] + skeleton_3d[18]) / 4
        base = wrist + (fingers - wrist) / 2

        gripper_end = (thumb + fingers) / 2
        gripper_direction = (gripper_end - base)

        #gripper is bigger than hand, so want to slide gripper back a little
        base -= 2 * gripper_direction

        grasp_direction = fingers - thumb

        thumbtip = skeleton_3d[4]
        fingertips = np.array((skeleton_3d[8], skeleton_3d[12], skeleton_3d[16], skeleton_3d[20]))

        finger_graspnesses = np.linalg.norm(fingertips - thumbtip, axis=1)

        min_graspness = np.min(finger_graspnesses)


        bases.append(base)
        gripper_directions.append(gripper_direction)
        grasp_directions.append(grasp_direction)
        graspnesses.append(min_graspness)

    return bases, gripper_directions, grasp_directions, graspnesses


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

def get_hands_centroid(meshes):
    all_vertices = np.concatenate(meshes, axis=0)
    return np.mean(all_vertices, axis=0)

def render_mesh(handle, mesh, faces, centroid=None):
    rendering_mesh = mesh.copy()
    if centroid is not None:
        rendering_mesh -= centroid
    handle.vertices = rendering_mesh
    handle.faces = faces
    handle.visible = True

def render_meshes(handles, meshes, faces, centroid=None):
    for i in range(len(meshes)):
        render_mesh(handles[i], meshes[i], faces, centroid)

def initialize_mesh(server, name, color):
    handle = server.scene.add_mesh_simple(
        name=name,
        vertices=np.zeros((2, 3)),
        faces=np.zeros((1, 3)),
        color=color
    )
    handle.visible = False
    return handle

def render_grippers(frame_handles, urdf_handles, bases, gripper_directions, grasp_directions, graspnesses, centroid=None):

    for i in range(len(bases)):
        frame_handle = frame_handles[i]
        urdf_handle = urdf_handles[i]
        base = bases[i]
        gripper_direction = gripper_directions[i]
        grasp_direction = grasp_directions[i]
        graspness = graspnesses[i]

        rendering_position = base.copy()

        if centroid is not None:
            rendering_position -= centroid

        #print(rendering_position)

        rotation_quaternion = generate_rotation_quaternion(gripper_direction, grasp_direction, initial_approach, initial_lateral)

        frame_handle.position = rendering_position
        frame_handle.wxyz = rotation_quaternion

        #0.08 arbitrary threshold
        if graspness < GRASPNESS_THRESHOLD:
            urdf_handle.update_cfg(np.array([0.8])) #0.8 for closed.
        else:
            urdf_handle.update_cfg(np.array([0.1]))

def initialize_gripper(server, name):
    handle = server.scene.add_frame(
        name=name,
        show_axes=False,
        position = np.zeros(3)
    )

    gripper = ViserUrdf(server, urdf, root_node_name=name)

    return handle, gripper

def render_box(axis_handle, box_handle, quaternion, position, centroid=None):
    rendering_position = position.copy()
    if centroid is not None:
        rendering_position -= centroid

    axis_handle.wxyz = quaternion
    axis_handle.position = rendering_position

    box_handle.wxyz = quaternion
    box_handle.position = rendering_position

    axis_handle.visible = True
    box_handle.visible = True


def render_clouds(handles, clouds, colors, centroid=None):
    for i in range(len(clouds)):
        render_cloud(handles[i], clouds[i], colors, centroid)

def render_cloud(handle, points, colors, centroid=None):
    rendering_points = points.copy()
    if centroid is not None:
        rendering_points -= centroid

    handle.points = rendering_points
    handle.colors = colors
    handle.visible = True

def initialize_cloud(server, name, point_size):
    handle = server.scene.add_point_cloud(
        name=name,
        points=np.zeros((1, 3)),
        colors=np.zeros((1, 3)),
        point_size=point_size,
    )
    handle.visible = False
    return handle

def initialize_axes(server, name):
    handle = server.scene.add_frame(
        name=name,
        wxyz=(1, 0, 0, 0),
        position=np.array((0, 0, 0)),
    )
    handle.visible = False
    return handle

def initialize_box(server, name):
    handle = server.scene.add_box(
        name=name,
        dimensions=(.304, .304, .004), 
        color=(255, 0, 0),
        position=np.zeros(3),   
    )
    handle.visible = False
    return handle

server = viser.ViserServer()

    

#point cloud reconstructed from rgbd data

point_cloud_handle = initialize_cloud(server, "point cloud", point_size=0.001)

axes_handle = initialize_axes(server, "aruco pose")

box_handle = initialize_box(server, "cardboard sheet")

left_original_handle = initialize_mesh(server, "left hand original", COLOR_LEFT)
right_original_handle = initialize_mesh(server, "right hand original", COLOR_RIGHT)
original_handles = [left_original_handle, right_original_handle]

left_gripper_handle, left_urdf_handle = initialize_gripper(server, "/left gripper")
right_gripper_handle, right_urdf_handle = initialize_gripper(server, "/right gripper")
gripper_frame_handles = [left_gripper_handle, right_gripper_handle]
gripper_urdf_handles = [left_urdf_handle, right_urdf_handle]


left_gripper_poses = []
right_gripper_poses = []
left_graspnesses = []
right_graspnesses = []
box_poses = []


hands_centroid = np.array((0.03,0.07,0.45))
hands_centroid = None

print("wahoo made it")
fps = 100
dt = 1.0 / fps

start_frame = 400
end_frame = 1100

for i in range(start_frame, end_frame):
    t0 = time.time()

    frame_no = str(i).zfill(6)

    wilor_data = np.load(npz_prefix + str(i).zfill(6) + '.npz')
    meshes_3d, meshes_2d, skeletons_2d, skeletons_3d, faces, handednesses = wilor_data['meshes_3d'], wilor_data['meshes_2d'], wilor_data['skeletons_2d'], wilor_data['skeletons_3d'], wilor_data['faces'], wilor_data['handednesses']

    points, depth, colors = get_point_cloud(i)

    color_img = cv2.imread(color_prefix + frame_no + '.png')
    
    _, rvec, tvec = reconstruct.detect_aruco_pose(color_img, intrinsic_matrix, None, 0.05)[0]
    box_quaternion, box_position = reconstruct.quatnpos_from_vector(tvec, rvec)

    #original hands -> gripper
    gripper_bases, gripper_directions, grasp_directions, graspnesses = gripperify_skeletons(skeletons_3d)

    gripper_bases, gripper_directions, grasp_directions = snap_gripper(gripper_bases, gripper_directions, grasp_directions, box_quaternion, box_position)


    left_index, right_index = determine_hands(handednesses)

    render_cloud(point_cloud_handle, points, colors, hands_centroid)

    render_box(axes_handle, box_handle, box_quaternion, box_position, hands_centroid)
    box_pose = generate_xyzrpy(box_quaternion, box_position)
    box_poses.append(box_pose)

    if left_index is not None:
        render_mesh(left_original_handle, meshes_3d[left_index], faces, hands_centroid)
        rotation_quaternion = generate_rotation_quaternion(gripper_directions[left_index], grasp_directions[left_index], initial_approach, initial_lateral)
        left_pose = generate_xyzrpy(rotation_quaternion, gripper_bases[left_index])
        left_gripper_poses.append(left_pose)
        if graspnesses[left_index] < GRASPNESS_THRESHOLD:
            left_graspnesses.append(1)
        else:
            left_graspnesses.append(0)
    else:
        left_original_handle.visible = False
        left_gripper_poses.append(np.full(6, np.inf))
        left_graspnesses.append(0)
    if right_index is not None:
        render_mesh(right_original_handle, meshes_3d[right_index], faces, hands_centroid)
        rotation_quaternion = generate_rotation_quaternion(gripper_directions[right_index], grasp_directions[right_index], initial_approach, initial_lateral)
        right_pose = generate_xyzrpy(rotation_quaternion, gripper_bases[right_index])
        right_gripper_poses.append(right_pose)
        if graspnesses[right_index] < GRASPNESS_THRESHOLD:
            right_graspnesses.append(1)
        else:
            right_graspnesses.append(0)
    else:
        right_original_handle.visible = False
        right_gripper_poses.append(np.full(6, np.inf))
        right_graspnesses.append(0)
    

    render_grippers(gripper_frame_handles, gripper_urdf_handles, gripper_bases, gripper_directions, grasp_directions, graspnesses, hands_centroid)


    elapsed = time.time() - t0
    sleep_time = max(0.0, dt - elapsed)
    time.sleep(sleep_time)


box_poses = np.array(box_poses)
right_gripper_poses = np.array(right_gripper_poses)
left_gripper_poses = np.array(left_gripper_poses)
right_gripper_grasps = np.array(right_graspnesses)
left_gripper_grasps = np.array(left_graspnesses)

np.savez('demo_0_poses.npz', box_poses=box_poses, left_gripper_poses=left_gripper_poses, right_gripper_poses=right_gripper_poses, left_gripper_grasps=left_gripper_grasps, right_gripper_grasps=right_gripper_grasps)



while True:
    time.sleep(1)

