import numpy as np
from PIL import Image
import time

from quaternion_utils import generate_rotation_quaternion
import viser
import yourdfpy
from viser.extras import ViserUrdf

#intrinsics
fx, fy, cx, cy = 1366.3287, 1366.3287, 957.5452, 722.60974

fx *= 0.1333333333
fy *= 0.1333333333
cx *= 0.1333333333
cy *= 0.1333333333

DEPTHIMG_HEIGHT = 192
DEPTHIMG_WIDTH = 256

COLOR_2D = (250, 250, 150) #yellow
COLOR_3D = (250, 150, 250) #pink
COLOR_CORRECTED = (150, 250, 250) #teal
COLOR_GRIPPER = (40, 40, 40)

URDF_PATH = "../gripper_model/robots/robotiq_arg85_description.URDF" 
urdf = yourdfpy.URDF.load(
        URDF_PATH,
        build_scene_graph=True,
        load_meshes=True,
    )

initial_approach = np.array((0, 0, 1.0))
initial_lateral = np.array((-1.0, 0, 0))


npz_path = '../hand_npzs/'
frame_no = '000450'
img_path = 'frame_'+frame_no+'.png'

def correct_hand_depths(meshes_depthified, skeletons_3d, meshes):

    corrected_skeletons = []
    corrected_meshes = []
    for i in range(len(meshes_depthified)):
        cloud_skeleton = meshes_depthified[i]
        skeleton_3d = skeletons_3d[i]
        mesh = meshes[i]

        cloud_norms = np.linalg.norm(cloud_skeleton, axis=1)
        mesh_norms = np.linalg.norm(mesh, axis=1)

        ratios = cloud_norms / mesh_norms


        scale_factor = np.sort(ratios)[-50]

        corrected_skeletons.append(scale_factor * skeleton_3d)
        corrected_meshes.append(scale_factor * mesh)
    return corrected_skeletons, corrected_meshes

def depthify_2d_hands(depth, hands_2d):

    hands_depthified = []

    for i in range(len(hands_2d)):

        
        hand_2d = hands_2d[i]

        uv = np.floor(hand_2d).astype(int)
        u = np.clip(uv[:, 0], a_min=0, a_max=DEPTHIMG_HEIGHT-1)
        v = np.clip(uv[:, 1], a_min=0, a_max=DEPTHIMG_WIDTH-1)

        # Gather depths in one shot
        Z = depth[v, u]

        # Compute XYZ
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        hand_depthified = np.stack([X, Y, Z], axis=1)

        hands_depthified.append(hand_depthified)

    return hands_depthified

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
        base -= gripper_direction

        grasp_direction = fingers - thumb

        thumbtip = skeleton_3d[4]
        fingertips = np.array((skeleton_3d[8], skeleton_3d[12], skeleton_3d[16], skeleton_3d[20]))

        finger_graspnesses = np.linalg.norm(fingertips - thumbtip, axis=1)

        #print("graspnesses")
        #print(finger_graspnesses)
        min_graspness = np.min(finger_graspnesses)


        bases.append(base)
        gripper_directions.append(gripper_direction)
        grasp_directions.append(grasp_direction)
        graspnesses.append(min_graspness)

    return bases, gripper_directions, grasp_directions, graspnesses


def get_point_cloud(frame_no):
    depth_img = Image.open('../depth_data/depth/'+frame_no+".png")
    color_img = Image.open('../frames/frame_'+frame_no+".png")

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


def render_meshes(handles, meshes, faces, centroid=None):

    for i in range(len(meshes)):
        mesh = meshes[i]
        handle = handles[i]

        rendering_mesh = mesh.copy()
        if centroid is not None:
            rendering_mesh -= centroid

        handle.vertices = rendering_mesh
        handle.faces = faces

def initialize_mesh(server, name, color):

    return server.scene.add_mesh_simple(
        name=name,
        vertices=np.zeros((2, 3)),
        faces=np.zeros((1, 3)),
        color=color
    )

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
        if graspness < 0.05:
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

def render_dots(handle, points, centroid=None):

    range_start = 30
    range_end = 60

    rendering_points = points.copy()
    if centroid is not None:
        rendering_points -= centroid

    handle.points = rendering_points

    colors = np.zeros((778, 3))
    for i in range(778):
        if i >= range_start and i < range_end:
            colors[i] = np.array((1, 0, 0))
        else:
            colors[i] = np.array((0, 0, 0))

    print(colors)

    handle.colors = colors

def render_cloud(handle, points, colors, centroid=None):
    rendering_points = points.copy()
    if centroid is not None:
        rendering_points -= centroid

    handle.points = rendering_points
    handle.colors = colors

def initialize_cloud(server, name, point_size):
    return server.scene.add_point_cloud(
        name=name,
        points=np.zeros((1, 3)),
        colors=np.zeros((1, 3)),
        point_size=point_size,
    )

server = viser.ViserServer()

    

#point cloud reconstructed from rgbd data

point_cloud_handle = initialize_cloud(server, "point cloud", point_size=0.001)

left_corrected_handle = initialize_mesh(server, "left hand corrected", COLOR_CORRECTED)
right_corrected_handle = initialize_mesh(server, "right hand corrected", COLOR_CORRECTED)
corrected_handles = [left_corrected_handle, right_corrected_handle]

left_original_handle = initialize_mesh(server, "left hand original", COLOR_3D)
right_original_handle = initialize_mesh(server, "right hand original", COLOR_3D)
original_handles = [left_original_handle, right_original_handle]

left_gripper_handle, left_urdf_handle = initialize_gripper(server, "/left gripper")
right_gripper_handle, right_urdf_handle = initialize_gripper(server, "/right gripper")
gripper_frame_handles = [left_gripper_handle, right_gripper_handle]
gripper_urdf_handles = [left_urdf_handle, right_urdf_handle]

#corrected_hand_cloud_handle = initialize_cloud(server, "corrected hand cloud", point_size=0.01)





hands_centroid = np.array((0.03,0.07,0.45))
print("wahoo made it")
fps = 12
dt = 1.0 / fps

start_frame = 200
end_frame = 201

for i in range(start_frame, end_frame):
    t0 = time.time()

    frame_no = str(i).zfill(6)

    wilor_data = np.load('../hand_npzs/frame_' + frame_no + '.npz')
    meshes_3d, meshes_2d, skeletons_2d, skeletons_3d, faces = wilor_data['meshes_3d'], wilor_data['meshes_2d'], wilor_data['skeletons_2d'], wilor_data['skeletons_3d'], wilor_data['faces']

    #hands_centroid = get_hands_centroid(meshes_3d)
    #hands_centroid = None
    #print(hands_centroid)
    points, depth, colors = get_point_cloud(frame_no)
    skeletons_depthified = depthify_2d_hands(depth, skeletons_2d)
    meshes_depthified = depthify_2d_hands(depth, meshes_2d)

    corrected_skeletons, corrected_meshes = correct_hand_depths(meshes_depthified, skeletons_3d, meshes_3d)
    #corrected hands
    gripper_bases, gripper_directions, grasp_directions, graspnesses = gripperify_skeletons(corrected_skeletons)
    #original hands
    #gripper_bases, gripper_directions, grasp_directions, graspnesses = gripperify_skeletons(skeletons_3d)


    render_cloud(point_cloud_handle, points, colors, hands_centroid)

    #render_dots(corrected_hand_cloud_handle, corrected_meshes[0], hands_centroid)

    render_meshes(corrected_handles, corrected_meshes, faces, hands_centroid)
    render_meshes(original_handles, meshes_3d, faces, hands_centroid)

    render_grippers(gripper_frame_handles, gripper_urdf_handles, gripper_bases, gripper_directions, grasp_directions, graspnesses, hands_centroid)


    elapsed = time.time() - t0
    sleep_time = max(0.0, dt - elapsed)
    time.sleep(sleep_time)


while True:
    time.sleep(1)

