import torch
import cv2
import numpy as np
from PIL import Image
import time

from quaternion_utils import generate_rotation_quaternion
from wilor.models import load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset
from ultralytics import YOLO
import viser
import yourdfpy
from viser.extras import ViserUrdf

COLOR_2D = (250, 250, 150) #yellow
COLOR_3D = (250, 150, 250) #pink
COLOR_CORRECTED = (150, 250, 250) #teal
COLOR_GRIPPER = (40, 40, 40)

URDF_PATH = "../gripper_model/robots/robotiq_arg85_description.URDF" 
initial_approach = np.array((0, 0, 1.0))
initial_lateral = np.array((-1.0, 0, 0))


#intrinsics
fx, fy, cx, cy = 1366.3287, 1366.3287, 957.5452, 722.60974

fx *= 0.1333333333
fy *= 0.1333333333
cx *= 0.1333333333
cy *= 0.1333333333


def cam_crop_to_full(cam_bbox, box_center, box_size, img_size, focal_length=5000.):
    img_w, img_h = img_size[:, 0], img_size[:, 1]
    cx, cy, b = box_center[:, 0], box_center[:, 1], box_size
    w_2, h_2 = img_w / 2., img_h / 2.
    bs = b * cam_bbox[:, 0] + 1e-9
    tz = 2 * focal_length / bs
    tx = (2 * (cx - w_2) / bs) + cam_bbox[:, 1]
    ty = (2 * (cy - h_2) / bs) + cam_bbox[:, 2]
    full_cam = torch.stack([tx, ty, tz], dim=-1)
    return full_cam

def get_wilor_hands():

    # Load models
    model, model_cfg = load_wilor(
        checkpoint_path='../pretrained_models/wilor_final.ckpt',
        cfg_path='../pretrained_models/model_config.yaml'
    )
    detector = YOLO('../pretrained_models/detector.pt')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    detector = detector.to(device)
    model.eval()

    #ask chatgpt for the ifelse tree back if this breaks
    faces = model.mano.faces
    
    print("everything loaded")

    img_path = '../frames/frame_000410.png'
    img_cv2 = cv2.imread(str(img_path))

    H, W, _ = img_cv2.shape

    detections = detector(img_cv2, conf=0.3, verbose=False)[0]

    bboxes = []
    is_right = []
    for det in detections:
        Bbox = det.boxes.data.cpu().detach().squeeze().numpy()
        is_right.append(det.boxes.cls.cpu().detach().squeeze().item())
        bboxes.append(Bbox[:4].tolist())

    if len(bboxes) == 0:
        return

    boxes = np.stack(bboxes)
    right = np.stack(is_right)

    dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right,
                            rescale_factor=2.0) #2.0 is what was in the demo code. idk what it means

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=False, num_workers=0)

    meshes_3d = []
    meshes_2d = []
    skeletons_3d = []
    skeletons_2d = []


    for batch in dataloader:
        batch = recursive_to(batch, device)

        with torch.no_grad():
            out = model(batch)


        multiplier = (2*batch['right']-1)
        pred_cam = out['pred_cam']
        pred_cam[:,1] = multiplier*pred_cam[:,1]

        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()
        
        scaled_focal_length = fx

        pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length)

        batch_size = batch['img'].shape[0]

        for hand in range(batch_size):
            verts = out['pred_vertices'][hand]
            kpts_3d = out['pred_keypoints_3d'][hand]
            kpts_2d = out['pred_keypoints_2d'][hand]
            verts_2d = out['pred_vertices_2d'][hand]
            

            is_right = batch['right'][hand]

            # WiLoR handedness flip
            verts[:,0]  = (2*is_right-1)*verts[:,0]
            verts_2d[:,0]  = (2*is_right-1)*verts_2d[:,0]
            kpts_3d[:,0] = (2*is_right-1)*kpts_3d[:,0]
            kpts_2d[:,0] = (2*is_right-1)*kpts_2d[:,0]

            cam_t = pred_cam_t_full[hand]

            # Move 3d stuff into camera space
            verts_cam = verts + cam_t
            kpts_3d_cam = kpts_3d + cam_t

            #uncrop 2d stuff
            kpts_2d_cam = wilor_to_2d(
                kpts_2d,
                box_center[hand],
                box_size[hand]
            )
            verts_2d_cam = wilor_to_2d(
                verts_2d,
                box_center[hand],
                box_size[hand]
            )

            #shift view 3d vertices
            verts_cam = wilor_to_metric_camera(verts_cam, W, H)
            kpts_3d_cam = wilor_to_metric_camera(kpts_3d_cam, W, H)

            meshes_3d.append(verts_cam)
            meshes_2d.append(verts_2d_cam)
            skeletons_2d.append(kpts_2d_cam)
            skeletons_3d.append(kpts_3d_cam)


    print("done with big method")

    return meshes_3d, meshes_2d, skeletons_2d, skeletons_3d, faces

#turn 2d points in bounding box crop space back into image space
def wilor_to_2d(kpts_crop, box_center, box_size):
    kpts_img = kpts_crop * (box_size)
    kpts_img[:, 0] += box_center[0]
    kpts_img[:, 1] += box_center[1]

    return kpts_img

def wilor_to_metric_camera(points_wilor, W, H):
    """
    Convert WiLoR-space points back into original metric camera XYZ.
    """
    Xw, Yw, Z = points_wilor.T

    Xc = Xw - (cx - W / 2.0) * Z / fx
    Yc = Yw - (cy - H / 2.0) * Z / fy

    return np.stack([Xc, Yc, Z], axis=1)

def correct_hand_depths(meshes_depthified, skeletons_3d, meshes):

    corrected_skeletons = []
    corrected_meshes = []
    for i in range(len(meshes_depthified)):
        cloud_skeleton = meshes_depthified[i]
        skeleton_3d = skeletons_3d[i]
        mesh = meshes[i]

        print(cloud_skeleton.shape)
        print(skeleton_3d.shape)
        print(mesh.shape)

        cloud_norms = np.linalg.norm(cloud_skeleton, axis=1)
        mesh_norms = np.linalg.norm(mesh, axis=1)

        ratios = cloud_norms / mesh_norms


        scale_factor = np.sort(ratios)[-105]

        print("scale_factor coming up")
        print(scale_factor)
        corrected_skeletons.append(scale_factor * skeleton_3d)
        corrected_meshes.append(scale_factor * mesh)
    return corrected_skeletons, corrected_meshes

def depthify_2d_hands(depth, hands_2d):

    hands_depthified = []

    for i in range(len(hands_2d)):

        
        hand_2d = hands_2d[i]

        uv = np.floor(hand_2d).int()
        u = uv[:, 0]
        v = uv[:, 1]

        # Gather depths in one shot
        Z = depth[v, u]

        # Compute XYZ
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        hand_depthified = np.stack([X, Y, Z], axis=1)

        hands_depthified.append(hand_depthified)

    return hands_depthified

def real_gripperify_skeletons(skeletons_3d):
    bases = []
    gripper_directions = []
    grasp_directions = []
    graspnesses = []
    for skeleton_3d in skeletons_3d:

        base = skeleton_3d[0]
        thumb = skeleton_3d[2]
        fingers = (skeleton_3d[6] + skeleton_3d[10] + skeleton_3d[14] + skeleton_3d[18]) / 4

        gripper_end = (thumb + fingers) / 2
        gripper_direction = (gripper_end - base)

        #gripper is bigger than hand, so want to slide gripper back a little
        base -= gripper_direction / 2

        grasp_direction = fingers - thumb

        thumbtip = skeleton_3d[4]
        fingertips = np.array((skeleton_3d[8], skeleton_3d[12], skeleton_3d[16], skeleton_3d[20]))

        finger_graspnesses = np.linalg.norm(fingertips - thumbtip, axis=1)

        print("graspnesses")
        print(finger_graspnesses)
        min_graspness = np.min(finger_graspnesses)


        bases.append(base)
        gripper_directions.append(gripper_direction)
        grasp_directions.append(grasp_direction)
        graspnesses.append(min_graspness)

    return bases, gripper_directions, grasp_directions, graspnesses

def gripperify_skeleton(skeletons_3d):

    bases = []
    gripper_directions = []
    for skeleton_3d in skeletons_3d:


        base = skeleton_3d[0]
        thumb = skeleton_3d[2]
        fingers = (skeleton_3d[5] + skeleton_3d[9] + skeleton_3d[13] + skeleton_3d[17]) / 4

        gripper_end = (thumb + fingers) / 2
        gripper_direction = (gripper_end - base)

        #gripper is bigger than hand, so want to slide gripper back a little
        base -= gripper_direction / 3

        gripper_direction /= np.linalg.norm(gripper_direction)

        

        bases.append(base)
        gripper_directions.append(gripper_direction)

    return bases, gripper_directions

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


def render_mesh(server, name, mesh, faces, color, centroid=None):

    rendering_mesh = mesh.copy()
    if centroid is not None:
        rendering_mesh -= centroid

    server.scene.add_mesh_simple(
        name=name,
        vertices=rendering_mesh,
        faces=faces,
        color=color
    )

def render_real_gripper(server, name, base, gripper_direction, grasp_direction, graspness, centroid=None):

    rendering_position = base.copy()

    if centroid is not None:
        rendering_position -= centroid

    print(rendering_position)

    rotation_quaternion = generate_rotation_quaternion(gripper_direction, grasp_direction, initial_approach, initial_lateral)

    server.scene.add_frame(
            name=name,
            show_axes=False,
            position = rendering_position,
            wxyz = rotation_quaternion
        )

    # Load a URDF from robot_descriptions package.
    open_gripper = ViserUrdf(server, urdf, root_node_name=name)

    #0.08 arbitrary threshold
    if graspness < 0.06:
        open_gripper.update_cfg(np.array([0.8])) #0.8 for closed.


def render_gripper(server, name, base, gripper_direction, centroid=None):

    if centroid is not None:
        base -= centroid
    
    destination = base + gripper_direction

    lines = np.vstack((base, destination)).reshape(1, 2, 3)

    print("lines upcoming")
    print(lines)

    server.scene.add_line_segments(
        name=name,
        points=lines,
        colors=COLOR_GRIPPER,
        line_width=7.0,
    )

def render_cloud(server, name, points, colors, point_size, centroid=None):

    rendering_points = points.copy()
    if centroid is not None:
        rendering_points -= centroid

    server.scene.add_point_cloud(
        name=name,
        points=rendering_points,
        colors=colors,
        point_size=point_size,
    )

frame_no = '000410'
server = viser.ViserServer()

urdf = yourdfpy.URDF.load(
        URDF_PATH,
        build_scene_graph=True,
        load_meshes=True,
    )

points, depth, colors = get_point_cloud(frame_no)
print(depth.shape)

#exit()
meshes_3d, meshes_2d, skeletons_2d, skeletons_3d, faces = get_wilor_hands()
hands_centroid = get_hands_centroid(meshes_3d)
#hands_centroid = None
points, depth, colors = get_point_cloud(frame_no)
skeletons_depthified = depthify_2d_hands(depth, skeletons_2d)
meshes_depthified = depthify_2d_hands(depth, meshes_2d)

#old way
#corrected_skeletons, corrected_meshes = correct_hand_depths(skeletons_depthified, skeletons_3d, meshes_3d)

#new way
corrected_skeletons, corrected_meshes = correct_hand_depths(meshes_depthified, skeletons_3d, meshes_3d)

gripper_bases, gripper_directions, grasp_directions, graspnesses = real_gripperify_skeletons(corrected_skeletons)



#This is for rendering all of the keypoint joints. needed for testing but make it noisy
for i in range(len(skeletons_3d)):
    #wilor's 3d keypoints
    render_cloud(server, f"3dkpts_{i}", skeletons_3d[i], COLOR_3D, point_size=0.01, centroid=hands_centroid)
    #wilor's 2d keypoints with rgbd depth data on top
    render_cloud(server, f"2dkpts_{i}", skeletons_depthified[i], COLOR_2D, point_size=0.01, centroid=hands_centroid)
    #wilor's 2d vertices with rgbd depth data on top
    #render_cloud(server, f"2dmesh_{i}", meshes_depthified[i], COLOR_2D, point_size=0.005, centroid=hands_centroid)
    #the 3d keypoints using the scale from the 2d depthification
    #render_cloud(server, f"corrected_kpts_{i}", corrected_skeletons[i], COLOR_CORRECTED, point_size=0.01, centroid=hands_centroid)
    #mesh before correction
    render_mesh(server, f"original_hand_{i}", meshes_3d[i], faces, COLOR_3D, centroid=hands_centroid)
    #2d depthified mesh
    #render_mesh(server, f"2d_depthified_hand{i}", meshes_depthified[i], faces, COLOR_2D, centroid=hands_centroid)
    #mesh after correction
    render_mesh(server, f"corrected_hand_{i}", corrected_meshes[i], faces, COLOR_CORRECTED, centroid=hands_centroid)
    #gripper after correction
    #render_gripper(server, f"gripper_{i}", gripper_bases[i], gripper_directions[i], centroid=hands_centroid)
    #real grippers
    render_real_gripper(server, f"/gripper_{i}", corrected_skeletons[i][0], gripper_directions[i], grasp_directions[i], graspnesses[i], centroid=hands_centroid)

    

#point cloud reconstructed from rgbd data
render_cloud(server, "point cloud", points, colors, point_size=0.001, centroid=hands_centroid)


print("wahoo made it")

while True:
    time.sleep(1)

