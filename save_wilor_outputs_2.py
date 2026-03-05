import torch
import cv2
import numpy as np

from wilor.models import load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset
from ultralytics import YOLO

COLOR_2D = (250, 250, 150) #yellow
COLOR_3D = (250, 150, 250) #pink
COLOR_CORRECTED = (150, 250, 250) #teal
COLOR_GRIPPER = (40, 40, 40)


#intrinsics
fx, fy, cx, cy = 1366.3287, 1366.3287, 957.5452, 722.60974

fx *= 0.1333333333
fy *= 0.1333333333
cx *= 0.1333333333
cy *= 0.1333333333


npz_path = '../hand_npzs/'

def load_model():
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
    
    return model, model_cfg, detector, device, faces


def save_wilor_hands(model, model_cfg, detector, device, faces, frame_no):

    img_path = '../frames/frame_'+frame_no+'.png'
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




    np.savez(npz_path + "frame_"+frame_no+'.npz', meshes_2d=meshes_2d, meshes_3d=meshes_3d, skeletons_2d=skeletons_2d, skeletons_3d=skeletons_3d, faces=faces)

    print("done with big method")

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


model, model_cfg, detector, device, faces = load_model()


for i in range(2):
    frame_no = str(i).zfill(6)
    save_wilor_hands(model, model_cfg, detector, device, faces, frame_no)


