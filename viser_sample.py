import numpy as np
from PIL import Image
import time
import viser

# intrinsics
fx, fy, cx, cy = 1366.3287, 1366.3287, 957.5452, 722.60974

fx *= 0.1333333333
fy *= 0.1333333333
cx *= 0.1333333333
cy *= 0.1333333333

def cloudify_frame(cloud_handle, frame_no):
    depth_img = Image.open(f'../depth_data/depth/{frame_no}.png')
    color_img = Image.open(f'../frames/frame_{frame_no}.png')

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



def cloudify_video(server, num_frames, fps=24):
    dt = 1.0 / fps

    cloud_handle = server.scene.add_point_cloud(
        name="depth_cloud",
        points=np.zeros((1, 3)),
        colors=np.zeros((1, 3)),
        point_size=0.001,
    )

    for i in range(num_frames):
        t0 = time.time()

        frame_no = str(i).zfill(6)

        cloudify_frame(cloud_handle, frame_no)

        elapsed = time.time() - t0
        sleep_time = max(0.0, dt - elapsed)
        time.sleep(sleep_time)

server = viser.ViserServer()

cloudify_video(server, 600)

print("check out http://localhost:8080")

while True:
    time.sleep(1)
