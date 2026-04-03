import numpy as np
from PIL import Image
import time

import cluster_utils

import viser
import sklearn

CLUSTER_PATH = "cluster_labels.npy"
N_CLUSTERS = 24
REFERENCE_VERTICES = 300

npzs_path = '../handed_npzs/'

#intrinsics
fx, fy, cx, cy = 1366.3287, 1366.3287, 957.5452, 722.60974

fx *= 0.1333333333
fy *= 0.1333333333
cx *= 0.1333333333
cy *= 0.1333333333

DEPTHIMG_HEIGHT = 192
DEPTHIMG_WIDTH = 256

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

def split_hands(wilor_data):
    left_data, right_data = None, None

    faces, handednesses = wilor_data['faces'], wilor_data['handednesses']


    if len(handednesses) == 0:
        print("no hands detected")
        return left_data, right_data
    if len(handednesses) > 2:
        print("too many hands in frame")
        return left_data, right_data
    if len(handednesses) == 2 and handednesses[0] == handednesses[1]:
        print("two of the same side hand")
        return left_data, right_data

    meshes_3d, meshes_2d = wilor_data['meshes_3d'], wilor_data['meshes_2d']
    skeletons_2d, skeletons_3d =  wilor_data['skeletons_2d'], wilor_data['skeletons_3d']

    if len(handednesses) == 1:
        if handednesses[0] == 0:
            left_data = meshes_3d[0], meshes_2d[0], skeletons_2d[0], skeletons_3d[0], faces
        else:
            right_data = meshes_3d[0], meshes_2d[0], skeletons_2d[0], skeletons_3d[0], faces
    elif len(handednesses) == 2:
        if handednesses[0] == 0:
            left_data = meshes_3d[0], meshes_2d[0], skeletons_2d[0], skeletons_3d[0], faces
            right_data = meshes_3d[1], meshes_2d[1], skeletons_2d[1], skeletons_3d[1], faces
        else:
            right_data = meshes_3d[0], meshes_2d[0], skeletons_2d[0], skeletons_3d[0], faces
            left_data = meshes_3d[1], meshes_2d[1], skeletons_2d[1], skeletons_3d[1], faces
    
    return left_data, right_data



def depthify_2d_hands(depth, hands_2d):

    hands_depthified = []

    for i in range(len(hands_2d)):

        
        hand_2d = hands_2d[i]

        uv = np.round(hand_2d).astype(int)
        u = np.clip(uv[:, 0], a_min=0, a_max=DEPTHIMG_WIDTH-1)
        v = np.clip(uv[:, 1], a_min=0, a_max=DEPTHIMG_HEIGHT-1)

        # Gather depths in one shot
        Z = depth[v, u]

        # Compute XYZ
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        hand_depthified = np.stack([X, Y, Z], axis=1)

        hands_depthified.append(hand_depthified)

    return hands_depthified


def get_ratios(meshes_depthified, meshes):

    hand_ratios = []
    for i in range(len(meshes_depthified)):
        cloud_skeleton = meshes_depthified[i]
        mesh = meshes[i]

        cloud_norms = np.linalg.norm(cloud_skeleton, axis=1)
        mesh_norms = np.linalg.norm(mesh, axis=1)

        ratios = (mesh_norms / cloud_norms)[:REFERENCE_VERTICES]

        hand_ratios.append(ratios)
    return hand_ratios


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


if __name__ == "__main__":
        
    frame_no = '000401'

    wilor_data = np.load(npzs_path + 'frame_' + frame_no + '.npz')
    meshes_3d, meshes_2d, skeletons_2d, skeletons_3d, faces, handednesses = wilor_data['meshes_3d'], wilor_data['meshes_2d'], wilor_data['skeletons_2d'], wilor_data['skeletons_3d'], wilor_data['faces'], wilor_data['handednesses']

    mesh = meshes_3d[0]

    REFERENCE_VERTICES = 300

    mesh = mesh[:REFERENCE_VERTICES]


    kmeans = sklearn.cluster.KMeans(n_clusters=N_CLUSTERS, init='k-means++', n_init=3, random_state=0)
    kmeans.fit(mesh)
    labels = kmeans.labels_


    np.save(CLUSTER_PATH, labels)

    labels = np.load(CLUSTER_PATH)

    server = viser.ViserServer()

    left_hand, right_hand = split_hands(wilor_data)
    left_index, right_index = determine_hands(handednesses)

    points, depth, colors = get_point_cloud(frame_no)
    cluster_rosters = cluster_utils.get_cluster_rosters(CLUSTER_PATH, N_CLUSTERS)

    meshes_depthified = depthify_2d_hands(depth, meshes_2d)

    hand_ratios = get_ratios(meshes_depthified, meshes_3d)

    ratios = hand_ratios[0]

    median_idx_sets = cluster_utils.get_cluster_median_sets(cluster_rosters, hand_ratios)
    median_point_sets_depthified = cluster_utils.pointify_median_idx_sets(median_idx_sets, meshes_depthified)
    median_point_sets_3d = cluster_utils.pointify_median_idx_sets(median_idx_sets, meshes_3d)

    transform = cluster_utils.find_transformation(median_idx_sets[0], meshes_3d[0], meshes_depthified[0])
    transformed_mesh = cluster_utils.apply_transformation(meshes_3d[0], transform)

    boring_mesh = cluster_utils.boring_transform(meshes_3d[0], ratios, median_idx_sets[0])

    centroid = np.mean(mesh, axis=0)
    #centroid = np.zeros(3)
    meshes_3d -= centroid
    points -= centroid
    transformed_mesh -= centroid
    median_point_sets_depthified -= centroid
    median_point_sets_3d -= centroid
    boring_mesh -= centroid





    #print(transform)


    #print(transformed_mesh)




    server.scene.add_point_cloud(
            name="point_cloud",
            points=points,
            colors=colors,
            point_size=0.001,
        )

    ratio_colors = np.zeros((REFERENCE_VERTICES, 3))
    for i in range(REFERENCE_VERTICES):
        ratio_colors[i] = np.array(((ratios[i] - 0.6) * 1.5, 0, 0))

    '''
    server.scene.add_point_cloud(
        name="ratios",
        points=mesh,
        colors=ratio_colors,
        point_size=0.01
    )
    '''

    server.scene.add_mesh_simple(
            name="mesh",
            vertices=meshes_3d[0],
            faces=faces,
            color=(250, 0, 0)
        )

    server.scene.add_mesh_simple(
        name="transformed mesh",
        vertices = transformed_mesh,
        faces=faces,
        color=(0,0,1)
    )


    server.scene.add_mesh_simple(
        name="boring mesh",
        vertices=boring_mesh,
        faces=faces,
        color=(0, 100, 0)
    )



    cluster_colors = np.zeros((REFERENCE_VERTICES, 3))
    for i in range(REFERENCE_VERTICES):
        label = labels[i]
        cluster_colors[i] = np.array((label // 4 / 6, label % 3 / 2, label % 2))


    server.scene.add_point_cloud(
            name="clusters",
            points=mesh,
            colors=cluster_colors,
            point_size=0.01,
        )

    for i in range(len(median_point_sets_depthified)):
        server.scene.add_point_cloud(
            name=f"depthified medians_{i}",
            points = median_point_sets_depthified[i],
            colors=np.array((0, 0, 250)),
            point_size=0.01)

    for i in range(len(median_point_sets_3d)):
        server.scene.add_point_cloud(
            name=f"wilor medians_{i}",
            points = median_point_sets_3d[i],
            colors=np.array((0, 250, 0)),
            point_size=0.01)

    while True:
        time.sleep(1)