import numpy as np
from PIL import Image
import time

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

def get_cluster_rosters():
    labels = np.load(CLUSTER_PATH)
    clusters = []
    for i in range(N_CLUSTERS):
        clusters.append([])

    for i in range(len(labels)):
        clusters[labels[i]].append(i)
 
    for i in range(N_CLUSTERS):
        clusters[i] = np.array(clusters[i])
    
    return clusters

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

def get_cluster_median_sets(clusters, ratiis):
    median_sets = []
    for ratii in ratiis:
        medians = []

        for cluster in clusters:
            # ratios belonging to this cluster
            cluster_ratios = ratii[cluster]

            # index of median within the cluster
            order = np.argsort(cluster_ratios)
            median_local_idx = order[len(order) // 2]

            # convert back to original ratio index
            medians.append(cluster[median_local_idx])


        medians = np.array(medians)

        # sort medians by their ratio values
        medians = medians[np.argsort(ratii[medians])][3:6]

        median_sets.append(medians)
    return median_sets

def pointify_median_idx_sets(median_idx_sets, meshes_3d):
    median_point_sets = []
    for i in range(len(meshes_3d)):
        median_idxs = median_idx_sets[i]
        median_points = meshes_3d[i][median_idxs]
        median_point_sets.append(median_points)
    return median_point_sets

def find_transformation(idx_set, meshes_3d, meshes_depthified):
    corrs_3d = meshes_3d[idx_set]                 # source
    corrs_depthified = meshes_depthified[idx_set] # target

    # 1. Compute centroids
    centroid_3d = np.mean(corrs_3d, axis=0)
    centroid_depth = np.mean(corrs_depthified, axis=0)

    # 2. Center the points
    Y = corrs_3d - centroid_3d
    X = corrs_depthified - centroid_depth

    # 3. Covariance
    H = X.T @ Y

    # 4. SVD
    U, S, Vt = np.linalg.svd(H)

    # 5. Rotation
    R = Vt.T @ U.T

    # 6. Reflection fix
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # 7. Translation
    t = centroid_depth - R @ centroid_3d

    # 8. Homogeneous transform
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T

def apply_transformation(points, transform):
    """
    points: (N, 3) numpy array
    T: (4, 4) homogeneous transformation matrix

    returns: (N, 3) transformed points
    """
    # convert to homogeneous coordinates (N, 4)
    ones = np.ones((points.shape[0], 1))
    points_h = np.hstack((points, ones))

    # apply transformation
    transformed_h = (transform @ points_h.T).T

    # convert back to (N, 3)
    return transformed_h[:, :3]


def boring_transform(points, ratios, idx_set):
    scale_factor = np.median(ratios[idx_set])
    return points / scale_factor
    




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


    print(handednesses.shape)

    mesh = meshes_3d[0]

    REFERENCE_VERTICES = 300

    mesh = mesh[:REFERENCE_VERTICES]


    kmeans = sklearn.cluster.KMeans(n_clusters=N_CLUSTERS, init='k-means++', n_init=3, random_state=0)
    kmeans.fit(mesh)
    labels = kmeans.labels_


    np.save("cluster_labels.npy", labels)

    labels = np.load("cluster_labels.npy")

    server = viser.ViserServer()

    points, depth, colors = get_point_cloud(frame_no)



    cluster_rosters = get_cluster_rosters()

    meshes_depthified = depthify_2d_hands(depth, meshes_2d)

    hand_ratios = get_ratios(meshes_depthified, meshes_3d)

    ratios = hand_ratios[0]

    median_idx_sets = get_cluster_median_sets(cluster_rosters, hand_ratios)
    median_point_sets_depthified = pointify_median_idx_sets(median_idx_sets, meshes_depthified)
    median_point_sets_3d = pointify_median_idx_sets(median_idx_sets, meshes_3d)

    transform = find_transformation(median_idx_sets[0], meshes_3d[0], meshes_depthified[0])
    transformed_mesh = apply_transformation(meshes_3d[0], transform)

    boring_mesh = boring_transform(meshes_3d[0], ratios, median_idx_sets[0])

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