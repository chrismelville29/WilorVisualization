import numpy as np

def get_cluster_rosters(cluster_path, n_clusters):
    labels = np.load(cluster_path)
    clusters = []
    for i in range(n_clusters):
        clusters.append([])

    for i in range(len(labels)):
        clusters[labels[i]].append(i)
 
    for i in range(n_clusters):
        clusters[i] = np.array(clusters[i])
    
    return clusters


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
    

