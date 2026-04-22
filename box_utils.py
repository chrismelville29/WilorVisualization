import numpy as np
import reconstruct

def condense_aruco_poses(aruco_poses):
    ids, rvecs, tvecs = aruco_poses
    agg_rvec = np.mean(rvecs, axis=0)
    agg_tvec = np.mean(tvecs, axis=0)
    quaternion, position = reconstruct.quatnpos_from_vector(agg_tvec, agg_rvec)

    return quaternion, position