import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

BIG_TAG = cv2.aruco.DICT_6X6_50
LIL_TAG = cv2.aruco.DICT_4X4_50

ACTIVE_TAG = LIL_TAG

def make_matrix_from_tvec_and_rvec(tvec, rvec):
    """
    Create a 4x4 matrix from a list of translation vectors and a list of rotation vectors.
    """
    T = np.eye(4)
    T[:3, 3] = tvec.reshape(3,)
    T[:3,:3] = cv2.Rodrigues(rvec)[0]
    return T

def quatnpos_from_vector(tvec, rvec):
    R, _ = cv2.Rodrigues(rvec)

    quaternion = R_scipy.from_matrix(R).as_quat()
    quaternion = np.roll(quaternion, 1)

    return quaternion, tvec.flatten()

def matrix_from_intrix(fx, fy, cx, cy):
    intrinsic_matrix = np.eye(3)
    intrinsic_matrix[0, 0] = fx
    intrinsic_matrix[1, 1] = fy
    intrinsic_matrix[0, 2] = cx
    intrinsic_matrix[1, 2] = cy
    return intrinsic_matrix


def detect_aruco_pose(image_bgr, K, dist_coeffs, marker_length):

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_RGB2GRAY)

    aruco_dict = cv2.aruco.getPredefinedDictionary(ACTIVE_TAG)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    corners, ids, _ = detector.detectMarkers(gray)

    poses = []
    tag_ids = []
    rvecs = []
    tvecs = []

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image_bgr, corners, ids)

        # --- Define marker 3D points (in marker frame) ---
        s = marker_length / 2.0
        objp = np.array([
            [-s, -s, 0],
            [ s, -s, 0],
            [ s,  s, 0],
            [-s,  s, 0]
        ], dtype=np.float32)

        for i in range(len(ids)):
            imgp = corners[i].reshape(4, 2).astype(np.float32)

            # --- Solve PnP ---
            success, rvec, tvec = cv2.solvePnP(
                objp,
                imgp,
                K,
                dist_coeffs,
                #flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                continue

            poses.append((int(ids[i][0]), rvec, tvec))
            tag_ids.append(int(ids[i][0]))
            rvecs.append(rvec)
            tvecs.append(tvec)

            '''# --- Draw axes ---
            cv2.drawFrameAxes(
                image_bgr,
                K,
                dist_coeffs,
                rvec,
                tvec,
                marker_length * 0.5
            )
            '''

    return np.array(tag_ids), np.array(rvecs), np.array(tvecs)


if __name__ == "__main__":

    img_path = '../board_captures/color_captures/20260404_122039/frame_000001.png'
    img = cv2.imread(img_path)



    fx, fy, cx, cy = 597.01702881, 597.04272461, 327.60223389, 240.29771423

    K = np.eye(3)
    K[0, 0] = 597.01702881
    K[1, 1] = 597.04272461
    K[0, 2] = 327.60223389
    K[1, 2] = 240.29771423

    _, rvec, tvec = detect_aruco_pose(img, K, None, 0.05)[0]


    print(make_matrix_from_tvec_and_rvec(tvec, rvec))

