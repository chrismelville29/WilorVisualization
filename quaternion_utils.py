import numpy as np

#needs unit quaternion
def aa_to_quaternion(axis, angle):
    sine = np.sin(angle / 2)
    return np.array((np.cos(angle / 2), sine * axis[0], sine * axis[1], sine * axis[2]))

def multiply_quaternions(q1, q2):
    one_term = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
    i_term = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
    j_term = q1[0] * q2[2] + q1[2] * q2[0] + q1[3] * q2[1] - q1[1] * q2[3]
    k_term = q1[0] * q2[3] + q1[3] * q2[0] + q1[1] * q2[2] - q1[2] * q2[1]
    return np.array((one_term, i_term, j_term, k_term))

#needs unit quaternion
def invert_quaternion(q):
    return np.array((q[0], -q[1], -q[2], -q[3]))

#needs unit quaternion
def rotate_point(point, quaternion):
    point_q = np.array((0, point[0], point[1], point[2]))
    inverse = invert_quaternion(quaternion)
    return multiply_quaternions(multiply_quaternions(quaternion, point_q), inverse)[1:]

#no inputs need to be unit. lateral_goal doesn't need to be perpendicular to approach_goal
def generate_rotation_quaternion(approach_goal, lateral_goal, approach_initial, lateral_initial):
    approach_goal /= np.linalg.norm(approach_goal)
    lateral_goal /= np.linalg.norm(lateral_goal)
    approach_initial /= np.linalg.norm(approach_initial)
    lateral_initial /= np.linalg.norm(lateral_initial)

    rightified_grasp_goal = lateral_goal - np.dot(lateral_goal, approach_goal) * approach_goal
    rightified_grasp_goal /= np.linalg.norm(rightified_grasp_goal)

    gripper_rotation_axis = np.cross(approach_goal, approach_initial)
    gripper_rotation_axis /= np.linalg.norm(gripper_rotation_axis)

    gripper_rotation_angle = np.acos(np.dot(approach_goal, approach_initial))

    gripper_quaternion = aa_to_quaternion(gripper_rotation_axis, -gripper_rotation_angle)

    grasp_rotated = rotate_point(lateral_initial, gripper_quaternion)

    dot_val = np.clip(np.dot(grasp_rotated, rightified_grasp_goal), -1.0, 1.0)
    cross = np.cross(grasp_rotated, rightified_grasp_goal)
    sign = np.sign(np.dot(cross, approach_goal))
    grasp_rotation_angle = -sign * np.arccos(dot_val)


    grasp_quaternion = aa_to_quaternion(approach_initial, -grasp_rotation_angle)

    full_quaternion = multiply_quaternions(gripper_quaternion, grasp_quaternion)

    return full_quaternion
