import time
import numpy as np
import viser
import yourdfpy
from viser.extras import ViserUrdf
import math
from quaternion_utils import *

URDF_PATH = "../gripper_model/robots/robotiq_arg85_description.URDF"

initial_direction = np.array((0, 0, 1.0))
initial_grasp = np.array((1.0, 0, 0))

gripper_direction = np.array((-3.0, -2, -1))
grasp_direction = np.array((1, -3, 1.0))

quaternion = generate_rotation_quaternion(gripper_direction, grasp_direction, initial_direction, initial_grasp)    

'''
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

    rightified_gripper_initial = approach_initial - np.dot(approach_initial, approach_goal) * approach_goal
    rightified_gripper_initial /= np.linalg.norm(rightified_gripper_initial)

    gripper_rotation_axis = np.cross(approach_goal, rightified_gripper_initial)
    gripper_rotation_axis /= np.linalg.norm(gripper_rotation_axis)

    gripper_rotation_angle = -np.acos(np.dot(approach_goal, approach_initial))

    gripper_quaternion = aa_to_quaternion(gripper_rotation_axis, gripper_rotation_angle)

    grasp_rotated = rotate_point(lateral_initial, gripper_quaternion)

    grasp_rotation_angle = -np.acos(np.dot(grasp_rotated, rightified_grasp_goal))

    grasp_quaternion = aa_to_quaternion(approach_initial, grasp_rotation_angle)

    full_quaternion = multiply_quaternions(gripper_quaternion, grasp_quaternion)

    return full_quaternion

'''


quaternion = generate_rotation_quaternion(gripper_direction, grasp_direction, initial_direction, initial_grasp)

gripper_direction /= np.linalg.norm(gripper_direction)


grasp_direction /= np.linalg.norm(grasp_direction)

rightified_grasp = grasp_direction - np.dot(grasp_direction, gripper_direction) * gripper_direction
rightified_grasp /= np.linalg.norm(rightified_grasp)

rightified_initial_gripper = initial_direction - np.dot(initial_direction, gripper_direction) * gripper_direction
rightified_initial_gripper /= np.linalg.norm(rightified_initial_gripper)

gripper_rotn_axis = np.cross(rightified_initial_gripper, gripper_direction)
gripper_rotn_axis /= np.linalg.norm(gripper_rotn_axis)

grippers_angle = np.acos(np.dot(gripper_direction, initial_direction))




gripper_quaternion = aa_to_quaternion(gripper_rotn_axis, grippers_angle)

rotated_initial_grasp = rotate_point(initial_grasp, gripper_quaternion)
rotated_initial_grasp /= np.linalg.norm(rotated_initial_grasp)

grasp_angle = 0 - np.acos(np.dot(rightified_grasp, rotated_initial_grasp))


grasp_quaternion = aa_to_quaternion(initial_direction, grasp_angle)

full_quaternion = multiply_quaternions(gripper_quaternion, grasp_quaternion)




origin = np.array((0.0, 0, 0))

def main():
    server = viser.ViserServer()

    #main wrist line
    line1 = np.vstack((origin, gripper_direction))
    #finger connection line
    line2 = np.vstack((gripper_direction / 6 - rightified_grasp / 2, gripper_direction / 6 + rightified_grasp / 2))
    #gripper axis so that its 90 from main wrist line
    #line3 = np.vstack((origin, rightified_initial_gripper))
    #rotation axis
    #line4 = np.vstack((origin, gripper_rotn_axis))
    #rotated grasp axis
    #line5 = np.vstack((gripper_direction / 6 - rotated_initial_grasp / 2, gripper_direction / 6 + rotated_initial_grasp / 2))

    lines = np.vstack((line1, line2)).reshape(2, 2, 3)


    server.scene.add_line_segments(
        name="la",
        points=lines,
        colors=(0, 0, 0),
        line_width=7.0,
    )

    urdf = yourdfpy.URDF.load(
        URDF_PATH,
        build_scene_graph=True,
        load_meshes=True,
    )

    server.scene.add_frame(
        "/robot",
        show_axes=False,
        position=(0, 0, 0),
        wxyz=quaternion
    )

    # ---- URDF ----
    gripper = ViserUrdf(server, urdf, root_node_name="/robot")
    gripper.update_cfg(np.array([0.0]))


    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()


