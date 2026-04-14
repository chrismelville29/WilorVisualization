import viser
import yourdfpy
from viser.extras import ViserUrdf
import numpy as np
from quaternion_utils import generate_rotation_quaternion


def render_mesh(handle, mesh, faces, centroid=None):
    rendering_mesh = mesh.copy()
    if centroid is not None:
        rendering_mesh -= centroid
    handle.vertices = rendering_mesh
    handle.faces = faces
    handle.visible = True

def render_gripper(frame_handle, urdf_handle, base, quaternion, graspness, centroid=None):

    rendering_position = base.copy()

    if centroid is not None:
        rendering_position -= centroid

    frame_handle.position = rendering_position
    frame_handle.wxyz = quaternion

    if graspness == 1:
        urdf_handle.update_cfg(np.array([0.8])) #0.8 for closed.
    else:
        urdf_handle.update_cfg(np.array([0.1]))

def render_box(axis_handle, box_handle, quaternion, position, centroid=None):
    rendering_position = position.copy()
    if centroid is not None:
        rendering_position -= centroid

    axis_handle.wxyz = quaternion
    axis_handle.position = rendering_position

    box_handle.wxyz = quaternion
    box_handle.position = rendering_position

    axis_handle.visible = True
    box_handle.visible = True

def render_clouds(handles, clouds, colors, centroid=None):
    for i in range(len(clouds)):
        render_cloud(handles[i], clouds[i], colors, centroid)

def render_cloud(handle, points, colors, centroid=None):
    rendering_points = points.copy()
    if centroid is not None:
        rendering_points -= centroid

    handle.points = rendering_points
    handle.colors = colors
    handle.visible = True

def initialize_mesh(server, name, color):
    handle = server.scene.add_mesh_simple(
        name=name,
        vertices=np.zeros((2, 3)),
        faces=np.zeros((1, 3)),
        color=color
    )
    handle.visible = False
    return handle

def initialize_gripper(server, name, urdf):
    handle = server.scene.add_frame(
        name=name,
        show_axes=False,
        position = np.zeros(3)
    )

    gripper = ViserUrdf(server, urdf, root_node_name=name)

    return handle, gripper

def initialize_cloud(server, name, point_size):
    handle = server.scene.add_point_cloud(
        name=name,
        points=np.zeros((1, 3)),
        colors=np.zeros((1, 3)),
        point_size=point_size,
    )
    handle.visible = False
    return handle

def initialize_axes(server, name):
    handle = server.scene.add_frame(
        name=name,
        wxyz=(1, 0, 0, 0),
        position=np.array((0, 0, 0)),
    )
    handle.visible = False
    return handle

def initialize_box(server, name):
    handle = server.scene.add_box(
        name=name,
        dimensions=(.304, .304, .004), 
        color=(255, 0, 0),
        position=np.zeros(3),   
    )
    handle.visible = False
    return handle