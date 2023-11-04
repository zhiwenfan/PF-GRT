"""
A simple script that uses bpy to render views of a single object by
move the camera around it.

Original source:
https://github.com/panmari/stanford-shapenet-renderer
"""

import os, sys
import bpy
import math
from math import radians
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2
import pickle
from blender_utils import obj_location, save_visual
import imutils
pi = math.pi
np.random.seed(0)

def reset_blend():
    bpy.ops.wm.read_factory_settings()

    for scene in bpy.data.scenes:
        for obj in scene.objects:
            scene.objects.unlink(obj)

    # only worry about data in the startup scene
    for bpy_data_iter in (
            bpy.data.objects,
            bpy.data.meshes,
            bpy.data.lamps,
            bpy.data.cameras,
    ):
        for id_data in bpy_data_iter:
            bpy_data_iter.remove(id_data)

def resize_padding(im, desired_size):
    # compute the new size
    old_size = im.size
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    im = im.resize(new_size, Image.ANTIALIAS)

    # create a new image and paste the resized on it
    new_im = Image.new("RGBA", (desired_size, desired_size))
    new_im.paste(im, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2))

    return new_im


def resize_padding_v2(im, desired_size_in, desired_size_out):
    # compute the new size
    old_size = im.size
    ratio = float(desired_size_in)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    im = im.resize(new_size, Image.ANTIALIAS)

    # create a new image and paste the resized on it
    new_im = Image.new("RGBA", (desired_size_out, desired_size_out))
    new_im.paste(im, ((desired_size_out - new_size[0]) // 2, (desired_size_out - new_size[1]) // 2))
    return new_im


# create a lamp with an appropriate energy
def makeLamp(lamp_name, rad):
    # Create new lamp data block
    lamp_data = bpy.data.lights.new(name=lamp_name, type='POINT')
    lamp_data.energy = rad
    # modify the distance when the object is not normalized
    # lamp_data.distance = rad * 2.5

    # Create new object with our lamp data block
    lamp_object = bpy.data.objects.new(name=lamp_name, object_data=lamp_data)
    # Link lamp object to the scene so it'll appear in this scene
    scene = bpy.context.collection
    scene.objects.link(lamp_object)
    return lamp_object


def parent_obj_to_camera(b_camera):
    # set the parenting to the origin
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty

    scn = bpy.context.collection
    scn.objects.link(b_empty)
    bpy.context.view_layer.objects.active = b_empty
    return b_empty


def clean_obj_lamp_and_mesh(context):
    scene = context.collection
    objs = bpy.data.objects
    meshes = bpy.data.meshes
    for obj in objs:
        if obj.type == "MESH" or obj.type == 'LAMP':
            scene.objects.unlink(obj)
            objs.remove(obj)
    for mesh in meshes:
        meshes.remove(mesh)

def add_shader_on_world():
    bpy.data.worlds['World'].use_nodes = True
    env_node = bpy.data.worlds['World'].node_tree.nodes.new(type='ShaderNodeTexEnvironment')
    back_node = bpy.data.worlds['World'].node_tree.nodes['Background']
    bpy.data.worlds['World'].node_tree.links.new(env_node.outputs['Color'], back_node.inputs['Color'])

def set_material_node_parameters(material):
    nodes = material.node_tree.nodes
    nodes['Principled BSDF'].inputs['Roughness'].default_value = 0.5#np.random.uniform(0.8, 1)


def add_shader_on_ply_object(obj):
    material = bpy.data.materials.new("VertCol")

    material.use_nodes = True
    material.node_tree.links.clear()

    mat_out = material.node_tree.nodes['Material Output']
    diffuse_node = material.node_tree.nodes['Principled BSDF']
    attr_node = material.node_tree.nodes.new(type='ShaderNodeAttribute')

    attr_node.attribute_name = 'Col'

    material.node_tree.links.new(attr_node.outputs['Color'], diffuse_node.inputs['Base Color'])
    material.node_tree.links.new(diffuse_node.outputs['BSDF'], mat_out.inputs['Surface'])

    obj.data.materials.append(material)

    return material

def setup(shape, light_main, light_add):
    clean_obj_lamp_and_mesh(bpy.context)
    # Set up rendering of depth map:
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    # clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)

    # Depth config
    rl = tree.nodes.new(type="CompositorNodeRLayers")
    depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.base_path = ''
    depth_file_output.format.file_format = 'PNG'
    depth_file_output.format.color_depth = '16'

    map_node = tree.nodes.new(type="CompositorNodeMapRange")
    map_node.inputs[1].default_value = 0
    map_node.inputs[2].default_value = 255
    map_node.inputs[3].default_value = 0
    map_node.inputs[4].default_value = 1
    links.new(rl.outputs['Depth'], map_node.inputs[0])
    links.new(map_node.outputs[0], depth_file_output.inputs[0])

    # Setting up the environment
    scene = bpy.context.scene
    context = bpy.context

    scene.render.engine = "CYCLES"
    scene.cycles.sample_clamp_indirect = 1.0
    scene.cycles.blur_glossy = 3.0
    scene.cycles.samples = 100

    for mesh in bpy.data.meshes:
        mesh.use_auto_smooth = True

    scene.render.resolution_x = shape[0]
    scene.render.resolution_y = shape[1]
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.file_format = 'PNG'

    # Camera setting
    cam = scene.objects['Camera']
    cam_constraint = scene.objects['Camera'].constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    cam_empty = parent_obj_to_camera(scene.objects['Camera'])
    cam_constraint.target = cam_empty

    # Light setting
    lamp_object = makeLamp('Lamp1', light_main)
    lamp_add = makeLamp('Lamp2', light_add)

    return cam, depth_file_output, lamp_object, lamp_add

def render(camera, lamp_object, lamp_add, depth_file_output, outfile, pose):
    bpy.context.scene.render.filepath = outfile
    depth_file_output.file_slots[0].path = bpy.context.scene.render.filepath + '_depth.png'

    theta, elevation, azimuth = pose[:3]

    azimuth = -(azimuth + 90)
    elevation = elevation - 90

    cam_dist = pose[-1]

    x, y, z = obj_location(cam_dist, azimuth, elevation)
    camera.location = (x, y, z)

    ## setup_light
    lamp_object.location = (0, 0, 4)
    lamp_add.location = (0, 0, -4)

    bpy.ops.render.render(write_still=True)

    im_path = outfile + '.png'
    im = Image.open(im_path).copy()
    im = np.array(im)

    im = imutils.rotate(im, angle=-theta)

    mask = (im[:, :, 3] > 0).astype(np.uint8) * 255
    im = cv2.cvtColor(im[:, :, :3], cv2.COLOR_RGB2BGR)
    return im, mask

def render_ply(obj, output_dir, pose_list, shape=[256, 256], light_main=5, light_add=1, normalize=False, forward=None, up=None, texture=True):
    # setup
    cam, depth_file_output, lamp_object, lamp_add = setup(shape, light_main, light_add)

    # import object
    bpy.ops.import_mesh.ply(filepath=obj)
    object = bpy.data.objects[os.path.basename(obj).replace('.ply', '')]

    ## texture
    if texture is True:
        material = add_shader_on_ply_object(object)
        set_material_node_parameters(material)

    for object in bpy.context.scene.objects:
        if object.name in ['Lamp'] or object.type in ['EMPTY', 'LAMP']:
            continue
        bpy.context.view_layer.objects.active = object
        max_dim = max(object.dimensions)

    # normalize the object
    if normalize:
        object.dimensions = object.dimensions / max_dim if max_dim != 0 else object.dimensions

    cam_ob = bpy.context.scene.camera
    bpy.context.view_layer.objects.active = cam_ob

    bbx_list = []
    for i, pose in enumerate(tqdm(pose_list)):
        # redirect output to log file
        logfile = 'render.log'
        open(logfile, 'a').close()
        old = os.dup(1)
        sys.stdout.flush()
        os.close(1)
        os.open(logfile, os.O_WRONLY)

        im, mask = render(cam_ob, lamp_object, lamp_add, depth_file_output, os.path.join(output_dir, 'rendered_img'), pose)

        bbx = np.where(mask > 0)
        bbx = np.asarray([bbx[1].min(), bbx[0].min(), bbx[1].max(), bbx[0].max()])
        bbx_list.append(bbx)

        save_visual(im, mask, output_dir, pose[:3])

        # disable output redirection
        os.close(1)
        os.dup(old)
        os.close(old)

    os.system("rm render.log")
    return bbx_list
