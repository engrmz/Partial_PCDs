'''

How to generate the partial PCDs from complete.


You should have the following files.folders in the currect directory
	1. partial_pcd.py
	2. cameras.npz
	3. dataset       (dataset/models/02691156/"object_id".ply)
			 where, "object_id" the id of the object





Step 1:
======
Run generate_img_depth() from the file "partial_pcd.py"  [python partial_pcd.py]
output:	It will generate a folder 'partialPCD' that contains RGB and depth images in 24 different poses
	images: (partialPCD/02691156/"object_id"/depth/**.png)



Step 1:
======
Run generate_partial_pcds() from the file "partial_pcd.py"  [python partial_pcd.py]
process:	It will read the RGB and depth images one by one, generates the partial pcd and save in the output directory.
output:		The generated PCDs will be saved in the same directory. See the next two lines;
		1. generic pose PCDs: partialPCD/02691156/"object_id"/pcds_24poses/**.xyz
		1. canonical pose PCDs: partialPCD/02691156/"object_id"/pcds_canonical/**.xyz



Mohammad Zohaib
PAVIS, IIT Genova Italy
engr.mz@hotmail.com


'''

import open3d as o3d
from numpy.linalg import inv
import scipy.io as sio
from scipy import misc
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import copy
import tqdm
import cv2
import pdb
from glob import glob

add = 11


ID2NAMES = {"02691156": "airplane",
            "02808440": "bathtub",
            "02818832": "bed",
            "02876657": "bottle",
            "02954340": "cap",
            "02958343": "car",
            "03001627": "chair",
            "03467517": "guitar",
            "03513137": "helmet",
            "03624134": "knife",
            "03642806": "laptop",
            "03790512": "motorcycle",
            "03797390": "mug",
            "04225987": "skateboard",
            "04379243": "table",
            "04530566": "vessel", }


# dir_point = '/home/mz/code/KeypointNet/dataset/pcds/03467517/'
dir_point = 'dataset/models/02691156'

o_dir = 'partialPCD/{}/'.format(dir_point.split('/')[-1])


if not os.path.exists(o_dir):
    os.makedirs(o_dir)

# out_color_file = o_dir + 'color_image/'
# out_depth_file = o_dir + 'depth_image/'
# out_scene_point = o_dir + 'scene_3dpoint/'


def TransformToCenter(pcd):
    # Centering the objects
    pcd_point = np.asarray(pcd.vertices)

    # pcd_point = pcd_point - pcd_point.mean(0)
    # pcd_point /= np.max(np.linalg.norm(pcd_point, axis=-1))

    centroid = np.mean(pcd_point, axis=0)
    pcd_point = pcd_point - centroid
    m = np.max(np.sqrt(np.sum(pcd_point ** 2, axis=1)))
    pcd_point = pcd_point / m
    pcd.vertices = o3d.utility.Vector3dVector(pcd_point*0.2)
    pcd.compute_vertex_normals()

    return pcd

def max_vertic(vertices):
  xmin = min(vertices[:, 0])
  xmax = max(vertices[:, 0])
  ymin = min(vertices[:, 1])
  ymax = max(vertices[:, 1])
  zmin = min(vertices[:, 2])
  zmax = max(vertices[:, 2])
  return max(xmax-xmin,ymax-ymin,zmax-zmin), (xmax+xmin)/2, (ymax+ymin)/2, (zmax+zmin)/2

def get_normalized_mesh(mesh):
  out_mesh = copy.deepcopy(mesh)

  """Step 1: Get min and max from very vertecs"""
  m_v = np.asarray(out_mesh.vertices)
  max_v, avg_x, avg_y, avg_z = max_vertic(m_v)

  # Step 2: Normalization
  m_v[:, 0] = m_v[:, 0] - avg_x #np.mean(m_v[:, 0])
  m_v[:, 1] = m_v[:, 1] - avg_y #np.mean(m_v[:, 1])
  m_v[:, 2] = m_v[:, 2] - avg_z #np.mean(m_v[:, 2])

  # Step 2:
  m_v /= max_v
  for i in range(len(m_v)):
      out_mesh.vertices[i] = m_v[i] * 0.7

  return out_mesh

def visualize(depth):
    depth.asarray()

def poi(dir_point):
    dataList = list()
    for (dir, _, files) in os.walk(dir_point):
        for f in sorted(files):
            path = os.path.join(dir, f)
            if os.path.exists(path):
                dataList.append(path)

    po = list()
    prog = re.compile('.ply$')
    for d in range(len(dataList)):
        binMatch = prog.search((dataList[d]))
        if binMatch:
            po.append(binMatch.string)

    return po


PO = poi(dir_point)

list_scene_name = list()
for data in range(len(PO)):
    scene = os.path.split(PO[data])[-1][:-4]
    list_scene_name.append(scene)

#### For partial point cloud loading
def read_point_cloud(path):
    pc = o3d.io.read_point_cloud(path)
    return np.array(pc.points, np.float32)



def generate_img_depth():
    # Loading Numpy file containing parameters: Intrinsic and Extrinsic Matrix
    cam_mat = np.load('cameras.npz')

    # Reading only extrinsic [4x4] matrix: 24 camera poses
    extrinsic_mat = []
    for i in range(int(len(cam_mat)/2)):
        extrinsic_mat.append(np.vstack([cam_mat['world_mat_{}'.format(i)], [0, 0, 0, 1]]))

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=137, height=137)  # size of the image width=137, height=137

    count = 0
    for scene_name in list_scene_name:
        point = dir_point +'/'+ scene_name + '.ply'  # Directory of related scene
        # point = '/home/mz/code/KeypointNet/dataset/models/03797390/10c2b3eac377b9084b3c42e318f3affc.ply'
        print("{}/{}   -   {}".format(count, len(list_scene_name), point))
        count += 1

        # consider only first 10 objects for debug
        if count >= 10:
            print('considered only {} objects'.format(count))
            break

        for k in range(1):
            pcd = o3d.io.read_triangle_mesh(point)
            pcd = get_normalized_mesh(pcd)
            pcd.compute_vertex_normals()
            poset = np.asarray(extrinsic_mat)
            vis.add_geometry(pcd)

            cam = vis.get_view_control().convert_to_pinhole_camera_parameters()


            for i in range(poset.shape[0]):
                number1 = poset[i]
                cam.extrinsic = number1
                vis.get_view_control().convert_from_pinhole_camera_parameters(cam)
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()

                file_name = os.path.split(point)[-1][:-4]
                image_directory = o_dir + file_name + '/rgb'
                if not os.path.exists(image_directory):
                    os.makedirs(image_directory)
                vis.capture_screen_image(os.path.join(image_directory, '{:03d}.png'.format(i)))

                depth_directory = (o_dir + file_name + '/depth')
                if not os.path.exists(depth_directory):
                    os.makedirs(depth_directory)
                vis.capture_depth_image(os.path.join(depth_directory, '{:03d}.png'.format(i)))

            vis.clear_geometries()
        # storing camera parameters
        kwargs = {key: cam_mat[key] for key in cam_mat.keys()}
        np.savez(os.path.join(o_dir + file_name, 'cameras.npz'), **kwargs)



def generate_partial_pcds():
    # Loading Numpy file containing parameters: Intrinsic and Extrinsic Matrix
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=137, height=137)  # size of the image width=137, height=137
    cam = vis.get_view_control().convert_to_pinhole_camera_parameters()

    cam_mat = np.load('/home/mz/code/generate_images/cameras.npz')

    # Reading only extrinsic [4x4] matrix: 24 camera poses
    extrinsic_mat = []
    # intrinsic_mat = []
    for i in range(int(len(cam_mat)/2)):
        extrinsic_mat.append(np.vstack([cam_mat['world_mat_{}'.format(i)], [0, 0, 0, 1]]))
        # intrinsic_mat.append(cam_mat['camera_mat_{}'.format(i)])
    extrinsic_mat = np.asarray(extrinsic_mat)
    # intrinsic_mat = np.asarray(intrinsic_mat)

    objects_list = glob(o_dir+'*')
    for object_ in objects_list:
        for i in range(24):
            # cam.extrinsic = extrinsic_mat[i]
            img_path = object_+'/rgb'
            depth_path = object_ + '/depth'

            color_raw = o3d.io.read_image(os.path.join(img_path, '{:03d}.png'.format(i)))
            depth_raw = o3d.io.read_image(os.path.join(depth_path, '{:03d}.png'.format(i)))

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, depth_trunc=4,
                                                                      convert_rgb_to_intensity=False)
            # 24 different poses
            pcdi = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, cam.intrinsic)
            pcdi.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            out_scene = object_ + '/pcds_24poses'
            if not os.path.exists(out_scene):
                os.makedirs(out_scene)
            o3d.io.write_point_cloud(os.path.join(out_scene, '{:03d}.xyz'.format(i)), pcdi)

            # 24 partial pcds in canonical pose
            pcdi = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, cam.intrinsic, extrinsic_mat[i])
            out_scene = object_ + '/pcds_canonical'
            if not os.path.exists(out_scene):
                os.makedirs(out_scene)
            o3d.io.write_point_cloud(os.path.join(out_scene, '{:03d}.xyz'.format(i)), pcdi)


if __name__ == '__main__':
    # generate_img_depth()
    generate_partial_pcds()