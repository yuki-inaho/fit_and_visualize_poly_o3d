#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import open3d as o3d
import pdb
import cv2
import click
import toml
import glob
import numpy as np
import polynomial2d 
import functools

class CameraParameter:
    def __init__(self):
        fx = None
        fy = None
        cx = None
        cy = None

global DATA_PATH 
global TOML_PATH
global IntrinsicParam
camera_param = CameraParameter()

def loadIntrinsicParam():
    dict_intrinsic_toml = toml.load(open("camera_parameter.toml"))
    camera_param.fx = dict_intrinsic_toml["CameraParameter"]["fx"]
    camera_param.fy = dict_intrinsic_toml["CameraParameter"]["fy"]
    camera_param.cx = dict_intrinsic_toml["CameraParameter"]["cx"]
    camera_param.cy = dict_intrinsic_toml["CameraParameter"]["cy"]
    return camera_param

def get_depth_images():
    dimg_names = np.sort(glob.glob(os.path.join(DATA_PATH,"depth_*.png")))
    return dimg_names

def _cvt_depth2pc(depth_img, camera_param):
    arr_y = np.arange(depth_img.shape[0], dtype=np.float32)
    arr_x = np.arange(depth_img.shape[1], dtype=np.float32)
    val_x, val_y = np.meshgrid(arr_x, arr_y)
    
    tmp_x = depth_img * (val_x - camera_param.cx) * (1. / camera_param.fx) * 0.001
    tmp_y = depth_img * (val_y - camera_param.cy) * (1. / camera_param.fy) * -0.001
    tmp_z = depth_img * 0.001

    pc = np.stack([tmp_x, tmp_y, tmp_z], axis=-1)
    return pc

def cvt_depth2pc(depth_img, camera_param):
    _pc = _cvt_depth2pc(depth_img, camera_param)
    pc = _pc.reshape(_pc.shape[0]*_pc.shape[1], 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    return pcd

def get_merged(near, far):
    '''
    bflg_near_empty = (near == 0)
    depth_merged = near.copy()
    depth_merged[bflg_near_empty] = far[bflg_near_empty]
    '''
    bflg_near_empty = np.where(near == 0)
    depth_merged = near.copy()
    depth_merged[bflg_near_empty] = far[bflg_near_empty]
    return depth_merged

def cvt_gridvtx2mesh(grid_vtx, double_sided=True):
    ngrid_x = grid_vtx.shape[0]
    ngrid_y = grid_vtx.shape[1]
    vertices = np.array(grid_vtx.reshape(-1,3))

    triangles = []
    for i_x in range(grid_vtx.shape[0] - 1):
        for i_y in range(grid_vtx.shape[1] - 1):
            ivert_base = i_x * ngrid_y + i_y
            triangles.append([ivert_base, ivert_base+ngrid_y, ivert_base+1])
            triangles.append([ivert_base+ngrid_y+1, ivert_base+1, ivert_base+ngrid_y])
    triangles = np.array(triangles)

    if double_sided:
        triangles = np.concatenate([triangles, triangles[:,::-1]], axis=0)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.paint_uniform_color([0.4, 0.4, 0.4])
    mesh.compute_vertex_normals()


    return mesh

def main():
    global camera_param
    loadIntrinsicParam()
    dimgs = get_depth_images()
    #dict_toml = toml.load(open(TOML_PATH))

    depth_near = cv2.imread(dimgs[0], cv2.IMREAD_ANYDEPTH)
    depth_far = cv2.imread(dimgs[1], cv2.IMREAD_ANYDEPTH)
    depth_img = get_merged(depth_near, depth_far)
    near_pc = cvt_depth2pc(depth_near, camera_param)
    far_pc = cvt_depth2pc(depth_far, camera_param)
    pc = cvt_depth2pc(depth_img, camera_param)
    pc_ary = np.asarray(pc.points)

    poly_coeff = polynomial2d.polyfit2d(pc_ary[:,0], pc_ary[:,1], pc_ary[:,2], deg=3)
    N_GRID = 25
    _grid_x = np.linspace(np.min(pc_ary[:,0]), np.max(pc_ary[:,0]), N_GRID)
    _grid_y = np.linspace(np.min(pc_ary[:,1]), np.max(pc_ary[:,1]), N_GRID)

    val_x, val_y = np.meshgrid(_grid_x, _grid_y)
    landmark_grid_2d = np.dstack([val_x, val_y])
    eval_poly2d = functools.partial(polynomial2d.polyval2d, c=poly_coeff)

    #https://codereview.stackexchange.com/questions/90005/efficient-element-wise-function-computation-in-python
    grid_depth = np.vectorize(eval_poly2d)(*np.meshgrid(_grid_x, _grid_y, sparse=True))
    grid_x, grid_y= np.meshgrid(_grid_x, _grid_y)

    grid_depth_flt = grid_depth.flatten()
    grid_x_flt = grid_x.flatten()
    grid_y_flt = grid_y.flatten()
    grid_vtx = np.dstack([grid_x, grid_y, grid_depth])
    mesh = cvt_gridvtx2mesh(grid_vtx)

    near_pc.paint_uniform_color([1, 0.706, 0])
    far_pc.paint_uniform_color([0, 0.706, 1])

    #o3d.visualization.draw_geometries([near_pc, far_pc, mesh])
    o3d.visualization.draw_geometries([pc, mesh])

if __name__ == "__main__":
    argc = len(sys.argv)
    if argc > 1:
        DATA_PATH = sys.argv[1]
        TOML_PATH = os.path.join(DATA_PATH,"camera_parameters.toml")        
    else:
        DATA_PATH = "./data"
        #DATA_PATH = "/home/inaho-00/デスクトップ/aspara_detector_out/20191004_20-11-50"
        TOML_PATH = os.path.join(DATA_PATH,"camera_parameters.toml")
    main()
