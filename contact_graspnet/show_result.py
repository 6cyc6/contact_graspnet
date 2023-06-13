import argparse
import os
import sys

import numpy as np
import polyscope as ps

from visualization_utils import visualize_grasps, show_image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))

parser = argparse.ArgumentParser()
parser.add_argument('--file_name', type=str, default="test_cup")
parser.add_argument('--id', type=int, default=1)
parser.add_argument('--obj', type=str, default="cups")
parser.add_argument('--obj_id', type=str, default=None)
parser.add_argument('--can', action='store_true', help='if canonical shape')
parser.add_argument('--ycb', action='store_true', help='if ycb object')

arg = parser.parse_args()

if arg.can:
    dir_result = os.path.join("results", arg.obj, "canonical", arg.file_name, 'predictions_' + str(arg.id) + '.npz')
elif arg.ycb:
    dir_result = os.path.join("results", arg.obj, "ycb", arg.obj_id, arg.file_name, 'predictions_' + str(arg.id) + '.npz')
else:
    dir_result = os.path.join("results", arg.obj, arg.obj_id, arg.file_name, 'predictions_' + str(arg.id) + '.npz')

data = np.load(dir_result, allow_pickle=True)
pc_full = data["pcd"]
rgb = data["rgb"]
segmap = data["segmap"]
pred_grasps_cam = data["pred_grasps_cam"].tolist()
scores = data["scores"].tolist()
pc_colors = data["pc_colors"]
mat_trans = data["mat_trans"]

show_image(rgb, segmap)
visualize_grasps(pc_full, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=pc_colors)

# mat = np.eye(4)
# mat[1, 1] = -1
# mat[2, 2] = -1
# pcd = pc_full.T
# ones = np.ones((1, pcd.shape[1]))
# hom_pcd = np.vstack([pcd, ones])
# trans_pcd = mat_trans @ mat @ hom_pcd
# pcd = trans_pcd[:3, :].T
# pcd_obj = data["pcd_obj"]
#
# ps.init()
# ps.set_up_dir("z_up")
# ps1 = ps.register_point_cloud("pcd_obj", pcd_obj, radius=0.001, enabled=True)
# ps2 = ps.register_point_cloud("scene", pcd, radius=0.001, enabled=True)
# ps.show()
