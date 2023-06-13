import argparse
import os
import sys

import numpy as np
import glob
import polyscope as ps

from scipy.spatial.transform import Rotation as R

from visualization_utils import visualize_grasps, show_image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))


def quat_to_mat(quat):
    # first change the order to use scipy package: scalar-last (x, y, z, w) format
    # id_ord = [1, 2, 3, 0]
    # quat = quat[id_ord]
    r = R.from_quat(quat)
    return r.as_matrix()


def matrix_from_pose(pos, quat):
    """
    homogeneous transformation matrix from transformation and quaternion
    :param pos: (3, ) numpy array
    :param quat: (4, ) numpy array
    :return: 4x4 numpy array
    """
    t = np.eye(4)
    t[:3, :3] = quat_to_mat(quat)
    pos = pos.reshape((1, 3))
    t[0:3, 3] = pos

    return t


def pose_from_matrix(t):
    """
    transformation and quaternion from homogeneous transformation matrix
    :param t: 4x4 numpy array
    :return: pos: (3, ) numpy array
             quat: (4, ) numpy array
    """
    pos = t[0:3, -1]
    r_mat = t[0:3, 0:3]
    quat = mat_to_quat(r_mat)

    return pos, quat


def mat_to_quat(mat):
    r = R.from_matrix(mat)
    quat = r.as_quat()
    # change order
    # id_ord = [3, 0, 1, 2]

    return quat
    # return quat[id_ord]


def save_results(obj_id=None, obj='cups', ycb=False, can=True, pick_part=0.9, file_name='test_cups', save_name='test'):
    if can:
        input_paths = os.path.join("results", obj, file_name, "*.npz")
    elif ycb:
        input_paths = os.path.join("results", obj, "ycb", file_name, "*.npz")
    else:
        input_paths = os.path.join("results", obj, str(obj_id), file_name, "*.npz")
    # transform to simulation
    mat = np.eye(4)
    mat[1, 1] = -1
    mat[2, 2] = -1

    idx = 0
    pcd_dict = {}
    pcd_local_dict = {}
    obj_pos_dict = {}
    obj_quat_dict = {}
    grasps = []
    grasps_obj = []
    grasps_scores = []
    ids = []

    # for plotting
    gripper_control_points = np.array([[0.00000, 0.00000, 0.00000],
                                       [0.00000, 0.00000, 0.05840],
                                       [0.05269, -0.00006, 0.05840],
                                       [0.05269, -0.00006, 0.10527],
                                       [0.05269, -0.00006, 0.05840],
                                       [-0.05269, 0.00006, 0.05840],
                                       [-0.05269, 0.00006, 0.10527]])
    all_nodes = []
    all_edges = []
    index = 0
    n_pts = 7
    probs_all = []

    ps.init()
    # load data
    for p in glob.glob(input_paths):
        data = np.load(p, allow_pickle=True)
        # load info
        pc_full = data["pcd"]
        rgb = data["rgb"]
        segmap = data["segmap"]
        pred_grasps_cam = data["pred_grasps_cam"].tolist()
        scores = data["scores"].tolist()
        pc_colors = data["pc_colors"]
        obj_pos = data["obj_pos"]
        obj_quat = data["obj_quat"]
        mat_trans = data["mat_trans"]
        mat_trans_local = data["mat_trans_local"]
        pcd_obj = data["pcd_obj"]
        pcd_obj_local = data["pcd_obj_local"]

        # get world pcd of the whole scene
        pcd = pc_full.T
        ones = np.ones((1, pcd.shape[1]))
        hom_pcd = np.vstack([pcd, ones])
        trans_pcd = mat_trans @ mat @ hom_pcd  # global, for visualization
        # trans_pcd = mat_trans_local @ mat @ hom_pcd
        pcd = trans_pcd[:3, :].T

        # get grasps for the object and filter out grasps with lower scores
        pred_grasps_cam_obj = pred_grasps_cam[1]
        scores_obj = scores[1]

        n_grasps = pred_grasps_cam_obj.shape[0]
        n_pick = int(n_grasps * pick_part)
        id_order = np.argsort(-scores_obj)  # from higher scores to lower scores
        pred_grasps_cam_obj = pred_grasps_cam_obj[id_order]
        scores_obj = scores_obj[id_order]
        grasps_filtered = pred_grasps_cam_obj[0:n_pick, ...]
        scores_filtered = scores_obj[0:n_pick, ...]

        # get grasps in world and object frame
        pose_obj = matrix_from_pose(obj_pos, obj_quat)
        mat_trans = mat_trans @ mat
        mat_trans_local = mat_trans_local @ mat

        grasps_world = []
        for j in range(grasps_filtered.shape[0]):
            # save local, visualize global
            # grasp_world = pose_obj @ grasp_obj, here get grasp_obj
            grasp_world = mat_trans @ grasps_filtered[j, :, :]  # grasps in world frame, global
            grasp_world_local = mat_trans_local @ grasps_filtered[j, :, :]  # local

            grasp_obj = np.linalg.inv(pose_obj) @ grasp_world_local  # grasps in object frame

            # for visualization
            grasps_world.append(grasp_world)

            # to save
            pos, quat = pose_from_matrix(grasp_world_local)
            grasps.append(np.hstack([pos, quat]))

            pos, quat = pose_from_matrix(grasp_obj)
            grasps_obj.append(np.hstack([pos, quat]))

            grasps_scores.append(scores_filtered[j])

        # add to list and dict
        ids.append(idx)
        pcd_dict[idx] = pcd_obj_local
        obj_pos_dict[idx] = obj_pos
        obj_quat_dict[idx] = obj_quat

        # visualization
        ps.set_up_dir("z_up")
        ps.register_point_cloud("pcd_obj" + str(idx), pcd_obj, radius=0.001, enabled=True)
        ps.register_point_cloud("scene" + str(idx), pcd, radius=0.001, enabled=True)

        # show grasp (gripper)
        for i in range(len(grasps_world)):
            coords = np.concatenate((gripper_control_points, np.ones((7, 1))), axis=1)
            coords = grasps_world[i] @ coords.T
            coords = coords[0:3, :]
            coords = coords.T
            nodes = coords
            all_nodes.append(nodes)
            all_edges.append(np.array([[index, index + 1],
                                       [index + 2, index + 5],
                                       [index + 2, index + 3],
                                       [index + 5, index + 6]]))
            index += n_pts

        idx += 1

    all_nodes = np.vstack(all_nodes)
    all_edges = np.vstack(all_edges)
    probs_all = np.repeat(grasps_scores, 4)
    ps3 = ps.register_curve_network("gripper", all_nodes, all_edges, radius=0.00015, enabled=True)
    ps3.add_scalar_quantity("probs", probs_all, defined_on='edges', cmap='coolwarm', enabled=True)
    ps.show()

    if can:
        save_dir = os.path.join(BASE_DIR, "grasp", obj, "canonical", file_name)
    elif ycb:
        save_dir = os.path.join(BASE_DIR, "grasp", obj, "ycb", obj_id, file_name)
    else:
        save_dir = os.path.join(BASE_DIR, "grasp", obj, obj_id, file_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.savez(save_dir + '/grasps',
             ids=ids, pcd_dict=pcd_dict, obj_pos_dict=obj_pos_dict, obj_quat_dict=obj_quat_dict,
             grasps_world=grasps, grasps_obj=grasps_obj, grasps_scores=grasps_scores)

    print(str(len(grasps)) + " grasps are saved.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj', type=str, default='cups')
    parser.add_argument('--obj_id', type=str, default=None)
    parser.add_argument('--can', action='store_true')
    parser.add_argument('--ycb', action='store_true')
    parser.add_argument('--part', type=float, default=0.9)
    parser.add_argument('--file_name', type=str, default='test_cups')
    parser.add_argument('--save_name', type=str)

    args = parser.parse_args()

    # transform the grasps to object coordinate system
    save_results(obj=args.obj, obj_id=args.obj_id, ycb=args.ycb, can=args.can, pick_part=args.part,
                 file_name=args.file_name, save_name=args.save_name)
