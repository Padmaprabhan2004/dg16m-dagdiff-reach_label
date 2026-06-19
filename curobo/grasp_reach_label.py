import torch
import os
import argparse
import sys
import time
import copy
from curobo.inverse_kinematics import InverseKinematics,InverseKinematicsCfg
from curobo.types import GoalToolPose, ContentPath
from curobo.scene import Scene,Cuboid,Sphere, Mesh
from curobo.types import Pose
from curobo._src.geom.types import SceneCfg
from curobo.viewer import ViserVisualizer
import numpy as np
import trimesh
import viser.transforms as vtf
import h5py

##DO ALL OPS IN device=cuda!!
#grounding the mesh
def make_grounded_mesh(file_path, name="object", x=0.0, y=0.7, z_floor=0.0):
    mesh = trimesh.load(file_path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    z_offset = float(-mesh.bounds[0, 2] + z_floor)

    return Mesh(
        name=name,
        pose=[float(x), float(y) + 0.001, z_offset, 1, 0, 0, 0],
        file_path=file_path,
    )


def pose_to_matrix(pose_4x4):
    """Convert a 4x4 grasp matrix to numpy if needed."""
    mat = np.asarray(pose_4x4, dtype=np.float32)
    if mat.shape != (4, 4):
        raise ValueError(f"Expected 4x4 pose matrix, got {mat.shape}")
    return mat


def matrix_to_pose(mat):
    mat = np.asarray(mat, dtype=np.float32)
    position = torch.tensor(mat[:3, 3][None, :], device="cuda", dtype=torch.float32)
    # cuRobo Pose expects wxyz
    rot = trimesh.transformations.quaternion_from_matrix(mat)
    quaternion = torch.tensor(rot[None, :], device="cuda", dtype=torch.float32)
    #quaternion = torch.tensor([1,0,0,0], device="cuda", dtype=torch.float32)
    return Pose(position=position, quaternion=quaternion)


def transform_grasp_to_world(T_world_obj, grasp):
    return T_world_obj @ pose_to_matrix(grasp)


def build_dual_arm_goal(ik, left_T_world, right_T_world):
    # ee_link is the right arm; ee_link_1 is the left arm in the dual configs.
    goal_dict = {
        "ee_link": matrix_to_pose(right_T_world),
        "ee_link_1": matrix_to_pose(left_T_world),
    }
    return GoalToolPose.from_poses(
        goal_dict,
        ordered_tool_frames=ik.tool_frames,
        num_goalset=1,
    )


def load_grasp_object_file(grasp_path, object_name):
    grasp_file = os.path.join(grasp_path, f"{object_name}.h5")
    if not os.path.exists(grasp_file):
        raise FileNotFoundError(f"Grasp file not found: {grasp_file}")
    return h5py.File(grasp_file, "r")


def get_reach_label_from_ik(ik, goal):
    """Return 1 if both arms solve the grasp pose, else 0."""
    result = ik.solve_pose(goal)
    return float(bool(result.success.item())), result


def _default_current_state(ik: InverseKinematics):
    # Keep it consistent with other examples in this repo.
    # get_active_js expects a joint-state tensor shaped like default_joint_state.
    return ik.get_active_js(ik.default_joint_state.clone()).unsqueeze(0)





def add_frame_axes(server, name, pose_mat, axis_len=0.08):
    pose_mat = np.asarray(pose_mat, dtype=np.float32)
    origin = pose_mat[:3, 3]
    rot = pose_mat[:3, :3]
    axes = {
        "x": (np.array([1.0, 0.0, 0.0], dtype=np.float32), np.array([255, 0, 0], dtype=np.uint8)),
        "y": (np.array([0.0, 1.0, 0.0], dtype=np.float32), np.array([0, 255, 0], dtype=np.uint8)),
        "z": (np.array([0.0, 0.0, 1.0], dtype=np.float32), np.array([0, 0, 255], dtype=np.uint8)),
    }
    for axis_name, (axis_vec, color) in axes.items():
        world_vec = rot @ axis_vec
        points = np.stack([origin, origin + axis_len * world_vec], axis=0)
        server.scene.add_line_segments(
            f"{name}/{axis_name}",
            points=points[None, :, :],
            colors=color,
            line_width=4.0,
        )


#debug for visualizing the target gripper poses
def visualize_grasp_frame(
    robot_file="dual_franka.yml",
    port=8080,
    sample_grasp=4,
    grasp_path="/home/prabhu2004/Desktop/curobo/grasps",
    mesh_path="/home/prabhu2004/Desktop/curobo/meshes",
    object="model_normalized",
):
    viser_viz = ViserVisualizer(
        content_path=ContentPath(robot_config_file=robot_file),
        connect_ip="0.0.0.0",
        connect_port=port,
        add_control_frames=False,
        visualize_robot_spheres=True,
        add_robot_to_scene=True,
    )
    server = viser_viz._server

    obj = make_grounded_mesh(
        os.path.join(mesh_path, f"{object}.obj"),
        name="object",
        x=0.0,
        y=0.7,
        z_floor=0.0,
    )
    scene_cfg = SceneCfg(mesh=[obj])
    viser_viz.add_scene(scene_cfg, add_control_frames=True)

    grasp_file = os.path.join(grasp_path, f"{object}.h5")
    with h5py.File(grasp_file, "r") as data:
        grasps = data["grasps/grasps"][()]
        print(grasps.ndim,grasps.shape)
        grasp_idx = int(sample_grasp)

        sample_grasp_pair = grasps[grasp_idx]
        obj_pose = np.asarray(obj.pose, dtype=np.float32)

        #T_world_mesh 
        T_world_obj = trimesh.transformations.quaternion_matrix(
            np.array([obj_pose[3], obj_pose[4], obj_pose[5], obj_pose[6]], dtype=np.float32)
        )
        T_world_obj[:3, 3] = obj_pose[:3]

        #T_world_grasp, grasps in world frame
        left_T_world = T_world_obj @ pose_to_matrix(sample_grasp_pair[0])
        right_T_world = T_world_obj @ pose_to_matrix(sample_grasp_pair[1])

        add_frame_axes(server, "/grasp_debug/object", T_world_obj, axis_len=0.06)
        add_frame_axes(server, "/grasp_debug/left_grasp", left_T_world, axis_len=0.08)
        add_frame_axes(server, "/grasp_debug/right_grasp", right_T_world, axis_len=0.08)

        server.scene.add_transform_controls(
            "/grasp_debug/left_handle",
            scale=0.05,
            position=tuple(left_T_world[:3, 3].tolist()),
            wxyz=tuple(trimesh.transformations.quaternion_from_matrix(left_T_world).tolist()),
        )
        server.scene.add_transform_controls(
            "/grasp_debug/right_handle",
            scale=0.05,
            position=tuple(right_T_world[:3, 3].tolist()),
            wxyz=tuple(trimesh.transformations.quaternion_from_matrix(right_T_world).tolist()),
        )

    while True:
        time.sleep(0.1)





#reach ik


def get_single_arm_reachability(
    ik: InverseKinematics,
    left_T_world,
    right_T_world,
    obj=None,
):
    """two single-arm grasps using a dual-arm robot model."""

    tool_frames = ik.tool_frames
    home_kin = ik.compute_kinematics(ik.default_joint_state.clone())
    home_tool_poses = home_kin.tool_poses

    if len(tool_frames) < 2:
        raise ValueError(f"Expected a dual-arm robot, got tool frames {tool_frames}")

    def _goal_for(target_frame: str, target_pose, stationary_frame: str):
        return GoalToolPose.from_poses(
            {
                target_frame: matrix_to_pose(target_pose),
                stationary_frame: home_tool_poses[stationary_frame].clone(),
            },
            ordered_tool_frames=tool_frames,
            num_goalset=1,
        )

    if "ee_link" in tool_frames and "ee_link_1" in tool_frames:
        right_frame = "ee_link"
        left_frame = "ee_link_1"
    else:
        left_frame = tool_frames[0]
        right_frame = tool_frames[1]

    left_goal = _goal_for(left_frame, left_T_world, right_frame)
    right_goal = _goal_for(right_frame, right_T_world, left_frame)

    if obj is not None:
        ik.update_world(Scene(mesh=[obj]))

    left_result = ik.solve_pose(left_goal, current_state=_default_current_state(ik), return_seeds=1)
    right_result = ik.solve_pose(right_goal, current_state=_default_current_state(ik), return_seeds=1)

    left_success = bool(left_result.success.item())
    right_success = bool(right_result.success.item())

    dual_success = False
    dual_result = None
    if left_success and right_success:
        ik.update_world(Scene(mesh=[]))
        dual_goal = GoalToolPose.from_poses(
            {
                left_frame: matrix_to_pose(left_T_world),
                right_frame: matrix_to_pose(right_T_world),
            },
            ordered_tool_frames=tool_frames,
            num_goalset=1,
        )
        dual_result = ik.solve_pose(
            dual_goal,
            current_state=_default_current_state(ik),
            return_seeds=1,
        )
        dual_success = bool(dual_result.success.item())

    return {
        "left_success": float(left_success),
        "right_success": float(right_success),
        "dual_success": float(dual_success),
        "left_result": left_result,
        "right_result": right_result,
        "dual_result": dual_result,
    }



def check_one_grasp_reachabilbilty(
    robot_file="dual_franka.yml",
    port=8080,
    sample_grasp=1,
    grasp_path="/home/prabhu2004/Desktop/curobo/grasps",
    mesh_path="/home/prabhu2004/Desktop/curobo/meshes",
    object="monitor",
):
    '''
    viser_viz = ViserVisualizer(
        content_path=ContentPath(robot_config_file=robot_file),
        connect_ip="0.0.0.0",
        connect_port=port,
        add_control_frames=False,
        visualize_robot_spheres=False,
        add_robot_to_scene=True,
    )
    '''

    
    obj = make_grounded_mesh(
        os.path.join(mesh_path, f"{object}.obj"),
        name="object",
        x=0.0,
        y=0.7,
        z_floor=0.0,
    )

    #scene_cfg = SceneCfg(mesh=[obj])
    #server = viser_viz._server
    #obstacle_frames = viser_viz.add_scene(scene_cfg, add_control_frames=True)

    config = InverseKinematicsCfg.create(
        robot=robot_file,
        scene_model="collision_table.yml",
        self_collision_check=True,
        collision_cache={"mesh": 10},
    )

    ik = InverseKinematics(config)
    #ik.update_world(Scene(mesh=[obj]))

    grasp_file = os.path.join(grasp_path, f"{object}.h5")
    with h5py.File(grasp_file, "r") as data:
        grasps = data["grasps/grasps"][()]
        if grasps.ndim != 4 or grasps.shape[1:] != (2, 4, 4):
            raise ValueError(
                f"Expected grasps with shape [N, 2, 4, 4], got {grasps.shape}"
            )

        grasp_idx = int(sample_grasp)
        if grasp_idx < 0 or grasp_idx >= grasps.shape[0]:
            raise IndexError(
                f"sample_grasp={grasp_idx} outside range [0, {grasps.shape[0] - 1}]"
            )

        object_T_world = trimesh.transformations.translation_matrix(obj.pose[:3])
        object_T_world[:3, :3] = trimesh.transformations.quaternion_matrix(
            np.array([obj.pose[3], obj.pose[4], obj.pose[5], obj.pose[6]], dtype=np.float32)
        )[:3, :3]

        sample_grasp_pair = grasps[grasp_idx]
        left_T_world = transform_grasp_to_world(object_T_world, sample_grasp_pair[0])
        right_T_world = transform_grasp_to_world(object_T_world, sample_grasp_pair[1])

        reachability = get_single_arm_reachability(ik, left_T_world, right_T_world, obj=obj)
        reach_label = reachability["dual_success"]
        result = reachability["dual_result"] if reachability["dual_result"] is not None else reachability["left_result"]

        print(f"left_single={reachability['left_success']}, right_single={reachability['right_success']}, dual={reachability['dual_success']}")

        if result is not None and hasattr(result, "position_error"):
            print("\nPosition Error (m):")
            print(result.position_error)

            print(
                f"Position Error Norm: "
                f"{result.position_error.squeeze().item()*1000:.3f} mm"
            )

        if hasattr(result, "rotation_error"):
            print("\nRotation Error:")
            print(result.rotation_error)

            print(
                f"Rotation Error: "
                f"{np.rad2deg(result.rotation_error.squeeze().item()):.3f} deg"
            )

        print("========")

    return reach_label


def write_grasp_reachability_labels(grasp_path,mesh_path,robot_file="dual_franka.yml",):
    """ADDS:
    - `grasps/reach_labels`
    - `grasps/reach_passing_indices`
    - `grasps/reach_failed_indices`
    """
    obj_name = os.path.basename(mesh_path).split(".")[0]
    obj = make_grounded_mesh(
        os.path.join(mesh_path, f"{obj_name}.obj"),
        name="object",
        x=0.0,
        y=0.7,
        z_floor=0.0,
    )
    object_T_world = trimesh.transformations.translation_matrix(obj.pose[:3])
    object_T_world[:3, :3] = trimesh.transformations.quaternion_matrix(
        np.array([obj.pose[3], obj.pose[4], obj.pose[5], obj.pose[6]], dtype=np.float32)
    )[:3, :3]

    config = InverseKinematicsCfg.create(
        robot=robot_file,
        scene_model="collision_table.yml",
        self_collision_check=True,
        collision_cache={"mesh": 10},
    )

    ik = InverseKinematics(config)
    ik.update_world(Scene(mesh=[obj]))

    grasp_file = os.path.join(grasp_path, f"{obj_name}.h5")
    with h5py.File(grasp_file, "r+") as data:
        grasps = data["grasps/grasps"][()]
        if grasps.ndim != 4 or grasps.shape[1:] != (2, 4, 4):
            raise ValueError(
                f"Expected grasps with shape [N, 2, 4, 4], got {grasps.shape}"
            )

        reach_labels = np.zeros((grasps.shape[0],), dtype=np.float32)
        reach_passing_indices = []
        reach_failed_indices = []

        for i in range(grasps.shape[0]):
            grasp_pair = grasps[i]
            left_T_world = transform_grasp_to_world(object_T_world, grasp_pair[0])
            right_T_world = transform_grasp_to_world(object_T_world, grasp_pair[1])
            reachability = get_single_arm_reachability(ik, left_T_world, right_T_world, obj=obj)
            label = reachability["dual_success"]
            reach_labels[i] = label
            if label > 0.5:
                reach_passing_indices.append(i)
            else:
                reach_failed_indices.append(i)

        def _write_or_replace(name, arr):
            if f"grasps/{name}" in data:
                del data[f"grasps/{name}"]
            data.create_dataset(f"grasps/{name}", data=np.asarray(arr))

        _write_or_replace("reach_labels", reach_labels)
        _write_or_replace("reach_passing_indices", np.asarray(reach_passing_indices, dtype=np.int64))
        _write_or_replace("reach_failed_indices", np.asarray(reach_failed_indices, dtype=np.int64))

        print(
            f"Wrote reach labels for {grasps.shape[0]} grasps: "
            f"{len(reach_passing_indices)} pass, {len(reach_failed_indices)} fail"
        )

import argparse

# dual panda is the one with more offset
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--robot_file",
        type=str,
        default="dual_panda.yml",
        help="Path to robot configuration file."
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize grasp frames before running reachability check."
    )

    args = parser.parse_args()


    if args.visualize:
        visualize_grasp_frame(
            robot_file=args.robot_file,
        )
    else:
        reach=0
        total=100
        for i in range (1,100):

            if check_one_grasp_reachabilbilty(robot_file=args.robot_file,sample_grasp=i):
                reach+=1
            
        print("Reached ",reach,"/",total," poses")
