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
        pose=[float(x), float(y) + 0.002, z_offset, 1, 0, 0, 0],
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


def transform_grasp_to_world(T_world_obj, grasp, pregrasp_distance=0.0, approach_axis="y"):
    grasp_T_obj = pose_to_matrix(grasp)
    if pregrasp_distance != 0.0:
        axis_map = {
            "x": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "y": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "z": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }
        if approach_axis not in axis_map:
            raise ValueError(f"Unsupported approach_axis={approach_axis!r}")
        world_grasp = T_world_obj @ grasp_T_obj
        local_axis_world = world_grasp[:3, :3] @ axis_map[approach_axis]
        # Move backward along the grasp's local -y axis in world coordinates.
        world_grasp = world_grasp.copy()
        world_grasp[:3, 3] = world_grasp[:3, 3] - pregrasp_distance * local_axis_world
        return world_grasp
    return T_world_obj @ grasp_T_obj


def build_dual_arm_goal(ik, left_T_world, right_T_world):
    #use ee_link,ee_link1
    goal_dict = {
        "ee_link": matrix_to_pose(right_T_world),
        "ee_link_1": matrix_to_pose(left_T_world),
    }
    return GoalToolPose.from_poses(
        goal_dict,
        ordered_tool_frames=ik.tool_frames,
        num_goalset=1,
    )

def build_dual_arm_goal_batch(ik, left_T_world_batch, right_T_world_batch):
    """Build a batched dual-arm goal from stacked world-frame poses."""
    
    left_positions = torch.tensor(
        left_T_world_batch[:, :3, 3], device="cuda", dtype=torch.float32
    )
    right_positions = torch.tensor(
        right_T_world_batch[:, :3, 3], device="cuda", dtype=torch.float32
    )
    left_quats = torch.tensor(
        np.stack(
            [trimesh.transformations.quaternion_from_matrix(m) for m in left_T_world_batch],
            axis=0,
        ),
        device="cuda",
        dtype=torch.float32,
    )
    right_quats = torch.tensor(
        np.stack(
            [trimesh.transformations.quaternion_from_matrix(m) for m in right_T_world_batch],
            axis=0,
        ),
        device="cuda",
        dtype=torch.float32,
    )

    goal_dict = {
        "ee_link": Pose(position=right_positions, quaternion=right_quats),
        "ee_link_1": Pose(position=left_positions, quaternion=left_quats),
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
    result = ik.solve_pose(goal)
    return float(bool(result.success.item())), result


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
    sample_grasp=1,
    grasp_path="/home/prabhu2004/Desktop/curobo/grasps",
    mesh_path="/home/prabhu2004/Desktop/curobo/meshes",
    object="monitor",
    pregrasp_distance=0.08
):
    #pregrasp_distance = 0.03
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
        left_T_world = transform_grasp_to_world(
            T_world_obj, sample_grasp_pair[0], pregrasp_distance=pregrasp_distance
        )
        right_T_world = transform_grasp_to_world(
            T_world_obj, sample_grasp_pair[1], pregrasp_distance=pregrasp_distance
        )

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


def check_one_grasp_reachabilbilty(
    robot_file="dual_franka.yml",
    port=8080,
    sample_grasp=1,
    grasp_path="/home/prabhu2004/Desktop/curobo/grasps",
    mesh_path="/home/prabhu2004/Desktop/curobo/meshes",
    object="monitor",
    pregrasp_distance = 0.08
):

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
        #max_batch_size=total_batch,
        collision_cache={"mesh":10}
    )

    ik = InverseKinematics(config)
    ik.update_world(Scene(mesh=[obj]))
    print("\nObject pose:")
    print(obj.pose)


    #print(ik.scene_collision_checker)
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
        left_T_world = transform_grasp_to_world(
            object_T_world, sample_grasp_pair[0], pregrasp_distance=pregrasp_distance
        )
        right_T_world = transform_grasp_to_world(
            object_T_world, sample_grasp_pair[1], pregrasp_distance=pregrasp_distance
        )

        goal = build_dual_arm_goal(ik, left_T_world, right_T_world)
        reach_label, result = get_reach_label_from_ik(ik, goal)

        print("Success:")
        print(result.success.item())

        if hasattr(result, "position_error"):
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

from tqdm import tqdm
def write_grasp_reachability_labels(robot_file="dual_panda.yml",pregrasp_distance=0.08,grasp_path="/home/prabhu2004/Desktop/curobo/grasps/",
    mesh_path="/home/prabhu2004/Desktop/curobo/meshes/",obj_name="monitor.obj",batch_size=64):

    """batch_ik, gives
    - `reach_labels`
    - `reach_passing_indices`
    - `reach_failed_indices` 
    as {object_name}_reachability.h5
    """
    obj = make_grounded_mesh(
        os.path.join(mesh_path,obj_name),
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
        num_seeds=32,
        max_batch_size=batch_size,
        collision_cache={"mesh":10}
    )

    ik = InverseKinematics(config)
    ik.update_world(Scene(mesh=[obj]))


    obj_name = obj_name.split(".")[0]
    grasp_file = os.path.join(grasp_path, f"{obj_name}.h5")
    with h5py.File(grasp_file, "r") as data:
        grasps = data["grasps/grasps"][()]
        #grasps=grasps[:100]
        if grasps.ndim != 4 or grasps.shape[1:] != (2, 4, 4):
            raise ValueError(
                f"Expected grasps with shape [N, 2, 4, 4], got {grasps.shape}"
            )

        reach_labels = np.zeros((grasps.shape[0],), dtype=np.float32)
        reach_passing_indices = []
        reach_failed_indices = []
        for start in tqdm(range(0, grasps.shape[0], batch_size), desc=f"{obj_name}"):
            end = min(start + batch_size, grasps.shape[0])
            batch_grasps = grasps[start:end]

            left_T_world_batch = np.stack(
                [
                    transform_grasp_to_world(
                        object_T_world,
                        grasp_pair[0],
                        pregrasp_distance=pregrasp_distance,
                    )
                    for grasp_pair in batch_grasps
                ],
                axis=0,
            )
            right_T_world_batch = np.stack(
                [
                    transform_grasp_to_world(
                        object_T_world,
                        grasp_pair[1],
                        pregrasp_distance=pregrasp_distance,
                    )
                    for grasp_pair in batch_grasps
                ],
                axis=0,
            )

            goal = build_dual_arm_goal_batch(ik, left_T_world_batch, right_T_world_batch)
            result = ik.solve_pose(goal)
            batch_success = np.asarray(result.success.detach().cpu()).reshape(-1).astype(bool)

            for idx, ok in enumerate(batch_success):
                reach_labels[start + idx] = float(ok)
                if ok:
                    reach_passing_indices.append(start + idx)
                else:
                    reach_failed_indices.append(start + idx)


        reach_file = os.path.join(
            grasp_path,
            f"{obj_name}_reachability.h5"
        )

        print("\nSaving reachability labels:")
        print(reach_file)

        with h5py.File(reach_file, "w") as out:

            out.create_dataset(
                "reach_labels",
                data=reach_labels
            )

            out.create_dataset(
                "reach_passing_indices",
                data=np.asarray(
                    reach_passing_indices,
                    dtype=np.int64,
                )
            )

            out.create_dataset(
                "reach_failed_indices",
                data=np.asarray(
                    reach_failed_indices,
                    dtype=np.int64,
                )
            )

            out.attrs["object_name"] = obj_name
            out.attrs["num_grasps"] = grasps.shape[0]
            out.attrs["num_reachable"] = len(
                reach_passing_indices
            )

        success_rate = (
            100.0 *
            len(reach_passing_indices)
            / grasps.shape[0]
        )

        print("\nFinished")
        print(f"Object: {obj_name}")
        print(f"Total grasps: {grasps.shape[0]}")
        print(f"Reachable: {len(reach_passing_indices)}")
        print(f"Unreachable: {len(reach_failed_indices)}")
        print(f"Success Rate: {success_rate:.2f}%")



import argparse

# dual panda is the one with more offset ie 1.2, dual franka is 0.9
#|
#|
#dual_panda.yml is has negligible collision sphere radius for the gripper,
#dual_panda_full_coll_sphere.yml is complete dual_panda
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--robot_file",
        type=str,
        default="dual_panda.yml",
        help="path to robot configuration file."
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="visualize grasp frames before running reachability check."
    )

    parser.add_argument(
        "--pregrasp_distance",
        type=float,
        default=0.08,
        help="select the pre grasping distance along the approach vector to the object"
    )

    args = parser.parse_args()

    if args.visualize:
        visualize_grasp_frame(
            robot_file=args.robot_file,pregrasp_distance=args.pregrasp_distance
        )
    else:
        '''reach=0
        total=100
        for i in range (1,total):

            if check_one_grasp_reachabilbilty(robot_file=args.robot_file,sample_grasp=i,pregrasp_distance=args.pregrasp_distance):
                reach+=1
            
        print("Reached",reach,"/",total," poses")'''
        mesh_path = "/home/prabhu2004/Desktop/curobo/meshes"
        grasp_path = "/home/prabhu2004/Desktop/curobo/grasps"

        mesh_files = sorted(
            [
                f for f in os.listdir(mesh_path)
                if f.endswith(".obj")
            ]
        )

        print(f"\nFound {len(mesh_files)} mesh files\n")

        for obj_name in mesh_files:

            grasp_file = os.path.join(
                grasp_path,
                obj_name.replace(".obj", ".h5")
            )

            if not os.path.exists(grasp_file):
                print(
                    f"[SKIP] Missing grasp file: "
                    f"{os.path.basename(grasp_file)}"
                )
                continue

            print("\n" + "=" * 80)
            print(f"Processing: {obj_name}")
            print("=" * 80)

            write_grasp_reachability_labels(
                robot_file=args.robot_file,
                pregrasp_distance=args.pregrasp_distance,
                obj_name=obj_name,
                grasp_path=grasp_path,
                mesh_path=mesh_path,
            )
