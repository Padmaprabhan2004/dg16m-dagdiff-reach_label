
import argparse
import sys
import time
import copy

import numpy as np
import torch

from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.scene import Cuboid, Scene, Mesh
from curobo.types import ContentPath, GoalToolPose, Pose
from curobo.viewer import ViserVisualizer
from curobo._src.geom.types import SceneCfg
import trimesh





def make_grounded_mesh(file_path, name="object", x=0.0, y=0.7, z_floor=0.0):
    mesh =trimesh.load(file_path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    z_offset = float(-mesh.bounds[0, 2] + z_floor)
    return Mesh(
        name=name,
        pose=[float(x), float(y)+0.1, z_offset, 1,0,0,0], #0.2 offset form the arms
        file_path=file_path,
    )

#impt function
def interactive_ik_example(robot_file="dual_panda.yml", port=8080):
    """Launch an interactive dual-arm IK viewer."""
    import time

    from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
    from curobo.scene import Scene
    from curobo.types import ContentPath, GoalToolPose, Pose
    from curobo.viewer import ViserVisualizer

    viser_viz = ViserVisualizer(
        content_path=ContentPath(robot_config_file=robot_file),
        connect_ip="0.0.0.0",
        connect_port=port,
        add_control_frames=True,
        visualize_robot_spheres=True,
        add_robot_to_scene=True,
    )

    obj = make_grounded_mesh(
        "/home/prabhu2004/Desktop/curobo/meshes/monitor.obj",
        name="object",
        x=0.0,
        y=0.7,
        z_floor=0.0,
    )
    scene_cfg = SceneCfg(mesh=[obj])
    viser_viz.add_scene(scene_cfg, add_control_frames=False)

    config = InverseKinematicsCfg.create(
        robot=robot_file,
        scene_model="collision_table.yml",
        metrics_rollout="metrics_base.yml",
        transition_model="ik/transition_ik.yml",
        use_cuda_graph=True,
        num_seeds=1,
        seed_solver_num_seeds=1,
        self_collision_check=True,
        collision_cache={"mesh": 10},
    )
    ik_solver = InverseKinematics(config)
    ik_solver.update_world(Scene(mesh=[obj]))
    ik_solver.config.use_lm_seed = False
    ik_solver.config.exit_early = False

    goal_state = ik_solver.default_joint_state.clone()
    kin_state = ik_solver.compute_kinematics(goal_state).clone()
    goal_tool_poses = kin_state.tool_poses.to_dict()

    print(f"Tool frames: {ik_solver.tool_frames}")

    current_state = ik_solver.get_active_js(ik_solver.default_joint_state.clone()).unsqueeze(0)
    ik_solver.solve_pose(
        goal_tool_poses=GoalToolPose.from_poses(
            goal_tool_poses,
            ordered_tool_frames=ik_solver.tool_frames,
            num_goalset=1,
        ),
        current_state=current_state.clone(),
        return_seeds=1,
    )

    print(f"\nInteractive IK running at http://localhost:{port}")
    print("Drag the two end-effector target frames.")
    print("Monitor is static.")
    print("Ctrl+C to exit.\n")

    previous_target_poses = None
    pose_changed = False
    while True:
        target_poses = viser_viz.get_control_frame_pose()
        if previous_target_poses is None:
            previous_target_poses = {k: v.clone() for k, v in target_poses.items()}
        else:
            for frame_name in target_poses.keys():
                if not torch.allclose(
                    target_poses[frame_name].position,
                    previous_target_poses[frame_name].position,
                ) or not torch.allclose(
                    target_poses[frame_name].quaternion,
                    previous_target_poses[frame_name].quaternion,
                ):
                    previous_target_poses = {k: v.clone() for k, v in target_poses.items()}
                    pose_changed = True
                    break

        if pose_changed:
            active_js = ik_solver.get_active_js(current_state)
            target_link_poses = {
                k.replace("target_", ""): v for k, v in target_poses.items()
            }
            result = ik_solver.solve_pose(
                goal_tool_poses=GoalToolPose.from_poses(
                    target_link_poses,
                    ordered_tool_frames=ik_solver.tool_frames,
                    num_goalset=1,
                ),
                current_state=active_js.squeeze(1).clone(),
                return_seeds=1,
            )

            success = bool(result.success.any().item()) if torch.is_tensor(result.success) else bool(result.success)
            if success or result.js_solution is not None:
                current_state = result.js_solution.clone()
                viser_viz.set_joint_state(result.js_solution.squeeze(0).squeeze(0))
                pose_changed = False

        time.sleep(0.001)




if __name__ == "__main__":
    interactive_ik_example()
    #differential_ik_example()
