import torch
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
#For testing the dual panda setup, batched_ik, single_ik, ik_with collisions and visualization
import viser.transforms as vtf


def dual_panda_current_pose_test():

    config = InverseKinematicsCfg.create(
        robot="dual_franka.yml",
        num_seeds=32,
    )

    ik = InverseKinematics(config)

    #print("Tool frames:")
    #print(ik.tool_frames)
    left_goal = Pose(
        position=torch.tensor(
            [[0.45, -0.20, 0.45]],
            device="cuda",
            dtype=torch.float32,
        ),
        quaternion=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        ),
    )
    right_goal = Pose(
        position=torch.tensor(
            [[0.45, 0.20, 0.45]],
            device="cuda",
            dtype=torch.float32,
        ),
        quaternion=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        ),
    )

    goal = GoalToolPose.from_poses(
        {
            "panda_hand": left_goal,
            "panda_hand_1": right_goal,
        },
        ordered_tool_frames=ik.tool_frames,
        num_goalset=1,
    )

    result = ik.solve_pose(goal)
    #print(result.success)
    result = ik.solve_pose(goal)

    print("Success:", result.success)

    if result.success.item():
        print("IK solved!")
        print(result.js_solution.position)
    else:
        print("IK failed")



def batched_ik_dual_panda_2():
    n_poses = 100
    config = InverseKinematicsCfg.create(
        robot="dual_franka.yml",
        num_seeds=32,
        max_batch_size=n_poses,
    )
    ik = InverseKinematics(config)
    target_link = ik.tool_frames
    #print(target_link)
    #first arm
    left_positions = torch.zeros(
        n_poses,
        3,
        device="cuda",
        dtype=torch.float32,
    )

    left_positions[:, 0] = torch.linspace(
        0.35,
        0.55,
        n_poses,
    )

    left_positions[:, 1] = -0.20
    left_positions[:, 2] = 0.45

    left_quaternions = torch.zeros(
        n_poses,
        4,
        device="cuda",
        dtype=torch.float32,
    )

    left_quaternions[:, 0] = 1.0

    left_goal = Pose(
        position=left_positions,
        quaternion=left_quaternions,
    )


    #second arm
    right_positions = torch.zeros(
        n_poses,
        3,
        device="cuda",
        dtype=torch.float32,
    )

    right_positions[:, 0] = torch.linspace(
        0.35,
        0.55,
        n_poses,
    )

    right_positions[:, 1] = 0.20
    right_positions[:, 2] = 0.45

    right_quaternions = torch.zeros(
        n_poses,
        4,
        device="cuda",
        dtype=torch.float32,
    )

    right_quaternions[:, 0] = 1.0
    right_goal = Pose(
        position=right_positions,
        quaternion=right_quaternions,
    )
    goal = GoalToolPose.from_poses({"panda_hand": left_goal,"panda_hand_1": right_goal,},ordered_tool_frames=ik.tool_frames,num_goalset=1,)
    result = ik.solve_pose(goal)
    n_success = result.success.sum().item()


    #diagnostics
    print(
        f"Solved {n_success}/{n_poses} "
        f"({100*n_success/n_poses:.1f}% success)"
    )

    successful = result.success.squeeze()

    if n_success > 0:

        pos_errors = result.position_error[successful]

        print(
            f"Mean position error: "
            f"{pos_errors.mean().item()*1000:.3f} mm"
        )

        print(
            f"Max position error: "
            f"{pos_errors.max().item()*1000:.3f} mm"
        )

    return n_success > 0



#works
def batched_ik_dual_panda():



    from curobo.inverse_kinematics import (
        InverseKinematics,
        InverseKinematicsCfg,
    )

    from curobo.types import (
        Pose,
        GoalToolPose,
    )

    n_poses = 100

    config = InverseKinematicsCfg.create(
        robot="dual_franka.yml",
        num_seeds=32,
        max_batch_size=n_poses,
    )

    ik = InverseKinematics(config)

    #print("Tool Frames:")
    #print(ik.tool_frames)

    kin_state = ik.compute_kinematics(
        ik.default_joint_state
    )

    left_current = kin_state.tool_poses["panda_hand"]
    right_current = kin_state.tool_poses["panda_hand_1"]

    print("\nCurrent Left Position:")
    print(left_current.position)

    print("\nCurrent Right Position:")
    print(right_current.position)

    offsets = torch.zeros(
        n_poses,
        3,
        device="cuda",
        dtype=torch.float32,
    )

    #can change this to get wider range of perts for testing ik, +-0.5 gives <100% reach
    offsets[:, 0] = torch.linspace(
        -0.5,
        0.5,
        n_poses,
    )

    
    angle_deg = 20.0
    angle_rad = torch.deg2rad(torch.tensor(angle_deg))

    rand_axis = torch.randn(
        n_poses,
        3,
        device="cuda",
    )

    rand_axis = rand_axis / rand_axis.norm(
        dim=1,
        keepdim=True,
    )

    rand_angle = (
        torch.rand(
            n_poses,
            device="cuda",
        ) * 2.0 - 1.0
    ) * angle_rad

    delta_q = torch.zeros(
        n_poses,
        4,
        device="cuda",
    )

    delta_q[:,0] = torch.cos(rand_angle/2)

    delta_q[:,1:] = (
        rand_axis
        * torch.sin(rand_angle/2).unsqueeze(-1)
    )

    def quat_mul(q1, q2):
        w1,x1,y1,z1 = q1.unbind(-1)
        w2,x2,y2,z2 = q2.unbind(-1)

        return torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ], dim=-1)

    left_positions = (
        left_current.position
        + offsets
    )
    right_positions = (
        right_current.position
        + offsets
    )

    left_quaternions = quat_mul(
        delta_q,
        left_current.quaternion.expand(n_poses,-1)
    )

    right_quaternions = quat_mul(
        delta_q,
        right_current.quaternion.expand(n_poses,-1)
    )


    left_goal = Pose(
        position=left_positions,
        quaternion=left_quaternions,
    )
    right_goal = Pose(
        position=right_positions,
        quaternion=right_quaternions,
    )


    goal = GoalToolPose.from_poses(
        {
            "panda_hand": left_goal,
            "panda_hand_1": right_goal,
        },
        ordered_tool_frames=ik.tool_frames,
        num_goalset=1,
    )

    result = ik.solve_pose(goal)
    print("\nSuccess tensor shape:")
    print(result.success.shape)
    n_success = result.success.sum().item()
    print(
        f"\nSolved {n_success}/{n_poses} "
        f"({100*n_success/n_poses:.1f}% success)"
    )

    print(
        "Orientation error:",
        result.rotation_error.mean().item()
    )

    if n_success > 0:

        successful = result.success.squeeze()

        pos_errors = result.position_error[
            successful
        ]

        print(
            f"Mean position error: "
            f"{pos_errors.mean().item()*1000:.3f} mm"
        )

        print(
            f"Max position error: "
            f"{pos_errors.max().item()*1000:.3f} mm"
        )

    return n_success > 0


def batched_ik_with_obstacle():



    from curobo.inverse_kinematics import (
        InverseKinematics,
        InverseKinematicsCfg,
    )

    from curobo.types import (
        Pose,
        GoalToolPose,
    )

    n_poses = 100

    #robots collision spheres might be intersecting (0/100)
    new_obstacle = Cuboid(
        name="box_1",
        pose=[0.5, 0.0, 0.3, 1, 0, 0, 0],
        dims=[0.1, 0.3, 0.2],
    )


    #this one works, with the cuboid exactly in the middle (100/100)
    new_obstacle = Cuboid(
        name="box_1",
        pose=[0.0, 0.58, 0.65, 1, 0, 0, 0],
        dims=[0.15, 0.15, 0.15],
    )
    config_with_cache = InverseKinematicsCfg.create(
        robot="dual_franka.yml",
        scene_model="collision_table.yml",
        num_seeds=32,
        max_batch_size=n_poses,
        self_collision_check=True,
        collision_cache={"cuboid":10}
    )

    ik = InverseKinematics(config_with_cache)
    ik.update_world(Scene(cuboid=[new_obstacle]))

    #print("Tool Frames:")
    #print(ik.tool_frames)

    kin_state = ik.compute_kinematics(
        ik.default_joint_state
    )

    left_current = kin_state.tool_poses["panda_hand"]
    right_current = kin_state.tool_poses["panda_hand_1"]

    print("\nCurrent Left Position:")
    print(left_current.position)

    print("\nCurrent Right Position:")
    print(right_current.position)

    offsets = torch.zeros(
        n_poses,
        3,
        device="cuda",
        dtype=torch.float32,
    )

    #can change this to get wider range of perts for testing ik, +-0.5 gives <100% reach
    offsets[:, 0] = torch.linspace(
        -0.6,
        0.6,
        n_poses,
    )

    left_positions = (
        left_current.position
        + offsets
    )
    right_positions = (
        right_current.position
        + offsets
    )
    left_quaternions = (
        left_current.quaternion
        .expand(n_poses, -1)
        .contiguous()
    )
    right_quaternions = (
        right_current.quaternion
        .expand(n_poses, -1)
        .contiguous()
    )
    left_goal = Pose(
        position=left_positions,
        quaternion=left_quaternions,
    )
    right_goal = Pose(
        position=right_positions,
        quaternion=right_quaternions,
    )


    goal = GoalToolPose.from_poses(
        {
            "panda_hand": left_goal,
            "panda_hand_1": right_goal,
        },
        ordered_tool_frames=ik.tool_frames,
        num_goalset=1,
    )

    result = ik.solve_pose(goal)
    print("\nSuccess tensor shape:")
    print(result.success.shape)
    n_success = result.success.sum().item()
    print(
        f"\nSolved {n_success}/{n_poses} "
        f"({100*n_success/n_poses:.1f}% success)"
    )

    if n_success > 0:
        successful = result.success.squeeze()
        pos_errors = result.position_error[
            successful
        ]
        print(
            f"Mean position error: "
            f"{pos_errors.mean().item()*1000:.3f} mm"
        )
        print(
            f"Max position error: "
            f"{pos_errors.max().item()*1000:.3f} mm"
        )

    return n_success > 0


def make_grounded_mesh(file_path, name="object", x=0.0, y=0.7, z_floor=0.0):
    mesh =trimesh.load(file_path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    z_offset = float(-mesh.bounds[0, 2] + z_floor)
    return Mesh(
        name=name,
        pose=[float(x), float(y)+0.2, z_offset, 1, 0, 0, 0], #0.2 offset form the arms
        file_path=file_path,
    )


def reachability_map(robot_file="dual_franka.yml", port=8080):



    BATCH_TARGET = 100
    n_per_axis = int(BATCH_TARGET ** 0.5)
    actual_batch = n_per_axis * n_per_axis
    total_batch = actual_batch + 1

    viser_viz = ViserVisualizer(
        content_path=ContentPath(robot_config_file=robot_file),
        connect_ip="0.0.0.0",
        connect_port=port,
        add_control_frames=False,
        visualize_robot_spheres=False,
        add_robot_to_scene=True,
    )

    #change the location in front of the dual arm
    '''cube = Cuboid(
        name="workspace_cuboid",
        pose=[0.0, 0.7, 0.1, 1, 0, 0, 0],
        dims=[0.15, 0.15, 0.15],
    )'''

    obj = make_grounded_mesh(
        "/home/prabhu2004/Desktop/curobo/meshes/monitor.obj",
        name="object",
        x=0.0,
        y=0.7,
        z_floor=0.0,
    )


    scene_cfg = SceneCfg(mesh=[obj])
    server = viser_viz._server
    obstacle_frames = viser_viz.add_scene(scene_cfg, add_control_frames=True)



    config = InverseKinematicsCfg.create(
        robot=robot_file,
        scene_model="collision_table.yml",
        self_collision_check=True,
        max_batch_size=total_batch,
        collision_cache={"mesh":10}
    )

    ik = InverseKinematics(config)
    ik.update_world(Scene(mesh=[obj]))
    ik.exit_early = False
    all_target_links = ik.tool_frames
    primary_link = all_target_links[0]

    kin_state = ik.compute_kinematics(ik.default_joint_state)
    arm_views = {}
    for idx, link_name in enumerate(all_target_links):
        tool_pose = kin_state.tool_poses[link_name]
        center = tool_pose.position.squeeze().cpu().numpy()
        arm_views[link_name] = {
            "slice_gizmo": server.scene.add_transform_controls(
                f"/reachability_gizmo_{link_name}",
                scale=0.15,
                position=tuple(center.tolist()),
                wxyz=(1.0, 0.0, 0.0, 0.0),
            ),
            "tool_frame_gizmos": {},
            "grid_name": f"/reachability_gizmo_{link_name}/slice_image",
            "bounds_name": f"/reachability_gizmo_{link_name}/bounds",
        }
        for other_link in all_target_links:
            link_pose = kin_state.tool_poses[other_link]
            link_pos = link_pose.position.squeeze().cpu().numpy()
            link_quat = link_pose.quaternion.squeeze().cpu().numpy()
            arm_views[link_name]["tool_frame_gizmos"][other_link] = server.scene.add_transform_controls(
                f"/tool_frame_{link_name}_{other_link}",
                scale=0.10,
                position=tuple(link_pos.tolist()),
                wxyz=tuple(link_quat.tolist()),
            )

    cube_frame = server.scene.add_transform_controls(
        "/workspace_cuboid",
        scale=0.12,
        position=tuple(obj.pose[:3]),
        wxyz=tuple(obj.pose[3:]),
    )

    with server.gui.add_folder("Reachability"):
        grid_extent_slider = server.gui.add_slider(
            "Grid Extent (m)",
            min=0.1,
            max=2.0,
            step=0.05,
            initial_value=1.0,
        )
        grid_height_slider = server.gui.add_slider(
            "Grid Height (m)",
            min=0.15,
            max=1.2,
            step=0.05,
            initial_value=0.45,
        )

    old_obstacle_poses = {
        "workspace_cuboid": Pose.from_numpy(cube_frame.position, cube_frame.wxyz)
    }
    prev_views = {
        link_name: {
            "pos": np.array(view["slice_gizmo"].position, dtype=np.float32),
            "wxyz": np.array(view["slice_gizmo"].wxyz, dtype=np.float32),
            "tool_poses": {
                name: (
                    np.array(g.position, dtype=np.float32),
                    np.array(g.wxyz, dtype=np.float32),
                )
                for name, g in view["tool_frame_gizmos"].items()
            },
        }
        for link_name, view in arm_views.items()
    }
    prev_extent = grid_extent_slider.value
    prev_height = grid_height_slider.value

    print(f"\nReachability viewer running at http://localhost:{port}")
    print(f"Slice: {n_per_axis}x{n_per_axis} = {actual_batch} IK queries per update")
    print("Drag the gizmo to move/rotate the slice plane.")
    print("Adjust the 'Grid Extent' and 'Grid Height' sliders to reshape it.")
    print("Drag the cuboid obstacle to test different workspace layouts.")
    print("Press Ctrl+C to exit.\n")

    needs_update = True
    while True:
        obstacle_pose = Pose.from_numpy(cube_frame.position, cube_frame.wxyz)
        if obstacle_pose != old_obstacle_poses["workspace_cuboid"]:
            ik.scene_collision_checker.update_obstacle_pose(
                "workspace_cuboid", obstacle_pose
            )
            needs_update = True
            old_obstacle_poses["workspace_cuboid"] = obstacle_pose.clone()

        cur_extent = grid_extent_slider.value
        cur_height = grid_height_slider.value
        cur_view_states = {}
        for link_name, view in arm_views.items():
            cur_view_states[link_name] = {
                "pos": np.array(view["slice_gizmo"].position, dtype=np.float32),
                "wxyz": np.array(view["slice_gizmo"].wxyz, dtype=np.float32),
                "tool_poses": {
                    name: (
                        np.array(g.position, dtype=np.float32),
                        np.array(g.wxyz, dtype=np.float32),
                    )
                    for name, g in view["tool_frame_gizmos"].items()
                },
            }
        if cur_extent != prev_extent or cur_height != prev_height:
            needs_update = True
            prev_extent = cur_extent
            prev_height = cur_height
        for link_name in all_target_links:
            if (
                not np.allclose(cur_view_states[link_name]["pos"], prev_views[link_name]["pos"])
                or not np.allclose(cur_view_states[link_name]["wxyz"], prev_views[link_name]["wxyz"])
                or any(
                    not np.allclose(cur_view_states[link_name]["tool_poses"][n][0], prev_views[link_name]["tool_poses"][n][0])
                    or not np.allclose(cur_view_states[link_name]["tool_poses"][n][1], prev_views[link_name]["tool_poses"][n][1])
                    for n in all_target_links
                )
            ):
                needs_update = True
                prev_views[link_name] = cur_view_states[link_name]

        if not needs_update:
            time.sleep(0.02)
            continue
        needs_update = False

        extent = cur_extent
        half = extent / 2.0
        height = cur_height
        half_h = height / 2.0
        lin = torch.linspace(-half, half, n_per_axis, device="cuda", dtype=torch.float32)
        uu, vv = torch.meshgrid(lin, lin, indexing="xy")
        local_pts = torch.stack(
            [
                uu.reshape(-1),
                vv.reshape(-1),
                torch.full((actual_batch,), half_h, device="cuda", dtype=torch.float32),
                torch.ones(actual_batch, device="cuda", dtype=torch.float32),
            ],
            dim=-1,
        )

        total_success = {}
        for link_name in all_target_links:
            view = arm_views[link_name]
            cur_pos = cur_view_states[link_name]["pos"]
            cur_wxyz = cur_view_states[link_name]["wxyz"]
            rot = vtf.SO3(cur_wxyz).as_matrix().astype(np.float32)
            pose_matrix = np.eye(4, dtype=np.float32)
            pose_matrix[:3, :3] = rot
            pose_matrix[:3, 3] = cur_pos
            pose_t = torch.tensor(pose_matrix, device="cuda", dtype=torch.float32)
            grid_world_pts = (pose_t @ local_pts.T).T[:, :3]

            goal_dict = {}
            for target_link in all_target_links:
                lp, lq = cur_view_states[link_name]["tool_poses"][target_link]
                link_pos = torch.tensor(lp, device="cuda", dtype=torch.float32).unsqueeze(0).expand(total_batch, -1).contiguous()
                link_quat = torch.tensor(lq, device="cuda", dtype=torch.float32).unsqueeze(0).expand(total_batch, -1).contiguous()
                goal_dict[target_link] = Pose(position=link_pos, quaternion=link_quat)

            primary_quat = torch.tensor(
                cur_view_states[link_name]["tool_poses"][link_name][1],
                device="cuda",
                dtype=torch.float32,
            )
            goal_dict[link_name] = Pose(
                position=torch.cat([
                    grid_world_pts,
                    torch.tensor(cur_view_states[link_name]["tool_poses"][link_name][0], device="cuda", dtype=torch.float32).unsqueeze(0),
                ], dim=0),
                quaternion=primary_quat.unsqueeze(0).expand(total_batch, -1).contiguous(),
            )

            result = ik.solve_pose(
                GoalToolPose.from_poses(
                    goal_dict,
                    ordered_tool_frames=all_target_links,
                    num_goalset=1,
                ),
            )

            all_success = result.success.squeeze().cpu().numpy().astype(bool)
            grid_success = all_success[:actual_batch].reshape(n_per_axis, n_per_axis)
            gizmo_success = all_success[actual_batch]
            total_success[link_name] = int(grid_success.sum())

            img = np.zeros((n_per_axis, n_per_axis, 3), dtype=np.uint8)
            img[grid_success] = [0, 200, 0]
            img[~grid_success] = [200, 0, 0]
            server.scene.add_image(
                name=view["grid_name"],
                image=img,
                render_width=extent,
                render_height=extent,
            )

            corners_local = np.array(
                [
                    [-half, -half, half_h],
                    [half, -half, half_h],
                    [half, half, half_h],
                    [-half, half, half_h],
                ],
                dtype=np.float32,
            )
            corners_world = (rot @ corners_local.T).T + cur_pos
            edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
            lines = np.array(
                [[corners_world[i], corners_world[j]] for i, j in edges],
                dtype=np.float32,
            )
            yellow = np.array([255, 255, 0], dtype=np.uint8)
            server.scene.add_line_segments(
                view["bounds_name"],
                points=lines,
                colors=yellow,
                line_width=3.0,
            )

            if gizmo_success and link_name == primary_link:
                gizmo_js = result.js_solution[actual_batch]
                viser_viz.set_joint_state(gizmo_js.squeeze(0))

        print(
            "Reachability: "
            + " | ".join(
                f"{name} {total_success[name]}/{actual_batch} ({100 * total_success[name] / actual_batch:.0f}%)"
                for name in all_target_links
            )
            + f" | Grid: {n_per_axis}x{n_per_axis} = {actual_batch} | Extent: {extent:.2f} m | Height: {height:.2f} m"
        )






if __name__ == "__main__":

    #dual_panda_current_pose_test()
    batched_ik_dual_panda()
    #batched_ik_with_obstacle()
    #reachability_map()
