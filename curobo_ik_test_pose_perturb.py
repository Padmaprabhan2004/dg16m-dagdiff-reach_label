import torch

from curobo.inverse_kinematics import InverseKinematics,InverseKinematicsCfg
from curobo.types import Pose,GoalToolPose


config= InverseKinematicsCfg.create(robot="dual_franka.yml",num_seeds=32)

ik =  InverseKinematics(config)
print("Tool frames:",ik.tool_frames)

js=ik.compute_kinematics(ik.default_joint_state)

left_current = js.tool_poses["panda_hand"]
right_current = js.tool_poses["panda_hand_1"]
#small pert
left_target_position = (
    left_current.position
    + torch.tensor(
        [[0.05, 0.00, 0.00]],
        device="cuda",
        dtype=torch.float32,
    )
)

right_target_position = (
    right_current.position
    + torch.tensor(
        [[0.05, 0.00, 0.00]],
        device="cuda",
        dtype=torch.float32,
    )
)

left_target = Pose(position=left_target_position,quaternion=left_current.quaternion)
right_target = Pose(position=right_target_position,quaternion=right_current.quaternion)

goal_dict = {"panda_hand":left_target,"panda_hand_1":right_target}
goal = GoalToolPose.from_poses(goal_dict,ordered_tool_frames=ik.tool_frames,num_goalset=1)

result = ik.solve_pose(goal)
print(result.success)


if result.success.item():

    print("\nIK Solved!")

    print("\nJoint Solution:")
    print(result.js_solution.position)

    print("\nPosition Error (meters):")
    print(result.position_error)

else:

    print("\nIK Failed")