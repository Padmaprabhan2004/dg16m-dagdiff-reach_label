# For dual panda yml:
python -m curobo.examples.getting_started.build_robot_model     --urdf curobo/content/assets/robot/franka_description/dual_franka_panda.urdf     --asset-path curobo/content/assets/robot/franka_description     --output dual_panda.yml     --tool-frames panda_hand panda_hand_1     --compute-metrics



# For computing reachability labels:
python grasp_reachability_label.py --pregrasp_distance 0.08 (Store meshes and grasps in meshes/ and grasps/)


TODO:
Need to visialize the accuracy of the collision spheres.
Need to import objects, erase them and write to .h5


