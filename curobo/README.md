For dual panda yml:
python -m curobo.examples.getting_started.build_robot_model     --urdf curobo/content/assets/robot/franka_description/dual_franka_panda.urdf     --asset-path curobo/content/assets/robot/franka_description     --output dual_panda.yml     --tool-frames panda_hand panda_hand_1     --compute-metrics


TODO:
Need to visialize the accuracy of the collision spheres.
Need to import objects, erase them and write to .h5


