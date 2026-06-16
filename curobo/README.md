For dual panda yml:
python -m curobo.examples.getting_started.build_robot_model     --urdf curobo/content/assets/robot/franka_description/dual_franka_panda.urdf     --asset-path curobo/content/assets/robot/franka_description     --output dual_panda.yml     --tool-frames panda_hand panda_hand_1     --compute-metrics



#DOUBTS:
WHAT SHOULD I ADD AS END EFFECTOR? panda_hand, panda_hand_1. or the grippers, ee_link and ee_link_1
Dual Panda base link config still to be decided. Can be changed in urdf and then generate the corresponding .yml
should i do batch ik? 
or reachability map? and then sample from it?



TODO:
Need to visialize the accuracy of the collision spheres.
Need to import objects, erase them and write to .h5


