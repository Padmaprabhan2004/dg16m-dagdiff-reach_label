import os
import random

grasps_dir = "/scratch/dualarm/DG16M/dg16m/grasps/"
reach_dir = "/scratch/dualarm/DG16M/dg16m/reach_labels/"
meshes_dir = "/scratch/dualarm/DA2_15mar/meshes/"
sdf_dir = "/scratch/dualarm/DA2_15mar/sdf/"

train_out = "train_final.txt"
test_out = "test_final.txt"

valid_meshes = []

for mesh_file in sorted(os.listdir(meshes_dir)):
    if not mesh_file.endswith(".obj"):
        continue

    name = os.path.splitext(mesh_file)[0]

    grasp_file = os.path.join(grasps_dir, name + ".h5")
    reach_file = os.path.join(reach_dir, name + "_reachability.h5")
    sdf_file = os.path.join(sdf_dir, name + ".json")   

    if (
        os.path.exists(grasp_file)
        and os.path.exists(reach_file)
        and os.path.exists(sdf_file)
    ):
        valid_meshes.append(mesh_file)

print(f"Found {len(valid_meshes)} valid meshes.")

random.seed(42)
random.shuffle(valid_meshes)

train = valid_meshes[:10]
test = valid_meshes[10:20]

with open(train_out, "w") as f:
    f.write("\n".join(train))

with open(test_out, "w") as f:
    f.write("\n".join(test))

print(f"Wrote {len(train)} meshes to {train_out}")
print(f"Wrote {len(test)} meshes to {test_out}")