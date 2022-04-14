import argparse
import os
import sys
import tqdm
import torch
import pytorch3d.io, pytorch3d.structures, pytorch3d.ops, pytorch3d.loss

# insert project directory into the module search path
PROJECT_DIR = os.path.join(sys.path[0], '../..')
sys.path.insert(1, PROJECT_DIR)

import im2mesh.data


POINT_CLOUD_SAMPLE_SIZE = 10000

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # read & sample point clouds on the generated meshes
    print("Reading generated meshes...")
    io = pytorch3d.io.IO()

    meshes = []
    for mesh_file in tqdm.tqdm(os.listdir(args.mesh_dir)):
        mesh_path = os.path.join(args.mesh_dir, mesh_file)
        meshes.append(io.load_mesh(
            path=mesh_path, include_textures=False, device=device
        ))
    meshes = pytorch3d.structures.join_meshes_as_batch(
        meshes=meshes, include_textures=False
    )

    print("Sampling generated meshes...")
    generated_pcls = pytorch3d.ops.sample_points_from_meshes(
        meshes=meshes, num_samples=POINT_CLOUD_SAMPLE_SIZE
    )

    # read ShapeNet testing dataset point clouds
    test_dataset = im2mesh.data.Shapes3dDataset(
        dataset_folder=args.shapenet_dir,
        fields={
            "inputs": im2mesh.data.PointCloudField(
                        file_name="pointcloud.npz",
                        transform=im2mesh.data.SubsamplePointcloud(POINT_CLOUD_SAMPLE_SIZE)
                      )
        },
        split="test",
        categories=[ args.category_id ]
    )

    print("Reading test point clouds...")
    test_pcls = [ 
        torch.from_numpy(test_dataset[index]["inputs"]).to(device)
        for index in tqdm.tqdm(range(len(test_dataset))) 
    ]
    test_pcls = torch.stack(test_pcls, dim=0)

    # compute COV-CD & MMD-CD
    print("Computing COV-CD & MMD-CD...")

    is_covered = torch.zeros(len(test_pcls), device=device)
    mmd_cd = torch.full(
        size=[ len(test_pcls) ], fill_value=float("inf"), device=device
    )
    for generated_pcl in tqdm.tqdm(generated_pcls):
        batch_generated_pcl = generated_pcl.unsqueeze(dim=0) \
                                           .expand(len(test_pcls), -1, -1)      # (len(test_pcls), POINT_CLOUD_SAMPLE_SIZE, 3)
        chamfer_dists, _ = pytorch3d.loss.chamfer_distance(                     # (len(test_pcls))
            batch_generated_pcl, test_pcls, batch_reduction=None
        )
        is_covered[chamfer_dists.argmin()] = 1
        mmd_cd = torch.minimum(mmd_cd, chamfer_dists)

    cov_cd = is_covered.mean()
    mmd_cd = mmd_cd.mean()
    
    print("COV-CD:", cov_cd.item())
    print("MMD-CD:", mmd_cd.item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluation script for generated meshes."
    )
    parser.add_argument(
        "mesh_dir", type=str,
        help="Directory to generated meshes."
    )
    parser.add_argument(
        "category_id", type=str,
        help="Shape category."
    )
    parser.add_argument(
        "--shapenet_dir", type=str, default="data/ShapeNet",
        help="Directory to ShapeNet dataset."
    )
    args = parser.parse_args()
    main(args)
