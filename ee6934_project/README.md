# EE6934 Project 2: Generating Neural Occupancy Fields with Latent Flows for 3D Shape Synthesis

## Project Code
All additional code implemented for this project are placed in the `ee6934_project/` folder. Most of it are built upon the code base of Occupancy Networks.

## Installation
1. Create the `onet` conda environment with:
    ```bash
    conda env create -f environment.yaml
    ```
    and activate it with:
    ```bash
    conda activate onet
    ```
2. Compile extension modules with:
   ```bash
   python setup.py build_ext --inplace
   python setup_c.py build_ext --inplace
   ```

## Dataset
Download the Occupancy Networks-preprocessed ShapeNet dataset with:
```bash
bash scripts/download_data.sh
```
which should download and unpack the data automatically into the `data/ShapeNet/` folder.

## Training
1. To train the point cloud completion network, run:
   ```bash
   python train.py ee6934_project/configs/pcae/pcae_{CATEGORY}.yaml
   ```
   which saves its checkpoints and logs at `out/ee6934_project/pcae/pcae_{CATEGORY}/`.
2. To train the latent flow network, first generate the latent code dataset with:
   ```bash
   python ee6934_project/scripts/generate_latent_dataset.py ee6934_project/configs/dataset/pcae_{CATEGORY}_pretrained.yaml train
   python ee6934_project/scripts/generate_latent_dataset.py ee6934_project/configs/dataset/pcae_{CATEGORY}_pretrained.yaml val
   ```
   which saves the dataset at `ee6934_project/dataset/{CATEGORY_ID}`. Then, start the training with:
   ```bash
   python ee6934_project/scripts/train_latent_flow.py ee6934_project/configs/latent_flow/latent_flow_{CATEGORY}.yaml
   ```
   which saves its checkpoints and logs at `out/ee6934_project/latent_flow/latent_flow_{CATEGORY}/`.
1. The training progress can be monitored with:
   ```bash
   tensorboard --logdir out
   ```

## Pretrained Models
Instead of training the point cloud completion & latent flow network from scratch, you can download their pretrained models from [this link](https://drive.google.com/file/d/1bYcyjRHm0u1Ynzw-0MSJoVVgBYyuaCpD/view?usp=sharing) and place them in the `out/ee6934_project/` folder.

## Evaluation
1. To generate meshes using the pretrained CVAE model, run:
   ```bash
   python generate.py configs/unconditional/onet_{CATEGORY}_pretrained.yaml
   ```
   which saves its outputs at `out/unconditional/onet_{CATEGORY}/`.
2. To generate meshes using the pretrained latent flow model, run:
   ```bash
   python ee6934_project/scripts/generate_mesh.py ee6934_project/configs/mesh/mesh_{CATEGORY}.yaml
   ```
   which saves its outputs at `out/ee6934_project/generated_mesh/{CATEGORY_ID}`.
3. To compute the COV-CD and MMD-CD metrics on the generated meshes, run:
   ```bash
   python ee6934_project/scripts/eval_generated_mesh.py {MESH_FOLDER} {CATEGORY_ID}
   ```
4. To render images of the generated meshes, run:
   ```bash
   python ee6934_project/scripts/render_meshes.py {MESH_FOLDER} {RENDER_OUTPUT_FOLDER}
   ```
