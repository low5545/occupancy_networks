import torch
import sys
import os
import glob
import argparse
from tqdm import tqdm
import easydict
import pytorch_lightning as pl

# insert the project directory into the module search path
PROJECT_DIR = os.path.join(sys.path[0], '../..')
sys.path.insert(1, PROJECT_DIR)

from im2mesh import config
from im2mesh.checkpoints import CheckpointIO
import ee6934_project


# ------------- nflows hotfix ------------- #
def weight_inverse_and_logabsdet(self):
    """
    Cost:
        inverse = O(D^3)
        logabsdet = O(D)
    where:
        D = num of features
    """
    # If both weight inverse and logabsdet are needed, it's cheaper to compute both together.
    identity = torch.eye(self.features, self.features, device=self._weight.device)  # hotfix to place it to the correct device
    weight_inv, lu = torch.solve(identity, self._weight)  # Linear-system solver.
    logabsdet = torch.sum(torch.log(torch.abs(torch.diag(lu))))
    return weight_inv, logabsdet


def bind(instance, func, as_name=None):
    """
    Bind the function *func* to *instance*, with either provided name *as_name*
    or the existing name of *func*. The provided *func* should accept the 
    instance as the first argument, i.e. "self".

    Reference: https://stackoverflow.com/a/1015405
    """
    if as_name is None:
        as_name = func.__name__
    bound_method = func.__get__(instance, instance.__class__)
    setattr(instance, as_name, bound_method)
    return bound_method
# ----------------------------------------- #


parser = argparse.ArgumentParser(
    description='Extract meshes from occupancy process.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')

args = parser.parse_args()
cfg = easydict.EasyDict(config.load_config(args.config, 'configs/default.yaml'))
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

out_dir = cfg['training']['out_dir']
category_id = cfg.data.classes[0]
generation_dir = os.path.join(cfg['generation']['generation_dir'], category_id)

if not os.path.exists(generation_dir):
    os.makedirs(generation_dir)

# seed all pseudo-random generators
pl.seed_everything(workers=True)

# Models
pcae_model = config.get_model(cfg, device=device, dataset=None)
checkpoint_io = CheckpointIO(out_dir, model=pcae_model)
checkpoint_io.load(cfg['test']['model_file'])

latent_flow_ckpt_dir = os.path.join(
    cfg.logger.save_dir, cfg.logger.name, "version_0", "checkpoints"
)
latent_flow_ckpt_paths = glob.glob(
    os.path.join(latent_flow_ckpt_dir, "*.ckpt")
)
latent_flow_model = ee6934_project.models.latent_flow.LatentFlow.load_from_checkpoint(
    checkpoint_path=latent_flow_ckpt_paths[0]
)
latent_flow_model.to(device)

# Generator
generator = config.get_generator(pcae_model, cfg, device=device)

# Generate
pcae_model.eval()
latent_flow_model.eval()

# nflows hotfix
for index in [ 1, 3, 5, 7 ]:
    bind(
        instance=latent_flow_model.core._transform._transforms[index],
        func=weight_inverse_and_logabsdet
    )

test_dataset_len = len(config.get_dataset("test", cfg, return_idx=True))
for it in tqdm(range(test_dataset_len)):
    while True:
        with torch.no_grad():
            c = latent_flow_model.core.sample(num_samples=1)

        z = pcae_model.get_z_from_prior((1,), sample=generator.sample).to(device)
        generated_mesh = generator.generate_from_latent(z, c)

        if not generated_mesh.is_empty:
            break
    
    mesh_out_file = os.path.join(generation_dir, '%d.off' % it)
    generated_mesh.export(mesh_out_file)
