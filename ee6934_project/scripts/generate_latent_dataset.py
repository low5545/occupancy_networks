import torch
import sys
import os
import argparse
from tqdm import tqdm

# insert the project directory into the module search path
PROJECT_DIR = os.path.join(sys.path[0], '../..')
sys.path.insert(1, PROJECT_DIR)

from im2mesh import config
from im2mesh.checkpoints import CheckpointIO

parser = argparse.ArgumentParser(
    description='Extract latent code from auto-encoder.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('stage', type=str, help='train or val.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

out_dir = cfg['training']['out_dir']
dataset_dir = cfg['generation']['generation_dir']

# Dataset
assert args.stage in [ "train", "val" ]
dataset = config.get_dataset(args.stage, cfg, return_idx=True)

# Model
model = config.get_model(cfg, device=device, dataset=dataset)

checkpoint_io = CheckpointIO(out_dir, model=model)
checkpoint_io.load(cfg['test']['model_file'])

# Loader
loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, num_workers=0, shuffle=False)

# Generate
latent_dataset = torch.empty(0, cfg['model']['c_dim'], device=device)
model.eval()

for it, data in enumerate(tqdm(loader)):
    # Get index etc.
    idx = data['idx'].item()

    try:
        model_dict = dataset.get_model_dict(idx)
    except AttributeError:
        model_dict = {'model': str(idx), 'category': 'n/a'}
    
    category_id = model_dict.get('category', 'n/a')

    # Generate latent dataset
    model.eval()
    inputs = data.get('inputs', torch.empty(1, 0)).to(device)
    with torch.no_grad():
        c = model.encode_inputs(inputs)
    latent_dataset = torch.cat([ latent_dataset, c ], dim=0)

# save dataset
dataset_dir = os.path.join(dataset_dir, str(category_id))
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

dataset_filepath = os.path.join(dataset_dir, f"{args.stage}_latent_code.pt")
torch.save(latent_dataset.cpu(), dataset_filepath)
