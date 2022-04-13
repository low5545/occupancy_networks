import sys
import os
import argparse
import easydict
import pytorch_lightning as pl

# insert project directory into the module search path
PROJECT_DIR = os.path.join(sys.path[0], '../..')
sys.path.insert(1, PROJECT_DIR)

import im2mesh.config
import ee6934_project

def main(args):
    # load the config from the config file
    config = easydict.EasyDict(im2mesh.config.load_config(args.config))

    # seed all pseudo-random generators
    config.seed = pl.seed_everything(workers=True)
    
    # instantiate the data module & model
    datamodule = ee6934_project.data.datamodule.DataModule(
        config.data.path,
        config.data.classes,
        config.training.batch_size
    )
    model = ee6934_project.models.latent_flow.LatentFlow()

    # instantiate the trainer & its components
    if getattr(config.trainer, "checkpoint_callback", True):
        checkpoint_callback = pl.callbacks.ModelCheckpoint(**config.checkpoint)
        callbacks = [ checkpoint_callback ]
    else:
        callbacks = None

    if getattr(config.trainer, "logger", True):
        logger = pl.loggers.tensorboard.TensorBoardLogger(
            default_hp_metric=False, **config.logger
        )   
    else:
        logger = False
    if hasattr(config.trainer, "logger"):
        config.trainer.pop("logger")

    plugins = {
        None: None,
        "ddp_cpu": pl.plugins.DDPPlugin(find_unused_parameters=False),
        "ddp": pl.plugins.DDPPlugin(find_unused_parameters=False),
        "ddp_spawn": pl.plugins.DDPSpawnPlugin(find_unused_parameters=False)
    }[config.trainer.accelerator]

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        plugins=plugins,
        replace_sampler_ddp=True,
        sync_batchnorm=True,
        terminate_on_nan=True,
        multiple_trainloader_mode="min_size",
        **config.trainer
    )

    # train the model
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training script of Latent Flow"
    )
    parser.add_argument(
        "config", type=str, help="Path to a configuration file in yaml format."
    )
    args = parser.parse_args()
    main(args)
