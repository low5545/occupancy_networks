from turtle import forward
import torch
import pytorch_lightning as pl
import nflows.transforms, nflows.nn.nets, nflows.flows, nflows.distributions


class LatentFlow(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.core = self._build_core()

    def _build_core(
        self,
        latent_dims=128,
        hidden_layer_dims=128,
        num_residual_blocks=1,
        num_coupling_layers=4,
        hidden_activation_fn=torch.nn.functional.relu
    ):
        # define alternating mask for each coupling transform
        mask = torch.ones(latent_dims)
        mask[::2] = -1

        # define helper function to create scale & translation inference
        # networks
        def _create_resnet(input_dims, output_dims):
            return nflows.nn.nets.ResidualNet(
                in_features=input_dims,
                out_features=output_dims,
                hidden_features=hidden_layer_dims,
                context_features=None,
                num_blocks=num_residual_blocks,
                activation=hidden_activation_fn,
                dropout_probability=0,
                use_batch_norm=False
            )

        # define the invertible transformation composed of affine coupling
        # layers with feature reversing in between
        invertible_transforms = []
        for index in range(num_coupling_layers):
            layer = nflows.transforms.AffineCouplingTransform(
                mask=mask,
                transform_net_create_fn=_create_resnet,
                scale_activation=lambda x : (torch.nn.functional.softplus(x) + 1e-3)
            )
            linear = nflows.transforms.NaiveLinear(
                features=latent_dims,
                orthogonal_initialization=True,
                using_cache=True
            )
            invertible_transforms.extend([ layer, linear ])

        # instantiate the normalizing flow
        return nflows.flows.Flow(
            transform=nflows.transforms.CompositeTransform(invertible_transforms),
            distribution=nflows.distributions.StandardNormal([ latent_dims ])
        )
    
    def forward(self, batch):
        pass

    def training_step(self, batch, batch_index):
        return self._step(batch, batch_index, stage="train")

    def validation_step(self, batch, batch_index):
        self._step(batch, batch_index, stage="val")

    def _step(self, batch, batch_index, stage):
        batch = batch[0]    # remove redundant packing
        batch_size = batch.shape[0]

        nll_loss = -self.core.log_prob(inputs=batch).mean()
        self.log(
            f"{stage}/nll_loss", nll_loss, batch_size=batch_size, prog_bar=True
        )
        return nll_loss

    def configure_optimizers(self):
        # instantiate optimizer
        optimizer = torch.optim.Adam(
            self.parameters(), lr=0.001, weight_decay=0.1
        )

        # instantiate learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[ 150000 ],
            gamma=0.1
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step"
            }
        }
