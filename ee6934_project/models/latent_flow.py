from turtle import forward
import torch
import pytorch_lightning as pl
import nflows.transforms, nflows.nn.nets, nflows.flows, nflows.distributions


class InputsOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs, context=None):
        return self.model(inputs)


class LatentFlow(pl.LightningModule):
    def __init__(
        self,
        latent_dims=128,
        hidden_layer_dims=128,
        num_hidden_layers=2,
        num_coupling_layers=4,
        hidden_activation_fn=torch.nn.functional.relu
    ):
        super().__init__()
        self.core = self._build_core(
            latent_dims,
            hidden_layer_dims,
            num_hidden_layers,
            num_coupling_layers,
            hidden_activation_fn
        )

    def _build_core(
        self,
        latent_dims,
        hidden_layer_dims,
        num_hidden_layers,
        num_coupling_layers,
        hidden_activation_fn
    ):
        # define alternating mask for each coupling transform
        mask = torch.ones(latent_dims)
        mask[::2] = -1

        # define helper function to create scale & translation inference
        # networks
        def _create_mlp(input_dims, output_dims):
            hidden_sizes = num_hidden_layers * [ hidden_layer_dims ]
            return InputsOnly(
                nflows.nn.nets.MLP(
                    in_shape=[ input_dims ],
                    out_shape=[ output_dims ],
                    hidden_sizes=hidden_sizes,
                    activation=hidden_activation_fn,
                    activate_output=False
                )
            )

        # define the invertible transformation composed of affine coupling
        # layers with feature reversing in between
        invertible_transforms = []
        for index in range(num_coupling_layers):
            layer = nflows.transforms.AffineCouplingTransform(
                mask=mask,
                transform_net_create_fn=_create_mlp,
                scale_activation=nflows.transforms.AffineCouplingTransform
                                                  .GENERAL_SCALE_ACTIVATION
            )
            reverse = nflows.transforms.ReversePermutation(
                features=latent_dims
            )
            invertible_transforms.extend([ layer, reverse ])

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
            self.parameters(), lr=0.001
        )

        # instantiate learning rate scheduler
        return {
            "optimizer": optimizer
        }
