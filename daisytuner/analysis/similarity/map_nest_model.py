# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
from __future__ import annotations

import torch
import dace
import pytorch_lightning as pl

from pathlib import Path

from torch_geometric.nn import DeepGCNLayer, TransformerConv, GlobalAttention

from daisytuner.analysis.similarity.map_nest_encoding import MapNestEncoding
from daisytuner.analysis.similarity.device_encoding import CPUEncoding, GPUEncoding

from daisytuner.analysis.similarity.profiling_features.norm_coeffs import *
from daisytuner.analysis.similarity.profiling_features.targets import (
    TARGETS,
    TARGETS_GPU,
)


class MapNestModel(pl.LightningModule):

    __create_key = object()

    def __init__(
        self,
        create_key: object,
        device: dace.DeviceType,
        node_features: int,
        edge_features: int,
        device_features: int,
        profiling_features: int,
        num_targets: int,
        hidden_channels: int = 256,
    ):
        assert create_key == MapNestModel.__create_key
        super().__init__()

        heads = 4
        assert hidden_channels % heads == 0
        self.backbone = GNNBackbone(
            node_features=node_features,
            edge_features=edge_features,
            hidden_channels=int(hidden_channels / heads),
            heads=heads,
            num_layers=8,
        )
        self.device_encoder = torch.nn.Sequential(
            torch.nn.Linear(in_features=device_features, out_features=hidden_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=hidden_channels, out_features=hidden_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=hidden_channels, out_features=hidden_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=hidden_channels, out_features=hidden_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=hidden_channels, out_features=hidden_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=hidden_channels, out_features=hidden_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=hidden_channels, out_features=hidden_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=hidden_channels, out_features=hidden_channels),
        )
        self.neck_lower = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=2 * hidden_channels, out_features=hidden_channels
            ),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=hidden_channels, out_features=hidden_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=hidden_channels, out_features=hidden_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=hidden_channels, out_features=hidden_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=hidden_channels, out_features=hidden_channels),
        )

        self.profiling_encoder = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=profiling_features, out_features=hidden_channels
            ),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=hidden_channels, out_features=hidden_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=hidden_channels, out_features=hidden_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=hidden_channels, out_features=hidden_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=hidden_channels, out_features=hidden_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=hidden_channels, out_features=hidden_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=hidden_channels, out_features=hidden_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=hidden_channels, out_features=hidden_channels),
        )

        self.neck_upper = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=2 * hidden_channels, out_features=hidden_channels
            ),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=hidden_channels, out_features=hidden_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=hidden_channels, out_features=hidden_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=hidden_channels, out_features=hidden_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=hidden_channels, out_features=hidden_channels),
        )

        self.head = torch.nn.Linear(
            in_features=hidden_channels, out_features=num_targets
        )

        self.loss = torch.nn.L1Loss()

        # Normalization coefficients
        if device == dace.DeviceType.CPU:
            self.register_buffer(
                "DEVICE_MEAN", torch.tensor(ARCHS_MEAN_CPU, dtype=torch.float32)
            )
            self.register_buffer(
                "DEVICE_STD", torch.tensor(ARCHS_STD_CPU, dtype=torch.float32) + 1e-6
            )
            self.register_buffer(
                "COUNTERS_MEAN", torch.tensor(COUNTERS_MEAN_CPU, dtype=torch.float32)
            )
            self.register_buffer(
                "COUNTERS_STD",
                torch.tensor(COUNTERS_STD_CPU, dtype=torch.float32) + 1e-6,
            )
            self.register_buffer(
                "TARGETS_MEAN", torch.tensor(TARGETS_MEAN_CPU, dtype=torch.float32)
            )
            self.register_buffer(
                "TARGETS_STD", torch.tensor(TARGETS_STD_CPU, dtype=torch.float32) + 1e-6
            )
        elif device == dace.DeviceType.GPU:
            self.register_buffer(
                "DEVICE_MEAN", torch.tensor(ARCHS_MEAN_GPU, dtype=torch.float32)
            )
            self.register_buffer(
                "DEVICE_STD", torch.tensor(ARCHS_STD_GPU, dtype=torch.float32) + 1e-6
            )
            self.register_buffer(
                "COUNTERS_MEAN", torch.tensor(COUNTERS_MEAN_GPU, dtype=torch.float32)
            )
            self.register_buffer(
                "COUNTERS_STD",
                torch.tensor(COUNTERS_STD_GPU, dtype=torch.float32) + 1e-6,
            )
            self.register_buffer(
                "TARGETS_MEAN", torch.tensor(TARGETS_MEAN_GPU, dtype=torch.float32)
            )
            self.register_buffer(
                "TARGETS_STD", torch.tensor(TARGETS_STD_GPU, dtype=torch.float32) + 1e-6
            )

    @staticmethod
    def create(
        device: dace.DeviceType,
        dest: str = "cpu",
        pretrained: bool = True,
        model_path: Path = None,
    ) -> MapNestModel:
        if device == dace.DeviceType.CPU:
            model = MapNestModel(
                create_key=MapNestModel.__create_key,
                device=device,
                node_features=MapNestEncoding.node_dimensions(),
                edge_features=MapNestEncoding.edge_dimensions(),
                device_features=CPUEncoding.dimensions(),
                profiling_features=64,
                num_targets=len(TARGETS),
            )
        elif device == dace.DeviceType.GPU:
            model = MapNestModel(
                create_key=MapNestModel.__create_key,
                device=device,
                node_features=MapNestEncoding.node_dimensions(),
                edge_features=MapNestEncoding.edge_dimensions(),
                device_features=GPUEncoding.dimensions(),
                profiling_features=68,
                num_targets=len(TARGETS_GPU),
            )
        else:
            raise ValueError(f"Device type {device} not yet supported")

        if pretrained:
            if device == dace.DeviceType.CPU:
                model_path = (
                    Path(__file__).parent.parent.parent
                    / "data"
                    / "MapNestModel_v4_cpu.ckpt"
                )
            else:
                model_path = (
                    Path(__file__).parent.parent.parent
                    / "data"
                    / "MapNestModel_v4_gpu.ckpt"
                )
            checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
            model.load_state_dict(checkpoint["state_dict"])
        else:
            if model_path is not None:
                checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
                model.load_state_dict(checkpoint["state_dict"])

        model.to(dest)
        model.eval()

        return model

    def forward(self, data):
        # 1. Compute static embedding from static features
        static_embedding, node_embeddings = self.backbone(
            data.x, data.edge_index, data.edge_attr, data.batch
        )

        # 2. Compute device embedding from device features
        device_featues = (data.device_features - self.DEVICE_MEAN) / self.DEVICE_STD
        device_embedding = self.device_encoder(device_featues)

        # 3. Lower neck: Static + device
        lower_embedding = torch.hstack([static_embedding, device_embedding])
        lower_embedding = self.neck_lower(lower_embedding)

        # 4. Compute optional profiling embeddings from profiling features
        profiling_embedding = torch.zeros_like(lower_embedding).to(
            lower_embedding.device
        )
        if hasattr(data, "profiling_features"):
            profiling_featues = (
                data.profiling_features - self.COUNTERS_MEAN
            ) / self.COUNTERS_STD
            profiling_embedding = self.profiling_encoder(profiling_featues)

        # 4. Compute final embedding
        upper_embedding = torch.hstack([lower_embedding, profiling_embedding])
        upper_embedding = self.neck_upper(upper_embedding)

        # Predict runtime metrics
        preds = self.head(upper_embedding)

        return (
            preds,
            upper_embedding,
            lower_embedding,
            node_embeddings,
            device_embedding,
        )

    def training_step(self, data, batch_idx):
        batch_size = data.ptr.size(dim=0) - 1
        targets = (data.y - self.TARGETS_MEAN) / self.TARGETS_STD

        preds = self(data)[0]
        l = self.loss(preds, targets)

        self.log(
            "train_loss",
            l,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        return l

    def validation_step(self, data, batch_idx):
        batch_size = data.ptr.size(dim=0) - 1
        targets = (data.y - self.TARGETS_MEAN) / self.TARGETS_STD

        preds = self(data)[0]
        l = self.loss(preds, targets)

        self.log(
            "val_loss",
            l,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

    def test_step(self, data, batch_idx):
        batch_size = data.ptr.size(dim=0) - 1
        targets = (data.y - self.TARGETS_MEAN) / self.TARGETS_STD

        preds = self(data)[0]
        l = self.loss(preds, targets)

        self.log(
            "test_loss",
            l,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


class GNNBackbone(torch.nn.Module):
    def __init__(
        self,
        node_features: int,
        edge_features: int,
        hidden_channels: int,
        num_layers: int,
        heads: int,
    ) -> None:
        super().__init__()

        self.node_encoder = torch.nn.Linear(
            in_features=node_features, out_features=heads * hidden_channels, bias=True
        )
        self.edge_encoder = torch.nn.Linear(
            in_features=edge_features, out_features=heads * hidden_channels, bias=True
        )

        self.layers = torch.nn.Sequential()
        for i in range(1, num_layers + 1):
            conv = TransformerConv(
                in_channels=heads * hidden_channels,
                out_channels=hidden_channels,
                heads=heads,
                concat=True,
                dropout=0.0,
                edge_dim=heads * hidden_channels,
            )
            act = torch.nn.LeakyReLU(inplace=True)

            layer = DeepGCNLayer(
                conv, act=act, block="res+", dropout=0.0, ckpt_grad=i % 3
            )
            self.layers.append(layer)

        self.pooling_layer = GlobalAttention(
            gate_nn=torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=heads * hidden_channels,
                    out_features=heads * hidden_channels,
                ),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(
                    in_features=heads * hidden_channels,
                    out_features=heads * hidden_channels,
                ),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(
                    in_features=heads * hidden_channels,
                    out_features=heads * hidden_channels,
                ),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(in_features=heads * hidden_channels, out_features=1),
            )
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        x = self.layers[0].conv(x, edge_index, edge_attr)
        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)

        y = self.pooling_layer(x, batch)
        return y, x
