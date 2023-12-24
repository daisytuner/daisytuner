# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
from __future__ import annotations

import torch
import dace
import platform
import pytorch_lightning as pl

from pathlib import Path
from typing import Union

from torch_geometric.nn import DeepGCNLayer, TransformerConv, GlobalAttention, BatchNorm
from torchmetrics.regression import PearsonCorrCoef

from daisytuner.analysis.similarity.map_nest import MapNest
from daisytuner.analysis.similarity.map_nest_encoding import MapNestEncoding
from daisytuner.analysis.similarity.benchmarking import CPUBenchmark, GPUBenchmark
from daisytuner.analysis.similarity.profiling import (
    CPUProfiling,
    GPUProfiling,
    TARGETS_CPU,
    TARGETS_GPU,
)
from daisytuner.profiling.likwid_helpers import cpu_codename, gpu_codename


class MapNestModel(pl.LightningModule):

    __create_key = object()

    def __init__(
        self,
        create_key: object,
        model_type: dace.DeviceType,
        node_features: int,
        edge_features: int,
        device_features: int,
        profiling_features: int,
        num_targets: int,
        hidden_channels: int = 256,
    ):
        assert create_key == MapNestModel.__create_key
        super().__init__()

        self._model_type = model_type

        # Trainings metrics
        self.train_pearson_corr_runtime_log = PearsonCorrCoef(num_outputs=1)
        self.train_pearson_corr_ipc_log = PearsonCorrCoef(num_outputs=1)
        self.train_pearson_corr_runtime = PearsonCorrCoef(num_outputs=1)
        self.train_pearson_corr_ipc = PearsonCorrCoef(num_outputs=1)
        self.val_pearson_corr_runtime_log = PearsonCorrCoef(num_outputs=1)
        self.val_pearson_corr_ipc_log = PearsonCorrCoef(num_outputs=1)
        self.val_pearson_corr_runtime = PearsonCorrCoef(num_outputs=1)
        self.val_pearson_corr_ipc = PearsonCorrCoef(num_outputs=1)
        self.test_pearson_corr_runtime_log = PearsonCorrCoef(num_outputs=1)
        self.test_pearson_corr_ipc_log = PearsonCorrCoef(num_outputs=1)
        self.test_pearson_corr_runtime = PearsonCorrCoef(num_outputs=1)
        self.test_pearson_corr_ipc = PearsonCorrCoef(num_outputs=1)

        # Encode dynamic features
        self.device_encoder = torch.nn.Linear(
            in_features=device_features, out_features=hidden_channels
        )
        self.profiling_encoder = torch.nn.Linear(
            in_features=profiling_features, out_features=hidden_channels
        )

        # Encode static features
        heads = 4
        assert hidden_channels % heads == 0
        self.backbone = GNNBackbone(
            node_features=node_features,
            edge_features=edge_features,
            hidden_channels=int(hidden_channels / heads),
            heads=heads,
            num_layers=12,
        )

        # Fuse embeddings
        self.neck_lower = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=2 * hidden_channels, out_features=hidden_channels
            ),
            torch.nn.BatchNorm1d(num_features=hidden_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(in_features=hidden_channels, out_features=hidden_channels),
            torch.nn.BatchNorm1d(num_features=hidden_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(in_features=hidden_channels, out_features=hidden_channels),
            torch.nn.BatchNorm1d(num_features=hidden_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(in_features=hidden_channels, out_features=hidden_channels),
            torch.nn.BatchNorm1d(num_features=hidden_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(in_features=hidden_channels, out_features=hidden_channels),
        )

        self.neck_upper = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=3 * hidden_channels + 1, out_features=hidden_channels
            ),
            torch.nn.BatchNorm1d(num_features=hidden_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(in_features=hidden_channels, out_features=hidden_channels),
            torch.nn.BatchNorm1d(num_features=hidden_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(in_features=hidden_channels, out_features=hidden_channels),
            torch.nn.BatchNorm1d(num_features=hidden_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(in_features=hidden_channels, out_features=hidden_channels),
            torch.nn.BatchNorm1d(num_features=hidden_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(in_features=hidden_channels, out_features=hidden_channels),
        )

        self.head = torch.nn.Linear(
            in_features=hidden_channels, out_features=num_targets
        )

        self.loss = torch.nn.L1Loss()

        # Normalization coefficients
        if model_type == dace.DeviceType.CPU:
            self.register_buffer(
                "BENCHMARK_MEAN", torch.tensor(BENCHMARK_CPU_MEAN, dtype=torch.float32)
            )
            self.register_buffer(
                "BENCHMARK_STD", torch.tensor(BENCHMARK_CPU_STD, dtype=torch.float32)
            )
            self.register_buffer(
                "TARGETS_MEAN", torch.tensor(TARGETS_CPU_MEAN, dtype=torch.float32)
            )
            self.register_buffer(
                "TARGETS_STD", torch.tensor(TARGETS_CPU_MEAN, dtype=torch.float32)
            )
        # else:
        #     self.register_buffer(
        #         "BENCHMARK_MEAN", torch.tensor(BENCHMARK_GPU_MEAN, dtype=torch.float32)
        #     )
        #     self.register_buffer(
        #         "BENCHMARK_STD", torch.tensor(BENCHMARK_GPU_STD, dtype=torch.float32)
        #     )
        #     self.register_buffer(
        #         "TARGETS_MEAN", torch.tensor(TARGETS_GPU_MEAN, dtype=torch.float32)
        #     )
        #     self.register_buffer(
        #         "TARGETS_STD", torch.tensor(TARGETS_GPU_MEAN, dtype=torch.float32)
        #     )

    @staticmethod
    def create(
        device: dace.DeviceType,
        dest: str = "cuda:0",
        pretrained: bool = True,
        model_path: Path = None,
    ) -> MapNestModel:
        if device == dace.DeviceType.CPU:
            model = MapNestModel(
                create_key=MapNestModel.__create_key,
                model_type=device,
                node_features=MapNestEncoding.node_dimensions(),
                edge_features=MapNestEncoding.edge_dimensions(),
                device_features=CPUBenchmark.dimensions(),
                profiling_features=CPUProfiling.dimensions(),
                num_targets=len(TARGETS_CPU),
            )
        elif device == dace.DeviceType.GPU:
            model = MapNestModel(
                create_key=MapNestModel.__create_key,
                model_type=device,
                node_features=MapNestEncoding.node_dimensions(),
                edge_features=MapNestEncoding.edge_dimensions(),
                device_features=GPUBenchmark.dimensions(),
                profiling_features=GPUProfiling.dimensions(),
                num_targets=len(TARGETS_GPU),
            )
        else:
            raise ValueError(f"Device type {device} not yet supported")

        if pretrained:
            if device == dace.DeviceType.CPU:
                model_path = (
                    Path(__file__).parent.parent.parent
                    / "data"
                    / "MapNestModel_v5_cpu.ckpt"
                )
            else:
                model_path = (
                    Path(__file__).parent.parent.parent
                    / "data"
                    / "MapNestModel_v5_gpu.ckpt"
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

    def predict(
        self,
        map_nest: MapNest,
        benchmark: Union[CPUBenchmark, GPUBenchmark],
    ):
        cutout = map_nest.as_cutout()
        MapNestEncoding.preprocess(cutout)

        # Static features
        static_encoding = MapNestEncoding(cutout)
        data = static_encoding.encode()

        # Device features
        data.device_features = benchmark.encode()

        # Profiling features
        host = platform.node()
        if self._model_type == dace.DeviceType.CPU:
            if data.is_data_dependent:
                codename = cpu_codename()
                profiling_encoding = CPUProfiling(cutout, host, codename)
                data.profiling_features = profiling_encoding.encode()
            else:
                data.profiling_features = torch.zeros((1, CPUProfiling.dimensions()))
        else:
            if data.is_data_dependent:
                codename = gpu_codename()
                profiling_encoding = GPUProfiling(cutout, host, codename)
                data.profiling_features = profiling_encoding.encode()
            else:
                data.profiling_features = torch.zeros((1, GPUProfiling.dimensions()))

        data = data.to(self.device)
        preds, *rem = self.forward(data)
        preds = torch.exp2((preds * self.TARGETS_STD) + self.TARGETS_MEAN)

        preds = preds.cpu().detach().numpy()[0]
        return (preds, *rem)

    def forward(self, data):
        # 1. Compute static embedding from static features
        static_embedding, node_embeddings = self.backbone(
            data.x, data.edge_index, data.edge_attr, data.batch
        )

        # 2. Compute device embedding from device features
        device_featues = (
            data.device_features - self.BENCHMARK_MEAN
        ) / self.BENCHMARK_STD
        device_embedding = self.device_encoder(device_featues)

        # 3. Lower neck: Static + device
        lower_embedding = torch.hstack([static_embedding, device_embedding])
        lower_embedding = self.neck_lower(lower_embedding)

        # 4. Compute optional profiling embeddings from profiling features
        profiling_featues = data.profiling_features
        profiling_embedding = self.profiling_encoder(profiling_featues)

        mask = data.is_data_dependent
        if not isinstance(mask, torch.Tensor):
            mask = torch.BoolTensor([mask]).to(self.device)
        profiling_embedding[~mask, :] = 0

        # 5. Compute final embedding
        embedding = torch.hstack(
            [
                lower_embedding,
                profiling_embedding,
                static_embedding,
                mask[:, None],
            ]
        )
        embedding = self.neck_upper(embedding)

        # Predict runtime metrics
        preds = self.head(embedding)
        return (
            preds,
            embedding,
            node_embeddings,
        )

    def training_step(self, data, batch_idx):
        batch_size = data.ptr.size(dim=0) - 1
        targets = data.y[:, 0][:, None]
        targets = (targets - self.TARGETS_MEAN) / self.TARGETS_STD

        preds = self(data)[0]
        loss = self.loss(preds, targets)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        self.train_pearson_corr_runtime_log(preds, targets)
        # self.train_pearson_corr_ipc_log(preds[:, 1], targets[:, 1])
        self.train_pearson_corr_runtime(torch.exp2(preds), torch.exp2(targets))
        # self.train_pearson_corr_ipc(torch.exp2(preds[:, 1]), torch.exp2(targets[:, 1]))
        self.log(
            "train_pearson_corr_runtime_log",
            self.train_pearson_corr_runtime_log,
            on_step=True,
            on_epoch=False,
        )
        # self.log(
        #     "train_pearson_corr_ipc_log",
        #     self.train_pearson_corr_ipc_log,
        #     on_step=True,
        #     on_epoch=False,
        # )
        self.log(
            "train_pearson_corr_runtime",
            self.train_pearson_corr_runtime,
            on_step=True,
            on_epoch=False,
        )
        # self.log(
        #     "train_pearson_corr_ipc",
        #     self.train_pearson_corr_ipc,
        #     on_step=True,
        #     on_epoch=False,
        # )

        return loss

    def validation_step(self, data, batch_idx):
        batch_size = data.ptr.size(dim=0) - 1
        targets = data.y[:, 0][:, None]
        targets = (targets - self.TARGETS_MEAN) / self.TARGETS_STD

        preds = self(data)[0]
        loss = self.loss(preds, targets)

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        self.val_pearson_corr_runtime_log(preds, targets)
        # self.val_pearson_corr_ipc_log(preds[:, 1], targets[:, 1])
        self.val_pearson_corr_runtime(torch.exp2(preds), torch.exp2(targets))
        # self.val_pearson_corr_ipc(torch.exp2(preds[:, 1]), torch.exp2(targets[:, 1]))
        self.log(
            "val_pearson_corr_runtime_log",
            self.val_pearson_corr_runtime_log,
            on_step=False,
            on_epoch=True,
        )
        # self.log(
        #     "val_pearson_corr_ipc_log",
        #     self.val_pearson_corr_ipc_log,
        #     on_step=False,
        #     on_epoch=True,
        # )
        self.log(
            "val_pearson_corr_runtime",
            self.val_pearson_corr_runtime,
            on_step=False,
            on_epoch=True,
        )
        # self.log(
        #     "val_pearson_corr_ipc",
        #     self.val_pearson_corr_ipc,
        #     on_step=False,
        #     on_epoch=True,
        # )

    def test_step(self, data, batch_idx):
        batch_size = data.ptr.size(dim=0) - 1
        targets = data.y[:, 0][:, None]
        targets = (targets - self.TARGETS_MEAN) / self.TARGETS_STD

        preds = self(data)[0]
        loss = self.loss(preds, targets)

        self.log(
            "test_loss",
            loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        self.test_pearson_corr_runtime_log(preds, targets)
        # self.test_pearson_corr_ipc_log(preds[:, 1], targets[:, 1])
        self.test_pearson_corr_runtime(torch.exp2(preds), torch.exp2(targets))
        # self.test_pearson_corr_ipc(torch.exp2(preds[:, 1]), torch.exp2(targets[:, 1]))
        self.log(
            "test_pearson_corr_runtime_log",
            self.test_pearson_corr_runtime_log,
            on_step=False,
            on_epoch=True,
        )
        # self.log(
        #     "test_pearson_corr_ipc_log",
        #     self.test_pearson_corr_ipc_log,
        #     on_step=False,
        #     on_epoch=True,
        # )
        self.log(
            "test_pearson_corr_runtime",
            self.test_pearson_corr_runtime,
            on_step=False,
            on_epoch=True,
        )
        # self.log(
        #     "test_pearson_corr_ipc",
        #     self.test_pearson_corr_ipc,
        #     on_step=False,
        #     on_epoch=True,
        # )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            "min",
            threshold=1e-2,
        )
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
                conv,
                norm=BatchNorm(in_channels=heads * hidden_channels),
                act=act,
                block="res+",
                dropout=0.0,
                ckpt_grad=i % 3,
            )
            self.layers.append(layer)

        self.pooling_layer = GlobalAttention(
            gate_nn=torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=heads * hidden_channels,
                    out_features=heads * hidden_channels,
                ),
                torch.nn.BatchNorm1d(num_features=heads * hidden_channels),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(p=0.1),
                torch.nn.Linear(
                    in_features=heads * hidden_channels,
                    out_features=heads * hidden_channels,
                ),
                torch.nn.BatchNorm1d(num_features=heads * hidden_channels),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(p=0.1),
                torch.nn.Linear(
                    in_features=heads * hidden_channels,
                    out_features=heads * hidden_channels,
                ),
                torch.nn.BatchNorm1d(num_features=heads * hidden_channels),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(p=0.1),
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


BENCHMARK_CPU_MEAN = [
    1.3622,
    4.1023,
    1.5205,
    16.6664,
    19.5052,
    16.3520,
    15.2979,
    15.8072,
    16.0396,
]
BENCHMARK_CPU_STD = [
    0.5897,
    1.0049,
    0.1832,
    0.8020,
    1.3262,
    1.3951,
    1.5845,
    1.6859,
    1.6109,
]
TARGETS_CPU_MEAN = [-4.5443]
TARGETS_CPU_STD = [2.1218]
