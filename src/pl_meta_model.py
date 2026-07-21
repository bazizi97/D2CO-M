"""A meta PyTorch Lightning model for training and evaluating DIFUSCO models."""

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader as GraphDataLoader
from pytorch_lightning.utilities import rank_zero_info
from models.gnn_encoder import GNNEncoder
from utils.lr_schedulers import get_schedule_fn
from utils.diffusion_schedulers import CategoricalDiffusion


class COMetaModel(pl.LightningModule):
    def __init__(self, param_args, node_feature_only=True):
        super(COMetaModel, self).__init__()
        self.args = param_args
        self.diffusion_schedule = self.args.diffusion_schedule
        self.diffusion_steps = self.args.diffusion_steps
        self.sparse = self.args.sparse_factor > 0

        out_channels = getattr(self.args, "num_classes", 2)
        self.diffusion = CategoricalDiffusion(
            T=self.diffusion_steps,
            schedule=self.diffusion_schedule,
            num_classes=out_channels,
        )

        self.model = GNNEncoder(
            n_layers=self.args.n_layers,
            hidden_dim=self.args.hidden_dim,
            n_edges_features=self.args.edges_dim,
            n_nodes_features=self.args.nodes_dim,
            n_out_features=out_channels,
            aggregation=self.args.aggregation,
            sparse=self.args.sparse_factor > 0,
            use_activation_checkpoint=self.args.use_activation_checkpoint,
            node_feature_only=node_feature_only,
        )
        self.num_training_steps_cached = None

    def on_test_epoch_end(self):
        unmerged_metrics = {}
        for metrics in self.outputs:
            for k, v in metrics.items():
                if k not in unmerged_metrics:
                    unmerged_metrics[k] = []
                unmerged_metrics[k].append(v)

        merged_metrics = {}
        for k, v in unmerged_metrics.items():
            merged_metrics[k] = float(np.mean(v))
        self.logger.log_metrics(merged_metrics, step=self.global_step)

    def get_total_num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.num_training_steps_cached is not None:
            return self.num_training_steps_cached
        dataset = self.train_dataloader()
        if self.trainer.max_steps and self.trainer.max_steps > 0:
            return self.trainer.max_steps

        dataset_size = (
            self.trainer.limit_train_batches * len(dataset)
            if self.trainer.limit_train_batches != 0
            else len(dataset)
        )

        num_devices = max(1, self.trainer.num_devices)
        effective_batch_size = self.trainer.accumulate_grad_batches * num_devices
        self.num_training_steps_cached = (
            dataset_size // effective_batch_size
        ) * self.trainer.max_epochs
        return self.num_training_steps_cached

    def configure_optimizers(self):
        rank_zero_info(
            "Parameters: %d" % sum([p.numel() for p in self.model.parameters()])
        )
        rank_zero_info("Training steps: %d" % self.get_total_num_training_steps())

        if self.args.lr_scheduler == "constant":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            )

        else:
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            )
            scheduler = get_schedule_fn(
                self.args.lr_scheduler, self.get_total_num_training_steps()
            )(optimizer)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }

    def categorical_posterior(self, target_t, t, x0_pred_prob, xt):
        """Sample from the categorical posterior for a given time step."""
        diffusion = self.diffusion

        if target_t is None:
            target_t = t - 1
        
        device = x0_pred_prob.device

        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=device)
        t = t.long()
        
        if not isinstance(target_t, torch.Tensor):
            target_t = torch.as_tensor(target_t, device=device)
        target_t = target_t.long()

        if t.numel() == 1:
            t = t.view([])
        if target_t.numel() == 1:
            target_t = target_t.view([])

        Q_bar_all = torch.as_tensor(diffusion.Q_bar, dtype=torch.float32, device=device)

        Q_bar_t = Q_bar_all[t]
        Q_bar_target_t = Q_bar_all[target_t]

        Q_t = torch.linalg.inv(Q_bar_target_t) @ Q_bar_t
        
        Q_bar_t_source = Q_bar_t
        Q_bar_t_target = Q_bar_target_t

        num_classes = getattr(diffusion, "num_classes", 2)
        xt = F.one_hot(xt.long(), num_classes=num_classes).float()
        xt = xt.reshape(x0_pred_prob.shape)

        p_xt_given_xtarget = torch.matmul(xt, Q_t.permute((1, 0)).contiguous())
        p_xt_given_x0 = torch.matmul(xt, Q_bar_t_source.permute((1, 0)).contiguous())

        weight = x0_pred_prob / p_xt_given_x0.clamp(min=1e-8)
        term2 = torch.matmul(weight, Q_bar_t_target)

        posterior_prob = p_xt_given_xtarget * term2
        posterior_prob = posterior_prob / posterior_prob.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        posterior_prob = posterior_prob.clamp(min=0.0)

        if target_t > 0:
            if num_classes == 2:
                xt = torch.bernoulli(posterior_prob[..., 1].clamp(0, 1))
            else:
                xt = torch.distributions.Categorical(probs=posterior_prob.clamp(min=1e-8)).sample()
        else:
            xt = torch.argmax(x0_pred_prob, dim=-1)

        if self.sparse:
            xt = xt.reshape(-1)
        return xt

    def duplicate_edge_index(self, edge_index, num_nodes, device):
        """Duplicate edge_index for parallel sampling."""
        edge_index = edge_index.reshape((2, 1, -1))
        edge_index_indent = (
            torch.arange(0, self.args.parallel_sampling).view(1, -1, 1).to(device)
        )
        edge_index_indent = edge_index_indent * num_nodes
        edge_index = edge_index + edge_index_indent
        edge_index = edge_index.reshape((2, -1))
        return edge_index

    def train_dataloader(self):
        batch_size = self.args.batch_size
        train_dataloader = GraphDataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )
        return train_dataloader

    def test_dataloader(self):
        batch_size = getattr(self.args, "test_batch_size", 32)
        test_dataloader = GraphDataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=self.args.num_workers,
        )
        return test_dataloader

    def val_dataloader(self):
        batch_size = getattr(self.args, "val_batch_size", 32)
        n_val = min(self.args.validation_examples, len(self.validation_dataset))
        val_dataset = torch.utils.data.Subset(
            self.validation_dataset, range(n_val)
        )
        val_dataloader = GraphDataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=self.args.num_workers,
        )
        return val_dataloader
