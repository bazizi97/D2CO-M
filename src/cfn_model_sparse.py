"""Lightning module for training the DIFUSCO MIS model."""

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from co_datasets.Ingraham_dataset import CFNDataset
from utils.diffusion_schedulers import InferenceSchedule
from cfn_meta_sparse import COMetaModel
from utils.effie_utils import compute_nsr, compute_masked_nsr, compute_effie_score


class CFNSPARSEModel(COMetaModel):
    def __init__(self, param_args=None):
        super(CFNSPARSEModel, self).__init__(param_args=param_args, node_feature_only=True)

        data_label_dir = None
        if self.args.training_split_label_dir is not None:
            data_label_dir = os.path.join(self.args.storage_path, self.args.training_split_label_dir)

        self.train_dataset = CFNDataset(
            data_file=os.path.join(self.args.storage_path, self.args.training_split),
            data_label_dir=data_label_dir,
        )

        self.test_dataset = CFNDataset(
            data_file=os.path.join(self.args.storage_path, self.args.test_split),
        )

        self.validation_dataset = CFNDataset(
            data_file=os.path.join(self.args.storage_path, self.args.validation_split),
        )

        # ── Momentum-inference hyper-parameters ──────────────────────────────
        self.epsilon = getattr(self.args, "momentum_epsilon", 1e-3)
        self.a = getattr(self.args, "momentum_a", 0.9)
        self.b = getattr(self.args, "momentum_b", 0.3)
        self.c = getattr(self.args, "momentum_c", 0.99)
        norm = (self.a ** 2 + self.b ** 2) ** 0.5
        self.a /= norm
        self.b /= norm

    def forward(self, x, t, edge_index, edge_attr, batch_vector=None):
        return self.model(x, t, edge_index=edge_index, edge_attr=edge_attr)

    def categorical_training_step(self, batch, batch_idx):
        node_labels = batch[1].x

        t = np.random.randint(1, self.diffusion.T + 1, batch[-1].shape[0]).astype(int)

        # Sample from diffusion process
        node_labels_onehot = F.one_hot(node_labels.long(), num_classes=20).float()
        t = torch.from_numpy(t).long()
        t = t.repeat_interleave(batch[-1].reshape(-1).cpu(), dim=0).numpy()
        xt = self.diffusion.sample(node_labels_onehot, t)

        t = torch.from_numpy(t).float().reshape(-1)

        x0_pred = self.forward(
            xt.long().to(node_labels.device),
            t.float().to(node_labels.device),
            edge_index=batch[1].edge_index,
            edge_attr=batch[1].edge_attr,
            batch_vector=batch[1].batch,
        )

        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(x0_pred, node_labels) 
        self.log("train/loss", loss, batch_size=8)
        return loss

    def training_step(self, batch, batch_idx):
        return self.categorical_training_step(batch, batch_idx)

    def categorical_denoise_step(self, xt, t, device, edge_index=None, edge_attr=None,
                                   target_t=None, batch_vector=None):
        with torch.no_grad():
            t = torch.from_numpy(t).view(1)
            t_node = t.float().to(device).expand(xt.shape[0])
            x0_pred = self.forward(
                xt.long().to(device),
                t_node,
                edge_index=edge_index,
                edge_attr=edge_attr,
                batch_vector=batch_vector,
            )
            x0_pred_prob = x0_pred.softmax(dim=-1)
            xt = self.categorical_posterior(target_t, t, x0_pred_prob, xt)

        return xt

    def test_step(self, batch, batch_idx, draw=False, split='test'):
        device = self.device
        _, graph_data, _ = batch
        node_labels = graph_data.x

        stacked_predict_labels = []
        use_momentum = getattr(self.args, "use_momentum_inference", False)

        for _ in range(self.args.sequential_sampling):
            n_nodes = node_labels.shape[0]

            steps = self.args.inference_diffusion_steps
            time_schedule = InferenceSchedule(
                inference_schedule=self.args.inference_schedule,
                T=self.diffusion.T, inference_T=steps,
            )
            batch_size = 1

            if use_momentum:
                _run_seed = getattr(self.args, "init_seed", None)
                if _run_seed is not None:
                    torch.manual_seed(_run_seed)
                logits_t = torch.randn(n_nodes, 20, device=device)
                xt = torch.argmax(F.softmax(logits_t, dim=-1), dim=-1)
                momentum_state = self.init_momentum_state(n_nodes, device)

                for i in range(steps):
                    t1, t2 = time_schedule(i)
                    t1 = np.array([t1] * batch_size, dtype=int)
                    t2 = np.array([t2] * batch_size, dtype=int)
                    xt, logits_t = self.categorical_denoise_step_momentum(
                        xt, t1, device,
                        edge_index=graph_data.edge_index,
                        edge_attr=graph_data.edge_attr,
                        target_t=t2,
                        batch_vector=graph_data.batch,
                        momentum_state=momentum_state,
                        logits_t=logits_t,
                    )
            else:
                _run_seed = getattr(self.args, "init_seed", None)
                if _run_seed is not None:
                    torch.manual_seed(_run_seed)
                logits_t = torch.randn(n_nodes, 20, device=device)
                xt = torch.argmax(F.softmax(logits_t, dim=-1), dim=-1)
                if self.args.parallel_sampling > 1:
                    xt = xt.repeat(self.args.parallel_sampling, 1, 1)
                    xt = torch.randn_like(xt)
                    edge_index = self.duplicate_edge_index(
                        graph_data.edge_attr, n_nodes, device
                    )

                for i in range(steps):
                    t1, t2 = time_schedule(i)
                    t1 = np.array([t1] * batch_size, dtype=int)
                    t2 = np.array([t2] * batch_size, dtype=int)
                    xt = self.categorical_denoise_step(
                        xt, t1, device,
                        edge_index=graph_data.edge_index,
                        edge_attr=graph_data.edge_attr,
                        target_t=t2,
                        batch_vector=graph_data.batch,
                    )

            stacked_predict_labels.append(xt)

        solved_solutions = []
        accuracy = []
        masked_nsr = []
        for predict_labels in stacked_predict_labels:
            solved_solutions.append(compute_nsr(graph_data.nat, predict_labels).item())
            masked_nsr.append(compute_masked_nsr(graph_data.nat, predict_labels, graph_data.missing).item())
            accuracy.append(compute_nsr(graph_data.x, predict_labels).item())

        best_solved_cost = np.max(solved_solutions)
        best_acc = np.max(accuracy)
        best_masked_nsr = np.max(masked_nsr)

        metrics = {
            f"{split}/Accuracy": best_acc,
        }
        self.outputs = [metrics]
        for k, v in metrics.items():
            self.log(k, v, on_epoch=True, sync_dist=True, batch_size=32)
        self.log(f"{split}/NSR", best_masked_nsr, prog_bar=True, on_epoch=True, sync_dist=True, batch_size=32)

        return metrics
    
    def infer_solution(self, edge_index, edge_attr, n_nodes):
        device = edge_index.device
        edge_index_proc = edge_index.squeeze(0) if edge_index.dim() == 3 else edge_index
        edge_attr_proc = edge_attr.squeeze(0) if edge_attr.dim() == 3 else edge_attr

        stacked_predict_labels = []
        evol_sat = []

        for _ in range(self.args.sequential_sampling):
            _run_seed = getattr(self.args, "init_seed", None)
            if _run_seed is not None:
                torch.manual_seed(_run_seed)
            
            # Initialise random 20-class categorical logits
            logits_t = torch.randn(n_nodes, 20, device=device)
            xt = torch.argmax(F.softmax(logits_t, dim=-1), dim=-1)

            if self.args.parallel_sampling > 1:
                edge_index_proc = self.duplicate_edge_index(edge_index_proc, n_nodes, device)

            batch_size = 1
            steps = self.args.inference_diffusion_steps
            time_schedule = InferenceSchedule(
                inference_schedule=self.args.inference_schedule,
                T=self.diffusion.T,
                inference_T=steps
            )
            
            edge_attr_proc = edge_attr_proc.to(device)
            edge_index_proc = edge_index_proc.to(device)
            
            run_evol_sat = []
            for i in range(steps):
                t1, t2 = time_schedule(i)
                t1 = np.array([t1] * batch_size, dtype=int)
                t2 = np.array([t2] * batch_size, dtype=int)
                xt = self.categorical_denoise_step(
                    xt, t1, device,
                    edge_index=edge_index_proc,
                    edge_attr=edge_attr_proc,
                    target_t=t2,
                    batch_vector=None,
                )
                run_evol_sat.append(
                    compute_effie_score(
                        xt.long(),
                        edge_index=edge_index_proc,
                        edge_attr=edge_attr_proc,
                        batch_size=1
                    )
                )
            stacked_predict_labels.append(xt)
            evol_sat.append(run_evol_sat)

        solved_solutions = [
            compute_effie_score(
                predict_labels.long(),
                edge_index=edge_index_proc,
                edge_attr=edge_attr_proc,
                batch_size=1
            )
            for predict_labels in stacked_predict_labels
        ]
        best_idx = int(np.argmax(solved_solutions))
        best_solved_cost = solved_solutions[best_idx]
        best_predict_labels = stacked_predict_labels[best_idx]
        best_evol_sat = evol_sat[best_idx]

        return best_solved_cost, best_predict_labels, best_evol_sat
    
    # ── Momentum inference helpers ───────────────────────────────────────────

    def init_momentum_state(self, n_nodes: int, device: torch.device) -> dict:
        return {
            "m": torch.zeros(n_nodes, 20, device=device, dtype=torch.float32),
            "v": torch.ones(n_nodes, device=device, dtype=torch.float32),
        }

    def categorical_denoise_step_momentum(
        self,
        xt,
        t,
        device,
        edge_index=None,
        edge_attr=None,
        target_t=None,
        batch_vector=None,
        momentum_state=None,
        logits_t=None,
    ):
        with torch.no_grad():
            t_arr = torch.from_numpy(t).view(1)
            t_node = t_arr.float().to(device).expand(xt.shape[0])
            x0_pred = self.forward(
                xt.long().to(device),
                t_node,
                edge_index=edge_index,
                edge_attr=edge_attr,
                batch_vector=batch_vector,
            )

            d_logits = x0_pred - logits_t
            momentum_state["m"] = (
                self.a * momentum_state["m"] + self.b * d_logits
            )
            momentum_state["v"] = (
                (1 - self.c) * momentum_state["v"]
                + self.c * torch.norm(d_logits, dim=-1) ** 2
            )
            momentum_update = momentum_state["m"] / (
                torch.sqrt(momentum_state["v"]).unsqueeze(-1) + self.epsilon
            )
            logits_next = logits_t + momentum_update

        return torch.argmax(F.softmax(logits_next, dim=-1), dim=-1), logits_next

    def infer_solution_momentum(
        self,
        edge_index,
        edge_attr,
        n_nodes,
        batch_vector=None,
    ):
        device = edge_index.device
        edge_index_proc = edge_index.squeeze(0)
        edge_attr_proc  = edge_attr.squeeze(0)

        stacked_predict_labels: list = []
        evol_sat:  list = []
        denoise_time: list = []

        for _ in range(self.args.sequential_sampling):
            _run_seed = getattr(self.args, "init_seed", None)
            if _run_seed is not None:
                torch.manual_seed(_run_seed)
            logits_t = torch.randn(n_nodes, 20, device=device)
            xt = torch.argmax(F.softmax(logits_t, dim=-1), dim=-1)

            if self.args.parallel_sampling > 1:
                raise NotImplementedError(
                    "parallel_sampling > 1 is not supported in momentum inference"
                )

            momentum_state = self.init_momentum_state(n_nodes, device)

            batch_size = 1
            steps = self.args.inference_diffusion_steps
            time_schedule = InferenceSchedule(
                inference_schedule=self.args.inference_schedule,
                T=self.diffusion.T,
                inference_T=steps,
            )

            edge_attr_proc  = edge_attr_proc.to(device)
            edge_index_proc = edge_index_proc.to(device)

            for i in range(steps):
                t1, t2 = time_schedule(i)
                t1 = np.array([t1] * batch_size, dtype=int)
                t2 = np.array([t2] * batch_size, dtype=int)

                t0 = time.time()
                xt, logits_t = self.categorical_denoise_step_momentum(
                    xt,
                    t1,
                    device,
                    edge_index=edge_index_proc,
                    edge_attr=edge_attr_proc,
                    target_t=t2,
                    batch_vector=batch_vector,
                    momentum_state=momentum_state,
                    logits_t=logits_t,
                )
                denoise_time.append(time.time() - t0)
                evol_sat.append(
                    compute_effie_score(
                        xt.long(),
                        edge_index=edge_index,
                        edge_attr=edge_attr.squeeze(0) if edge_attr.dim() > 2 else edge_attr,
                        batch_size=1,
                    )
                )

            stacked_predict_labels.append(xt)

        solved_solutions = [
            compute_effie_score(
                pred.long(),
                edge_index=edge_index,
                edge_attr=edge_attr.squeeze(0) if edge_attr.dim() > 2 else edge_attr,
                batch_size=1,
            )
            for pred in stacked_predict_labels
        ]
        best_nsr = float(np.max(solved_solutions))
        return best_nsr, stacked_predict_labels, evol_sat, denoise_time

    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx, split='val')
