# ---------------------------------------------------------
# basic_client2.py - Extended client with Task1/Task2 support
# Task1: Observed nodes (default)
# Task2: Unobserved nodes
# ---------------------------------------------------------

import os
import torch
import pickle
import numpy as np
from copy import deepcopy
from keeper.third_party.fedlab.logger import Logger
from keeper.third_party.fedlab.serialization import SerializationTool
from keeper.third_party.fedlab.trainer import ClientTrainer, SerialClientTrainer

from keeper.backbone.utils import *
from keeper.federated.secure_compression import compress_and_encrypt, compute_graph_sensitivity
from madgrad import MADGRAD

def _flat_to_param_shapes(flat_tensor, param_iter):
    out = []
    offset = 0
    for p in param_iter:
        numel = p.numel()
        out.append(flat_tensor[offset:offset+numel].reshape(p.shape))
        offset += numel
    return out
import torch.optim as optim_mod
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

class SGDClientTrainer(ClientTrainer):
    def __init__(self, model:torch.nn.Module, cuda:bool=False, device:str=None, logger:Logger=None):
        super(SGDClientTrainer, self).__init__(model, cuda, device)
        self._LOGGER = Logger() if logger is None else logger
        self.criterion = torch.nn.CrossEntropyLoss()

    @property
    def uplink_package(self):
        return [self.model_parameters]

    def setup_dataset(self, dataset):
        self.dataset = dataset

    def setup_optim(self, epochs, batch_size, lr):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

    def local_process(self, payload, id): 
        model_parameters = payload[0]
        train_loader = self.dataset.get_dataloader(id, self.batch_size)
        self.train(model_parameters, train_loader)

    def train(self, model_parameters, train_loader) -> None:
        SerializationTool.deserialize_model(self._model, model_parameters)          
        self._LOGGER.info("Local train procedure is running")
        optimizer = torch.optim.SGD(self._model.parameters(), self.lr)
        for epoch in range(self.epochs):
            self._model.train()
            for data, target in train_loader:
                if self.cuda:
                    data, target = data.to(self.device), target.to(self.device)

                outputs = self._model(data)
                loss = self.criterion(outputs, target)

                optimizer.zero_grad()
                loss_pred.backward()
                optimizer.step()

        self._LOGGER.info("Local train procedure is finished")


class SGDSerialClientTrainer(SerialClientTrainer):
    def __init__(self, model, num_clients, cuda=False, device=None, logger=None, personal=False, task="Task1", args=None) -> None:
        super().__init__(model, num_clients, cuda, device, personal)
        self._LOGGER = Logger() if logger is None else logger
        self.cache = []
        self.error_accumulators = {}  # For Momentum-Aware SPQSA Gradient Selection
        self.client_best_performance = []
        self.device = device
        self.experimental_setting = task  # "Task1" = observed, "Task2" = unobserved
        self._args = args  # store for STE-HFM access
        self.fed_algo = getattr(args, 'fed_algo', 'fedavg') if args else 'fedavg'
        self.local_c = {}
        self.feddyn_h_local = {}
        self.feddyn_alpha = getattr(args, 'feddyn_alpha', 0.0) if args else 0.0
        self.moon_mu = getattr(args, 'moon_mu', 0.0) if args else 0.0
        self.moon_tau = getattr(args, 'moon_tau', 0.5) if args else 0.5
        self.moon_prev_flat = {}
        self._moon_global_flat = None
    
    def set_experimental_setting(self, setting):
        """Set experimental setting: 'Task1' for observed nodes, 'Task2' for unobserved nodes."""
        self.experimental_setting = setting
        print(f"Experimental setting set to: {self.experimental_setting}")
                
    def setup_dataset(self, gs):
        self.gs = gs

    def set_malicious_clients(self, clients):
        self.malicious_clients = clients

    def setup_optim(self, epochs, lr, weight_decay):
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay

    def _select_eval_edges(self, test_edges, test_labels, train_snapshot_edges, train_time_slots):
        obs_test_edges = []
        obs_test_labels = []
        unobs_test_edges = []
        unobs_test_labels = []

        train_edges = np.concatenate(train_snapshot_edges[:train_time_slots], axis=0)
        train_set = set(list(train_edges.flatten()))

        if self.experimental_setting == "Task2":
            if train_time_slots >= 2:
                train_pre_edges = np.concatenate(train_snapshot_edges[:train_time_slots-1], axis=0)
                train_pre = set(list(train_pre_edges.flatten()))
            else:
                train_pre = set()

            train_last_edges = train_snapshot_edges[train_time_slots-1]
            train_t = set(list(train_last_edges.flatten()))

            for i in range(len(test_edges)):
                tr = test_edges[i][0]
                te = test_edges[i][1]
                if tr in train_t and tr not in train_pre and te not in train_t and te in train_pre:
                    unobs_test_edges.append(test_edges[i])
                    unobs_test_labels.append(test_labels[i])
                elif te in train_t and te not in train_pre and tr not in train_t and tr in train_pre:
                    unobs_test_edges.append(test_edges[i])
                    unobs_test_labels.append(test_labels[i])
                elif tr in train_t and te in train_t and tr not in train_pre and te not in train_pre:
                    unobs_test_edges.append(test_edges[i])
                    unobs_test_labels.append(test_labels[i])

            return np.array(unobs_test_edges), np.array(unobs_test_labels)

        for i in range(len(test_edges)):
            tr = test_edges[i][0]
            te = test_edges[i][1]
            if tr in train_set and te in train_set:
                obs_test_edges.append(test_edges[i])
                obs_test_labels.append(test_labels[i])

        return np.array(obs_test_edges), np.array(obs_test_labels)

    def _select_eval_edges_both(self, test_edges, test_labels, train_snapshot_edges, train_time_slots):
        """Return (obs_edges, obs_labels, unobs_edges, unobs_labels) in one pass."""
        obs_edges, obs_labels = [], []
        unobs_edges, unobs_labels = [], []

        train_edges = np.concatenate(train_snapshot_edges[:train_time_slots], axis=0)
        train_set = set(list(train_edges.flatten()))

        if train_time_slots >= 2:
            train_pre_edges = np.concatenate(train_snapshot_edges[:train_time_slots-1], axis=0)
            train_pre = set(list(train_pre_edges.flatten()))
        else:
            train_pre = set()
        train_last_edges = train_snapshot_edges[train_time_slots-1]
        train_t = set(list(train_last_edges.flatten()))

        for i in range(len(test_edges)):
            tr, te = int(test_edges[i][0]), int(test_edges[i][1])
            # Task 1: both nodes observed
            if tr in train_set and te in train_set:
                obs_edges.append(test_edges[i])
                obs_labels.append(test_labels[i])
            # Task 2: at least one node new in last slot
            if (tr in train_t and tr not in train_pre and te not in train_t and te in train_pre) or \
               (te in train_t and te not in train_pre and tr not in train_t and tr in train_pre) or \
               (tr in train_t and te in train_t and tr not in train_pre and te not in train_pre):
                unobs_edges.append(test_edges[i])
                unobs_labels.append(test_labels[i])

        return (np.array(obs_edges) if obs_edges else np.empty((0, 2), dtype=np.int64),
                np.array(obs_labels) if obs_labels else np.empty(0),
                np.array(unobs_edges) if unobs_edges else np.empty((0, 2), dtype=np.int64),
                np.array(unobs_labels) if unobs_labels else np.empty(0))

    def evaluate_global_model(self, model_parameters):
        self.set_model(model_parameters)
        self.model.eval()

        per_client_t1 = []
        per_client_t2 = []
        for client_id in sorted(self.gs.main_dictionary.keys()):
            cdata_dict = self.gs.get_dataloader(client_id)
            
            if isinstance(cdata_dict, dict) and 'clean' in cdata_dict:
                eval_data = cdata_dict['clean']
            else:
                eval_data = cdata_dict

            eval_snapshot_edges = eval_data.get('client_edges')
            eval_snapshot_labels = eval_data.get('client_labels')
            if eval_snapshot_edges is None or len(eval_snapshot_edges) <= 2:
                continue

            client_t1_rows = []
            client_t2_rows = []
            with torch.no_grad():
                for testing_time_slots in range(2, len(eval_snapshot_edges)):
                    test_edges = eval_snapshot_edges[testing_time_slots]
                    test_labels = eval_snapshot_labels[testing_time_slots]
                    obs_e, obs_l, unobs_e, unobs_l = self._select_eval_edges_both(
                        test_edges, test_labels, eval_snapshot_edges, testing_time_slots,
                    )

                    if len(obs_e) == 0 and len(unobs_e) == 0:
                        continue

                    # Build client-specific training data for this time slot
                    client_train_edges_np = np.concatenate(eval_snapshot_edges[:testing_time_slots], axis=0)
                    client_train_labels_np = np.concatenate(eval_snapshot_labels[:testing_time_slots], axis=0)
                    
                    if self.cuda:
                        client_train_edges = torch.from_numpy(np.array(client_train_edges_np, dtype=np.int64).T).type(torch.long).to(self.device)
                        client_train_labels_t = torch.from_numpy(np.array(client_train_labels_np, dtype=np.float32)).type(torch.float).to(self.device)
                        client_y = client_train_labels_t[:, 1].long().to(self.device)
                    else:
                        client_train_edges = torch.from_numpy(np.array(client_train_edges_np, dtype=np.int64).T).type(torch.long)
                        client_train_labels_t = torch.from_numpy(np.array(client_train_labels_np, dtype=np.float32)).type(torch.float)
                        client_y = client_train_labels_t[:, 1].long()

                    self._args.train_time_slots = testing_time_slots
                    self.model.args.train_time_slots = testing_time_slots
                    # Build client-specific index_list for evaluate
                    cumsum = 0
                    eval_index_list = []
                    for si in range(testing_time_slots):
                        cumsum += len(eval_snapshot_edges[si])
                        eval_index_list.append(cumsum - 1)
                    self.model.index_list = eval_index_list
                    _, train_z = self.model(client_train_edges, client_y, client_train_labels_t)
                    weights = self.model.regression_weights.to(self.device)

                    if len(obs_e) > 0:
                        se = torch.from_numpy(np.array(obs_e, dtype=np.int64).T).type(torch.long).to(self.device)
                        tz = torch.cat((train_z[se[0], :], train_z[se[1], :]), 1)
                        preds = F.softmax(torch.mm(tz, weights), dim=1)
                        mcc, auc, ab, ap, f1m, f1M = calculate_auc(preds, obs_l)
                        client_t1_rows.append([mcc, auc, ab, ap, f1m, f1M, 0.0])

                    if len(unobs_e) > 0:
                        se = torch.from_numpy(np.array(unobs_e, dtype=np.int64).T).type(torch.long).to(self.device)
                        tz = torch.cat((train_z[se[0], :], train_z[se[1], :]), 1)
                        preds = F.softmax(torch.mm(tz, weights), dim=1)
                        mcc, auc, ab, ap, f1m, f1M = calculate_auc(preds, unobs_l)
                        # Reference-style filter: drop degenerate T2 rows (any of MCC/AUC/BAcc/F1_Macro == 0)
                        if not (mcc == 0.0 or auc == 0.0 or ab == 0.0 or f1M == 0.0):
                            client_t2_rows.append([mcc, auc, ab, ap, f1m, f1M, 0.0])

            if client_t1_rows:
                per_client_t1.append(np.nanmean(np.array(client_t1_rows, dtype=np.float64), axis=0))
            if client_t2_rows:
                per_client_t2.append(np.nanmean(np.array(client_t2_rows, dtype=np.float64), axis=0))

        nan7 = np.array([np.nan]*7)
        t1 = np.nanmean(np.array(per_client_t1, dtype=np.float64), axis=0) if per_client_t1 else nan7
        t2 = np.nanmean(np.array(per_client_t2, dtype=np.float64), axis=0) if per_client_t2 else nan7
        return np.concatenate([t1, t2])  # 14 elements: [t1_mcc..t1_loss, t2_mcc..t2_loss]
        
    
    @property
    def uplink_package(self):
        package = deepcopy(self.cache)
        self.cache = []
        return package

    def compute_loss(self, z, train_edges, y):
        """Reference-style per-step class-weighted loss (re-weights at every training step)."""
        start_node, end_node = z[train_edges[0], :], z[train_edges[1], :]
        features = torch.cat((start_node, end_node), 1)
        weights = self.model.regression_weights.to(self.device)
        predictions = torch.mm(features, weights)
        class_weight = torch.FloatTensor(1.0 / np.bincount(y.cpu()) * features.size(0)).to(self.device)
        crossEntropyLoss = torch.nn.CrossEntropyLoss(weight=class_weight).to(self.device)
        return crossEntropyLoss(predictions, y)

    def local_process(self, payload, id_list):
        model_parameters = payload[0]
        global_c = payload[1] if (self.fed_algo == "scaffold" and len(payload) > 1) else None
        feddyn_global_h = payload[1] if (self.fed_algo == "feddyn" and len(payload) > 1) else None
        self.client_best_performance = []
        # FedProx: store global model parameters as a list (matching model.parameters())
        self._global_params_list = [p.detach().clone() for p in self.model.parameters()]
        for id in id_list:
            data_dict = self.gs.get_dataloader(id)
            # MOON: freeze global model params via serialization
            if self.fed_algo == "moon" and self.moon_mu > 0:
                self.set_model(model_parameters)
                self._moon_global_flat = SerializationTool.serialize_model(self.model).detach().clone()
            pack, best_epoch = self.train(
                model_parameters,
                data_dict,
                id,
                global_c=global_c,
                feddyn_global_h=feddyn_global_h,
            )
            self.cache.append(pack)
            self.client_best_performance.append(best_epoch)

        if self.client_best_performance:
            analyze = np.array(self.client_best_performance)[:, :].astype(np.float64)
            m = np.nanmean(analyze, axis=0)            
            if np.isnan(m).all():
                m = np.zeros_like(m)     
                          
            return m
        return np.array([])
    
    def train(self, model_parameters, cdata_dict, id, global_c=None, feddyn_global_h=None):
        single_prediction = True
        self.best_epochs = []
        init_params = SerializationTool.serialize_model(self.model)
        num_local_steps = 0

        # SCAFFOLD: expand control variates
        if self.fed_algo == "scaffold" and global_c is not None:
            if id not in self.local_c:
                self.local_c[id] = torch.zeros_like(model_parameters)
            self._scaffold_global_c_list = [
                g.detach().clone() for g in _flat_to_param_shapes(global_c, self.model.parameters())
            ]
            self._scaffold_local_c_list = [
                g.detach().clone() for g in _flat_to_param_shapes(self.local_c[id], self.model.parameters())
            ]

        if isinstance(cdata_dict, dict) and 'poisoned' in cdata_dict:
            train_data = cdata_dict['poisoned']
            test_data = cdata_dict['clean']
        else:
            train_data = cdata_dict
            test_data = cdata_dict

        self.train_snapshot_edges   = train_data.get('client_edges')
        self.train_snapshot_labels  = train_data.get('client_labels')
        self.test_snapshot_edges    = test_data.get('client_edges')
        self.test_snapshot_labels   = test_data.get('client_labels')

        avg = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        if single_prediction:
            for train_time_slots in range(2, len(self.train_snapshot_edges)):
                self.best_epoch = [0, 0, 0, 0, 0, 0]
                self.obs_test_edges = []
                self.obs_test_labels = []
                self.unobs_test_edges = []
                self.unobs_test_labels = []

                # Per-time-slot model reset to global params (matches old code)
                self.set_model(model_parameters)
                optimizer = optim_mod.Adam(
                    self._model.parameters(),
                    lr=self.lr,
                    weight_decay=self.weight_decay
                )

                self.train_edges = np.concatenate(self.train_snapshot_edges[:train_time_slots], axis=0)
                self.train_labels = np.concatenate(self.train_snapshot_labels[:train_time_slots], axis=0)
                # Build client-specific index_list matching concatenated snapshot sizes
                cumsum = 0
                client_index_list = []
                for si in range(train_time_slots):
                    cumsum += len(self.train_snapshot_edges[si])
                    client_index_list.append(cumsum - 1)
                # train_set = set(list(self.train_edges.flatten()))

                if self.cuda:
                    self.train_edges  = torch.from_numpy(np.array(self.train_edges, dtype=np.int64).T).type(torch.long).to(self.device)
                    self.train_labels = torch.from_numpy(np.array(self.train_labels, dtype=np.float32)).type(torch.float).to(self.device)
                    col = self.train_labels[:, 1]
                    self.y = col.long().to(self.device)
                else:
                    self.train_edges  = torch.from_numpy(np.array(self.train_edges, dtype=np.int64).T).type(torch.long)
                    self.train_labels = torch.from_numpy(np.array(self.train_labels, dtype=np.float32)).type(torch.float)
                    col = self.train_labels[:, 1]
                    self.y = col.long()

                self.test_edges =  self.test_snapshot_edges[train_time_slots]
                self.test_labels = self.test_snapshot_labels[train_time_slots]
                selected_edges, selected_labels = self._select_eval_edges(
                    self.test_edges,
                    self.test_labels,
                    self.train_snapshot_edges,
                    train_time_slots,
                )

                if self.experimental_setting == "Task2":
                    self.unobs_test_edges = selected_edges
                    self.unobs_test_labels = selected_labels
                else:
                    self.obs_test_edges = selected_edges
                    self.obs_test_labels = selected_labels

                if self.experimental_setting == "Task2":
                    if len(self.unobs_test_edges) == 0:
                        self.best_epoch = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
                else:
                    if len(self.obs_test_edges) == 0:
                        self.best_epoch = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

                c_loss = 0.0
                self._args.train_time_slots = train_time_slots
                self.model.index_list = client_index_list
                self.model.args.train_time_slots = train_time_slots

                # MOON: precompute frozen model representations for contrastive loss
                _moon_z_global = None
                _moon_z_prev = None
                if self.fed_algo == "moon" and self.moon_mu > 0 and id in self.moon_prev_flat:
                    self.model.index_list = client_index_list
                    self.model.args.train_time_slots = train_time_slots
                    # z_global: model currently has global params (set_model called above)
                    self.model.eval()
                    with torch.no_grad():
                        _, _moon_z_global = self.model(self.train_edges, self.y, self.train_labels, return_loss=False)
                    # z_prev: temporarily load previous round's local params
                    current_flat = SerializationTool.serialize_model(self.model).detach().clone()
                    SerializationTool.deserialize_model(self.model, self.moon_prev_flat[id])
                    self.model.eval()
                    with torch.no_grad():
                        _, _moon_z_prev = self.model(self.train_edges, self.y, self.train_labels, return_loss=False)
                    SerializationTool.deserialize_model(self.model, current_flat)

                # FedDyn: expand h_i and w_t for loss computation
                _feddyn_h_i_list = None
                _feddyn_w_t_list = None
                if self.fed_algo == "feddyn" and feddyn_global_h is not None:
                    h_i = self.feddyn_h_local.get(id, torch.zeros_like(model_parameters))
                    _feddyn_h_i_list = _flat_to_param_shapes(h_i.to(self.device), self.model.parameters())
                    _feddyn_w_t_list = _flat_to_param_shapes(model_parameters.to(self.device), self.model.parameters())

                for epoch in range(self.epochs):
                    self.model.train()
                    optimizer.zero_grad()
                    _, z = self.model(self.train_edges, self.y, self.train_labels, return_loss=False)
                    loss_pred = self.compute_loss(z, self.train_edges, self.y)

                    # FedDyn: (alpha/2) * ||w - w_t||^2 - <h_i, w>
                    if _feddyn_h_i_list is not None:
                        quad_term = torch.tensor(0.0, device=self.device)
                        linear_term = torch.tensor(0.0, device=self.device)
                        for p, w_t_p in zip(self.model.parameters(), _feddyn_w_t_list):
                            quad_term = quad_term + torch.norm(p - w_t_p) ** 2
                        for p, h_i_p in zip(self.model.parameters(), _feddyn_h_i_list):
                            linear_term = linear_term + (h_i_p * p).sum()
                        loss_pred = loss_pred + (self.feddyn_alpha / 2.0) * quad_term - linear_term

                    # MOON: contrastive loss
                    if _moon_z_global is not None:
                        z_l = z.mean(dim=0)
                        z_g = _moon_z_global.to(z_l.device).mean(dim=0)
                        z_p = _moon_z_prev.to(z_l.device).mean(dim=0)
                        pos = F.cosine_similarity(z_l.unsqueeze(0), z_g.unsqueeze(0))
                        neg = F.cosine_similarity(z_l.unsqueeze(0), z_p.unsqueeze(0))
                        moon_loss = -torch.log(
                            torch.exp(pos / self.moon_tau) /
                            (torch.exp(pos / self.moon_tau) + torch.exp(neg / self.moon_tau) + 1e-8)
                        )
                        loss_pred = loss_pred + self.moon_mu * moon_loss

                    c_loss += loss_pred.item()

                    loss_pred.backward()
                    if self.fed_algo == "scaffold" and global_c is not None:
                        for p, c_i, c_g in zip(self.model.parameters(), self._scaffold_local_c_list, self._scaffold_global_c_list):
                            p.grad -= (c_i.to(p.device) - c_g.to(p.device))
                    optimizer.step()
                    num_local_steps += 1
                    self.test_model(train_time_slots, epoch, id)

                BEST_EPOCH_METRICS_IDX = [0, 1, 2, 5] # MCC, AUC, ACC_Balanced, F1_Macro
                self.best_epoch.append(c_loss / self.epochs)

                if all(self.best_epoch[i] != 0 for i in BEST_EPOCH_METRICS_IDX):
                    self.best_epochs.append(self.best_epoch)
                
                if len(self.best_epochs) > 0:
                    best_epochs = np.array(self.best_epochs, dtype=float)
                    avg = np.mean(best_epochs, axis=0)
                else:
                    avg = np.array(self.best_epoch)
        else:
            if len(self.best_epochs) > 0:
                best_epochs = np.array(self.best_epochs, dtype=float)
                avg = np.mean(best_epochs, axis=0)
            else:
                avg = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        current_params = self.model_parameters
        uploaded_params = current_params

        # ── SPQSA Compression-After-Encryption ───────────────────────────
        if self._args and getattr(self._args, 'secure_compress', False):
            delta = current_params - model_parameters  # model update
            
            # ── Adaptive Error Accumulation (AdaEF) ──────────────────────
            adaef_enabled = getattr(self._args, 'adaef', False)
            adaef_K = getattr(self._args, 'adaef_K', 5)
            
            if id not in self.error_accumulators:
                self.error_accumulators[id] = torch.zeros_like(delta)
            
            if adaef_enabled:
                # AdaEF: weight past errors by cosine similarity to current gradient
                if not hasattr(self, 'error_history'):
                    self.error_history = {}
                if id not in self.error_history:
                    self.error_history[id] = []
                
                # Compute weighted error from history
                history = self.error_history[id]
                if len(history) > 0:
                    delta_flat = delta.flatten()
                    delta_norm = delta_flat.norm().clamp(min=1e-8)
                    weighted_error = torch.zeros_like(delta)
                    total_weight = 0.0
                    sims = []
                    for past_delta, past_error in history:
                        past_flat = past_delta.flatten().to(delta.device)
                        past_norm = past_flat.norm().clamp(min=1e-8)
                        sim = (delta_flat @ past_flat) / (delta_norm * past_norm)
                        sim_val = abs(sim.item())
                        sims.append(sim_val)
                        weighted_error = weighted_error + sim_val * past_error.to(delta.device)
                        total_weight += sim_val
                    
                    # Standard EF accumulator for comparison
                    standard_acc = sum(pe.to(delta.device) for _, pe in history)
                    
                    if total_weight > 0:
                        self.error_accumulators[id] = weighted_error / total_weight
                    else:
                        self.error_accumulators[id] = torch.zeros_like(delta)
                    
                    adaef_norm = self.error_accumulators[id].norm().item()
                    std_norm = standard_acc.norm().item()
                    sims_str = ", ".join(f"{s:.3f}" for s in sims)
                    print(f"  [AdaEF] Client {id}: history={len(history)} | "
                          f"sims=[{sims_str}] | "
                          f"weight={total_weight:.4f} | "
                          f"adaef_norm={adaef_norm:.4f} | "
                          f"std_ef_norm={std_norm:.4f}")
                else:
                    sims = []
                    print(f"  [AdaEF] Client {id}: round 0, no history (standard EF)")
            else:
                sims = []
            
            delta_with_mem = delta + self.error_accumulators[id].to(delta.device)
            
            # Temporal TAP-Noise: compute topology-based noise sensitivity
            noise_scale = 1.0
            tap_enabled = getattr(self._args, 'tap_noise', False)
            if tap_enabled:
                # Compute temporal churn if snapshots available
                temporal_churn = 0.0
                if hasattr(self, 'train_snapshot_edges') and len(self.train_snapshot_edges) > 1:
                    try:
                        # Use numpy for fast set operations on edge tuples
                        prev = self.train_snapshot_edges[-2]
                        curr = self.train_snapshot_edges[-1]
                        if hasattr(prev, 'shape') and prev.ndim == 2 and prev.shape[0] > 0:
                            prev_set = set(map(tuple, prev))
                            curr_set = set(map(tuple, curr))
                            new_edges = len(curr_set - prev_set)
                            removed_edges = len(prev_set - curr_set)
                            temporal_churn = (new_edges + removed_edges) / (2 * len(prev_set))
                    except Exception:
                        temporal_churn = 0.0
                
                noise_scale = compute_graph_sensitivity(
                    self.train_edges.to(delta.device),
                    self.train_labels.to(delta.device),
                    num_nodes=getattr(self._args, 'num_nodes', 0),
                    temporal_churn=temporal_churn,
                )
                base_eps = getattr(self._args, 'dp_epsilon', 8.0)
                effective_eps = base_eps / noise_scale
                churn_str = f" | churn={temporal_churn:.3f}" if temporal_churn > 0 else ""
                print(f"  [TAP-Noise] Client {id}: noise_scale={noise_scale:.3f} | "
                      f"ε={base_eps} → effective ε={effective_eps:.2f}{churn_str}")
            
            packet = compress_and_encrypt(
                delta_with_mem,
                round_id=getattr(self, '_current_round', 0),
                client_id=int(id),
                shared_secret=self._args.shared_secret,
                target_sparsity=getattr(self._args, 'target_sparsity', 0.90),
                num_bits=getattr(self._args, 'num_bits', 8),
                dp_epsilon=getattr(self._args, 'dp_epsilon', 8.0),
                noise_scale=noise_scale,
            )
            
            
            # Store last client's delta for pipeline visualization
            self.last_client_delta = delta_with_mem.detach().cpu().clone()
            self.last_client_meta = dict(
                round_id=getattr(self, '_current_round', 0),
                client_id=int(id),
            )

            # Update error accumulator and history
            # DP-Aware AdaEF: compute residual against pre-DP reconstruction.
            # This prevents stochastic DP noise from accumulating in the error
            # buffer across rounds, which would degrade gradient quality.
            pre_dp_reconstructed = packet['stats'].pop('reconstructed_pre_dp').to(delta.device)
            post_dp_reconstructed = packet['stats'].pop('reconstructed_dense').to(delta.device)
            # Error buffer uses pre-DP reconstruction (clean quantization/sparsification error only)
            compression_error = delta_with_mem - pre_dp_reconstructed
            self.error_accumulators[id] = compression_error
            
            # AdaEF: store gradient-error pair in history
            if adaef_enabled:
                self.error_history[id].append((delta.detach().cpu(), compression_error.cpu()))
                if len(self.error_history[id]) > adaef_K:
                    self.error_history[id] = self.error_history[id][-adaef_K:]

            st = packet['stats']
            ef_type = "AdaEF" if adaef_enabled else "EF"
            print(f"  [SPQSA-{ef_type}] client {id}: "
                  f"{st['original_bytes']/1024:.1f}KB → {st['compressed_bytes']/1024:.1f}KB  "
                  f"({st['compression_ratio']:.1f}× | ε={st['dp_epsilon']})")
            uploaded_params = packet

        # ── Algorithm-specific extras ──────────────────────────────────────
        algo_extra = None
        if self.fed_algo == "scaffold" and global_c is not None:
            final_params = SerializationTool.serialize_model(self.model)
            total_steps = max(num_local_steps, 1)
            new_delta_c = (init_params - final_params) / (self.lr * total_steps) - global_c
            self.local_c[id] = self.local_c[id] + new_delta_c
            algo_extra = new_delta_c

        if self.fed_algo == "fednova":
            algo_extra = torch.tensor(num_local_steps, dtype=torch.float32)

        if self.fed_algo == "feddyn" and feddyn_global_h is not None:
            final_params = SerializationTool.serialize_model(self.model)
            h_old = self.feddyn_h_local.get(id, torch.zeros_like(model_parameters))
            h_new = h_old - self.feddyn_alpha * (model_parameters.to(final_params.device) - final_params)
            self.feddyn_h_local[id] = h_new
            algo_extra = h_new - h_old

        # MOON: store current model params as previous for next round
        if self.fed_algo == "moon" and self.moon_mu > 0:
            self.moon_prev_flat[id] = SerializationTool.serialize_model(self.model).detach().clone()

        return [uploaded_params, self.model_gradients, avg[0], id, algo_extra], avg

    
    def compute_loss(self, z, train_edges, y):
        start_node, end_node = z[train_edges[0]], z[train_edges[1]]
        features = torch.cat((start_node, end_node), 1)
        if hasattr(self.model, 'regression_weights'):
            weights = self.model.regression_weights.to(self.device)
        else:
            raise AttributeError("Model is missing 'regression_weights'")
        predictions = torch.mm(features, weights)
        class_weight = torch.FloatTensor(1 / np.bincount(y.cpu()) * features.size(0))
        crossEntropyLoss = torch.nn.CrossEntropyLoss(weight=class_weight).to(self.device)
        loss_term = crossEntropyLoss(predictions, y)

        # FedProx: proximal regularization term
        # L_prox = L_task + (mu/2) * ||theta_local - theta_global||^2
        fedprox_mu = getattr(self._args, 'fedprox_mu', 0.0) if hasattr(self, '_args') else 0.0
        if fedprox_mu > 0 and hasattr(self, '_global_params_list'):
            prox_term = torch.tensor(0.0, device=self.device)
            for p_local, p_global in zip(self.model.parameters(), self._global_params_list):
                prox_term = prox_term + torch.norm(p_local - p_global.to(self.device)) ** 2
            loss_term = loss_term + (fedprox_mu / 2.0) * prox_term

        return loss_term
    

    def test_model(self, train_time_slots, epoch, id):
        self.model.eval()
        with torch.no_grad():
            self._args.train_time_slots = train_time_slots
            self.model.args.train_time_slots = train_time_slots
            cumsum = 0
            test_il = []
            for si in range(train_time_slots):
                cumsum += len(self.train_snapshot_edges[si])
                test_il.append(cumsum - 1)
            self.model.index_list = test_il
            _, train_z = self.model(self.train_edges, self.y, self.train_labels)
            
            if self.experimental_setting == "Task2":
                # Task2: Test on unobserved nodes
                if len(self.unobs_test_edges) == 0:
                    return 
                score_edges = torch.from_numpy(np.array(self.unobs_test_edges, dtype=np.int64).T).type(torch.long).to(self.device)
                test_labels_curr = self.unobs_test_labels
            else:
                # Task1 (default): Test on observed nodes
                if len(self.obs_test_edges) == 0:
                    return
                score_edges = torch.from_numpy(np.array(self.obs_test_edges, dtype=np.int64).T).type(torch.long).to(self.device)
                test_labels_curr = self.obs_test_labels

            test_z = torch.cat((train_z[score_edges[0, :], :], train_z[score_edges[1, :], :]), 1)
            
            weights = self.model.regression_weights.to(self.device)
            scores = torch.mm(test_z, weights)
            predictions = F.softmax(scores, dim=1)
            mcc, auc, acc_balanced, precision, f1_micro, f1_macro = calculate_auc(predictions, test_labels_curr)
            current_score = float(mcc + auc + acc_balanced + f1_macro)
            
            # Handle NaNs in best_epoch
            if np.isnan(self.best_epoch[0]):
                self.best_epoch = [mcc, auc, acc_balanced, precision, f1_micro, f1_macro]
            else:
                best_score = (self.best_epoch[0] + self.best_epoch[1] + self.best_epoch[2] + self.best_epoch[5])
                if current_score > best_score:
                    self.best_epoch = [mcc, auc, acc_balanced, precision, f1_micro, f1_macro]
        

