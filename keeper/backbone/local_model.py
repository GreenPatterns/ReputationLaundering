import torch
import random
import numpy as np
import torch.nn as nn
import importlib
from typing import Tuple
import torch.nn.init as init
from torch.nn import Parameter
from keeper.backbone.s3r import S3R
from keeper.backbone.structure_layer import StructuralNetwork
from keeper.backbone.attacks import *
from keeper.backbone.utils import *
# ===================================================================
# 2. LocalGDTE MODEL CLASS
# ===================================================================
class LocalGDTE(torch.nn.Module):
    def __init__(self, device, args, X, num_labels, index_list, class_weight=None, strategy=None):
        super(LocalGDTE, self).__init__()
        self.args = args
        self.set_seed(self.args.seed)
        self.device = device
        self.strategy = strategy or {}
        self.pipeline_mode = self.strategy.get("mode") == "pipeline"
        self.X = X
        self.nodes = X.shape[0]
        self.num_labels = num_labels
        self.index_list = index_list
        self.dropout = self.args.dropout
        self.input_dim = self.args.layers[-1]
        self.num_heads = self.args.attention_head
        self.timeslots = self.args.train_time_slots
        self.class_weight = class_weight
        
        self.attack = self.args.attack
        
        self._build_model()
        self._initialize_loss_functions()
        
    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        
    def _build_model(self) -> None:
        if self.pipeline_mode:
            self.gpu_structural = torch.device(f"cuda:{self.strategy['gpu_structural']}")
            self.gpu_s3r = torch.device(f"cuda:{self.strategy['gpu_s3r']}")
            X_struct = self.X.to(self.gpu_structural)
        else:
            self.gpu_structural = self.device
            self.gpu_s3r = self.device
            X_struct = self.X

        self.structural_layer = StructuralNetwork(
                self.gpu_structural,
                self.args,
                X_struct,
                self.num_labels,
        ).to(self.gpu_structural)

        self.s3r = S3R(input_dim=self.input_dim,
                         n_heads=self.num_heads,
                         max_seq_len=int(getattr(self.args, 'max_snapshots_inaClient', 0) or self.args.train_time_slots),
                         dropout=self.dropout,
                         top_k_ratio=0.3,
                         ).to(self.gpu_s3r)
        del self.X  # free tensor; only needed for StructuralNetwork init
                         
    def _initialize_loss_functions(self) -> None:
        self.mse_loss_func = nn.MSELoss()
        loss_device = self.gpu_s3r if self.pipeline_mode else self.device
        # Output head. Previously `Parameter(torch.Tensor(...)).to(loss_device)`:
        # on a CUDA device Tensor.to() returns a plain Tensor, so the head was
        # never registered -> absent from parameters()/state_dict(), never
        # optimised, never serialised (frozen at its xavier init). Allocate it on
        # the target device and wrap it, so it is a real Parameter. The random
        # draw is unchanged (same shape, same device generator), so seeds give
        # the same initial head as before.
        # args.legacy_unregistered_head=True reproduces the old behaviour, only
        # for controlled comparison with pre-fix results.
        w = torch.empty(self.args.layers[-1] * 2, self.num_labels, device=loss_device)
        init.xavier_normal_(w)
        if getattr(self.args, "legacy_unregistered_head", False):
            self.regression_weights = w
        else:
            self.regression_weights = Parameter(w)
        if self.class_weight is not None:
            self.criterion = torch.nn.CrossEntropyLoss(weight=self.class_weight).to(loss_device)
        else:
            self.criterion = torch.nn.CrossEntropyLoss().to(loss_device)
        del self.class_weight
        
    def _compute_structural_trust(self, mu, edge_index, edge_labels):
        """Compute structural trust using JS divergence (same as robust_aggr)"""
        if not hasattr(self, 'use_dual_graph') or not self.use_dual_graph:
            return None
            
        _js_ra = importlib.import_module("mycode.js_ra")
        prepare_probabilities = _js_ra.prepare_probabilities
        ro_coefficient_dense_js = _js_ra.ro_coefficient_dense_js
        
        row, col = edge_index
        feature_prob = prepare_probabilities(mu)
        structural_trust = ro_coefficient_dense_js(
            feature_prob,
            0.4,
            row,
            col,
        ).to(self.device)
        return structural_trust
        
    def _process_structural_layer(self, train_edges, train_labels, force_attack: bool = False) -> torch.Tensor:
        structural_out = []
        index0 = 0
        eng={}
        edges_dev = train_edges.to(self.gpu_structural) if self.pipeline_mode else train_edges
        labels_dev = train_labels.to(self.gpu_structural) if self.pipeline_mode else train_labels
        for i in range(self.args.train_time_slots):
            end = self.index_list[i] + 1
            lim = slice(index0, end)
            edges = edges_dev[:, lim]
            labels = labels_dev[lim, :]

            if self.attack and (self.training or force_attack):
                edges, labels = self._apply_attack(edges, labels, i)
                current = torch.utils.checkpoint.checkpoint(
                    self.structural_layer, edges, labels, use_reentrant=False
                )
            else:
                current = torch.utils.checkpoint.checkpoint(
                    self.structural_layer, edges, labels, use_reentrant=False
                )

            structural_out.append(current)
            index0 = end

        x_sequence = torch.stack(structural_out, dim=1)
        if self.pipeline_mode:
            x_sequence = x_sequence.to(self.gpu_s3r)
        return x_sequence
    

    def _apply_attack(self, edges, labels, time_slot: int) -> Tuple[torch.Tensor, torch.Tensor]:
        attack_type_to_use = None
        fixed_attack_intensity = 1.0 

        max_victim_pct = getattr(self.args, 'victim_percentage', 0.10) 
        total_snapshots = getattr(self.args, 'time_slots', 10)
        current_victim_pct = max_victim_pct 

        if self.args.attack_type == 'bad_mouthing': attack_type_to_use = 'bad_mouthing'
        elif self.args.attack_type == 'good_mouthing':attack_type_to_use = 'good_mouthing'
        elif self.args.attack_type == 'on-off':
            if time_slot % 2 == 0:
                attack_type_to_use = 'bad_mouthing'
        elif self.args.attack_type == 'slow_poisoning':
            attack_type_to_use = 'good_mouthing'
            current_victim_pct = calculate_victim_drift(
                current_time=time_slot,
                total_time_steps=total_snapshots,
                max_victim_percentage=0.15,
                mode='linear')

        if attack_type_to_use:
            poisoned_edges, poisoned_labels = initiate_attack(
                original_edges=edges,
                original_labels=labels,
                attack_type=attack_type_to_use,
                attack_percentage=fixed_attack_intensity, # Always 1.0
                victim_percentage=current_victim_pct,     # Grows over time
                verbose=getattr(self.args, 'verbose_attack', False),
                seed=self.args.seed)
        else:
            poisoned_edges, poisoned_labels = edges, labels
              
        poisoned_edges = poisoned_edges.to(edges.device)
        poisoned_labels = poisoned_labels.to(labels.device)
        return poisoned_edges, poisoned_labels

    
    def _calculate_prediction_loss(self, node_embeddings, train_edges, target) -> torch.Tensor:
        if self.pipeline_mode:
            train_edges = train_edges.to(self.gpu_s3r)
            target = target.to(self.gpu_s3r)
        start_node, end_node = node_embeddings[train_edges[0], :], node_embeddings[train_edges[1], :]
        features = torch.cat((start_node, end_node), 1)
        predictions = torch.mm(features, self.regression_weights)
        return self.criterion(predictions, target)
    
    def forward(self, train_edges, y, train_labels, force_attack: bool = False, return_loss: bool = True) -> Tuple:
        mu = self._process_structural_layer(train_edges, train_labels, force_attack)
        temporal_all = self.s3r(mu)
        temporal_out = temporal_all[:, self.args.train_time_slots-1, :].squeeze()

        if return_loss:
            prediction_loss = self._calculate_prediction_loss(temporal_out, train_edges, y)
            return prediction_loss, temporal_out
        return None, temporal_out
        


