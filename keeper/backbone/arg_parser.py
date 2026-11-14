from easydict import EasyDict as edict
import os

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def parameter_parser(dataset=None, **overrides):
    """
    Get default args with dynamic paths.
    
    Args:
        dataset: One of 'bitcoinotc', 'bitcoinalpha', 'epinions'. If None, uses epinions defaults.
        **overrides: Additional kwargs to override specific args.
    """
    # Dataset-specific defaults
    dataset_configs = {
        "bitcoinotc": {
            "edge_path": os.path.join(PROJECT_ROOT, "data/cyberdata/bitcoinotc-rating.txt"),
            "data_path": os.path.join(PROJECT_ROOT, "data/cyberdata/bitcoinotc.csv"),
            "num_nodes": 5881,
            "attention_head": 16,
            "layers": [32, 64, 32],
            "total_client": 4,
            "time_slots": 22,
            "epochs": 50,
            "com_round": 50,
        },
        "bitcoinalpha": {
            "edge_path": os.path.join(PROJECT_ROOT, "data/cyberdata/bitcoinalpha-rating.txt"),
            "data_path": os.path.join(PROJECT_ROOT, "data/cyberdata/bitcoinalpha.csv"),
            "num_nodes": 3783,
            "attention_head": 16,
            "layers": [32, 64, 32],
            "total_client": 4,
            "time_slots": 22,
            "epochs": 50,
            "com_round": 50,
        },
        "epinions": {
            "edge_path": os.path.join(PROJECT_ROOT, "data/cyberdata/epinions-rating.txt"),
            "data_path": os.path.join(PROJECT_ROOT, "data/cyberdata/epinions.csv"),
            "num_nodes": 131828,
            "attention_head": 8,
            "layers": [32, 32, 32],
            "total_client": 10,
            "time_slots": 100,
            "epochs": 10,
            "com_round": 10,
        },
    }
    
    # Base defaults (epinions as fallback)
    base = {
        "dataset": "epinions",
        "single_prediction": True,
        "train_time_slots": 2,
        "seed": 42,
        "lr": 0.005,
        "weight_decay": 1e-5,
        "dropout": 0.0,
        "sample_ratio": 1,
        "max_snapshots_inaClient": 0,
        "attack": False,
        "defense": False,
        "attacked_clients": [0],
        "victim_percentage": 0.15,
        "attack_type": "slow_poisoning",
        "attack_percentage": 1.0,
        "task": "Task1",
        "secure_compress": False,
        "shared_secret": 98765,
        "target_sparsity": 0.9,
        "num_bits": 8,
        "dp_epsilon": 8.0,
        "tap_noise": False,
        "ablation": False,
        "adaef": False,
        "adaef_K": 5,
        "fedprox_mu": 0.0,
        "fed_algo": "fedavg",
        "fedskip_interval": 0,
        "fednova_p": 1.0,
        "feddyn_alpha": 0.0,
        "moon_mu": 0.0,
        "moon_tau": 0.5,
    }
    
    # Merge dataset-specific config
    if dataset and dataset in dataset_configs:
        base.update(dataset_configs[dataset])
        base["dataset"] = dataset
    elif not dataset:
        # Default to epinions config
        base.update(dataset_configs["epinions"])
    
    # Apply any additional overrides
    base.update(overrides)
    
    return edict(base)
