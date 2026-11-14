# Copyright 2021 Peng Cheng Laboratory (http://www.szpclab.com/) and FedLab Authors (smilelab.group)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import numpy as np


_SELECTED_DEVICE = None


def _normalize_device(device):
    if device is None:
        return None
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def get_best_gpu_index():
    """Return the visible CUDA index with the most free memory."""
    if not torch.cuda.is_available():
        return None

    try:
        import pynvml

        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"]:
            visible_devices = [int(d.strip()) for d in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if d.strip()]
        else:
            visible_devices = list(range(device_count))

        gpu_scores = []
        for gpu_index in visible_devices:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_scores.append((gpu_index, int(mem_info.free), int(util_info.gpu)))

        idle_gpus = [gpu for gpu in gpu_scores if gpu[2] < 10]
        candidates = idle_gpus if idle_gpus else gpu_scores
        best_gpu_index = max(candidates, key=lambda item: item[1])[0]
        return best_gpu_index
    except Exception:
        return 0
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def resolve_device(device=None, prefer_cuda=True, cache=True, set_current=True):
    """Resolve and optionally cache the project-wide execution device."""
    global _SELECTED_DEVICE

    normalized_device = _normalize_device(device)
    if normalized_device is not None:
        selected_device = normalized_device
    elif cache and _SELECTED_DEVICE is not None:
        selected_device = _SELECTED_DEVICE
    elif prefer_cuda and torch.cuda.is_available():
        best_gpu_index = get_best_gpu_index()
        selected_device = torch.device(f"cuda:{best_gpu_index}") if best_gpu_index is not None else torch.device("cuda:0")
    else:
        selected_device = torch.device("cpu")

    if set_current and selected_device.type == "cuda":
        torch.cuda.set_device(selected_device)

    if cache:
        _SELECTED_DEVICE = selected_device

    return selected_device


def get_best_gpu():
    """Return gpu (:class:`torch.device`) with largest free memory."""
    return resolve_device(prefer_cuda=True)
