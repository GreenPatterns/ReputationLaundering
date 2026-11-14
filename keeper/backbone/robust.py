import torch
import numpy as np
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
from scipy.sparse import lil_matrix
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn.functional import normalize as torch_normalize 
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ CPU original +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def ro_coefficient(node_embedding, prune, row, col):
    """
    Similar with GNNGuard: https://arxiv.org/abs/2006.08149
    """
    node_embedding_copy = node_embedding.cpu().data.numpy()
    sim_matrix = cosine_similarity(node_embedding_copy)
    # print('sim_matrix',sim_matrix.max(),sim_matrix.min())  # Sim(X, Y) = <X, Y> / (||X||*||Y||)
    sim_matrix = 1 + sim_matrix  # range: [0,2]！！！
    sim = sim_matrix[row, col]
    sim[sim < prune] = 0
    # print('testing',sim.max(),sim.min())
    loop_list = [row[i] for i in range(len(row)) if row[i]==col[i]]

    att_dense = lil_matrix((node_embedding_copy.shape[0], node_embedding_copy.shape[0]), dtype=np.float32)
    att_dense[row, col] = sim
    att_dense = att_dense - sp.diags(att_dense.diagonal(), offsets=0, format='lil')
    att_dense_norm = normalize(att_dense, axis=1, norm='l1')

    degree = (att_dense_norm != 0).sum(1)
    att_dense_tran = lil_matrix(att_dense_norm.A * degree.A)
    ones = np.zeros((len(degree),))
    for i in range(len(loop_list)):
        ones[i] = 1
    self_loop = sp.diags(ones, offsets=0, format='lil')
    att_final = att_dense_tran + self_loop
    att_final_norm = normalize(att_final, axis=1, norm='l1')

    att_real = att_final_norm[row, col]
    att_real_final = np.reshape(att_real.A, (len(row), 1))
    att_real_final = torch.from_numpy(att_real_final).type(torch.float)

    return att_real_final

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ CPU fixed +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def pearson_correlation(matrix, epsilon=1e-8):
    mean_matrix = matrix - matrix.mean(dim=1, keepdim=True)
    std_matrix = matrix.std(dim=1, keepdim=True) + epsilon
    norm_matrix = mean_matrix / std_matrix
    correlation_matrix = torch.mm(norm_matrix, norm_matrix.t()) / (matrix.size(1) - 1)
    return correlation_matrix

def cosine_similarity_gpu(embeddings):
    norms = embeddings.norm(dim=1, keepdim=True)
    norms = torch.clamp(norms, min=1e-8)
    normalized_embeddings = embeddings / norms
    return torch.mm(normalized_embeddings, normalized_embeddings.t())

def ro_coefficient(node_embedding, prune, row, col):
    device = node_embedding.device
    num_nodes = node_embedding.shape[0]
    sim_matrix = cosine_similarity_gpu(node_embedding)
    sim_matrix = 1 + sim_matrix
    sim = sim_matrix[row, col]
    sim[sim < prune] = 0
    att_dense = torch.zeros((num_nodes, num_nodes), 
                            device=device, 
                            dtype=torch.float32)
    att_dense[row, col] = sim
    att_dense.fill_diagonal_(0)
    att_dense_norm = torch_normalize(att_dense, p=1, dim=1)
    degree = (att_dense_norm != 0).sum(dim=1, keepdim=True)
    att_dense_tran = att_dense_norm * degree
    loop_list = [row[i] for i in range(len(row)) if row[i] == col[i]]
    ones = torch.zeros((len(degree),), device=node_embedding.device)
    for i in range(len(loop_list)):
        ones[i] = 1
    self_loop = torch.diag(ones)
    att_final = att_dense_tran + self_loop
    att_final_norm = torch_normalize(att_final, p=1, dim=1)
    att_real = att_final_norm[row, col]
    return att_real.view(len(row), 1)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ Ensemble fixed +++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def generalized_jaccard_optimized(x, y, eps=1e-8):
    min_xy = torch.min(x, y)
    max_xy = torch.max(x, y)
    score = torch.sum(min_xy, dim=1) / (torch.sum(max_xy, dim=1) + eps)
    return score

def ro_coefficient_dense_jaccard(feature_mu, row, col, prune=0.2):
    device = feature_mu.device
    num_nodes = feature_mu.shape[0]
    sim = generalized_jaccard_optimized(feature_mu[row], feature_mu[col])
    prune_jaccard = prune 
    sim[sim < prune_jaccard] = 0
    adj_dense = torch.zeros(num_nodes, num_nodes, device=device, dtype=sim.dtype)
    mask = sim > 0
    if mask.sum() > 0:
        adj_dense[row[mask], col[mask]] = sim[mask]

    adj_dense.fill_diagonal_(0)
    row_sums = adj_dense.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    adj_normalized = adj_dense / row_sums
    degree = (adj_normalized > 0).sum(dim=1, keepdim=True).float()
    adj_scaled = adj_normalized * degree
    row_cpu = row.cpu().numpy()
    col_cpu = col.cpu().numpy()
    loop_nodes = list(set([row_cpu[i] for i in range(len(row_cpu)) if row_cpu[i] == col_cpu[i]]))
    
    if loop_nodes:
        adj_scaled.fill_diagonal_(0)  # Clear first
        for node in loop_nodes[:num_nodes]:  # Safety check
            adj_scaled[node, node] = 1.0
    final_row_sums = adj_scaled.sum(dim=1, keepdim=True)
    final_row_sums[final_row_sums == 0] = 1
    adj_final = adj_scaled / final_row_sums
    return adj_final[row, col]

def ro_coefficient_dense_cosine(feature_mu, prune, row, col):
    device = feature_mu.device
    num_nodes = feature_mu.shape[0]
    node_embedding_norm = F.normalize(feature_mu, p=2, dim=1)
    sim = torch.sum(node_embedding_norm[row] * node_embedding_norm[col], dim=1)
    sim = 1 + sim  # Shift to [0, 2] range
    sim[sim < prune] = 0
    adj_dense = torch.zeros(num_nodes, num_nodes, device=device, dtype=sim.dtype)
    mask = sim > 0
    if mask.sum() > 0:
        adj_dense[row[mask], col[mask]] = sim[mask]
    adj_dense.fill_diagonal_(0)
    row_sums = adj_dense.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1
    adj_normalized = adj_dense / row_sums
    degree = (adj_normalized > 0).sum(dim=1, keepdim=True).float()
    adj_scaled = adj_normalized * degree
    row_cpu = row.cpu().numpy()
    col_cpu = col.cpu().numpy()
    loop_nodes = list(set([row_cpu[i] for i in range(len(row_cpu)) if row_cpu[i] == col_cpu[i]]))
    if loop_nodes:
        adj_scaled.fill_diagonal_(0)
        for node in loop_nodes[:num_nodes]:
            adj_scaled[node, node] = 1.0
    final_row_sums = adj_scaled.sum(dim=1, keepdim=True)
    final_row_sums[final_row_sums == 0] = 1
    adj_final = adj_scaled / final_row_sums
    return adj_final[row, col]


def ro_coefficient_dense_ensemble_complete(feature_mu, prune, row, col, ensemble_method='CWAE'):
    device = feature_mu.device    
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    coeff_cosine  = ro_coefficient_dense_cosine(feature_mu, prune, row, col)
    coeff_jaccard = ro_coefficient_dense_jaccard(feature_mu, row, col)

    if ensemble_method == 'adaptive':
        total_scores = coeff_cosine + coeff_jaccard
        mask = total_scores > 0
        coeff_ensemble = torch.zeros_like(coeff_cosine)
        if mask.sum() > 0:
            weight_cosine = coeff_cosine[mask] / total_scores[mask]
            weight_jaccard = coeff_jaccard[mask] / total_scores[mask]
            coeff_ensemble[mask] = weight_cosine * coeff_cosine[mask] + weight_jaccard * coeff_jaccard[mask]

    elif ensemble_method == 'CWAE':
        T = coeff_cosine + coeff_jaccard + 1e-8
        lambda_ij = coeff_cosine / T
        S_fused = lambda_ij * coeff_cosine + (1.0 - lambda_ij) * coeff_jaccard
        return S_fused

    elif ensemble_method == 'SBEN':
        stacked = torch.stack([coeff_cosine , coeff_jaccard], dim=0)
        weights = F.softmax(stacked, dim=0)
        S_fused = weights[0] * coeff_cosine + weights[1] * coeff_jaccard
        return S_fused
    else:
        raise ValueError(f"Unknown ensemble method: {ensemble_method}")
    
    del coeff_cosine, coeff_jaccard
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return coeff_ensemble
