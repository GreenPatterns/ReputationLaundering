
import torch
import datetime
import numpy as np
import pandas as pd
from .basic_dataset import FedDataset
from keeper.backbone.utils import *
from itertools import accumulate

# single_prediction: {1-2}-->3, {1-3}-->4, {1-4}-->5, {1-5}-->6, {1-6}-->7, {1-7}-->8, {1-8}-->9, {1-9}-->10
# multi_prediction: {1-2}-->{3-5}, {1-3}-->{4-6}, {1-4}-->{5-7}, {1-5}-->{6-8}, {1-6}-->{7-9}, {1-7}-->{8-10}
# snapshot index [2384, 7411, 10542, 16618, 23885, 30172, 32990, 34569, 35265, 35591] 

class PartitioningCyberData(FedDataset):
    def __init__(self, args,  wrkr = 5 ) -> None:
        self.args = args
        self.train_time_slots = self.args.train_time_slots
        self.main_dictionary = {}
        self.maax = 0
        self.wrkr = wrkr # type: ignore
        self._perform_partition()  

    def get_dataloader(self, id):
        return self.main_dictionary.get(id)

    def _perform_partition(self):
        self.graph = self._read_graph()
        self.index_list = self._get_snapshot_index(self.args.time_slots, data_path=self.args.data_path)
        # print("total snapshots:", len(self.index_list))
        # print("total edges:", self.graph['ecount'])

        counts, starts, slices = self.build_slices(self.index_list, workers=self.wrkr)
        per_slice_snapshot_edges = self.per_slice_snapshot_edges_np(self.graph['edges'], self.index_list, slices)
        per_slice_snapshot_labels = self.per_slice_snapshot_edges_np(self.graph['labels'], self.index_list, slices)

        # print("\n\n\nlen(nested_snapshot_edges[0][0])", len(per_slice_snapshot_edges[0][0]))
        grand_total_edges_all_clients = 0

        for i in range(self.wrkr):
            siz = len(per_slice_snapshot_edges[i])
            self.maax = siz if siz > self.maax else self.maax
            # print(f"client {i} is having {siz} snapshots")

            client_snapshots_edges = per_slice_snapshot_edges[i]
            client_snapshots_labels = per_slice_snapshot_labels[i]

            # Snapshot count (your existing log)
            num_snapshots = len(client_snapshots_edges)
            self.maax = num_snapshots if num_snapshots > self.maax else self.maax
            # print(f"client {i} is having {num_snapshots} snapshots")

            # Per-snapshot edge counts
            snapshot_edge_counts = [snap.shape[0] for snap in client_snapshots_edges]

            # print(f"Client {i} per-snapshot edge counts (global_snapshot_id -> edges):")
            for local_j, c in enumerate(snapshot_edge_counts):
                global_snapshot_id = starts[i] + local_j
                # print(f"  {global_snapshot_id} -> {c}")

            # Sum of the per-snapshot figures for this client
            client_total = int(sum(snapshot_edge_counts))
            # print(f"Client {i} SUM (edges over its snapshots) = {client_total}")
            # Accumulate grand total
            grand_total_edges_all_clients += client_total

            client_data = {
                "client_edges": client_snapshots_edges, 
                "client_labels": client_snapshots_labels
            }
            self.main_dictionary[i] = client_data

    def _read_graph(self):
        edges = {}
        ecount = 0
        ncount = []
        edg = []
        lab = []
        with open(self.args.edge_path) as dataset:
            for edge in dataset:
                ecount += 1
                ncount.append(edge.split()[0])
                ncount.append(edge.split()[1])
                edg.append(list(map(float, edge.split()[0:2])))
                lab.append(list(map(float, edge.split()[2:])))

        edges["labels"] = np.array(lab)
        edges["edges"] = np.array(edg) 
        edges["ecount"] = ecount
        edges["ncount"] = len(set(ncount)) # Node Count
        return edges


    
    def _get_snapshot_index(self, time_slots, data_path):
        df = pd.read_csv(data_path)
        df.columns = ['source','target','rating','time']
        time_array = np.sort(df['time'].to_numpy())
        ts_begin = int(time_array[0])
        ts_finish = int(time_array[-1])
        if time_slots <= 1 or len(time_array) == 1 or ts_finish == ts_begin:
            return [len(time_array) - 1]
        split_points = np.linspace(ts_begin, ts_finish, num=time_slots + 1, endpoint=True)[1:]
        index_list = []
        for split_point in split_points[:-1]:
            idx = int(np.searchsorted(time_array, split_point, side='right') - 1)
            idx = max(0, min(idx, len(time_array) - 2))
            if not index_list or idx > index_list[-1]:
                index_list.append(idx)
        index_list.append(len(time_array) - 1)
        return index_list

    def partition_counts_front_loaded(self, n, workers=5):
        if n < workers:
            raise ValueError(f"len(indexes)={n} must be >= number of partitions k={workers}")
        base, rem = divmod(n, workers)
        return [base + 1] * rem + [base] * (workers - rem)

    def partition_starts_from_counts(self, counts):
        return [0] + list(accumulate(counts))[:-1]

    def build_slices(self, index_list, workers=5):
        # index_list contains inclusive ends for each snapshot
        n = len(index_list)  # number of snapshots
        counts = self.partition_counts_front_loaded(n, workers=workers)
        starts = self.partition_starts_from_counts(counts)
        slices = [index_list[s:s + c] for s, c in zip(starts, counts)]
        return counts, starts, slices

    def snapshot_ranges(self, index_list):
        # Produce (start_inclusive, end_inclusive) per snapshot given inclusive ends
        ends = np.asarray(index_list, dtype=np.int64)
        starts = np.concatenate(([0], ends[:-1] + 1))  # FIXED: correct syntax with leading [0]
        return list(zip(starts.tolist(), ends.tolist()))

    def group_ranges_by_slices(self, ranges, counts):
        out, pos = [], 0
        for c in counts:
            out.append(ranges[pos:pos + c])
            pos += c
        return out

    def per_slice_snapshot_edges_np(self, T_Edges, index_list, slices):
        T = np.asarray(T_Edges)
        # index_list uses inclusive ends; last should be len(T)-1 when edges are 0..len(T)-1
        assert index_list[-1] == len(T) - 1, (
            f"last inclusive index ({index_list[-1]}) must equal len(T_Edges)-1 ({len(T)-1})"
        )
        # Convert inclusive ends to np.split boundaries (positions BEFORE which to split)
        split_points = (np.asarray(index_list[:-1], dtype=np.int64) + 1).tolist()
        snapshots = np.split(T, split_points, axis=0)  # list of snapshot arrays
        out, pos = [], 0
        for ends in slices:
            L = len(ends)  # number of snapshots in this slice
            out.append(snapshots[pos:pos + L])
            pos += L
        return out
            