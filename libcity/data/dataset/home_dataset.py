import numpy as np
from libcity.data.dataset import AbstractDataset
from libcity.data.dataset import HeteroRoadFeatureDataset, HeteroRegionFeatureDataset
import torch
from torch_geometric.data import HeteroData
import json


class HOMEDataset(AbstractDataset):
    def __init__(self, config):
        self.config = config
        self.dataset = self.config.get('dataset', '')  # cd
        self.device = self.config.get('device')
        self.edge_types = self.config['edge_types']
        self.road_dataset = HeteroRoadFeatureDataset(config)
        self.region_dataset = HeteroRegionFeatureDataset(config)

        road2region = json.load(open('./raw_data/{}/road2region_{}.json'.format(self.dataset, self.dataset), 'r'))
        region2road = json.load(open('./raw_data/{}/region2road_{}.json'.format(self.dataset, self.dataset), 'r'))

        self.road2region_edge_index = []
        for k, v in road2region.items():
            if isinstance(v, list):
                for vi in v:
                    self.road2region_edge_index.append([int(k), vi])
            else:
                self.road2region_edge_index.append([int(k), v])
        self.road2region_edge_index = np.array(self.road2region_edge_index).T
        self.region2road_edge_index = []
        for k, v in region2road.items():
            if isinstance(v, list):
                for vi in v:
                    self.region2road_edge_index.append([int(k), vi])
            else:
                self.region2road_edge_index.append([int(k), v])
        self.region2road_edge_index = np.array(self.region2road_edge_index).T

    def get_data(self):
        road_x = torch.from_numpy(self.road_dataset.road_features).long()
        road_edge_index_rel = torch.tensor(self.road_dataset.edge_index_rel, dtype=torch.long)
        road_edge_index_sem = torch.tensor(self.road_dataset.edge_index_sem, dtype=torch.long)
        road_edge_index_mob = torch.tensor(self.road_dataset.edge_index_mob, dtype=torch.long)
        road_egde_weight_rel = torch.from_numpy(self.road_dataset.edge_weight_rel).float()
        road_egde_weight_sem = torch.from_numpy(self.road_dataset.edge_weight_sem).float()
        road_egde_weight_mob = torch.from_numpy(self.road_dataset.edge_weight_mob).float()

        region_x = torch.from_numpy(self.region_dataset.region_features).long()
        region_edge_index_rel = torch.tensor(self.region_dataset.edge_index_rel, dtype=torch.long)
        region_edge_index_sem = torch.tensor(self.region_dataset.edge_index_sem, dtype=torch.long)
        region_edge_index_mob = torch.tensor(self.region_dataset.edge_index_mob, dtype=torch.long)
        region_egde_weight_rel = torch.from_numpy(self.region_dataset.edge_weight_rel).float()
        region_egde_weight_sem = torch.from_numpy(self.region_dataset.edge_weight_sem).float()
        region_egde_weight_mob = torch.from_numpy(self.region_dataset.edge_weight_mob).float()

        road2region_edge_index = torch.tensor(self.road2region_edge_index, dtype=torch.long)
        region2road_edge_index = torch.tensor(self.region2road_edge_index, dtype=torch.long)

        data = HeteroData()
        data['road_node'].x = road_x
        data['region_node'].x = region_x

        if 'geo' in self.edge_types:
            data['road_node', 'geo', 'road_node'].edge_index = road_edge_index_rel
            data['road_node', 'geo', 'road_node'].edge_weight = road_egde_weight_rel
            data['region_node', 'geo', 'region_node'].edge_index = region_edge_index_rel
            data['region_node', 'geo', 'region_node'].edge_weight = region_egde_weight_rel
        if 'sem' in self.edge_types:
            data['road_node', 'sem', 'road_node'].edge_index = road_edge_index_sem
            data['road_node', 'sem', 'road_node'].edge_weight = road_egde_weight_sem
            data['region_node', 'sem', 'region_node'].edge_index = region_edge_index_sem
            data['region_node', 'sem', 'region_node'].edge_weight = region_egde_weight_sem
        if 'mob' in self.edge_types:
            data['road_node', 'mob', 'road_node'].edge_index = road_edge_index_mob
            data['road_node', 'mob', 'road_node'].edge_weight = road_egde_weight_mob
            data['region_node', 'mob', 'region_node'].edge_index = region_edge_index_mob
            data['region_node', 'mob', 'region_node'].edge_weight = region_egde_weight_mob
        data['road_node', 'belong', 'region_node'].edge_index = road2region_edge_index
        data['region_node', 'include', 'road_node'].edge_index = region2road_edge_index
        data = data.to(self.device)

        self.meta_data = data.metadata()
        return data, None, None

    def get_data_feature(self):
        res = {}
        res['meta_data'] = self.meta_data
        res.update(self.road_dataset.get_data_feature())
        res.update(self.region_dataset.get_data_feature())
        return res
