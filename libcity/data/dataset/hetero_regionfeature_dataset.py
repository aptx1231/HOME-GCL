import os
import pandas as pd
import numpy as np
from logging import getLogger
from libcity.data.dataset import NetWorkDataset
from tqdm import tqdm
from sklearn.preprocessing import KBinsDiscretizer
import warnings
warnings.filterwarnings('ignore')


class HeteroRegionFeatureDataset(NetWorkDataset):
    def __init__(self, config):
        self.config = config
        self.dataset = self.config.get('dataset', '')  # cd
        self.data_path = './raw_data/{}/regionmap_{}/'.format(self.dataset, self.dataset)
        self.model_name = self.config.get('model',)
        self.geo_file = 'regionmap_{}'.format(self.dataset)
        self.rel_file = 'regionmap_{}'.format(self.dataset)
        self.sem_file = 'regionmap_{}'.format(self.dataset)
        self.mob_file = 'regionmap_{}'.format(self.dataset)
        self.len_lon_encode = self.config.get('len_lon_encode', 15)
        self.len_lat_encode = self.config.get('len_lat_encode', 15)
        self.len_BLD_Count_encode = self.config.get('len_BLD_Count_encode', 30)
        self.len_BLD_FLOORN_encode = self.config.get('len_BLD_FLOORN_encode', 10)
        self.len_Shape_Area_encode = self.config.get('len_Shape_Area_encode', 1000)
        self.seed = self.config.get('seed', 0)
        self._logger = getLogger()
        assert os.path.exists(self.data_path + self.geo_file + '.geo')
        assert os.path.exists(self.data_path + self.rel_file + '.rel')
        assert os.path.exists(self.data_path + self.sem_file + '.sem')
        assert os.path.exists(self.data_path + self.sem_file + '.mob')
        self.geo_file = self._load_geo()
        self.rel_file = self._load_rel()
        self.sem_file = self._load_sem()
        self.mob_file = self._load_mob()
        self._cal_degree(self.geo_file, self.adj_mx_rel)
        self.region_features, self.region_fea_dim = self._load_geo_feature(self.geo_file)

    def _load_geo(self):
        geofile = pd.read_csv(self.data_path + self.geo_file + '.geo')
        self.geo_ids = list(geofile['geo_id'])
        self.num_regions = len(self.geo_ids)
        self.geo_to_ind = {}
        self.ind_to_geo = {}
        for index, geo_id in enumerate(self.geo_ids):
            self.geo_to_ind[geo_id] = index
            self.ind_to_geo[index] = geo_id
        self._logger.info("Loaded file " + self.geo_file + '.geo' + ', num_regions=' + str(len(self.geo_ids)))
        return geofile

    def _load_rel(self):
        relfile = pd.read_csv(self.data_path + self.rel_file + '.rel')
        relfile = relfile[['origin_id', 'destination_id', 'geographical_weight']]
        self.adj_mx_rel = np.zeros((len(self.geo_ids), len(self.geo_ids)), dtype=np.int32)
        self.edge_index_rel = []
        self.edge_weight_rel = []
        for row in relfile.values:
            if row[0] not in self.geo_to_ind or row[1] not in self.geo_to_ind:
                continue
            self.adj_mx_rel[self.geo_to_ind[row[0]], self.geo_to_ind[row[1]]] = 1
            self.edge_index_rel.append([self.geo_to_ind[row[0]], self.geo_to_ind[row[1]]])
            self.edge_weight_rel.append(row[-1])
        self.edge_index_rel = np.array(self.edge_index_rel).T
        self.num_edges_rel = self.edge_index_rel.shape[1]
        self.edge_weight_rel = np.array(self.edge_weight_rel, dtype='float32')
        self._logger.info("Loaded file " + self.rel_file + '.rel, shape=' + str(self.adj_mx_rel.shape) + ', edges=' + str(self.adj_mx_rel.sum()))
        self._logger.info("edge_index shape= " + str(self.edge_index_rel.shape) + ", edge_weight shape= "
                          + str(self.edge_weight_rel.shape) + ', edges=' + str(self.edge_index_rel.shape[1]))
        return relfile

    def _load_sem(self):
        semfile = pd.read_csv(self.data_path + self.sem_file + '.sem')
        semfile = semfile[['origin_id', 'destination_id', 'semantic_weight']]
        self.adj_mx_sem = np.zeros((len(self.geo_ids), len(self.geo_ids)), dtype=np.int32)
        self.edge_index_sem = []
        self.edge_weight_sem = []
        for row in semfile.values:
            if row[0] not in self.geo_to_ind or row[1] not in self.geo_to_ind:
                continue
            self.adj_mx_sem[self.geo_to_ind[row[0]], self.geo_to_ind[row[1]]] = 1
            self.edge_index_sem.append([self.geo_to_ind[row[0]], self.geo_to_ind[row[1]]])
            self.edge_weight_sem.append(row[-1])
        self.edge_index_sem = np.array(self.edge_index_sem).T
        self.num_edges_sem = self.edge_index_sem.shape[1]
        self.edge_weight_sem = np.array(self.edge_weight_sem, dtype='float32')
        self._logger.info("Loaded file " + self.sem_file + '.sem, shape=' + str(self.adj_mx_sem.shape) + ', edges=' + str(self.adj_mx_sem.sum()))
        self._logger.info("edge_index shape= " + str(self.edge_index_sem.shape) + ", edge_weight shape= "
                          + str(self.edge_weight_sem.shape) + ', edges=' + str(self.edge_index_sem.shape[1]))
        return semfile

    def _load_mob(self):
        mobfile = pd.read_csv(self.data_path + self.mob_file + '.mob')
        mobfile = mobfile[['origin_id', 'destination_id', 'mobility_weight']]
        self.adj_mx_mob = np.zeros((len(self.geo_ids), len(self.geo_ids)), dtype=np.int32)
        self.edge_index_mob = []
        self.edge_weight_mob = []
        for row in mobfile.values:
            if row[0] not in self.geo_to_ind or row[1] not in self.geo_to_ind:
                continue
            self.adj_mx_mob[self.geo_to_ind[row[0]], self.geo_to_ind[row[1]]] = 1
            self.edge_index_mob.append([self.geo_to_ind[row[0]], self.geo_to_ind[row[1]]])
            self.edge_weight_mob.append(row[-1])
        self.edge_index_mob = np.array(self.edge_index_mob).T
        self.num_edges_mob = self.edge_index_mob.shape[1]
        self.edge_weight_mob = np.array(self.edge_weight_mob, dtype='float32')
        self._logger.info("Loaded file " + self.mob_file + '.mob, shape=' + str(self.adj_mx_mob.shape) + ', edges=' + str(self.adj_mx_mob.sum()))
        self._logger.info("edge_index shape= " + str(self.edge_index_mob.shape) + ", edge_weight shape= "
                          + str(self.edge_weight_mob.shape) + ', edges=' + str(self.edge_index_mob.shape[1]))
        return mobfile

    def _cal_degree(self, road_info, adj_mx):
        outdegree = np.sum(adj_mx, axis=1)  # (N, )
        indegree = np.sum(adj_mx.T, axis=1)  # (N, )
        outdegree_list = []
        indegree_list = []

        for i, row in tqdm(road_info.iterrows(), total=road_info.shape[0]):
            outdegree_list.append(int(outdegree[self.geo_to_ind[row.geo_id]]))
            indegree_list.append(int(indegree[self.geo_to_ind[row.geo_id]]))

        road_info.insert(loc=road_info.shape[1], column='indegree', value=indegree_list)
        road_info.insert(loc=road_info.shape[1], column='outdegree', value=outdegree_list)

    def _load_geo_feature(self, region_info):
        node_features = region_info[['geo_id', 'FUNCTION', 'InCBD', 'FORM_TYPE', 'indegree', 'outdegree',
                                   'BLD_Count', 'BLD_FLOORN',
                                   'Shape_Area', 'm_lon', 'm_lat']]
        cat_features = ['FUNCTION', 'InCBD']
        num_features = ['BLD_Count', 'BLD_FLOORN', 'Shape_Area', 'm_lon', 'm_lat']
        for fe in cat_features:
            node_features.loc[:, fe] = node_features.loc[:, fe].astype('int32')
        for fe in num_features:
            node_features.loc[:, fe] = node_features.loc[:, fe].astype('float32')
        discretizer = KBinsDiscretizer(n_bins=self.len_lon_encode, encode='ordinal', strategy='uniform')
        node_features.loc[:, 'm_lon_encode'] = discretizer.fit_transform(
                np.array(node_features['m_lon'].tolist()).reshape(-1, 1))
        discretizer = KBinsDiscretizer(n_bins=self.len_lat_encode, encode='ordinal', strategy='uniform')
        node_features.loc[:, 'm_lat_encode'] = discretizer.fit_transform(
                np.array(node_features['m_lat'].tolist()).reshape(-1, 1))
        discretizer = KBinsDiscretizer(n_bins=self.len_BLD_Count_encode, encode='ordinal', strategy='uniform')
        node_features.loc[:, 'BLD_Count_encode'] = discretizer.fit_transform(
                np.array(node_features['BLD_Count'].tolist()).reshape(-1, 1))
        discretizer = KBinsDiscretizer(n_bins=self.len_BLD_FLOORN_encode, encode='ordinal', strategy='uniform')
        node_features.loc[:, 'BLD_FLOORN_encode'] = discretizer.fit_transform(
                np.array(node_features['BLD_FLOORN'].tolist()).reshape(-1, 1))
        discretizer = KBinsDiscretizer(n_bins=self.len_Shape_Area_encode, encode='ordinal', strategy='uniform')
        node_features.loc[:, 'Shape_Area_encode'] = discretizer.fit_transform(
                np.array(node_features['Shape_Area'].tolist()).reshape(-1, 1))
        node_features = node_features[['FUNCTION', 'InCBD',
                                       'BLD_Count_encode', 'BLD_FLOORN_encode', 'Shape_Area_encode',
                                       'm_lon_encode', 'm_lat_encode']]
        self.len_FUNCTION = node_features['FUNCTION'].max() + 1
        self.len_InCBD = node_features['InCBD'].max() + 1
        node_features.to_csv(self.data_path + 'region_features_{}.csv'.format(self.dataset), index=False)
        node_features = node_features.values
        self._logger.info('region_features: ' + str(node_features.shape))  # (N, fea_dim)
        return node_features, node_features.shape[1]

    def get_data(self):
        return self.region_features

    def get_data_feature(self):
        return {"region_edge_index_rel": self.edge_index_rel,
                "region_edge_index_sem": self.edge_index_sem,
                "region_edge_index_mob": self.edge_index_mob,
                "region_num_nodes": self.num_regions,
                "region_num_edges_rel": self.num_edges_rel,
                "region_num_edges_sem": self.num_edges_sem,
                "region_num_edges_mob": self.num_edges_mob,
                "region_geo_file": self.geo_file,
                "region_rel_file": self.rel_file,
                "region_geo_to_ind": self.geo_to_ind,
                "region_ind_to_geo": self.ind_to_geo,
                "region_fea_dim": self.region_fea_dim,
                'region_len_lon_encode': self.len_lon_encode,
                'region_len_lat_encode': self.len_lat_encode,
                'region_len_BLD_Count_encode': self.len_BLD_Count_encode,
                'region_len_BLD_FLOORN_encode': self.len_BLD_FLOORN_encode,
                'region_len_Shape_Area_encode': self.len_Shape_Area_encode,
                'region_len_FUNCTION': self.len_FUNCTION,
                'region_len_InCBD': self.len_InCBD,
                }
