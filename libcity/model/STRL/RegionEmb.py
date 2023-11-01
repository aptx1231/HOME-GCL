import torch
import torch.nn as nn

class RegionEmbedding(nn.Module):

    def __init__(self, config, data_feature):
        super().__init__()
        self.emb_dim = config['emb_dim']
        self.hid_dim = config['output_dim']

        self.region_len_InCBD = data_feature['region_len_InCBD']
        self.region_len_BLD_Count_encode = data_feature['region_len_BLD_Count_encode']
        self.region_len_BLD_FLOORN_encode = data_feature['region_len_BLD_FLOORN_encode']
        self.region_len_Shape_Area_encode = data_feature['region_len_Shape_Area_encode']
        self.region_len_lon_encode = data_feature['region_len_lon_encode']
        self.region_len_lat_encode = data_feature['region_len_lat_encode']

        self.region_len_InCBD_emb = nn.Embedding(self.region_len_InCBD, self.emb_dim)
        self.region_len_BLD_Count_encode_emb = nn.Embedding(self.region_len_BLD_Count_encode, self.emb_dim)
        self.region_len_BLD_FLOORN_encode_emb = nn.Embedding(self.region_len_BLD_FLOORN_encode, self.emb_dim)
        self.region_len_Shape_Area_encode_emb = nn.Embedding(self.region_len_Shape_Area_encode, self.emb_dim)
        self.region_len_lon_encode_emb = nn.Embedding(self.region_len_lon_encode, self.emb_dim)
        self.region_len_lat_encode_emb = nn.Embedding(self.region_len_lat_encode, self.emb_dim)

        self.seq_cat_emb = nn.Linear(self.emb_dim * 6, self.hid_dim)

    def forward(self, batch_seq_cat):
        """
            batch_seq_cat: (B, len_seqs_cat_feas)
        """
        region_len_InCBD_emb = self.region_len_InCBD_emb(batch_seq_cat[:, 1])
        region_len_BLD_Count_encode_emb = self.region_len_BLD_Count_encode_emb(batch_seq_cat[:, 2])
        region_len_BLD_FLOORN_encode_emb = self.region_len_BLD_FLOORN_encode_emb(batch_seq_cat[:, 3])
        region_len_Shape_Area_encode_emb = self.region_len_Shape_Area_encode_emb(batch_seq_cat[:, 4])
        region_len_lon_encode_emb = self.region_len_lon_encode_emb(batch_seq_cat[:, 5])
        region_len_lat_encode_emb = self.region_len_lat_encode_emb(batch_seq_cat[:, 6])

        cat_list = [region_len_InCBD_emb,
                                region_len_BLD_Count_encode_emb,
                                region_len_BLD_FLOORN_encode_emb,
                                region_len_Shape_Area_encode_emb,
                                region_len_lon_encode_emb,
                                region_len_lat_encode_emb]
        sparse_vec = torch.cat(cat_list, dim=1)  # (B, 6 * emb_dim)
        sparse_vec = self.seq_cat_emb(sparse_vec)   # (B, hid_dim)
        return sparse_vec
