import torch
from torch import nn
from otherlayers import *
import numpy as np

class ADVRI(nn.Module):
    def __init__(self, walk_emd, gcn_emd, GDP, param):
        super(ADVRI, self).__init__()
        self.Xwalk = walk_emd
        self.Xgcn = gcn_emd
        self.md_supernode = GDP 
        self.fm = param.fm
        self.num_heads = param.head

        self.nonlinear_fusion = nn.Sequential(
            nn.Linear(self.fm * 2, self.fm * 2),
            nn.ReLU(),
            nn.Linear(self.fm * 2, self.fm)
        )
        
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=self.fm,
            num_heads=self.num_heads,
            batch_first=True
        )

        self.output_layer = nn.Linear(self.fm, self.fm)
    
    def forward(self, simdata, train_data):
  
        Ewalk = self.Xwalk(simdata)  
        Egcn = self.Xgcn(simdata)    

        combined_features = torch.cat([Ewalk, Egcn], dim=1)
        fused_features = self.nonlinear_fusion(combined_features)
        
        attn_input = fused_features.unsqueeze(0)  # [1, max_nodes, fm]
        attn_output, _ = self.multihead_attention(attn_input, attn_input, attn_input)
        attn_output = attn_output.squeeze(0)  # [max_nodes, fm]

        final_features = self.output_layer(attn_output)

        mFea, dFea = pro_data(train_data, final_features, final_features)
        pre_asso = self.md_supernode(mFea, dFea)
        return pre_asso
def pro_data(data, em, ed):

    gene_idx = data[:, 0]
    disease_idx = data[:, 1]

    max_gene_idx = em.size(0) - 1
    max_disease_idx = ed.size(0) - 1
    gene_idx = torch.clamp(gene_idx, 0, max_gene_idx)
    disease_idx = torch.clamp(disease_idx, 0, max_disease_idx)
    Em = torch.index_select(em, 0, gene_idx)
    Ed = torch.index_select(ed, 0, disease_idx)

    return Em, Ed

class BaseEmbedding(nn.Module):
    def __init__(self, param, embedding_type='WALK'):
        super(BaseEmbedding, self).__init__()
        self.args = param
        self.fm = self.args.fm
        self.gcn_layers = self.args.gcn_layers
        self.num_heads = self.args.head
        self.embedding_type = embedding_type
        
        # GCN层
        self.gcn_layers_list = nn.ModuleList([
            nn.Linear(self.fm, self.fm) for _ in range(self.gcn_layers)
        ])
        for layer in self.gcn_layers_list:
            layer.activation = nn.ReLU()

        self.fc1 = nn.Linear(in_features=self.gcn_layers,
                            out_features=2 * self.args.head * self.args.gcn_layers)
        self.fc2 = nn.Linear(in_features=2 * self.args.head * self.args.gcn_layers,
                            out_features=self.gcn_layers)
        self.sigmoid = nn.Sigmoid()

        self.cnn = nn.Conv2d(in_channels=self.gcn_layers, out_channels=1,
                            kernel_size=(1, 1), stride=1, bias=True)

        self._init_parameters()
        self.dropout = nn.Dropout(0.5)
        
    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def sparse_gcn_forward(self, layer, x, adj):

        x = torch.nan_to_num(x, nan=0.0)
        support = torch.sparse.mm(adj, x)
        output = layer(support)
        if hasattr(layer, 'activation') and layer.activation is not None:
            output = layer.activation(output)
        return output

    def create_sparse_adj(self, matrix, edges, device):

        edge_values = matrix[edges[0], edges[1]]
        

        if self.embedding_type == 'GCN':
            edge_values = torch.abs(edge_values)
        

        sparse_adj = torch.sparse_coo_tensor(
            indices=edges,
            values=edge_values,
            size=matrix.shape
        ).to(device)

        num_nodes = sparse_adj.size(0)
        self_loop_idx = torch.tensor([[i, i] for i in range(num_nodes)]).t().to(device)
        self_loop_values = torch.ones(num_nodes, dtype=torch.float).to(device)

        all_indices = torch.cat([edges, self_loop_idx], dim=1)
        all_values = torch.cat([edge_values, self_loop_values])
        
        adj_with_loop = torch.sparse_coo_tensor(
            indices=all_indices,
            values=all_values,
            size=sparse_adj.shape
        ).to(device)
        

        deg = torch.sparse.sum(adj_with_loop, dim=1).to_dense()
        deg = deg + 1e-6  
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt = torch.clamp(deg_inv_sqrt, -1e3, 1e3)
        

        row, col = edges
        norm_values_edge = deg_inv_sqrt[row] * edge_values * deg_inv_sqrt[col]
        self_loop_norm = deg_inv_sqrt * deg_inv_sqrt

        all_norm_values = torch.cat([norm_values_edge, self_loop_norm])
        all_norm_values = torch.nan_to_num(all_norm_values, nan=0.0)

        norm_adj = torch.sparse_coo_tensor(
            indices=all_indices,
            values=all_norm_values,
            size=sparse_adj.shape
        ).to(device)
        
        return norm_adj

    def process_attention(self, X, node_count):
        kernel_size = (self.fm, X.size(3))
        globalAvgPool = nn.AvgPool2d(kernel_size, stride=(1, 1))
        x_channel_attention = globalAvgPool(X)
        x_channel_attention = x_channel_attention.reshape(x_channel_attention.size(0), -1)
        x_channel_attention = self.fc1(x_channel_attention)
        x_channel_attention = torch.relu(x_channel_attention)
        x_channel_attention = self.fc2(x_channel_attention)
        x_channel_attention = self.sigmoid(x_channel_attention)
        x_channel_attention = x_channel_attention.reshape(x_channel_attention.size(0), 
                                                        x_channel_attention.size(1), 1, 1)

        x_channel_attended = X * x_channel_attention

        x = self.cnn(x_channel_attended)
        x = x.squeeze(0).squeeze(0).t()  
        
        return x
class EmbeddingWALK(BaseEmbedding):

    def __init__(self, param):
        super(EmbeddingWALK, self).__init__(param, embedding_type='WALK')
    
    def forward(self, data, edge_weight=None):

        device = next(self.parameters(), torch.tensor(0.0)).device
        
        walk_matrix = torch.tensor(data['het_walk_mat']['data_matrix'], dtype=torch.float).to(device)
        edges_walk = torch.tensor(data['het_walk_mat']['edges'], dtype=torch.long).to(device)
        walk_number = walk_matrix.shape[0]
        
        original_ids = walk_matrix[:, 0]
        x_m = torch.randn(walk_number, self.fm, device=device)
        x_m[:, 0] = original_ids 
        

        norm_adj = self.create_sparse_adj(walk_matrix, edges_walk, device)

        x_m_layers = []
        x_input = x_m
        for layer in self.gcn_layers_list:
            x_out = self.sparse_gcn_forward(layer, x_input, norm_adj)
            x_out = self.dropout(x_out)
            x_m_layers.append(x_out)
            x_input = x_out

        XM = torch.cat(x_m_layers, 1)
        XM = XM.reshape(1, self.gcn_layers, self.fm, walk_number)

        x = self.process_attention(XM, walk_number)

        x[:, 0] = original_ids
        
        return x

class EmbeddingGCN(BaseEmbedding):

    def __init__(self, param):
        super(EmbeddingGCN, self).__init__(param, embedding_type='GCN')
    
    def forward(self, data, batch_indices=None):
        device = next(self.parameters(), torch.tensor(0.0)).device
        gcn_matrix = torch.tensor(data['het_gcn_mat']['data_matrix'], dtype=torch.float).to(device)
        edges_gcn = torch.tensor(data['het_gcn_mat']['edges'], dtype=torch.long).to(device)
        gcn_number = gcn_matrix.shape[0]

        original_ids = gcn_matrix[:, 0].long()
        x_d = torch.randn(gcn_number, self.fm, device=device)
        x_d[:, 0] = original_ids  

        norm_adj = self.create_sparse_adj(gcn_matrix, edges_gcn, device)

        x_d_layers = []
        x_input = x_d
        for layer in self.gcn_layers_list:
            x_out = self.sparse_gcn_forward(layer, x_input, norm_adj)
            x_out = self.dropout(x_out)
            x_d_layers.append(x_out)
            x_input = x_out

        XD = torch.cat(x_d_layers, 1)
        XD = XD.reshape(1, self.gcn_layers, self.fm, gcn_number)

        x = self.process_attention(XD, gcn_number)

        x[:, 0] = original_ids
        
        return x

class GDP(nn.Module):

    def __init__(self, param):
        super(GDP, self).__init__()
        self.inSize = param.inSize
        self.outSize = param.outSize
        self.fcDropout = param.fcDropout
        self.sigmoid = nn.Sigmoid()
        self.relu1 = nn.ReLU()

        self.mlp = nn.Sequential(
            nn.Linear(self.inSize * 2, self.outSize),
            nn.ReLU(),
            nn.Dropout(self.fcDropout),
            nn.Linear(self.outSize, self.outSize),
            nn.ReLU(),
            nn.Dropout(self.fcDropout),
            nn.Linear(self.outSize, 1)
        )
    
    def forward(self, em, ed):
        em = torch.nan_to_num(em, nan=0.0)
        ed = torch.nan_to_num(ed, nan=0.0)
        em = torch.clamp(em, -1e9, 1e9)
        ed = torch.clamp(ed, -1e9, 1e9)
        combined = torch.cat([em, ed], dim=1)
        pre_part = self.mlp(combined)
        pre_a = self.sigmoid(pre_part).squeeze(dim=1)
        return pre_a