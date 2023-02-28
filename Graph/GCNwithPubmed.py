#!/home/inje/anaconda3/envs/GNN_test/bin/python3.7
from torch_geometric.datasets import Planetoid, Reddit, Yelp, SNAPDataset
import numpy.random as rd
import numpy as np
import torch_geometric.transforms as T
import torch
import shutil
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes

from torch_geometric.nn.inits import glorot, zeros

import torch.cuda.nvtx as nvtx
import torch.cuda.profiler as profiler
import pyprof
import dgl
import sys
import time

# neighbors = [4886]

list = [0 for i in range(4887)]
import csv
with open('./test_Yelp_g.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ')
    for row in spamreader:
        for i in row:
            # print(i)
            if i != '':
                list[int(i)] += 1

import matplotlib.pyplot as plt
plt.plot(list)
# plt.plot(fc.val_losses)
plt.ylabel('Frequency')
plt.xlabel('Yelp')
# plt.legend(['train_loss', 'val_loss'])
plt.title("Neighbor Distribution")
plt.savefig('Yelp_g.png')
plt.clf()
exit()


if len(sys.argv) == 2:
    g_data = sys.argv[1]
    # g_method, g_data, g_split, *gcard = sys.argv[1:]
    # gcard.append('0')
else:
    g_data = 'cora'
    # g_method = 'lpa'
    #
    # g_split = '0'
    # gcard = [0]
from collections import defaultdict

graph = (
    dgl.data.CoraGraphDataset() if g_data == 'cora'
    else dgl.data.CiteseerGraphDataset() if g_data == 'citeseer'
    else dgl.data.PubmedGraphDataset() if g_data == 'pubmed'
    else dgl.data.CoraFullDataset() if g_data == 'corafull'
    else dgl.data.CoauthorCSDataset() if g_data == 'coauthor-cs'
    else dgl.data.CoauthorPhysicsDataset() if g_data == 'coauthor-phy'
    else dgl.data.RedditDataset() if g_data == 'reddit'
    else dgl.data.rdf.AIFBDataset(insert_reverse=False) if g_data == 'aifb'
    else dgl.data.AmazonCoBuyComputerDataset()
    if g_data == 'amazon-com'
    else dgl.data.AmazonCoBuyPhotoDataset() if g_data == 'amazon-photo'
    else None
)[0]
# print(graph.nodes[dgl.data.rdf.AIFBDataset(insert_reverse=False).predict_category])
X = node_features = graph.ndata['feat']
Y = node_labels = graph.ndata['label']
n_nodes = node_features.shape[0]
nrange = torch.arange(n_nodes)
n_features = node_features.shape[1]
n_labels = int(Y.max().item() + 1)
# graph.edges
src, dst = graph.edges()
n_edges = src.shape[0]
Y = Y.tolist()
X = X.tolist()
ZERO = float(0)
print("nrange is", len(nrange), "n_edges is", n_edges, "graph edge is", src, dst)
# print("n_features is", X[0], "label is", Y[0])
# exit()
src = src.tolist()
dst = dst.tolist()
# print(src, dst)
graph_file = open("test_"+g_data + '.graph', "w")
# print("start", len(data))
# exit()
ordict = defaultdict(list)
for i in range(0, len(src)):
    ordict[src[i]].append(str(dst[i]))


for i in range(0, len(nrange)):
    # print(ordict[i])
    print(" ".join(ordict[i]), file=graph_file)

graph_file.close()

svmlight_file = open("test_"+g_data + '.svmlight', "w")
ordict1 = defaultdict(list)


for i in range(len(X)):
    # ordict[i].append(str(rd.randint(0, 100)))
    ordict1[i].append(str(Y[i]))
    for idx, j in enumerate(X[i]):
        # print(j)
        if j  > 0:
            ordict1[i].append(str(idx)+":"+str(j))

    # print(ordict)

for i in range(len(ordict1)):
    # print(ordict[i])
        print(" ".join(ordict1[i]), file=svmlight_file)

svmlight_file.close()

# X = coradata[0].x
exit()

pyprof.init()
EVAL = 0
prefix = 'soc-Pokec'
"""
{'ego-facebook': ['facebook.tar.gz'], 'ego-gplus': ['gplus.tar.gz'], 'ego-twitter': ['twitter.tar.gz'], 
'soc-epinions1': ['soc-Epinions1.txt.gz'], 'soc-livejournal1': ['soc-LiveJournal1.txt.gz'], 'soc-pokec': ['soc-pokec-relationships.txt.gz'], 
'soc-slashdot0811': ['soc-Slashdot0811.txt.gz'], 'soc-slashdot0922': ['soc-Slashdot0902.txt.gz'], 'wiki-vote': ['wiki-Vote.txt.gz']}
[Data(edge_index=edge_index, num_nodes=num_nodes)]
"""
coradata = SNAPDataset(root='/home/inje/pytorch_geometric/data/'+prefix, name=prefix)
# coradata = Yelp(root='/home/inje/pytorch_geometric/data/'+prefix, transform=T.NormalizeFeatures())
# coradata = Planetoid(root='/home/inje/pytorch_geometric/data/'+prefix, name=prefix, transform=T.NormalizeFeatures())
# coradata = Reddit(root='home/inje/pytorch_geometric/data/Reddit', transform=T.NormalizeFeatures())
# dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
print(coradata[0].num_nodes)
# X = coradata[0].x
exit()

"""For Adjacency Matrix"""
data = torch.transpose(coradata[0].edge_index, 0, 1)
# print(len(coradata[0].x))
cnt = 0
data = data.tolist()
ordict = defaultdict(list)
graph_file = open("test_"+prefix + '.graph', "w")
print("start", len(data))
# exit()

for i in data:
    ordict[i[0]].append(str(i[1]))


for i in range(len(ordict)):
    # print(ordict[i])
    print(" ".join(ordict[i]), file=graph_file)

graph_file.close()



"""For Feature Matrix, use x and y"""
svmlight_file = open("test_"+prefix + '.svmlight', "w")
Fdata = coradata[0]
Feat_x = Fdata.x.tolist()
Feat_y = Fdata.y.tolist()
from collections import defaultdict
ordict = defaultdict(list)
for i in range(len(Feat_x)):
    ordict[i].append(str(rd.randint(0, 100)))
    # ordict[i].append(str(Feat_y[i]))
    for idx, j in enumerate(Feat_x[i]):
        # print(j)
        if j  > 0:
            ordict[i].append(str(idx)+":"+str(j))

    # print(ordict)

for i in range(len(ordict)):
    # print(ordict[i])
        print(" ".join(ordict[i]), file=svmlight_file)

svmlight_file.close()
"""Feature Matrix End && Split Start"""
"""For Feature Matrix, use x and y"""
split_file = open("test_"+prefix + '.split', "w")
Fdata = coradata[0]
from collections import defaultdict
ordict = defaultdict(list)
for i in range(len(Fdata.x)):
    ordict[i].append(str(1))

for i in range(len(ordict)):
    print(" ".join(ordict[i]), file=split_file)
split_file.close()
# from sklearn import datasets
# print(cnt)
exit()
# print(data.is_undirected(), "train=", data.train_mask.sum().item(), "val=", data.val_mask.sum().item(),"test=", data.test_mask.sum().item())
@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (Tensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> PairTensor  # noqa
    pass


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (SparseTensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> SparseTensor  # noqa
    pass


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t
    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


class Myconv(GCNConv):
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(GCNConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                if self.cached:
                    pass
                else:
                    edge_index, edge_weight = cache[0], cache[1]
            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache


        x = torch.matmul(x, self.weight)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        if edge_weight is None:
            return x_j
        else:
            return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class GATNet(torch.nn.Module):
    def __init__(self):
        super(GATNet, self).__init__()

        self.conv1 = GATConv(coradata.num_features, 16, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(16 * 8, coradata.num_classes, heads=1, concat=False,
                             dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        if EVAL == 1:
            with torch.autograd.profiler.emit_nvtx():
                torch.cuda.cudart().cudaProfilerStart()
                profiler.start()
                x = self.conv1(x, edge_index)
                x = F.elu(x)
                x = F.dropout(x, p=0.6, training=self.training)
                x = self.conv2(x, data.edge_index)
                x = F.log_softmax(x, dim=-1)
                torch.cuda.cudart().cudaProfilerStop()
                profiler.stop()
                # print("end")
                exit()
        else:
            x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, data.edge_index)
        return F.log_softmax(x, dim=-1)



class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = Myconv(coradata.num_node_features, 128, cached=True,
                             normalize=not False)
        # print('conv1:', self.conv1.weight.shape)
        # conv1: torch.Size([1433, 16])
        self.conv2 = Myconv(128, coradata.num_classes, cached=True,
                             normalize=not False)
        # print('conv2:',self.conv2.weight.shape)
        # conv2: torch.Size([16, 7])

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        if EVAL == 1:
            with torch.autograd.profiler.emit_nvtx():
                torch.cuda.cudart().cudaProfilerStart()
                profiler.start()
                x = self.conv1(x, edge_index)
                torch.cuda.cudart().cudaProfilerStop()
                profiler.stop()
                # print("end")
                exit()
        else:
            x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        nvtx.range_pop()

        return F.log_softmax(x, dim=1)
# pyprof.wrap(Net, 'forward')

def save_checkpoint(state, is_best, filename='GAT_Redditcheckpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'GAT_Reddit_model_best.pth.tar')


# for GAT
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = GATNet().to(device), coradata[0]
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
# for GCN
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Net().to(device)
# data = coradata[0].to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


# print(data)
model.train()
out = []
# best_acc1 = 0.0
for epoch in range(50):
    optimizer.zero_grad()
    out = model(data.x.to(device), data.edge_index.to(device))
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    # acc = int(out.max(dim=0)[data.train_mask].eq(data.y[data.train_mask]).sum()) / int(data.train_mask.sum())
    # is_best = acc > best_acc1
    # best_acc1 = max(acc, best_acc1)
    print(f'Epoch {epoch:02d}, Loss: {loss:.4f}')

    loss.backward()
    optimizer.step()
save_checkpoint({
        'epoch': epoch + 1,
        'arch': model,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, True)

# print("training end")
model.eval()
EVAL = 1
# with cpu profiler
# with profiler.profile(record_shapes=True) as prof:
#     with profiler.record_function("model_inference"):
#         model(data)
#
# prof.export_chrome_trace("trace.json")


# check for evaluation for nsight-cu
# torch.cuda.cudart().cudaProfilerStart()
# _, pred = model(data).max(dim=1)
# torch.cuda.cudart().cudaProfilerStop()

# check for evaluation for pytorch profiler

loc = 'cuda:{}'.format(0)
checkpoint = torch.load("GAT_Reddit_model_best.pth.tar", map_location=loc)
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])
# for GAT
_, pred = model(data.x, data.edge_index).max(dim=1)
#for GCN
# _, pred = model(data).max(dim=1)
correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / int(data.test_mask.sum())
print('Accuracy: {:.4f}'.format(acc))
