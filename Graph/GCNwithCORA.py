#!/home/inje/anaconda3/envs/GNN_test/bin/python

# edge size: (2,4), # of edge= 4
# edge_index = torch.tensor([[0, 1, 1, 2],
#                            [1, 0, 2, 1]], dtype=torch.long)
# # node size: (3,1), # of nodes: 3
# x = torch.tensor(([-1], [0], [1]), dtype=torch.float)
#
#
# from torch_geometric.datasets import TUDataset
#
# dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
#
# print(dataset, len(dataset), dataset.num_classes, dataset.num_node_features)
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# print(device)
# print(torch.__version__, torch.version.cuda)
# print(torch.cuda.device_count())
"""
class: torch_geometric.data.Data
data.keys : 해당 속성 이름
data.num_nodes : 노드 총 개수
data.num_edges : 엣지 총 개수
data.contains_isolated_nodes() : 고립 노드 여부 확인
data.contains_self_loops() : 셀프 루프 포함 여부 확인
data.is_directed() : 그래프의 방향성 여부 확인
"""

#Cora dataset
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
coradata = Planetoid(root='/tmp/Cora', name='Cora', transform=T.NormalizeFeatures())
data = coradata[0]
# cora feature: 1433 cora edge feature: 0 cora class: 7 cora node feature: 1433
# Data(edge_index=[2, 10556], test_mask=[2708], train_mask=[2708], val_mask=[2708], x=[2708, 1433], y=[2708])
""" 
Cora is a graph (1 graph data), network of paper citation
numbers of edge is 10556 / 2 (undirected) 
'x' means that it has 2708 nodes and 1433 node feature"""
print(data.is_undirected(), "train=", data.train_mask.sum().item(), "val=", data.val_mask.sum().item(),"test=", data.test_mask.sum().item())
# True 140 1000
#
# #   minibatch and dataloader example
# from torch_geometric.data import DataLoader
#
# enzymeset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)
# loader = DataLoader(enzymeset, batch_size=32, shuffle=True)

"""for batch in loader:
    print(batch)
    # >>> Batch(batch=[1082], edge_index=[2, 4066], x=[1082, 21], y=[32])

    print(batch.num_graphs)"""

#   Implement a two-layer GCN:
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor, PairTensor
import torch
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes

from torch_geometric.nn.inits import glorot, zeros


EVAL = 0
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
                        self._cached_edge_index = (edge_index, edge_weight)
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
        global EVAL

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


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = Myconv(coradata.num_node_features, 16, cached=True,
                             normalize=not False)
        # print('conv1:', self.conv1.weight.shape)
        # conv1: torch.Size([1433, 16])
        self.conv2 = Myconv(16, coradata.num_classes, cached=True,
                             normalize=not False)
        # print('conv2:',self.conv2.weight.shape)
        # conv2: torch.Size([16, 7])

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        # print('x:',x.shape, 'edge:',edge_index.shape)
        # x: torch.Size([2708, 1433]) edge: torch.Size([2, 10556])
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
# pyprof.wrap(Net, 'forward')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.autograd.profiler as profiler
model = Net().to(device)
data = coradata[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
# print(data)
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()


model.eval()
EVAL = 1
# with cpu profiler
# with profiler.profile(record_shapes=True) as prof:
#     with profiler.record_function("model_inference"):
#         model(data)
#
# prof.export_chrome_trace("trace.json")


# check for evaluation
torch.cuda.cudart().cudaProfilerStart()
_, pred = model(data).max(dim=1)
torch.cuda.cudart().cudaProfilerStop()
correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / int(data.test_mask.sum())
print('Accuracy: {:.4f}'.format(acc))
# Accuracy: 0.8040

# profiling with NVIDIA PyProf


# import pyprof
# pyprof.init()
#
# iters = 500
# iter_to_capture = 100

# Define network, loss function, optimizer etc.

# # PyTorch NVTX context manager
# with torch.autograd.profiler.emit_nvtx():
#
#     for iter in range(iters):
#
#         if iter == iter_to_capture:
#             cuprof.start()
#
#         model(data)
#
#         if iter == iter_to_capture:
#             cuprof.stop()
#













