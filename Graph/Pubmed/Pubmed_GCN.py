import torch
import torch_geometric

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

# Cora dataset
from torch_geometric.datasets import Planetoid

pubmed = Planetoid(root='/tmp/PubMed', name='PubMed')
# print()
# print("Pubmed feature:", pubmed.num_features,"Pubmed edge feature:", pubmed.num_edge_features, "Pubmed class:", pubmed.num_classes,
#       "Pubmed node feature:", pubmed.num_node_features)
# cora feature: 1433 cora edge feature: 0 cora class: 7 cora node feature: 1433
data = pubmed[0]
# print(data)
# Data(edge_index=[2, 10556], test_mask=[2708], train_mask=[2708], val_mask=[2708], x=[2708, 1433], y=[2708])
""" 
Pubmed feature: 500 Pubmed edge feature: 0 Pubmed class: 3 Pubmed node feature: 500
Data(edge_index=[2, 88648], test_mask=[19717], train_mask=[19717], val_mask=[19717], x=[19717, 500], y=[19717])
True train= 60 val= 500 test= 1000"""
print(data.keys)

# print(data.is_undirected(), "train=", data.train_mask.sum().item(), "val=", data.val_mask.sum().item(),"test=", data.test_mask.sum().item())
# True train= 60 val= 500 test= 1000
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
import torch.cuda.nvtx as nvtx


# # import torch.cuda.profiler as cuprof
# import pyprof
# pyprof.init()
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(pubmed.num_node_features, 16)
        # print('conv1:', self.conv1.weight.shape)
        # conv1: torch.Size([1433, 16])
        self.conv2 = GCNConv(16, pubmed.num_classes)
        # print('conv2:',self.conv2.weight.shape)
        # conv2: torch.Size([16, 7])

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # print('x:',x.shape, 'edge:',edge_index.shape)
        # x: torch.Size([2708, 1433]) edge: torch.Size([2, 10556])
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


# # pyprof.wrap(Net, 'forward')
# print

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)
import torch.autograd.profiler as profiler

model = Net().to(device)
from torchsummary import summary
import collections
# summary(model, input_size=(500, 1433))
# with torch.autograd.profiler.profile(use_cuda=True, record_shapes=True, profile_memory=True) as prof:
nvtx.range_push("Copy to device")
data = pubmed[0].to(device)
from torch_geometric.data import DataLoader
loader = DataLoader(data, batch_size=32, shuffle=True)
# print(isinstance(loader, collections.Iterable))
# for batch in loader:
#     print(batch, batch.shape)


#
# nvtx.range_pop()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
# # print(data)
# model.train()
# for epoch in range(1000):
#     nvtx.range_push("iter" + str(epoch))
#     nvtx.range_push("Forward pass")
#     optimizer.zero_grad()
#     out = model(data)
#     loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
#     nvtx.range_pop()
#     nvtx.range_push("Backward pass")
#     loss.backward()
#     optimizer.step()
#     nvtx.range_pop()
#     nvtx.range_pop()
# # print(prof.key_averages().table(sort_by="cuda_time_total"))  # sort_by="self_cpu_time_total"
# #
# model.eval()
# #
# # # with cpu profiler
# # # with profiler.profile(record_shapes=True) as prof:
# # #     with profiler.record_function("model_inference"):
# # #         model(data)
# # #
# # # prof.export_chrome_trace("trace.json")
# #
# #
# # # check for evaluation
# # with torch.autograd.profiler.profile(use_cuda=True, record_shapes=True, profile_memory=True) as prof:
# _, pred = model(data).max(dim=1)
# # print(prof.key_averages().table(sort_by="cuda_time_total"))  # sort_by="self_cpu_time_total"
#
# correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
# acc = correct / int(data.test_mask.sum())
# print('Accuracy: {:.4f}'.format(acc))
# # Accuracy: 0.8040
#
# # profiling with NVIDIA PyProf
#
#
# # import pyprof
# # pyprof.init()
# #
# # iters = 500
# # iter_to_capture = 100
#
# # Define network, loss function, optimizer etc.
#
# # # PyTorch NVTX context manager
# # with torch.autograd.profiler.emit_nvtx():
# #
# #     for iter in range(iters):
# #
# #         if iter == iter_to_capture:
# #             cuprof.start()
# #
# #         model(data)
# #
# #         if iter == iter_to_capture:
# #             cuprof.stop()
# #
#
#
