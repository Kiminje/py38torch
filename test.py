import torch
mat = [[2, 3], [1, 7]]

print(mat[0][0], mat[0][1])
print(mat[1][0], mat[1][1])

import torch_geometric
import torch_sparse
print(torch.__version__)
print("ok")
print(torch.cuda.get_device_capability(), torch.cuda.current_device())
import torch_geometric as geo
#   print(geo.__version__)
Dat_geo = geo.data.Data
print(geo.is_debug_enabled())
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
data = Dat_geo(x=x,edge_index=edge_index)

print(data)

edge_index = torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Dat_geo(x=x, edge_index=edge_index.t().contiguous())
print(data)
device = torch.device('cuda')
data = data.to(device)

import torch_geometric.datasets as DB

'''
dataset = DB.TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
print(dataset, len(dataset), dataset.num_classes, dataset.num_node_features)
data = dataset[0]
print(data)
print(data.is_undirected())

train_dataset = dataset[:540]
test_dataset = dataset[540:]
print("train is ", train_dataset, "test is  ", test_dataset)

#   data shuffling
dataset = dataset.shuffle()
print(dataset)
'''
# from Cora, do it semi-supervised graph node classification
dataset = DB.Planetoid(root='/tmp/Cora', name='Cora')
print(len(dataset), dataset.num_classes, dataset.num_node_features)

data = dataset[0]
print(data)
print(data.is_undirected(), data.train_mask.sum().item(), data.val_mask.sum().item(), data.test_mask.sum().item())
"""
1 7 1433
Data(edge_index=[2, 10556], test_mask=[2708], train_mask=[2708], val_mask=[2708], x=[2708, 1433], y=[2708])
True 140 500 1000
-> undirected edge = 10556 / 2 = 5278,
data holds new attributes = train_mask, val_mask, test_mask:
train_mask denotes against which nodes to train(140 nodes)
val_mask denotes which nodes to use for validation:(to perform early stopping[500 nodes])
# of classes is 7, node feature is 1433 vectors.
all mask has 2708 nodes of graph.
'''
validation test set과의 차이점은 test set은 모델의 '최종 성능' 을 평가하기 위해서 쓰이며, 
training의 과정에 관여하지 않는 차이가 있습니다. 
반면 validation set은 여러 모델 중에서 최종 모델을 선정하기 위한 성능 평가에 관여한다 보시면됩니다. 
따라서 validation set은 training과정에 관여하게 됩니다.
 즉, validation set은 training 과정에 관여를 하며, training이 된 여러가지 모델 중 가장 좋은 하나의 모델을 고르기 위한 셋입니다.
 test set은 모든 training 과정이 완료된 후에 최종적으로 모델의 성능을 평가하기 위한 셋입니다.
 만약 test set이 모델을 개선하는데 쓰인다면, 그건 test set이 아니라 validation set입니다. 
 만약 여러 모델을 성능 평가하여 그 중에서 가장 좋은 모델을 선택하고 싶지 않은 경우에는 validation set을 만들지 않아도 됩니다. 
 하지만 이 경우에는문제가 생길 것입니다. (test accuracy를 예측할 수도 없고, 모델 튜닝을 통해 overfitting을 방지할 수도 없습니다.)
'''
"""

########################################################################################3
#################MINI BATCH#############################################################
'''dataset = DB.TUDataset(root='tmp/ENZYMES', name='ENZYMES', use_node_attr=True)
loader = geo.data.DataLoader(dataset, batch_size=32, shuffle=True)
import torch_scatter

for data in loader:
    print(data, data.num_graphs)
    print(data.x.size())
    x = torch_scatter.scatter_mean(data.x, data.batch, dim=0)
    print(x.size())
    """
    Batch(batch=[908], edge_index=[2, 3538], x=[908, 21], y=[32]) 32
    torch.Size([908, 21])
    torch.Size([32, 21])
    """


print(loader.__sizeof__(), dataset.__sizeof__())
'''

import torch_geometric.transforms as Trans

dataset_ShapeNet = geo.datasets.ShapeNet(root='/tmp/ShapeNet', categories=['Airplane'],
                                         pre_transform=Trans.KNNGraph(k=6),
                                         transform=Trans.RandomTranslate(0.01))
print(dataset_ShapeNet[0])


