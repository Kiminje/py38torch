from ogb.nodeproppred import DglNodePropPredDataset
from torch_geometric.data import DataLoader
from collections import defaultdict
import torch
import shutil
import dgl
import sys
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.datasets import Yelp
import numpy.random as rd
# Download and process data at './dataset/ogbg_molhiv/'

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

# graph = DglNodePropPredDataset(name = "ogbn-products", root = 'dataset/')
# print(graph[0])
if g_data == 'yelp':
    coradata = Yelp(root='/home/inje/pytorch_geometric/data/' + g_data, transform=T.NormalizeFeatures())
    data = torch.transpose(coradata[0].edge_index, 0, 1)
    # print(len(coradata[0].x))
    cnt = 0
    data = data.tolist()
    ordict = defaultdict(list)
    graph_file = open("test_" + g_data + '.graph', "w")
    print("start", len(data))
    # exit()

    for i in data:
        ordict[i[0]].append(str(i[1]))

    for i in range(len(ordict)):
        # print(ordict[i])
        print(" ".join(ordict[i]), file=graph_file)

    graph_file.close()

    """For Feature Matrix, use x and y"""
    svmlight_file = open("test_" + g_data + '.svmlight', "w")
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
            if j > 0:
                ordict[i].append(str(idx) + ":" + str(j))

        # print(ordict)

    for i in range(len(ordict)):
        # print(ordict[i])
        print(" ".join(ordict[i]), file=svmlight_file)

    svmlight_file.close()
else:
    graph = (
        dgl.data.CoraGraphDataset() if g_data == 'cora'
        else DglNodePropPredDataset(name = "ogbn-products", root = 'dataset/') if g_data == 'ogbn-products'
        else DglNodePropPredDataset(name = "ogbn-papers100M", root = 'dataset/') if g_data == 'ogbn-papers100M'
        else DglNodePropPredDataset(name="ogbn-mag", root='dataset/') if g_data == 'ogbn-mag'
        else dgl.data.CiteseerGraphDataset() if g_data == 'citeseer'
        else dgl.data.PubmedGraphDataset() if g_data == 'pubmed'
        else dgl.data.CoraFullDataset() if g_data == 'corafull'
        else dgl.data.CoauthorCSDataset() if g_data == 'coauthor-cs'
        else dgl.data.CoauthorPhysicsDataset() if g_data == 'coauthor-phy'
        else dgl.data.RedditDataset() if g_data == 'reddit'
        else dgl.data.rdf.AIFBDataset(insert_reverse=False) if g_data == 'aifb'
        else dgl.data.AmazonCoBuyComputerDataset() if g_data == 'amazon-com'
        else dgl.data.AmazonCoBuyPhotoDataset() if g_data == 'amazon-photo'
        else None
    )[0]
    # print(graph[1][4444])
    if g_data == 'ogbn-products' or g_data == 'ogbn-papers100M' or g_data == 'ogbn-mag':
        print(graph)
        X = node_features = graph[0].ndata['feat']
        Y = node_labels = graph[1]
        src, dst = graph[0].edges()
        n_nodes = node_features.shape[0]
        nrange = torch.arange(n_nodes)
        n_features = node_features.shape[1]
        n_labels = int(Y.max().item() + 1)
        # graph.edges

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
        graph_file = open("test_" + g_data + '.graph', "w")
        # print("start", len(data))
        # exit()
        ordict = defaultdict(list)
        for i in range(0, len(src)):
            ordict[src[i]].append(str(dst[i]))

        for i in range(0, len(nrange)):
            # print(ordict[i])
            print(" ".join(ordict[i]), file=graph_file)

        graph_file.close()

        svmlight_file = open("test_" + g_data + '.svmlight', "w")
        ordict1 = defaultdict(list)

        for i in range(len(X)):
            # ordict[i].append(str(rd.randint(0, 100)))
            ordict1[i].append(str(Y[i][0]))
            for idx, j in enumerate(X[i]):
                # print(j)
                if j > 0:
                    ordict1[i].append(str(idx) + ":" + str(j))

            # print(ordict)

        for i in range(len(ordict1)):
            # print(ordict[i])
            print(" ".join(ordict1[i]), file=svmlight_file)

        svmlight_file.close()
    else:
        X = node_features = graph.ndata['feat']
        Y = node_labels = graph.ndata['label']
        src, dst = graph.edges()

        n_nodes = node_features.shape[0]
        nrange = torch.arange(n_nodes)
        n_features = node_features.shape[1]
        n_labels = int(Y.max().item() + 1)
        # graph.edges

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
