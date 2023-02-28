ExecutionDependency = 0
DataRequest = 0
Synchronization = 0
ImmConstant = 0
PipeBusy = 0
NotSelected =0

"""
We need to seperate execution kernel to aggregation and combination
Main kernels: sgemm, indexselect, scatter

LSU: Load Store Unit
FMA: Fused Multiply Add/Accumulate
ADU: Address Divergence Unit
CBU: Convergence Barrier Unit. (warp-level convergence, barrier, branch insts)

using batch size, make model size similar to DNN batch size
Metric
utilization(%):     SM, Memory, 
hit rate (%):       (L1h)L1 hit, (L2h)L2 hit, 
Throughput(GB/sec): (L12SMT)L1Cache to SM throughput, (MT) Memory Throughput
Etc:                (AO)Achieved Occupancy, (ATpR)Atomic Transactions per request, (IPC)Executed IPC, (SU)Issue Slot Utilization
Issue stall:        (MTS)Issue Stalled for Memory Throttle, (Sync)Synchronization, (PB)Pipe Busy, (ED)Execution Dependency, (DR)Data Request
layer 1: volta_sgemm_32x128_nn, indexSelectLargeIndex, 
layer 2: volta_sgemm_32x128_nn
"""
volta_sgemm_32x128_nn = {}
volta_sgemm_32x128_nn['SM'] = 25.4
volta_sgemm_32x128_nn['Memory'] = 35.62
volta_sgemm_32x128_nn['L1h'] = 15.65
volta_sgemm_32x128_nn['L2h'] = 70.64
volta_sgemm_32x128_nn['IPC'] = 0.77
volta_sgemm_32x128_nn['L12SMT'] = 137.25    # GB/sec
volta_sgemm_32x128_nn['MT'] = 154.32
volta_sgemm_32x128_nn['AO'] = 35.73
# % , Occupancy is the ratio of the number of active warps per multiprocessor to the maximum number of possible active warps.
# Another way to view occupancy is the percentage of the hardware's ability to process warps that is actively in use.
# Higher occupancy does not always result in higher performance, however, low occupancy always reduces the ability to hide latencies,
# resulting in overall performance degradation.
# Large discrepancies between the theoretical and the achieved occupancy during execution typically indicates highly imbalanced workloads.
# 32 warps can active in a multiprocessor
volta_sgemm_32x128_nn['ATpR'] = 0   #theorical warp: 8, active warp: 2.xx, and eligible warp is 0.28 -> stall by dependency
volta_sgemm_32x128_nn['SU'] = 19.27
volta_sgemm_32x128_nn['MTS'] = 0    # only activated in graph processing
volta_sgemm_32x128_nn['Sync'] = None
volta_sgemm_32x128_nn['PB'] = 81.59
volta_sgemm_32x128_nn['ED'] = None
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import squeeze
# #Kernel proportion

SAG_Pubmed_128 = {}
SAG_Pubmed_128['ScatterGather'] = 25.8
SAG_Pubmed_128['IndexSelect'] = 35.1
SAG_Pubmed_128['Sgemm'] = 26.6
SAG_Pubmed_128['ElementWise'] = 9.3

SAG_Citeseer_128 = {}
SAG_Citeseer_128['ScatterGather'] =24.3
SAG_Citeseer_128['IndexSelect'] = 28.1
SAG_Citeseer_128['Sgemm'] = 33.5
SAG_Citeseer_128['ElementWise'] = 13.2

SAG_Cora_128 = {}
SAG_Cora_128['ScatterGather'] =24.5
SAG_Cora_128['IndexSelect'] = 31.7
SAG_Cora_128['Sgemm'] = 30.5
SAG_Cora_128['ElementWise'] = 11.6

SAINT_Pubmed_128 = {}
SAINT_Pubmed_128['ScatterGather'] = 23.2
SAINT_Pubmed_128['Sgemm'] = 23.1
SAINT_Pubmed_128['IndexSelect'] = 22.2
SAINT_Pubmed_128['ElementWise'] = 23.1

SAINT_Citeseer_128 = {}
SAINT_Citeseer_128['ScatterGather'] = 32.6
SAINT_Citeseer_128['Sgemm'] = 32.7
SAINT_Citeseer_128['IndexSelect'] = 21.2
SAINT_Citeseer_128['ElementWise'] = 12.5

SAINT_Cora_128 = {}
SAINT_Cora_128['ScatterGather'] = 33.1
SAINT_Cora_128['Sgemm'] = 29.5
SAINT_Cora_128['IndexSelect'] = 25.6
SAINT_Cora_128['ElementWise'] = 10.1

GCN_Pubmed_128 = {}
GCN_Pubmed_128['ScatterGather'] = 23.0
GCN_Pubmed_128['Sgemm'] = 23.1
GCN_Pubmed_128['IndexSelect'] = 22.3
GCN_Pubmed_128['ElementWise'] = 25.2

GCN_Citeseer_128 = {}
GCN_Citeseer_128['ScatterGather'] = 4.1
GCN_Citeseer_128['Sgemm'] = 76.2
GCN_Citeseer_128['IndexSelect'] = 4.2
GCN_Citeseer_128['ElementWise'] = 6.1

GCN_Cora_128 = {}
GCN_Cora_128['ScatterGather'] = 10.3
GCN_Cora_128['Sgemm'] = 42.3
GCN_Cora_128['IndexSelect'] = 11.4
GCN_Cora_128['ElementWise'] = 14.9

GAT_Cora_128 = {}
GAT_Cora_128['ScatterGather'] = 10.3
GAT_Cora_128['Sgemm'] = 42.9
GAT_Cora_128['IndexSelect'] = 10.8
GAT_Cora_128['ElementWise'] = 14.9
GAT_Cora_128['Reduced'] = 10.7

GAT_Citeseer_128 = {}
GAT_Citeseer_128['ScatterGather'] = 5.1
GAT_Citeseer_128['Sgemm'] = 66
GAT_Citeseer_128['IndexSelect'] = 6.1
GAT_Citeseer_128['ElementWise'] = 10.3
GAT_Citeseer_128['Reduced'] = 6.8

GAT_Pubmed_128 = {}
GAT_Pubmed_128['ScatterGather'] = 17.8
GAT_Pubmed_128['Sgemm'] = 20.1
GAT_Pubmed_128['IndexSelect'] = 18.7
GAT_Pubmed_128['ElementWise'] = 27.5
GAT_Pubmed_128['Reduced'] = 10.3
# print(SAG_Reddit_128.keys(), SAG_Reddit_128.values())
fig, axs = plt.subplots(1, 2)
plt.rcParams.update({'font.size': 18})

SAG_Reddit_128 = {}
SAG_Reddit_128['ScatterGather'] = 29
SAG_Reddit_128['IndexSelect'] = 48.8
SAG_Reddit_128['Sgemm'] = 14.5
SAG_Reddit_128['ElementWise'] = 5.0
SAG_Reddit_128['etc'] = 100 -sum(SAG_Reddit_128.values())
# PubScatter = [SAG_Pubmed_128['ScatterGather'], GAT_Pubmed_128['ScatterGather'], GCN_Pubmed_128['ScatterGather'], SAINT_Pubmed_128['ScatterGather']]
# PubSelect = [SAG_Pubmed_128['IndexSelect'], GAT_Pubmed_128['IndexSelect'], GCN_Pubmed_128['ScatterGather'], SAINT_Pubmed_128['IndexSelect']]
# PubSgemm = [SAG_Pubmed_128['Sgemm'], GAT_Pubmed_128['Sgemm'], GCN_Pubmed_128['Sgemm'], SAINT_Pubmed_128['Sgemm']]
# PubElement = [SAG_Pubmed_128['ElementWise'], GAT_Pubmed_128['ElementWise'], GCN_Pubmed_128['ElementWise'], SAINT_Pubmed_128['ElementWise']]
# PubReduce = [0, GAT_Pubmed_128['Reduced'], 0, 0]
#
# CoraScatter = [SAG_Cora_128['ScatterGather'], GAT_Cora_128['ScatterGather'], GCN_Cora_128['ScatterGather'], SAINT_Cora_128['ScatterGather']]
# CoraSelect = [SAG_Cora_128['IndexSelect'], GAT_Cora_128['IndexSelect'], GCN_Cora_128['ScatterGather'], SAINT_Cora_128['IndexSelect']]
# CoraSgemm = [SAG_Cora_128['Sgemm'], GAT_Cora_128['Sgemm'], GCN_Cora_128['Sgemm'], SAINT_Cora_128['Sgemm']]
# CoraElement = [SAG_Cora_128['ElementWise'], GAT_Cora_128['ElementWise'], GCN_Cora_128['ElementWise'], SAINT_Cora_128['ElementWise']]
# CoraReduce = [0, GAT_Cora_128['Reduced'], 0, 0]
#
# CiteseerScatter = [SAG_Citeseer_128['ScatterGather'], GAT_Citeseer_128['ScatterGather'], GCN_Citeseer_128['ScatterGather'], SAINT_Citeseer_128['ScatterGather']]
# CiteseerSelect = [SAG_Citeseer_128['IndexSelect'], GAT_Citeseer_128['IndexSelect'], GCN_Citeseer_128['ScatterGather'], SAINT_Citeseer_128['IndexSelect']]
# CiteseerSgemm = [SAG_Citeseer_128['Sgemm'], GAT_Citeseer_128['Sgemm'], GCN_Citeseer_128['Sgemm'], SAINT_Citeseer_128['Sgemm']]
# CiteseerElement = [SAG_Citeseer_128['ElementWise'], GAT_Citeseer_128['ElementWise'], GCN_Citeseer_128['ElementWise'], SAINT_Citeseer_128['ElementWise']]
# CiteseerReduce = [0, GAT_Citeseer_128['Reduced'], 0, 0]
#
# RedditScatter = [SAG_Reddit_128['ScatterGather']]
# RedditSelect = [SAG_Reddit_128['IndexSelect']]
# RedditSgemm = [SAG_Reddit_128['Sgemm']]
# RedditElement = [SAG_Reddit_128['ElementWise']]
# RedditReduce = [0]
# print(PubScatter, PubSelect, PubSgemm, PubElement, PubReduce)
# print(CoraScatter, CoraSelect, CoraSgemm, CoraElement, CoraReduce)
# print(CiteseerScatter, CiteseerSelect, CiteseerSgemm, CiteseerElement, CiteseerReduce)
# print(RedditScatter, RedditSelect, RedditSgemm, RedditElement, RedditReduce)
# Frame = pd.DataFrame('ScatterGather':[PubScatter, CoraScatter, CiteseerScatter, RedditScatter], 'IndexSelect':[PubSelect, CoraSelect, CiteseerReduce, RedditSelect],
# 'Sgemm':[PubSgemm, CoraSgemm, CiteseerSgemm, RedditSgemm], 'ElementWise':[PubElement, CoraElement, CiteseerElement, RedditElement], 'Reduce'[None, None, None, RedditReduce],
# 'Index':[])
# print(SAG_Reddit_128.keys(), SAG_Reddit_128.values())
axs[0].pie(SAG_Reddit_128.values(), labels=SAG_Reddit_128.keys(), autopct='%.1f%%')
# plt.show()
IMC_Resnet50x4 = {}
IMC_Resnet50x4['GEMM'] = 68.2
IMC_Resnet50x4['BatchNormalization'] = 16.1
IMC_Resnet50x4['Activation'] = 12.9
IMC_Resnet50x4['etc'] = 100 - sum(IMC_Resnet50x4.values())
# SAG_Reddit_128 = pd.DataFrame(SAG_Reddit_128)


axs[1].pie(IMC_Resnet50x4.values(), labels=IMC_Resnet50x4.keys(), autopct='%.1f%%')
plt.show()

# SAG_Cora_128 = {}
# SAG_Cora_128['ScatterGather'] = 23.1
# SAG_Cora_128['IndexSelect'] = 29.9
# SAG_Cora_128['Sgemm'] = 34.6
# # print(SAG_Reddit_128.keys(), SAG_Reddit_128.values())
#
# axs[1].pie(SAG_Cora_128.values(), labels=SAG_Cora_128.keys(), autopct='%.1f%%')
# plt.show()

#
# width = 0.35
# # GPU Utilization: IndexSelect, ScatterGather, Sgemm / Reddit
# SM = [38.8, 39.2, 74.8]
# Memory = [70.4, 72.5, 37.1]
# Index1 = ['IndexSelect', 'ScatterGather', 'Sgemm']
# df_util = pd.DataFrame({'SM': SM, 'Memory': Memory, 'Index': Index1})
# fig1 = df_util.plot(kind='bar')
# fig1.set_title('GPU Utilization(%)', size=20)
# fig1.set_xticklabels(df_util['Index'], rotation=0)
# fig1.set_ylabel('Utilization(%)', size='large')
# fig1.set_ylim([0,100])
# plt.show()

# # GPU Utilization: IndexSelect, ScatterGather, Sgemm / Reddit
SM = [38.8, 39.2, 74.8]
Memory = [70.4, 72.5, 37.1]
Index1 = ['IndexSelect', 'ScatterGather', 'Sgemm']
df_util = pd.DataFrame({'SM': SM, 'Memory': Memory, 'Index': Index1})
fig1 = df_util.plot(kind='bar')
fig1.set_title('GPU Utilization(%)', size=20)
fig1.set_xticklabels(df_util['Index'], rotation=0)
fig1.set_ylabel('Utilization(%)', size='large')
fig1.set_ylim([0,100])
plt.show()
#
# # Compute Workload Analysis, Pipe Utilization(%)
# ALU   XU    FMA   LSU
IndexSelect1 = [34.81, 32.13, 28.56, 10.71]
Scatter1 = [35.83, 0, 19.21, 19.21]
Sgemm1 = [6.62, 0, 80.32, 36.99]
Index2 = ['ALU', 'XU', 'FMA', 'LSU']
df_pipe = pd.DataFrame({'IndexSelect': IndexSelect1, 'Scatter': Scatter1, 'Sgemm': Sgemm1, 'Index':Index2})
fig2 = df_pipe.plot(kind='barh', width=0.8)
fig2.set_title('Pipe Utilization(%)', size=20)
fig2.set_yticklabels(df_pipe['Index'], rotation=0)
fig2.set_xlabel('Utilization(%)', size='large')
fig2.set_xlim([0,100])
plt.show()

# analysis stall reasons, Warp cycles per instruction -> latency between two consecutive instructions.
# stall long scoreboard / stall wait  /   stall short scoreboard  /   stall not selected    / stall math pipe throttle  / stall MIO Throttle / stall no instruction / stall barrier
IndexSelect2 = [round(a  / 17.59*100, 2) for a in [10.65, 3.39, 1.42, 0.73, 0.53, 0.28, 0.24, 0.0]]
# tmp = np.array([IndexSelect2, IndexSelect2]).squeeze(axis=1)
#
# print(tmp)
# Sum0 = [10.6, 3.39, 1.42, 0.73, 0.53, 0.28, 0.24, 0.0]
# sum(Sum0)
stall_data = [10.6, 3.39, 1.42, 0.73, 0.53, 0.28, 0.24, 0.0,
                                        13.17, 2.54, 0.1, 0.68, 0.42, 0.0, 0.86, 0.0,
                                        0.73, 0.82, 0.1, 1.53, 1.47, 1.04, 0.04, 0.61]
print( stall_data)
stall_index = ['IndexSelect' for i in range(0,8)]  #.append(), 'Scatter' for i in range(0,8), 'Sgemm' for i in range(0,8)]
for i in range(0, 8):
    stall_index.append("Scatter")

for i in range(0, 8):
    stall_index.append("Sgemm")
# print(stall_index)
# IndexSelect2 = round(IndexSelect2, 2)
# LongScoreBoard = (61.0, 10.0, 73.0)
# Wait = (19.0, 13.0, 14.0)
# ShortScoreboard = (8.0, 1.0, 1.0)
# NotSelected = (4.0, 24.0, 4.0)
# MathPipeThrottle = (3.0, 24.0, 2.0)
# MIOThrottle = (2.0, 16.0, 0)
# NoInstruction = (1.0, 1.0, 5.0)
# Barrier = (0, 9, 0)
# N = 3
# ind = np.arange(N)
# p1 = plt.bar(ind, LongScoreBoard)
# p2 = plt.bar(ind, Wait)
# p3 = plt.bar(ind, ShortScoreboard)
# p4 = plt.bar(ind, NotSelected)
# p5 = plt.bar(ind, MathPipeThrottle)
# p6 = plt.bar(ind, MIOThrottle)
# p7 = plt.bar(ind, NoInstruction)
# p8 = plt.bar(ind, Barrier)
# plt.legend((p1[0], p2[0], p3[0], p4[0], p5[0], p6[0], p7[0], p8[0]), ('Long Scoreboard', 'Wait', 'Short Scoreboard' , 'Not Selected', 'Math Pipe Throttle' ,'MIO Throttle', 'No Instruction' ,'Barrier'))
# plt.xticks(ind, ('IndexSelect', 'Scatter', 'Sgemm'))
# plt.ylim([0,100])
# plt.show()


Sgemm2 = [round(b  / 6.32, 2) for b in [0.66, 0.82, 0.09, 1.54, 1.49, 1.01, 0.04, 0.6]]
Scatter2 = [round(c  / 17.91, 2) for c in [13.03, 2.54, 0.1, 0.68, 0.42, 0.0, 0.86, 0.0]]
Column3 = ['Long\nScoreboard', 'Wait', 'Short\nScoreboard' , 'Not\nSelected', 'Math Pipe\nThrottle' ,'MIO\nThrottle', 'No\nInstruction' ,'Barrier',
           'Long\nScoreboard', 'Wait', 'Short\nScoreboard' , 'Not\nSelected', 'Math Pipe\nThrottle' ,'MIO\nThrottle', 'No\nInstruction' ,'Barrier',
           'Long\nScoreboard', 'Wait', 'Short\nScoreboard' , 'Not\nSelected', 'Math Pipe\nThrottle' ,'MIO\nThrottle', 'No\nInstruction' ,'Barrier']
# Data = [stall_index, Column3, stall_data]
# rows = zip(Data[0], Data[1], Data[2])
headers = ['Kernels', 'Warp States', 'Values']
df_stall = pd.DataFrame({'Kernels':stall_index, 'Columns': Column3, 'Values': stall_data})
print(df_stall)
df_pivot = df_stall.pivot(columns='Kernels', index='Columns').fillna(0)
print(df_pivot.all())
fig3 = df_pivot.plot(kind='barh', width = 1.0)
# df_pivot = df_stall.pivot(index='Kernels', columns='Warp States', values='Values')
# df_pivot = df_pivot[['Long Scoreboard', 'Wait', 'Short Scoreboard' , 'Not Selected', 'Math Pipe Throttle' ,'MIO Throttle', 'No Instruction' ,'Barrier']].copy()
# fig3 = df_pivot.plot.bar(stacked=True, legend='reverse', width=1.0)
# fig3 = df_pivot.plot.bar(stacked=True, legend='reverse', width=1.0)
fig3.legend(['IndexSelect', 'Scatter', 'Sgemm'])
# fig3 = df_pivot.plot.bar(stacked=True, legend='reverse', width=1.0)
fig3.set_title('Stalled Cycles', size=20)
# fig3.set_yticklabels(df_pipe['Index'], rotation=0))
fig3.set_xlim([0,15])
plt.show()

# # MemThroughput, L1Hit, L2Hit, ExecutedIPC
IndexSelect = [308.62, 15.65, 57.66, 1.61]
Scatter = [314.77, 9.42, 51.08, 1.57]
Sgemm = [129.3, 2.08, 90.05, 1.94]
Index = ['DRAMThroughput', 'L1Hit', 'L2Hit', 'ExecutedIPC']
df_hwChar0 = pd.DataFrame({'IndexSelect': [IndexSelect[0]], 'Scatter': [Scatter[0]], 'Sgemm': Sgemm[0], 'Index':[Index[0]]})
df_hwChar1 = pd.DataFrame({'IndexSelect': [IndexSelect[1]], 'Scatter': [Scatter[1]], 'Sgemm': Sgemm[1], 'Index':[Index[1]]})
df_hwChar2 = pd.DataFrame({'IndexSelect': [IndexSelect[2]], 'Scatter': [Scatter[2]], 'Sgemm': Sgemm[2], 'Index':[Index[2]]})
df_hwChar3 = pd.DataFrame({'IndexSelect': [IndexSelect[3]], 'Scatter': [Scatter[3]], 'Sgemm': Sgemm[3], 'Index':[Index[3]]})
# print(df_hwChar)
fig, axes = plt.subplots(1, 4)
ax1 = df_hwChar0.plot(kind='bar', width=0.7, ax=axes[0], legend=False)
ax2 = df_hwChar1.plot(kind='bar', width=0.7, ax=axes[1])
ax3 = df_hwChar2.plot(kind='bar', width=0.7, ax=axes[2], legend=False)
ax4 = df_hwChar3.plot(kind='bar', width=0.7, ax=axes[3], legend=False)
ax1.set_xlabel(Index[0]+"[GB/s]", size='large')
ax1.set_ylim([0,350])
ax2.set_xlabel(Index[1]+"[%]", size='large')
ax2.set_ylim([0,100])
ax3.set_xlabel(Index[2]+"[%]", size='large')
ax3.set_ylim([0,100])
ax4.set_xlabel(Index[3], size='large')
ax4.set_ylim([0,2.5])
# fig2 = df_pipe.plot(kind='barh', width=0.8)
# fig2.set_title('Pipe Utilization(%)', size=20)
# fig2.set_yticklabels(df_pipe['Index'], rotation=0)
# fig2.set_xlabel('Utilization(%)', size='large')
# fig2.set_xlim([0,100])
plt.show()


kernel_utilization = 28
width = 0.35
# # GPU Utilization: IndexSelect, ScatterGather, Sgemm
#
# SM = [38.8, 39.2, 74.8]
# Memory = [70.4, 72.5, 37.1]
# Index1 = ['IndexSelect', 'ScatterGather', 'Sgemm']
# df_util = pd.DataFrame({'SM': SM, 'Memory': Memory, 'Index': Index1})
# fig1 = df_util.plot(kind='bar')
# fig1.set_title('GPU Utilization(%)', size=20)
# fig1.set_xticklabels(df_util['Index'], rotation=0)
# fig1.set_ylabel('Utilization(%)', size='large')
# fig1.set_ylim([0,100])
# plt.show()