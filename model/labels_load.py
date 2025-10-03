'''

'''
#导入依赖
import pandas as pd
import pickle 
import collections
import numpy as np
import torch
import dgl
import os
import torch
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

# 从PDB中读取seq数据（经过onehot处理）和结构数据，json中只用到了GO数据
# 这里的文件并不重要，你只需要导入将数据处理为一个字典，格式为 {"P01283":["GO:0002312","GO:0202312",...], }即可
with open("E:/Struct2Go/data/HUMAN_protein_info.json","r") as f:
    human_dict = json.load(f)
labels={}
for id in human_dict:
    labels[id] = human_dict[id]['go']

gos=[]
namespace=collections.defaultdict(str)
is_a=collections.defaultdict(list)
part=collections.defaultdict(list)
###根据规则来提取go term ，并依据其之间的依赖关系构建图谱
# 只取用is_a和part_of关系
# 边的方向：A is_a B  <=>  A -> B
# 同样，这里的文件也不重要，可以在Gene Ontology上下载
print("--------------1: go term processing")
with open('E:/Struct2Go/data/go-basic.obo','r') as fin:
    for line in fin:
        if '[Typedef]' in line:
            break
        if line[:5]=='id: G':
            line=line.strip().split()
            gos.append(line[1])
        elif line[:4]=='is_a':
            line=line.strip().split()
            is_a[gos[-1]].append(line[1])
        elif line[:4]=='rela' and 'part' in line:
            line=line.strip().split()
            part[gos[-1]].append(line[2])
        elif line[:5]=='names':
            line=line.strip().split()
            namespace[gos[-1]]=line[1]
# 把所有关系并入到is_a中，方便处理
for i in part:
    is_a[i].extend(part[i])
##划分子空间，每个子空间是一个集合
bp,mf,cc=set(),set(),set()
for i in namespace:
    if namespace[i]=='biological_process':
        bp.add(i)
    elif namespace[i]=='molecular_function':
        mf.add(i)
    elif namespace[i]=='cellular_component':
        cc.add(i)


print("--------------2: propagate labels")
# 使用暴力求go_term_set的传递闭包（TODO 可优化）
def propagate(l):
    while True:
        length=len(l)
        temp=[]
        for i in l:
            temp.extend(is_a[i])
        l.update(temp)
        if len(l)==length:
            return l

# 只选取有GO标签的蛋白质进行数据处理
# 对于每个蛋白质，求他们功能标签的传递闭包
pro_with_go={}
for i in tqdm(labels):
    if len(labels[i])>0:
        pro_with_go[i]=propagate(set(labels[i]))
print("protein_num:",len(labels),"->",len(pro_with_go))


print("--------------3: split protein set")
# TODO: 需要统一文件夹路径
df=pd.read_csv("../data/protein_list.csv",sep=" ")
tmp_list=df.values.tolist()
protein_list=[]
for i in tmp_list:
    protein_list.append(i[0])
print(len(protein_list))

label_bp=collections.defaultdict(list)
label_mf=collections.defaultdict(list)
label_cc=collections.defaultdict(list)

# 最终我们选取既有go又有struct_features的蛋白质
for i in pro_with_go:
    if i in protein_list:
        for j in pro_with_go[i]:
            if j in bp:
                label_bp[i].append(j)
            elif j in mf:
                label_mf[i].append(j)
            elif j in cc:
                label_cc[i].append(j)


print("--------------4: read all kinds of features")
# 处理蛋白质氨基酸序列的独热编码
# with open('../processed_data/protein_node2onehot','rb')as f:
#     protein_node2onehot = pickle.load(f)
# print("protein_node2onehot:",len(protein_node2onehot))
# 处理蛋白质氨基酸序列的node2vec编码 (论文方法)
with open('../processed_data/protein_node2vec','rb')as f:
    protein_node2vec = pickle.load(f)
print("protein_node2vec:",len(protein_node2vec))

# 处理蛋白质氨基酸序列的独热编码
try:
    with open('../processed_data/protein_node2onehot','rb') as f:
        protein_node2onehot = pickle.load(f)
    print("protein_node2onehot:", len(protein_node2onehot))
    
    # 合并node2vec和one-hot特征
    protein_combined_features = {}
    for protein_id in protein_node2vec:
        if protein_id in protein_node2onehot and protein_node2vec[protein_id].shape[0] == protein_node2onehot[protein_id].shape[0]:
            # 确保节点数量匹配
            combined_features = np.concatenate([protein_node2vec[protein_id], protein_node2onehot[protein_id]], axis=1)
            protein_combined_features[protein_id] = combined_features
    print("合并特征后的蛋白质数量:", len(protein_combined_features))
    
    # 使用合并的特征代替原始node2vec特征
    protein_node2vec = protein_combined_features
except Exception as e:
    print(f"无法加载或合并one-hot特征: {e}")
    print("将继续使用原始node2vec特征")

# 处理蛋白质氨基酸序列的SeqVec特征
with open('../processed_data/dict_sequence_feature','rb')as f:
    seqvec_feature_dic = pickle.load(f)
print("seqvec_feature_dic:",len(seqvec_feature_dic))

# 处理蛋白质的结构信息
graph_dic = {}
# 读取predicted_protein_struct2map.py处理出来的protein_edges
for path,dir_list,file_list in os.walk("../data/proteins_edges"):  
    for file_name in file_list: 
        trace = os.path.join(path, file_name)
        name = file_name.split(".")[0]
        if trace.endswith(".txt"):
            #graph_dic[name] = pd.read_csv(trace, names=['Src','Dst'],header=None, sep=" ")

            #5.22修改
            try:
                # 尝试使用正常方式读取
                graph_dic[name] = pd.read_csv(trace, names=['Src','Dst'], header=None, sep=" ", error_bad_lines=False, warn_bad_lines=True)
            except Exception as e:
                # 如果旧版pandas不支持error_bad_lines，使用try-except
                try:
                    # 对于更新版本的pandas (1.3.0+)
                    graph_dic[name] = pd.read_csv(trace, names=['Src','Dst'], header=None, sep=" ", on_bad_lines='skip')
                except Exception as e2:
                    print(f"警告: 无法读取文件 {trace}: {e2}. 尝试使用更强健的方法...")
                    try:
                        # 手动读取并过滤不符合格式的行
                        with open(trace, 'r') as f:
                            valid_lines = []
                            for i, line in enumerate(f):
                                parts = line.strip().split()
                                if len(parts) == 2:
                                    valid_lines.append(line)
                                else:
                                    print(f"警告: 跳过文件 {trace} 中的第 {i+1} 行: 格式不符 ({len(parts)} 字段)")
                        
                        # 从有效行创建DataFrame
                        from io import StringIO
                        graph_dic[name] = pd.read_csv(StringIO(''.join(valid_lines)), names=['Src','Dst'], header=None, sep=" ")
                    except Exception as e3:
                        print(f"错误: 依然无法读取文件 {trace}: {e3}")
                        # 创建一个空DataFrame作为替代
                        graph_dic[name] = pd.DataFrame(columns=['Src','Dst'])


print("graph_dic:",len(graph_dic))

def goterm2idx(term_set):
    term_dict={v:k for k,v in enumerate(term_set)}
    return term_dict

# 求出每个蛋白质筛选后的功能标签的multi-hot向量
def labels2onehot(protein2func_label,index):
    protein2func_onehot={}
    protein2func_label_filtered={}
    l=len(index)
    for i in protein2func_label:
        one_hot = [0]*l
        protein2func_label_filtered[i] = []
        for j in protein2func_label[i]:
            if j in index:
                one_hot[index[j]]=1
                protein2func_label_filtered[i].append(j)
        protein2func_onehot[i]=one_hot
    return protein2func_onehot,protein2func_label_filtered

# 根据bp、cc、mf三种GO标签进行分类
# label 表示需要处理的 {protein_id:[go_list]} 字典
# graph_node_feature_dic {protein_id:蛋白质接触图的点特征} 字典
# seq_feature_dic {protein_id:蛋白质序列的特征} 字典
# graph_dic {protein_id:蛋白质氨基酸接触图的边} 字典
def label_process(label,graph_node_feature_dic,seq_feature_dic,graph_dic,ns_type,go_dependency):
    print("now processing: ",ns_type)
    # 第一步: 过滤出go_term中出现次数大于thresh的标签
    print("----step 1----")
    counter=collections.Counter()
    for i in label:
        counter.update(label[i])
    tong=dict(counter)
    final_go=set()
    for i in tong:
        if ns_type=='bp' and tong[i]>=250:
            final_go.add(i)
        if ns_type!='bp' and tong[i]>=100:
            final_go.add(i)
    print("total_process",ns_type,"final_go_term_size",len(final_go))
    
    # 第二步：对筛选出来的go_term进行编号
    print("----step 2----")
    term2idx=goterm2idx(final_go)
    with open('../processed_data/'+ns_type+'_term2idx.json','w') as f:
        json.dump(term2idx,f,indent=4)

    # 第三步：求出每个蛋白质的功能标签序列对应的multihot向量, 并且把label筛选一遍
    # 其实这里用onehot并不严谨，应该称为multi-hot
    print("----step 3----")
    pro2func_multi_hot,pro2func_filtered = labels2onehot(label,term2idx)
    final_protein_list=list(pro2func_filtered.keys())

    # 第四步：作图统计
    # 统计每个list的长度
    print("----step 4----")
    lengths = [len(value) for value in pro2func_filtered.values()]
    plt.gca().set_prop_cycle(None)
    # 绘制直方图
    n, bins, patches = plt.hist(lengths, bins=[0, 100, 200, 300, 400, 500], edgecolor='black', facecolor='blue')  # 这里的bins定义了区间，您可以根据需要调整
    # 在每个柱子上标注数字
    for i in range(len(n)):
        plt.text(bins[i] + 0.5, n[i] + 0.2, str(int(n[i])), ha='center', va='bottom')
    plt.xlabel('Protein Numbers')
    plt.ylabel('GO term Number')
    plt.title(ns_type+'-go')
    plt.legend(loc='upper right')  # 显示图例
    plt.savefig('../processed_data/histogram_'+ns_type+'.png', format='png')
    # plt.show()

    # 将字典转换为一对一的键值对
    pairs = [(key, val) for key, values in pro2func_filtered.items() for val in values]
    # 创建DataFrame
    df = pd.DataFrame(pairs, columns=['Protein', ns_type+'-go'])
    # 保存为CSV文件
    df.to_csv('../processed_data/gos_'+ns_type+'.csv', index=False)

    # 第五步：输出完成处理之后的数据
    print("----step 5----")
    emb_graph = {}
    emb_seq_feature = {}
    emb_label = {}
    
    # 计算需要处理的蛋白质总数
    proteins_to_process = len(final_protein_list)
    batch_size = 1000  # 每批处理1000个蛋白质
    processed_count = 0
    
    # 分批处理蛋白质
    for batch_start in range(0, proteins_to_process, batch_size):
        batch_end = min(batch_start + batch_size, proteins_to_process)
        batch_proteins = final_protein_list[batch_start:batch_end]
        
        batch_emb_graph = {}
        batch_emb_seq_feature = {}
        batch_emb_label = {}
        
        for i in tqdm(batch_proteins):
            # 添加检查，确保蛋白质在所有字典中都存在
            if i not in graph_dic or i not in graph_node_feature_dic or i not in seq_feature_dic:
                continue
                
            # 构建蛋白质结构图，氨基酸作为节点，读取节点特征
            try:
                edges_data = graph_dic[i]
                src = edges_data['Src'].to_numpy()
                dst = edges_data['Dst'].to_numpy()
                g = dgl.graph((src, dst))
                g = dgl.add_self_loop(g)
                
                # 因为蛋白质数据有更新，部分数据长度已经无法匹配上了
                if graph_node_feature_dic[i].shape[0] != g.num_nodes():
                    continue
                    
                g.ndata['feature'] = torch.tensor(graph_node_feature_dic[i], dtype=torch.float32)
                
                # 功能标签的shape为[1,过滤后的标签数]
                multihot_go = torch.tensor(pro2func_multi_hot[i], dtype=torch.float32)
                multihot_go = torch.unsqueeze(multihot_go, 0)
                
                # 读取序列特征
                batch_emb_seq_feature[i] = torch.tensor(seq_feature_dic[i], dtype=torch.float32)
                batch_emb_graph[i] = g
                batch_emb_label[i] = multihot_go
                processed_count += 1
            except Exception as e:
                continue
        
        # 更新主字典
        emb_graph.update(batch_emb_graph)
        emb_seq_feature.update(batch_emb_seq_feature)
        emb_label.update(batch_emb_label)
        
        # 每批次保存一次，防止内存溢出
        print(f"已处理 {min(batch_end, processed_count)}/{proteins_to_process} 个蛋白质")
    
    print(ns_type, "dataset size:", len(emb_graph))
    
    # 分批保存数据，避免一次性写入大文件
    def save_in_batches(data_dict, file_path, batch_size=500):
        total = len(data_dict)
        keys = list(data_dict.keys())
        
        for i in range(0, total, batch_size):
            batch_keys = keys[i:min(i+batch_size, total)]
            batch_dict = {k: data_dict[k] for k in batch_keys}
            
            # 如果文件已存在，则追加；否则创建新文件
            mode = 'ab' if i > 0 else 'wb'
            with open(file_path, mode) as f:
                pickle.dump(batch_dict, f)
            
            print(f"保存进度: {min(i+batch_size, total)}/{total}")
    
    # 使用分批保存函数
    print(f"保存 {ns_type} 图数据...")
    save_in_batches(emb_graph, f'../processed_data/emb_graph_{ns_type}')
    
    print(f"保存 {ns_type} 序列特征...")
    save_in_batches(emb_seq_feature, f'../processed_data/emb_seq_feature_{ns_type}')
    
    print(f"保存 {ns_type} 标签数据...")
    save_in_batches(emb_label, f'../processed_data/emb_label_{ns_type}')
    
    # 第六步：输出GO graph
    print("----step 6----")
    # go_graph = dgl.DGLGraph()
    # go_graph = dgl.add_self_loop(go_graph)
    # go_graph.add_nodes(len(final_go))

    # term_to_idx = {term: idx for idx, term in enumerate(final_go)}
    # for child, parents in go_dependency.items():
    #     if child in term_to_idx:
    #         child_idx = term_to_idx[child]
    #         for parent in parents:
    #             if parent in term_to_idx:
    #                 parent_idx = term_to_idx[parent]
    #                 go_graph.add_edges(torch.tensor([child_idx]), torch.tensor([parent_idx]))

    # 5.29修改，首先收集所有边
    edges_to_add = []
    term_to_idx = {term: idx for idx, term in enumerate(final_go)}

    for child, parents in go_dependency.items():
        if child in term_to_idx:
            child_idx = term_to_idx[child]
            for parent in parents:
                if parent in term_to_idx:
                    parent_idx = term_to_idx[parent]
                    edges_to_add.append((child_idx, parent_idx))

    # 创建图
    if edges_to_add:
        src, dst = zip(*edges_to_add)
        go_graph = dgl.graph((src, dst), num_nodes=len(final_go))
    else:
        go_graph = dgl.graph(([], []), num_nodes=len(final_go))

    go_graph = dgl.add_self_loop(go_graph)

    with open('../processed_data/label_'+ns_type+'_network','wb')as f:
        pickle.dump(go_graph,f)  

print("--------------5: process all kinds of files")
label_process(label_bp,protein_node2vec,seqvec_feature_dic,graph_dic,"bp",is_a)
label_process(label_cc,protein_node2vec,seqvec_feature_dic,graph_dic,"cc",is_a)
label_process(label_mf,protein_node2vec,seqvec_feature_dic,graph_dic,"mf",is_a)
