import numpy as np
import os
import pickle

protein_node2vec = {}
for path, dir_list, file_list in os.walk("../data/struct_feature"):
    for file in file_list:
        try:
            protein_name = file.split('.')[0]
            print(protein_name)
            # 尝试使用loadtxt加载文件
            try:
                data = np.loadtxt(os.path.join(path, file))
                # 检查数据的维度
                if len(data.shape) == 2 and data.shape[1] == 30:
                    protein_node2vec[protein_name] = data
                    print(data.shape)
                else:
                    print(f"警告：{file}的形状不是预期的(N, 30)。跳过。")
            except ValueError as e:
                print(f"错误：无法读取{file}，错误为：{e}")
                continue
        except Exception as e:
            print(f"处理{file}时出现未知错误：{e}")
            continue

# 检查是否有任何蛋白质成功加载
if len(protein_node2vec) == 0:
    print("警告：没有成功加载任何蛋白质特征！")
else:
    print(f"成功加载了{len(protein_node2vec)}个蛋白质的特征")
    with open('../processed_data/protein_node2vec','wb') as f:
        pickle.dump(protein_node2vec, f)
    print("成功保存到'../processed_data/protein_node2vec'")