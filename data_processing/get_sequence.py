'''
通过读取pdb文件, 输出蛋白质序列的onehot表示
'''
import numpy as np
import pandas as pd 
from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser
import os
import pickle
import warnings

# 忽略警告信息
warnings.filterwarnings("ignore", category=UserWarning, module="Bio.SeqIO.PdbIO")

def load_predicted_PDB(pdbfile):
    """从PDB文件中提取氨基酸序列"""
    try:
        # 尝试从PDB文件中解析序列
        records = list(SeqIO.parse(pdbfile, 'pdb-atom'))
        if records:
            return str(records[0].seq)
        else:
            print(f"警告: 无法从{pdbfile}提取序列")
            return None
    except Exception as e:
        print(f"错误: 处理文件{pdbfile}时发生错误: {e}")
        return None

def seq2onehot(seq):
    """将氨基酸序列转换为26维one-hot编码"""
    if seq is None:
        return None
        
    chars = ['-', 'D', 'G', 'U', 'L', 'N', 'T', 'K', 'H', 'Y', 'W', 'C', 'P',
             'V', 'S', 'O', 'I', 'E', 'F', 'X', 'Q', 'A', 'B', 'Z', 'R', 'M']
    vocab_size = len(chars)
    vocab_embed = dict(zip(chars, range(vocab_size)))

    # 转换词汇到one-hot，并处理未知字符
    vocab_one_hot = np.zeros((vocab_size, vocab_size), int)
    for _, val in vocab_embed.items():
        vocab_one_hot[val, val] = 1

    # 处理可能存在的未知字符
    embed_x = []
    for v in seq:
        if v in vocab_embed:
            embed_x.append(vocab_embed[v])
        else:
            # 对于未知字符，使用'-'的编码
            embed_x.append(vocab_embed['-'])
    
    try:
        seqs_x = np.array([vocab_one_hot[j, :] for j in embed_x])
        return seqs_x
    except Exception as e:
        print(f"错误: 序列转换为one-hot编码时发生错误: {e}")
        return None

def main():
    # 创建输出目录
    os.makedirs("../processed_data", exist_ok=True)
    
    # 读取蛋白质列表
    try:
        df = pd.read_csv("../data/protein_list.csv", sep=" ")
        list1 = df.values.tolist()
        protein_list = []
        for i in list1:
            protein_list.append(i[0])
        print(f"找到{len(protein_list)}个蛋白质")
    except Exception as e:
        print(f"错误: 读取蛋白质列表时发生错误: {e}")
        return
    
    # 设置PDB文件夹路径
    PDB_FOLDER = "E:/Struct2Go/data/pdb"
    if not os.path.exists(PDB_FOLDER):
        print(f"错误: PDB文件夹不存在: {PDB_FOLDER}")
        return
        
    # 处理PDB文件并生成one-hot编码
    protein_node2one_hot = {}
    processed_count = 0
    skipped_count = 0
    
    # 列出所有PDB文件
    pdb_files = [f for f in os.listdir(PDB_FOLDER) if f.endswith('.pdb')]
    total_files = len(pdb_files)
    print(f"共找到{total_files}个PDB文件")
    
    for i, file_name in enumerate(pdb_files):
        try:
            # 从文件名中提取蛋白质ID
            parts = file_name.split("-")
            if len(parts) >= 2:
                protein_id = parts[1]
            else:
                protein_id = file_name.split('.')[0]
                
            # 检查是否在蛋白质列表中
            if protein_id in protein_list:
                # 处理PDB文件
                pdb_path = os.path.join(PDB_FOLDER, file_name)
                seq = load_predicted_PDB(pdb_path)
                
                if seq:
                    one_hot = seq2onehot(seq)
                    if one_hot is not None:
                        protein_node2one_hot[protein_id] = one_hot
                        processed_count += 1
                        print(f"{protein_id} {one_hot.shape}")
                    else:
                        skipped_count += 1
                else:
                    skipped_count += 1
            
            # 每处理100个文件显示进度
            if (i+1) % 100 == 0 or i+1 == total_files:
                print(f"进度: {i+1}/{total_files} ({processed_count}成功, {skipped_count}跳过)")
                
        except KeyboardInterrupt:
            print("\n处理被中断。保存已处理的数据...")
            break
        except Exception as e:
            print(f"错误: 处理文件{file_name}时发生未知错误: {e}")
            skipped_count += 1
    
    # 保存结果
    try:
        print(f"处理完成。成功处理了{processed_count}个蛋白质的one-hot特征，跳过了{skipped_count}个文件")
        if processed_count > 0:
            with open('../processed_data/protein_node2onehot', 'wb') as f:
                pickle.dump(protein_node2one_hot, f)
            print(f"特征已保存到'../processed_data/protein_node2onehot'")
        else:
            print("警告: 没有成功处理任何蛋白质，不保存文件")
    except Exception as e:
        print(f"错误: 保存特征文件时发生错误: {e}")

if __name__ == "__main__":
    main()