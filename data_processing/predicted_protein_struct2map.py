import os
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser
import warnings
from tqdm import tqdm
import pickle
import time
import multiprocessing
import argparse
from functools import partial

# 忽略警告
warnings.filterwarnings("ignore")

# 配置
PDB_FOLDER = "E:/Struct2Go/data/pdb"
OUTPUT_FOLDER = "../data/proteins_edges"
CHECKPOINT_FILE = "pdb_processing_checkpoint.pkl"

# 函数定义（保持原样）
def _load_cmap(filename, cmap_thresh=10.0):
    if filename.endswith('.pdb'):
        try:
            D, seq = load_predicted_PDB(filename)
            A = np.double(D < cmap_thresh)
        except Exception as e:
            raise Exception(f"Error in _load_cmap: {e}")
    else:
        raise Exception(f"File {filename} is not a PDB file")
    
    S = seq2onehot(seq)
    S = S.reshape(1, *S.shape)
    A = A.reshape(1, *A.shape)
    return A, S, seq

def load_predicted_PDB(pdbfile):
    try:
        parser = PDBParser()
        structure = parser.get_structure(pdbfile.split('/')[-1].split('.')[0], pdbfile)
        residues = [r for r in structure.get_residues()]
        
        # 检查是否有CA原子
        for r in residues:
            if 'CA' not in r:
                raise Exception(f"Residue {r} does not have CA atom")
        
        # 序列从原子线获取
        records = SeqIO.parse(pdbfile, 'pdb-atom')
        seqs = [str(r.seq) for r in records]
        if not seqs:
            raise Exception("No sequences found in PDB file")
        
        distances = np.empty((len(residues), len(residues)))
        for x in range(len(residues)):
            for y in range(len(residues)):
                one = residues[x]["CA"].get_coord()
                two = residues[y]["CA"].get_coord()
                distances[x, y] = np.linalg.norm(one-two)
        
        return distances, seqs[0]
    except Exception as e:
        raise Exception(f"Error in load_predicted_PDB: {e}")

def seq2onehot(seq):
    chars = ['-', 'D', 'G', 'U', 'L', 'N', 'T', 'K', 'H', 'Y', 'W', 'C', 'P',
             'V', 'S', 'O', 'I', 'E', 'F', 'X', 'Q', 'A', 'B', 'Z', 'R', 'M']
    vocab_size = len(chars)
    vocab_embed = dict(zip(chars, range(vocab_size)))
    
    # 转换词汇到one-hot
    vocab_one_hot = np.zeros((vocab_size, vocab_size), int)
    for _, val in vocab_embed.items():
        vocab_one_hot[val, val] = 1
    
    # 处理可能包含未知字符的序列
    embed_x = []
    for v in seq:
        if v in vocab_embed:
            embed_x.append(vocab_embed[v])
        else:
            # 对于未知字符，使用'-'的索引
            embed_x.append(vocab_embed['-'])
    
    seqs_x = np.array([vocab_one_hot[j, :] for j in embed_x])
    return seqs_x

# 处理单个文件的函数
def process_file(file_name, pdb_folder=PDB_FOLDER, output_folder=OUTPUT_FOLDER):
    try:
        full_path = os.path.join(pdb_folder, file_name)
        
        # 提取蛋白质ID
        parts = file_name.split("-")
        if len(parts) >= 2:
            protein_id = parts[1]
        else:
            protein_id = file_name.split('.')[0]
        
        # 处理PDB文件
        A, S, seqres = _load_cmap(full_path, cmap_thresh=10.0)
        B = np.reshape(A, (-1, len(A[0])))
        
        # 构建接触图
        result = []
        N = len(B)
        for i in range(N):
            for j in range(N):
                if B[i][j] and i != j:
                    result.append([i, j])
        
        # 保存结果
        output_path = os.path.join(output_folder, f"{protein_id}.txt")
        data = pd.DataFrame(result)
        if not data.empty:
            data.to_csv(output_path, sep=" ", index=False, header=False)
            return (file_name, True, "")
        else:
            return (file_name, False, "Empty contact graph")
                
    except Exception as e:
        return (file_name, False, str(e))

# 主程序
def main():
    parser = argparse.ArgumentParser(description='处理PDB文件并生成蛋白质接触图')
    parser.add_argument('--batch', type=int, default=1000, help='每批处理的文件数量')
    parser.add_argument('--processes', type=int, default=os.cpu_count(), help='使用的进程数量')
    parser.add_argument('--resume', action='store_true', help='是否从断点继续')
    args = parser.parse_args()
    
    # 确保输出文件夹存在
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # 检查是否有断点续传文件
    processed_files = set()
    if args.resume and os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'rb') as f:
                processed_files = pickle.load(f)
            print(f"从断点继续。已处理 {len(processed_files)} 个文件。")
        except Exception as e:
            print(f"加载断点时出错: {e}。从头开始处理。")
            processed_files = set()
    
    # 获取所有PDB文件列表
    all_files = [f for f in os.listdir(PDB_FOLDER) if f.endswith('.pdb')]
    print(f"找到 {len(all_files)} 个PDB文件。")
    
    # 过滤掉已处理的文件
    remaining_files = [f for f in all_files if f not in processed_files]
    print(f"剩余 {len(remaining_files)} 个文件需要处理。")
    
    # 确定进程数量
    processes = min(args.processes, os.cpu_count())
    print(f"将使用 {processes} 个进程并行处理。")
    
    # 分批处理
    batch_size = args.batch
    total_batches = (len(remaining_files) + batch_size - 1) // batch_size
    
    error_files = []
    
    try:
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(remaining_files))
            batch_files = remaining_files[start_idx:end_idx]
            
            print(f"正在处理第 {batch_idx+1}/{total_batches} 批 ({len(batch_files)} 个文件)...")
            
            # 使用多进程处理批次
            with multiprocessing.Pool(processes=processes) as pool:
                process_func = partial(process_file, pdb_folder=PDB_FOLDER, output_folder=OUTPUT_FOLDER)
                results = list(tqdm(pool.imap(process_func, batch_files), total=len(batch_files)))
            
            # 处理结果
            for file_name, success, error_msg in results:
                if success:
                    processed_files.add(file_name)
                else:
                    error_files.append((file_name, error_msg))
                    print(f"处理文件 {file_name} 时出错: {error_msg}")
            
            # 每批次保存一次断点
            with open(CHECKPOINT_FILE, 'wb') as f:
                pickle.dump(processed_files, f)
            
            print(f"批次 {batch_idx+1} 完成。当前已处理 {len(processed_files)} 个文件。")
    
    except KeyboardInterrupt:
        print("\n处理被用户中断。正在保存进度...")
    
    finally:
        # 保存断点
        with open(CHECKPOINT_FILE, 'wb') as f:
            pickle.dump(processed_files, f)
        
        # 保存错误文件列表
        if error_files:
            with open("pdb_processing_errors.txt", 'w') as f:
                for file_name, error in error_files:
                    f.write(f"{file_name}: {error}\n")
            print(f"有 {len(error_files)} 个文件处理出错。详情见 pdb_processing_errors.txt")
        
        print(f"共处理了 {len(processed_files)} 个文件。")
        print(f"断点信息已保存到 {CHECKPOINT_FILE}。")

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"总耗时: {elapsed_time/60:.2f} 分钟")