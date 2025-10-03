import pickle
import torch
from torch.utils.data import Dataset, DataLoader

class MyDataSet(Dataset):
    def __init__(self,emb_graph,emb_seq_feature,emb_label):
        super().__init__()
        self.list = list(emb_graph.keys())
        self.graphs = emb_graph
        self.seq_feature = emb_seq_feature
        self.label = emb_label

    def __getitem__(self,index): 
        protein = self.list[index] 
        graph = self.graphs[protein]
        seq_feature = self.seq_feature[protein]
        label = self.label[protein]

        return protein, graph, label, seq_feature 

    def __len__(self):
        return  len(self.list) 

# 修改 divide_data.py 中的数据加载部分
def load_data_in_batches(file_path):
    combined_dict = {}
    with open(file_path, 'rb') as f:
        while True:
            try:
                batch_dict = pickle.load(f)
                combined_dict.update(batch_dict)
            except EOFError:
                break
    return combined_dict

if __name__ == "__main__":
    ns_type_list = ['bp', 'mf', 'cc']
    for ns_type in ns_type_list:
        print("divide", ns_type, "dataset")
        
        # 使用新函数加载分批保存的数据
        emb_graph = load_data_in_batches(f'../processed_data/emb_graph_{ns_type}')
        emb_seq_feature = load_data_in_batches(f'../processed_data/emb_seq_feature_{ns_type}')
        emb_label = load_data_in_batches(f'../processed_data/emb_label_{ns_type}')
        
        # 修改数据划分比例为8:1:1
        dataset = MyDataSet(emb_graph=emb_graph, emb_seq_feature=emb_seq_feature, emb_label=emb_label)
        train_size = int(len(dataset) * 0.8)  # 改为80%
        valid_size = int(len(dataset) * 0.1)  # 改为10%
        test_size = len(dataset) - train_size - valid_size  # 剩余10%
        
        print(f"Dataset sizes - Total: {len(dataset)}, Train: {train_size}, Valid: {valid_size}, Test: {test_size}")
        
        train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])
        
        with open(f'../divided_data/{ns_type}_train_dataset', 'wb') as f:
            pickle.dump(train_dataset, f)
        with open(f'../divided_data/{ns_type}_valid_dataset', 'wb') as f:
            pickle.dump(valid_dataset, f)
        with open(f'../divided_data/{ns_type}_test_dataset', 'wb') as f:
            pickle.dump(test_dataset, f)
            
        print("train dataset size", len(train_dataset))
        print("valid dataset size", len(valid_dataset))
        print("test dataset size", len(test_dataset))