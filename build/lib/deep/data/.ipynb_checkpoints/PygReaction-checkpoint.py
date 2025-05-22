# dataset/PygReaction.py
import os
import os.path as osp
import csv
import re
from tqdm import tqdm
import torch
from torch_geometric.data import InMemoryDataset, Data
from rdkit import Chem
from rdkit.Chem import PeriodicTable
from sklearn.utils import shuffle
import pandas as pd
import torch

def read_xyz(file_path):
    """
    读取 xyz 文件，返回原子符号列表和坐标张量
    xyz 文件格式：
      第一行：原子数
      第二行：注释
      后续行：元素符号  x  y  z
    """
    atomic_symbols = []
    coords = []
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        if len(lines) < 3:
            raise ValueError(f"File {file_path} 格式错误：行数不足")
        natoms = int(lines[0].strip())
        if natoms != len(lines) - 2:
            print(f"警告：{file_path} 声明的原子数与实际行数不符")
        for line in lines[2:]:
            parts = line.split()
            if len(parts) < 4:
                continue
            atomic_symbols.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        coords = torch.tensor(coords, dtype=torch.float)
    except Exception as e:
        print(f"Error reading xyz file {file_path}: {e}")
        return None, None
    return atomic_symbols, coords

def symbols_to_atomic_numbers(symbols):
    """
    利用 RDKit 的 PeriodicTable 将原子符号转换为原子序数
    """
    pt = Chem.GetPeriodicTable()
    atomic_nums = []
    for s in symbols:
        try:
            atomic_nums.append(pt.GetAtomicNumber(s))
        except Exception as e:
            print(f"Error converting symbol {s} to atomic number: {e}")
            return None
    return torch.tensor(atomic_nums, dtype=torch.long)

class ReactionXYZDataset(InMemoryDataset):
    r"""
    数据集根目录下包含一个 CSV 文件（例如 DATASET_DA.csv），每一行记录一个反应信息，
    CSV 中字段包括：
      - ID: 反应标识，保存在 data.reaction_id
      - Rnber: 序号（可选）
      - R_dir: xyz 文件夹名称（例如 reaction_R0），保存在 data.id
      - reaction: 反应 SMILES 或其它描述
      - dG(ts): 目标值（预测值 y）
      
    对于每个 CSV 记录，程序将在数据集根目录下的对应文件夹中寻找三个 xyz 文件：
      - 去除文件夹名中 "reaction_" 前缀后的 {prefix}_reactant.xyz （作为输入0）
      - 去除 "reaction_" 前缀后的 {prefix}_ts.xyz       （作为输入1）
      - 去除 "reaction_" 前缀后的 {prefix}_product.xyz  （作为输入2）
      
    从 {prefix}_reactant.xyz 中提取原子符号（转换为原子序数 z），
    并分别从三个文件中提取原子坐标（pos0, pos1, pos2）。
    同时保存 CSV 中的 ID 到 data.reaction_id 和 R_dir 到 data.id。
    """
    def __init__(self, root, csv_file='DA_dataset_cleaned.csv', transform=None, pre_transform=None, pre_filter=None):
        self.csv_file = osp.join(root, csv_file)
        super(ReactionXYZDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_dir(self):
        return self.root

    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed')
    
    @property
    def raw_file_names(self):
        # raw 文件由 CSV 文件和各反应对应的文件夹构成，此处返回 CSV 文件名即可
        return [osp.basename(self.csv_file)]
    
    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def download(self):
        # 如有需要，可实现下载逻辑
        pass
    
    def process(self):
        data_list = []
        if not osp.exists(self.csv_file):
            raise FileNotFoundError(f"CSV 文件 {self.csv_file} 不存在")
        
        with open(self.csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        for row in tqdm(rows, desc="Processing reactions"):
            # 读取 CSV 中的各字段
            reaction_id = row.get('ID', '').strip()            # 反应标识
            R_dir = row.get('R_dir', '').strip()                 # 文件夹名称
            reaction_str = row.get('reaction', '').strip()
            dG_ts_str = row.get('dG(ts)', '').strip()
            if not reaction_id or not R_dir or not dG_ts_str:
                print(f"Warning: 缺少必要字段，跳过该记录：{row}")
                continue
            try:
                dG_ts = float(dG_ts_str)
            except Exception as e:
                print(f"Error parsing dG(ts) 值 {dG_ts_str} in reaction_id {reaction_id}: {e}")
                continue

            # 构造 xyz 文件夹路径
            folder_path = osp.join(self.raw_dir, R_dir)
            if not osp.isdir(folder_path):
                print(f"Warning: 文件夹 {folder_path} 不存在，跳过 reaction_id {reaction_id}")
                continue

            # 去掉文件夹名中的 "reaction_" 前缀（如果存在），用于构造文件名
            prefix = R_dir
            if prefix.startswith("reaction_"):
                prefix = prefix[len("reaction_"):]

            # 构造 xyz 文件路径
            reactant_file = osp.join(folder_path, f"{prefix}_reactant.xyz")
            ts_file       = osp.join(folder_path, f"{prefix}_ts.xyz")
            product_file  = osp.join(folder_path, f"{prefix}_product.xyz")
            if not (osp.exists(reactant_file) and osp.exists(ts_file) and osp.exists(product_file)):
                print(f"Warning: 在 {folder_path} 缺少一个或多个 xyz 文件，跳过 reaction_id {reaction_id}")
                continue

            # 读取反应物文件，提取原子符号和坐标
            symbols, pos0 = read_xyz(reactant_file)
            if symbols is None or pos0 is None:
                print(f"Warning: 读取 {reactant_file} 失败，跳过 reaction_id {reaction_id}")
                continue
            z = symbols_to_atomic_numbers(symbols)
            if z is None:
                print(f"Warning: 转换原子符号失败，跳过 reaction_id {reaction_id}")
                continue

            # 读取 ts 和 product 文件，提取坐标
            _, pos1 = read_xyz(ts_file)
            _, pos2 = read_xyz(product_file)
            if pos1 is None or pos2 is None:
                print(f"Warning: 读取 ts 或 product 文件失败，跳过 reaction_id {reaction_id}")
                continue
            # 检查原子数是否一致
            if not (pos0.size(0) == pos1.size(0) == pos2.size(0) == z.size(0)):
                print(f"Warning: 原子数不一致，跳过 reaction_id {reaction_id}")
                continue

            y = torch.tensor([dG_ts], dtype=torch.float)
            data = Data(z=z, pos0=pos0, pos1=pos1, pos2=pos2, y=y)
            # 保存属性：CSV 中的 ID 保存到 reaction_id，R_dir 保存到 id
            data.reaction_id = reaction_id
            data.id = R_dir
            data.reaction = reaction_str  # 可选：保存反应描述
            data.num_nodes = z.size(0)
            data_list.append(data)
        
        if len(data_list) == 0:
            raise RuntimeError("没有处理出任何反应数据，请检查 CSV 和 xyz 文件的格式。")
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


    # def get_idx_split(self, train_size, valid_size, seed):
    #     """
    #     数据集划分：按照 data.reaction_id 进行分组，保证同一 reaction_id 的所有样本被分到同一数据集，
    #     同时尽量满足各划分样本数的要求。
        
    #     参数：
    #       - train_size: 训练集样本数（按样本数量计，而非 group 数）
    #       - valid_size: 验证集样本数
    #       - seed: 随机种子
    #     返回：
    #       字典，包含 'train', 'valid', 'test' 三个键，其值为对应样本的索引张量。
    #     """
    #     # 以 data.reaction_id 为分组依据
    #     group_to_indices = {}
    #     for idx, data in enumerate(self):
    #         key = data.reaction_id  # 使用 reaction_id 作为分组依据
    #         if key not in group_to_indices:
    #             group_to_indices[key] = []
    #         group_to_indices[key].append(idx)
        
    #     # 获取所有分组的 key，并随机打乱
    #     group_keys = list(group_to_indices.keys())
    #     group_keys = shuffle(group_keys, random_state=seed)
        
    #     train_idx = []
    #     valid_idx = []
    #     test_idx = []
    #     n_assigned = 0
        
    #     # 按顺序将整个 group 分配到 train、valid、test 中，直至满足样本数量要求
    #     for key in group_keys:
    #         group = group_to_indices[key]
    #         group_size = len(group)
            
    #         if n_assigned + group_size <= train_size:
    #             train_idx.extend(group)
    #             n_assigned += group_size
    #         elif n_assigned + group_size <= train_size + valid_size:
    #             valid_idx.extend(group)
    #             n_assigned += group_size
    #         else:
    #             test_idx.extend(group)
    #             n_assigned += group_size
    
    #     # 检查划分后的数据集，确保每个 reaction_id 都只出现在一个数据集中
    #     reaction_id_to_split = {}
        
    #     # 记录每个 reaction_id 被分配到的数据集
    #     for idx in train_idx:
    #         reaction_id = self[idx].reaction_id
    #         reaction_id_to_split[reaction_id] = 'train'
    #     for idx in valid_idx:
    #         reaction_id = self[idx].reaction_id
    #         reaction_id_to_split[reaction_id] = 'valid'
    #     for idx in test_idx:
    #         reaction_id = self[idx].reaction_id
    #         reaction_id_to_split[reaction_id] = 'test'
        
    #     # 检查每个 reaction_id 是否出现在多个数据集中
    #     for reaction_id, split in reaction_id_to_split.items():
    #         indices = group_to_indices[reaction_id]
    #         assigned_splits = {reaction_id_to_split.get(idx, None) for idx in indices}
    #         if len(assigned_splits) > 1:
    #             raise ValueError(f"Reaction ID {reaction_id} has been split across multiple datasets: {assigned_splits}")
        
    #     # 保存划分结果到 CSV 文件
    #     split_data = []
    #     splits = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}
    #     for split_name, indices in splits.items():
    #         for idx in indices:
    #             data = self[idx]
    #             split_data.append({'reaction_id': data.reaction_id, 'id': data.id, 'split': split_name})
        
    #     # 转换为 DataFrame 并保存为 CSV 文件
    #     df = pd.DataFrame(split_data)
    #     df.to_csv('./split_data.csv', index=False)
    #     print(f"划分结果已保存到 'split_data.csv'。")
        
    #     # 返回划分后的索引
    #     train_idx = torch.tensor(train_idx, dtype=torch.long)
    #     valid_idx = torch.tensor(valid_idx, dtype=torch.long)
    #     test_idx  = torch.tensor(test_idx, dtype=torch.long)
        
    #     return {'train': train_idx, 'valid': valid_idx, 'test': test_idx}


    def get_idx_split(self, train_size, valid_size, seed):
        """
        数据集划分：按照 data.id (即 R_dir) 进行分组，保证同一 reaction_id 的样本（通常来自同一文件夹）被分到同一数据集，
        同时尽量满足各划分样本数的要求。
        
        参数：
          - train_size: 训练集样本数（按样本数量计，而非 group 数）
          - valid_size: 验证集样本数
          - seed: 随机种子
        返回：
          字典，包含 'train', 'valid', 'test' 三个键，其值为对应样本的索引张量。
        """
        # 以 data.id（即 R_dir）为分组依据
        group_to_indices = {}
        for idx, data in enumerate(self):
            key = data.id
            if key not in group_to_indices:
                group_to_indices[key] = []
            group_to_indices[key].append(idx)
        
        # 获取所有分组的 key，并随机打乱
        group_keys = list(group_to_indices.keys())
        group_keys = shuffle(group_keys, random_state=seed)
        
        train_idx = []
        valid_idx = []
        test_idx = []
        n_assigned = 0
        
        # 按顺序将整个 group 分配到 train、valid、test 中，直至满足样本数量要求
        for key in group_keys:
            group = group_to_indices[key]
            if n_assigned < train_size:
                train_idx.extend(group)
            elif n_assigned < train_size + valid_size:
                valid_idx.extend(group)
            else:
                test_idx.extend(group)
            n_assigned += len(group)
        
        train_idx = torch.tensor(train_idx, dtype=torch.long)
        valid_idx = torch.tensor(valid_idx, dtype=torch.long)
        test_idx  = torch.tensor(test_idx, dtype=torch.long)
        return {'train': train_idx, 'valid': valid_idx, 'test': test_idx}

if __name__ == '__main__':
    # 请确保数据集根目录中包含 CSV 文件和相应的 xyz 文件夹
    dataset = ReactionXYZDataset(root='DATASET_DA')
    print(dataset)
    print(f"Total reactions: {len(dataset)}")
    data0 = dataset[0]
    print("reaction_id (CSV中的ID):", data0.reaction_id)
    print("id (R_dir):", data0.id)
    print("dG(ts):", data0.y)
    print("Atomic numbers (z):", data0.z)
    print("Reactant coordinates (pos0):", data0.pos0)
    print("TS coordinates (pos1):", data0.pos1)
    print("Product coordinates (pos2):", data0.pos2)
