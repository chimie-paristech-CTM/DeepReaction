import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler


from .PygReaction import ReactionXYZDataset


def train_scaler(ds_list):
    """
    根据加载好的 Reaction 数据列表 ds_list（每个元素都是一个 Data 对象），
    取出所有样本的 y，拟合并返回一个 sklearn 的 StandardScaler。
    """
    ys = np.array([data.y.item() for data in ds_list]).reshape(-1, 1)
    scaler = StandardScaler().fit(ys)
    return scaler


def scale_reaction_dataset(ds_list, scaler):
    """
    将 Reaction 数据列表 ds_list（每个元素都是一个 Data 对象）的 y 做标准化。
    注意这里保留原有的 pos0, pos1, pos2，不做坐标归一化，只对标量目标值 y 做缩放。
    返回一个新的列表，其每个元素都是复制并替换了 y 的 Data 对象。
    """
    ys = np.array([data.y.item() for data in ds_list]).reshape(-1, 1)
    ys_scaled = scaler.transform(ys) if scaler else ys

    new_data_list = []
    for i, data in enumerate(ds_list):
        d = Data(
            # 保留原子序数
            z=data.z,
            # 保留三份坐标
            pos0=data.pos0,
            pos1=data.pos1,
            pos2=data.pos2,
            # 替换目标值
            y=torch.tensor(ys_scaled[i], dtype=torch.float),
            # PyG需要的 num_nodes
            num_nodes=data.num_nodes
        )
        # 如果你还想保留 reaction_id、id (R_dir)、reaction等字段，可手动拷贝
        if hasattr(data, 'reaction_id'):
            d.reaction_id = data.reaction_id
        if hasattr(data, 'id'):
            d.id = data.id
        if hasattr(data, 'reaction'):
            d.reaction = data.reaction
        new_data_list.append(d)
    
    return new_data_list


def load_reaction(
    random_seed: int,
    root: str,
    csv_file: str = 'DA_dataset_cleaned.csv',
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    use_scaler: bool = False,
):
    """
    类似 load_qm9.py 的函数，用来加载 Reaction 数据集，划分 train/val/test 并对目标值做标准化。

    参数：
    --------
    random_seed : int
        用于数据划分的随机种子。
    root : str
        数据集根目录，其中应包含 csv_file 和对应 xyz 文件夹（详见 PygReaction.py 的说明）。
    csv_file : str
        CSV 文件名，默认 'DA_dataset_cleaned.csv'。
    train_ratio : float
        训练集所占数据集比例。
    val_ratio : float
        验证集所占数据集比例。
      （剩余部分归到 test 集）
    
    返回：
    --------
    (train_scaled, val_scaled, test_scaled, scaler)
      - train_scaled, val_scaled, test_scaled: list[Data]
        分别为训练/验证/测试集的 Data 对象列表（对 y 做了标准化）
      - scaler: 训练集上拟合得到的 StandardScaler 对象，可对预测值做逆向变换
    """

    # 可根据需要对 random_seed 做固定或断言
    # assert random_seed in [844249, 787755, 420455, 700990, 791796]
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # 1) 加载自定义 ReactionXYZDataset
    dataset = ReactionXYZDataset(root=root, csv_file=csv_file)

    # 2) 按照比例计算 train_size, valid_size
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    # 剩下的就是 test

    # 3) 使用 PygReaction.py 中提供的 get_idx_split 进行分割
    splits = dataset.get_idx_split(train_size, val_size, seed=random_seed)
    train_indices = splits['train']
    val_indices   = splits['valid']
    test_indices  = splits['test']

    # 4) 根据索引取出对应数据
    train_data = [dataset[i] for i in train_indices]
    val_data   = [dataset[i] for i in val_indices]
    test_data  = [dataset[i] for i in test_indices]

    # 5) 用训练集数据拟合一个针对 y 的 StandardScaler

    # print(f"dataset use_scaler:::::{use_scaler}\n\n")
    scaler = train_scaler(train_data) if use_scaler else None

    # 6) 分别对 train/val/test 做 y 的缩放
    train_scaled = scale_reaction_dataset(train_data, scaler)
    val_scaled   = scale_reaction_dataset(val_data, scaler)
    test_scaled  = scale_reaction_dataset(test_data, scaler)

    return train_scaled, val_scaled, test_scaled, scaler


if __name__ == "__main__":
    # 使用示例
    train_set, val_set, test_set, sc = load_reaction(
        random_seed=42,
        root='./DATASET_DA',
        csv_file='./DATASET_DA/DA_dataset_cleaned.csv'
    )
    print(f"Train: {len(train_set)}  Val: {len(val_set)}  Test: {len(test_set)}")
    print("示例样本的y值(已标准化)：", train_set[0].y.item())
