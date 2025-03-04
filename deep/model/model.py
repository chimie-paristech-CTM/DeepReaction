"""
model/model.py
"""

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

# 从现有的代码中导入模型和组件
from .dimenet import DimeNet
from .dimenetplusplus import DimeNetPlusPlus
from .readout import ReadoutFactory
from .mlp import PredictionMLP


class MoleculeModel(nn.Module):
    def __init__(self, model_name: str = 'dimenet++', **model_kwargs):
        """
        初始化主模型接口，根据 model_name 参数选择加载不同的模型。

        Args:
            model_name (str): 模型名称，支持 'dimenet' 和 'dimenet++'（或 'dimenet_pp'）。
            **model_kwargs: 传递给对应模型构造函数的参数。
                常见参数包括：
                    - hidden_channels: 隐藏层维度
                    - out_channels: 输出维度（通常为节点的潜在表示维度）
                    - num_blocks: 交互块数目
                    - num_bilinear / int_emb_size / basis_emb_size / out_emb_channels: DimeNet++ 特有参数
                    - num_spherical, num_radial, cutoff, envelope_exponent 等
        """
        super(MoleculeModel, self).__init__()

        model_name = model_name.lower()
        if model_name == 'dimenet':
            # 调用原始 DimeNet 模型（请确保 model_kwargs 参数正确）
            self.model = DimeNet(**model_kwargs)
        elif model_name in ['dimenet++', 'dimenet_pp']:
            # 调用 DimeNet++ 模型
            self.model = DimeNetPlusPlus(**model_kwargs)
        else:
            raise ValueError(f"Unknown model_name: {model_name}. Supported: 'dimenet', 'dimenet++'.")

    def forward(self, *args, **kwargs):
        """
        forward 函数直接调用内部模型的 forward。

        参数说明：
            *args, **kwargs: 传递给具体模型的 forward 方法的参数。
        返回：
            模型的输出结果。
        """
        return self.model(*args, **kwargs)


class MoleculePredictionModel(nn.Module):
    """
    完整的分子预测模型，包括特征提取器、读出层和预测头部。
    这是一个统一的接口，可以组装不同的组件来构建完整的模型架构。
    """
    def __init__(
        self,
        base_model_name: str = 'dimenet++',
        readout_type: str = 'sum',
        max_num_atoms: int = 100,
        node_dim: int = 128,
        output_dim: int = 1,
        dropout: float = 0.0,
        use_layer_norm: bool = False,
        readout_kwargs: dict = None,
        base_model_kwargs: dict = None,
    ):
        """
        初始化分子预测模型。

        Args:
            base_model_name (str): 基础模型名称，支持 'dimenet' 和 'dimenet++'
            readout_type (str): 读出层类型，支持 'sum', 'mean', 'max', 'attention', 'set_transformer'
            max_num_atoms (int): 最大原子数量
            node_dim (int): 节点特征维度
            output_dim (int): 输出维度
            dropout (float): Dropout 比率
            use_layer_norm (bool): 是否使用层归一化
            readout_kwargs (dict): 读出层参数
            base_model_kwargs (dict): 基础模型参数
        """
        super(MoleculePredictionModel, self).__init__()
        
        # 初始化参数
        self.base_model_name = base_model_name
        self.readout_type = readout_type
        self.max_num_atoms = max_num_atoms
        self.node_dim = node_dim
        
        # 设置默认参数
        if readout_kwargs is None:
            readout_kwargs = {}
        
        if base_model_kwargs is None:
            base_model_kwargs = {}
        
        # 添加默认参数
        if 'out_channels' not in base_model_kwargs:
            base_model_kwargs['out_channels'] = node_dim
            
        # 初始化基础模型
        self.base_model = MoleculeModel(model_name=base_model_name, **base_model_kwargs)
        
        # 初始化读出层
        readout_params = {
            'node_dim': node_dim,
            'hidden_dim': readout_kwargs.get('hidden_dim', 128),
            'num_heads': readout_kwargs.get('num_heads', 4),
            'layer_norm': use_layer_norm,
            'num_sabs': readout_kwargs.get('num_sabs', 2)
        }
        self.readout = ReadoutFactory.create_readout(readout_type, **readout_params)
        
        # 初始化预测层
        self.prediction_mlp = PredictionMLP(
            input_dim=node_dim,
            output_dim=output_dim,
            hidden_dim=node_dim,
            dropout=dropout,
            use_layer_norm=use_layer_norm
        )
    
    def forward(self, pos0, pos1, pos2, atom_z, batch_mapping):
        """
        Forward pass through the model.
        
        Args:
            pos0: First position tensor
            pos1: Second position tensor
            pos2: Third position tensor
            atom_z: Atomic numbers
            batch_mapping: Batch mapping indices
            
        Returns:
            Tuple: (node_embeddings, graph_embeddings, predictions)
        """
        # Pass through DimeNet++
        _, node_embeddings = self.base_model(z=atom_z, pos0=pos0, pos1=pos1, pos2=pos2, batch=batch_mapping)
        
        # Convert to dense batch
        node_embeddings_dense, mask = to_dense_batch(node_embeddings, batch_mapping, 0, self.max_num_atoms)
        
        # Apply readout
        graph_embeddings = self.readout(node_embeddings_dense, mask)
        
        # Apply MLP for prediction
        predictions = torch.flatten(self.prediction_mlp(graph_embeddings))
        
        return node_embeddings_dense, graph_embeddings, predictions
    
    # 保留原有的forward方法作为一个别名，以保持兼容性
    def forward_legacy(self, z, pos0, pos1, pos2, batch):
        """
        完整的前向传播。

        Args:
            z (torch.Tensor): 原子类型
            pos0 (torch.Tensor): 第一种原子位置
            pos1 (torch.Tensor): 第二种原子位置
            pos2 (torch.Tensor): 第三种原子位置
            batch (torch.Tensor): 批处理索引

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                (节点嵌入, 图嵌入, 预测结果)
        """
        return self.forward(pos0, pos1, pos2, z, batch)


# 示例：如果需要添加其它模型，只需在此处扩展即可
if __name__ == '__main__':
    # 示例测试，构造一个简单的输入进行前向传播
    # 注意：这里的参数和输入数据需要根据具体情况调整
    dummy_atom_z = torch.randint(0, 95, (10,))
    dummy_pos = torch.rand((10, 3))
    dummy_batch = torch.zeros(10, dtype=torch.long)
    
    # 构造 DimeNet++ 模型参数（参数可根据实际需求修改）
    model_kwargs = {
        'hidden_channels': 128,
        'out_channels': 128,         # 这里输出作为节点嵌入，后续再进行 readout 和 MLP 层处理
        'num_blocks': 4,
        'int_emb_size': 64,
        'basis_emb_size': 8,
        'out_emb_channels': 256,
        'num_spherical': 7,
        'num_radial': 6,
        'cutoff': 5.0,
        'max_num_neighbors': 32,
        'envelope_exponent': 5,
        'num_before_skip': 1,
        'num_after_skip': 2,
        'num_output_layers': 3,
        'act': 'swish'
    }
    
    # 测试基本模型
    print("Testing basic model...")
    model = MoleculeModel(model_name='dimenet++', **model_kwargs)
    # 模拟传入三个不同的位置信息（例如不同视角）
    pos0 = dummy_pos.clone()
    pos1 = dummy_pos.clone()
    pos2 = dummy_pos.clone()
    
    # 调用 forward（这里的返回值与具体模型实现有关）
    _, predictions = model(z=dummy_atom_z, pos0=pos0, pos1=pos1, pos2=pos2, batch=dummy_batch)
    print("Forward output:", predictions.shape)
    
    # 测试完整预测模型
    print("\nTesting complete prediction model...")
    full_model = MoleculePredictionModel(
        base_model_name='dimenet++',
        readout_type='sum',
        max_num_atoms=20,
        node_dim=128,
        base_model_kwargs=model_kwargs
    )
    node_emb, graph_emb, pred = full_model(
        pos0=pos0, pos1=pos1, pos2=pos2, atom_z=dummy_atom_z, batch_mapping=dummy_batch
    )
    print("Node embeddings shape:", node_emb.shape)
    print("Graph embeddings shape:", graph_emb.shape)
    print("Predictions shape:", pred.shape)