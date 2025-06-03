import torch
import torch.nn as nn

class MatrixFactorization(nn.Module):
    """
    简单的矩阵分解模型。
    """
    def __init__(self, num_users, num_items, embedding_dim):
        super(MatrixFactorization, self).__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

        # 初始化嵌入层权重
        self.user_embeddings.weight.data.uniform_(-0.05, 0.05)
        self.item_embeddings.weight.data.uniform_(-0.05, 0.05)
        # print(f"MF Model Initialized: Users={num_users}, Items={num_items}, Dim={embedding_dim}")


    def forward(self, user_ids, item_ids):
        """
        前向传播。
        Args:
            user_ids (torch.Tensor): 用户ID张量, shape (batch_size, 1) or (batch_size,).
            item_ids (torch.Tensor): 物品ID张量, shape (batch_size, 1) or (batch_size,).
        Returns:
            torch.Tensor: 预测评分, shape (batch_size, 1).
        """
        user_embedded = self.user_embeddings(user_ids) 
        item_embedded = self.item_embeddings(item_ids)

        if user_embedded.dim() == 3: # (batch_size, 1, embedding_dim)
            user_embedded = user_embedded.squeeze(1) # (batch_size, embedding_dim)
        if item_embedded.dim() == 3: # (batch_size, 1, embedding_dim)
            item_embedded = item_embedded.squeeze(1) # (batch_size, embedding_dim)
        
        # 点积操作
        prediction = torch.sum(user_embedded * item_embedded, dim=1, keepdim=True)
        return prediction