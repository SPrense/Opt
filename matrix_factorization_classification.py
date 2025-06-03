import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from trak import TRAKer  # 新增导入TRAKer

def load_data():
    # 读取用户、电影和评分数据
    n_users = sum(1 for _ in open('users.dat'))
    n_items = sum(1 for _ in open('movies.dat'))
    
    # 创建评分矩阵
    ratings = torch.zeros((n_users, n_items), dtype=torch.long)
    with open('ratings.dat') as f:
        for line in f:
            user, item, rating, _ = line.strip().split('::')
            ratings[int(user)-1, int(item)-1] = int(rating) - 1  # 转换为0-4范围
    return n_users, n_items, ratings

class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, n_factors=20):
        super(MatrixFactorization, self).__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)
        self.classifier = nn.Linear(n_factors, 5)
        
        # 添加TRAK相关参数
        self.trak_enabled = False
        self.feature_projection = None
        
    def enable_trak(self, proj_dim=100):
        """启用TRAK特征提取"""
        self.trak_enabled = True
        self.feature_projection = nn.Linear(n_factors, proj_dim)
        
    def forward(self, user, item):
        features = self.user_factors(user) * self.item_factors(item)
        
        if self.trak_enabled:
            # 返回原始特征和投影特征
            return {
                'logits': self.classifier(features),
                'features': features,
                'proj_features': self.feature_projection(features)
            }
        return self.classifier(features)

# 初始化WandB
use_wandb = True
if use_wandb:
    wandb.init(project="matrix-factorization-classification")
    wandb.config.update({"n_factors": 20, "learning_rate": 0.01})

# 加载真实数据
n_users, n_items, ratings = load_data()

# 模型、损失函数和优化器
model = MatrixFactorization(n_users, n_items)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 初始化TRAKer
traker = TRAKer(
    model=model,
    task="classification",
    train_set_size=(ratings > 0).sum().item(),  # 使用有效评分数量作为训练集大小
    proj_dim=100,
    save_dir="./trak_results",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# 训练过程
model.enable_trak(proj_dim=100)

# 首先训练模型
for epoch in range(100):
    total_loss = 0
    for user in range(n_users):
        for item in range(n_items):
            if ratings[user, item] > 0:
                optimizer.zero_grad()
                output = model(torch.tensor(user), torch.tensor(item))
                loss = criterion(output['logits'].unsqueeze(0),
                               torch.tensor([ratings[user, item]]))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
    
    avg_loss = total_loss / (ratings > 0).sum().item()
    print(f'Epoch {epoch+1}, Loss: {avg_loss}')
    if use_wandb:
        wandb.log({"epoch": epoch, "loss": avg_loss})
    
    # 每10个epoch保存一次checkpoint
    if epoch % 10 == 0:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }
        torch.save(checkpoint, f'mf_checkpoint_epoch{epoch}.pth')
        if use_wandb:
            wandb.save(f'mf_checkpoint_epoch{epoch}.pth')

# 计算TRAK特征
traker.load_checkpoint(model.state_dict(), model_id=0)
for user in range(n_users):
    for item in range(n_items):
        if ratings[user, item] > 0:
            batch = (torch.tensor(user), torch.tensor(item))
            traker.featurize(batch=batch, num_samples=1)
traker.finalize_features()

# 计算TRAK分数（假设我们想评估所有用户-物品对）
test_users = torch.arange(n_users)
test_items = torch.arange(n_items)
traker.start_scoring_checkpoint(
    exp_name="mf_evaluation",
    checkpoint=model.state_dict(),
    model_id=0,
    num_targets=n_users * n_items
)
for user in test_users:
    for item in test_items:
        batch = (user, item)
        traker.score(batch=batch, num_samples=1)
scores = traker.finalize_scores(exp_name="mf_evaluation")

if use_wandb:
    torch.save(model.state_dict(), "mf_classifier.pth")
    wandb.save("mf_classifier.pth")