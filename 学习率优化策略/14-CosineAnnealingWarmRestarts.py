import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.optim as optim
import torch.nn as nn
import torch
import matplotlib.pyplot as plt


if __name__ == '__main__':

    # 模型参数
    model_parameters = [nn.Parameter(torch.tensor([1, 2, 3], dtype=torch.float32))]
    # 优化器
    optimizer = optim.SGD(model_parameters, lr=0.1)
    # 调度器
    # T_0 表示第 10 epoch 时，将学习率回归到初始学习率
    # T_mult=2时，表示2倍变化。第二次是 2*10，第三次是 4*10，第四次是 8*10
    scheduler = CosineAnnealingWarmRestarts(optimizer,
                                            T_0=10,
                                            T_mult=2,
                                            eta_min=0.01)

    learning_rates = [0.1]
    for _ in range(100):
        # 先优化器更新
        optimizer.step()
        # 再调度器更新
        scheduler.step()
        # 记录此刻学习率
        learning_rates.append(scheduler.get_last_lr()[0])

    # 横轴损失，纵轴学习率
    plt.plot(range(101), learning_rates)
    plt.title('CosineAnnealingWarmRestarts')
    plt.xticks(range(0, 100, 5))
    plt.grid()
    plt.show()




