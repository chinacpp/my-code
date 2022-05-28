import torch
from torch.optim.lr_scheduler import CyclicLR
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
    scheduler = CyclicLR(optimizer,
                         base_lr=0.1,
                         max_lr=1,
                         step_size_up=10,
                         step_size_down=10,
                         mode='triangular')

    learning_rates = [0.1]
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
    plt.title('CyclicLR')
    plt.grid()
    plt.show()




