import torch
from torch.optim.lr_scheduler import LinearLR
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
    scheduler = LinearLR(optimizer=optimizer, start_factor=0.001, end_factor=0.10, total_iters=50)

    learning_rates = []
    for _ in range(100):
        # 先优化器更新
        optimizer.step()
        # 再调度器更新
        scheduler.step()
        # 存储学习率
        print('%.5f' % scheduler.get_last_lr()[0])
        learning_rates.append(scheduler.get_last_lr()[0])

    # 绘制学习率变化
    plt.plot(range(100), learning_rates)
    plt.title('LinearLR')
    plt.grid()
    plt.show()
