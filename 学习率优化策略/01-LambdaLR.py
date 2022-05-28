import torch
from torch.optim.lr_scheduler import LambdaLR
import torch.optim as optim
import torch.nn as nn
import torch
import matplotlib.pyplot as plt


if __name__ == '__main__':

    # 模型参数
    model_parameters = [nn.Parameter(torch.tensor([1, 2, 3], dtype=torch.float32))]
    # 优化器
    optimizer = optim.Adam(model_parameters, lr=0.1)
    # 调度器
    scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.9 ** epoch)

    learning_rates = [0.1]
    for _ in range(100):
        # 先优化器更新
        optimizer.step()
        # 再调度器更新
        scheduler.step()
        # 存储学习率
        learning_rates.append(scheduler.get_last_lr()[0])
        print('%.10f' % scheduler.get_last_lr()[0])

    # 绘制学习率变化
    plt.plot(range(101), learning_rates)
    plt.title('LambdaLR')
    plt.grid()
    plt.show()

