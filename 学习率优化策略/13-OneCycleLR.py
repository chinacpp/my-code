import torch
from torch.optim.lr_scheduler import OneCycleLR
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
    # pct_start 学习率上升部分占比
    # total_steps 整个训练过程共有多少 step
    # max_lr / div_factor = 初始学习率
    # 初始学习率 / final_div_factor = 最终落脚的学习率
    # 过程从初始学习率到 max_lr 再到最终落脚的学习率，一个 cycle
    scheduler = OneCycleLR(optimizer,
                           max_lr=0.1,
                           pct_start=0.8,
                           total_steps=100,
                           div_factor=4,
                           final_div_factor=1e-1)

    print(0.1/4/1e-2)

    learning_rates = []
    for _ in range(100):
        # 先优化器更新
        optimizer.step()
        # 再调度器更新
        scheduler.step()
        # 记录此刻学习率
        learning_rates.append(scheduler.get_last_lr()[0])

    # 横轴损失，纵轴学习率
    plt.plot(range(100), learning_rates)
    plt.title('OneCycleLR')
    plt.grid()
    plt.show()




