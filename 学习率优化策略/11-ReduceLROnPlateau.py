import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
    # cooldown 学习修改之后，经过多少个 epoch 之后才进行正常的统计是否损失下降等
    # factor 当损失无法继续下降时，学习率 new_lr = lr * factor
    # patience 表示 patience epoch 仍然不降低损失则对学习率进行衰减
    scheduler = ReduceLROnPlateau(optimizer, factor=0.2, patience=2, cooldown=2, verbose=True)

    learning_rates = [0.1]
    torch.manual_seed(0)
    losses = torch.randint(1, 100, [100,]) / 100
    learning_rates = [0.1]
    for loss in losses:
        # 先优化器更新
        optimizer.step()
        # 再调度器更新
        scheduler.step(loss)
        # 记录此刻学习率
        learning_rates.append(scheduler.optimizer.param_groups[0]['lr'])

    # 横轴损失，纵轴学习率
    plt.plot(range(101), learning_rates)
    plt.title('ReduceLROnPlateau')
    plt.grid()
    plt.show()




