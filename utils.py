import torch
from torch import nn
from torch.nn import functional as F

# 计算准确率
def accuracy(outputs, labels):
    """
    计算模型预测的准确率
    :param outputs: 模型的输出
    :param labels: 真实标签
    :return: 准确率
    """
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

# 计算精确率
def precision(outputs, labels, num_classes):
    """
    计算模型预测的精确率
    :param outputs: 模型的输出
    :param labels: 真实标签
    :param num_classes: 类别数量
    :return: 平均精确率
    """
    _, preds = torch.max(outputs, dim=1)
    precision_scores = []
    for cls in range(num_classes):
        true_positives = torch.sum((preds == cls) & (labels == cls)).float()
        predicted_positives = torch.sum(preds == cls).float()
        precision_scores.append(true_positives / (predicted_positives + 1e-7))
    return torch.tensor(precision_scores).mean().item()


# 计算召回率
def recall(outputs, labels, num_classes):
    """
    计算模型预测的召回率
    :param outputs: 模型的输出
    :param labels: 真实标签
    :param num_classes: 类别数量
    :return: 平均召回率
    """
    _, preds = torch.max(outputs, dim=1)
    recall_scores = []
    for cls in range(num_classes):
        true_positives = torch.sum((preds == cls) & (labels == cls)).float()
        actual_positives = torch.sum(labels == cls).float()
        recall_scores.append(true_positives / (actual_positives + 1e-7))
    return torch.tensor(recall_scores).mean().item()

# 计算 F1 分数
def f1_score(outputs, labels, num_classes):
    """
    计算模型预测的 F1 分数
    :param outputs: 模型的输出
    :param labels: 真实标签
    :param num_classes: 类别数量
    :return: 平均 F1 分数
    """
    prec = precision(outputs, labels, num_classes)
    rec = recall(outputs, labels, num_classes)
    return 2 * (prec * rec) / (prec + rec + 1e-7)

# 训练步骤
def training_step(model, batch, device):
    """
    执行一个训练批次的前向传播和损失计算
    :param model: 模型
    :param batch: 一个批次的数据
    :param device: 设备（如 'cuda' 或 'cpu'）
    :return: 损失
    """
    images, labels = batch
    images, labels = images.to(device), labels.to(device)
    out, *_ = model(images)  # 生成预测
    loss = F.cross_entropy(out, labels)  # 计算损失
    return loss

# 验证步骤
def validation_step(model, batch, device, num_classes):
    """
    执行一个验证批次的前向传播，计算损失、准确率、精确率、召回率、F1 分数
    :param model: 模型
    :param batch: 一个批次的数据
    :param device: 设备（如 'cuda' 或 'cpu'）
    :param num_classes: 类别数量
    :return: 包含损失、准确率、精确率、召回率、F1 分数的字典
    """
    images, labels = batch
    images, labels = images.to(device), labels.to(device)
    out, *_ = model(images)  # 生成预测
    loss = F.cross_entropy(out, labels)  # 计算损失
    acc = accuracy(out, labels)  # 计算准确率
    prec = precision(out, labels, num_classes)
    rec = recall(out, labels, num_classes)
    f1 = f1_score(out, labels, num_classes)
    return {'Loss': loss.detach(), 'Acc': acc, 'Precision': prec, 'Recall': rec, 'F1': f1}

# 验证轮次结束处理
def validation_epoch_end(model, outputs):
    """
    汇总一个验证轮次的损失、准确率、精确率、召回率、F1 分数
    :param model: 模型
    :param outputs: 每个批次的验证结果
    :return: 包含汇总结果的字典
    """
    batch_losses = [x['Loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()  # 合并损失
    batch_accs = [x['Acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()  # 合并准确率
    batch_precs = [x['Precision'] for x in outputs]
    epoch_prec = sum(batch_precs) / len(batch_precs)
    batch_recs = [x['Recall'] for x in outputs]
    epoch_rec = sum(batch_recs) / len(batch_recs)
    batch_f1s = [x['F1'] for x in outputs]
    epoch_f1 = sum(batch_f1s) / len(batch_f1s)
    return {'Loss': epoch_loss.item(), 'Acc': epoch_acc.item(), 'Precision': epoch_prec, 'Recall': epoch_rec, 'F1': epoch_f1}

# 每个轮次结束时打印信息
def epoch_end(model, epoch, result):
    """
    打印每个训练轮次的训练和验证结果
    :param model: 模型
    :param epoch: 当前轮次
    :param result: 包含训练和验证结果的字典
    """
    print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}, val_prec: {:.4f}, val_rec: {:.4f}, val_f1: {:.4f}".format(
        epoch, result['lrs'][-1], result['train_loss'], result['Loss'], result['Acc'], result['Precision'], result['Recall'], result['F1']))

# 评估模型
@torch.no_grad()
def evaluate(model, val_loader, device='cuda', num_classes=10):
    """
    对模型在验证集上进行评估
    :param model: 模型
    :param val_loader: 验证集数据加载器
    :param device: 设备（如 'cuda' 或 'cpu'）
    :param num_classes: 类别数量
    :return: 包含评估结果的字典
    """
    model.eval()
    outputs = [validation_step(model, batch, device, num_classes) for batch in val_loader]
    return validation_epoch_end(model, outputs)

# 获取当前学习率
def get_lr(optimizer):
    """
    获取优化器当前的学习率
    :param optimizer: 优化器
    :return: 当前学习率
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

# 训练模型
def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader,
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD, device='cuda', num_classes=10):
    """
    使用 One Cycle 策略进行模型训练
    :param epochs: 训练轮次
    :param max_lr: 最大学习率
    :param model: 模型
    :param train_loader: 训练集数据加载器
    :param val_loader: 验证集数据加载器
    :param weight_decay: 权重衰减
    :param grad_clip: 梯度裁剪值
    :param opt_func: 优化器函数
    :param device: 设备（如 'cuda' 或 'cpu'）
    :param num_classes: 类别数量
    :return: 包含每个轮次训练和验证结果的历史记录
    """
    torch.cuda.empty_cache()
    history = []

    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    for epoch in range(epochs):
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            try:
                loss = training_step(model, batch, device)
                train_losses.append(loss)
                loss.backward()

                if grad_clip:
                    nn.utils.clip_grad_value_(model.parameters(), grad_clip)

                optimizer.step()
                optimizer.zero_grad()

                lrs.append(get_lr(optimizer))
            except Exception as e:
                print(f"Error in training batch: {e}")

        # 验证阶段
        try:
            result = evaluate(model, val_loader, device, num_classes)
            result['train_loss'] = torch.stack(train_losses).mean().item()
            result['lrs'] = lrs
            epoch_end(model, epoch, result)
            history.append(result)
            sched.step(result['Loss'])
        except Exception as e:
            print(f"Error in validation: {e}")

    return history



