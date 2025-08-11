# %%
# Necessary Imports
import os
import torch
import torchvision
import tarfile
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as F
import random
from sklearn.metrics import confusion_matrix
import seaborn as sns

from datasets import *
from datasets import CIC2017WithZeroDay
from utils import *
from models import *
from unlearning import *

# %%
torch.manual_seed(100)

# 初始化DataFrame来存储所有准确率数据
metrics_df = pd.DataFrame(columns=[
    'epoch',
    'original_model_train_acc',
    'teacher1_train_acc',
    'teacher2_train_acc',
    'student_train_acc',
    'original_model_val_acc',
    'teacher1_val_acc',
    'teacher2_val_acc',
    'student_val_acc',
    'teacher1_forget_acc',
    'teacher2_forget_acc',
    'student_forget_acc'
])

# 加载基础数据集
base_train_ds, base_valid_ds = cic2017()
train_ds = CIC2017WithZeroDay(base_train_ds, unknown_ratio=0.3)
valid_ds = CIC2017WithZeroDay(base_valid_ds, unknown_ratio=0.3)

print(f"Training set size: {len(train_ds)}")
print(f"Validation set size: {len(valid_ds)}")
print(f"Classes: {train_ds.classes}")
print(f"Number of classes: {len(train_ds.classes)}")

batch_size = 256
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0)
valid_dl = DataLoader(valid_ds, batch_size, num_workers=0)

num_classes = len(train_ds.classes)

# %%
device = 'cuda'
model = AllCNN().float().to(device)
epochs = 20
max_lr = 0.001
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam


# 定义绘制混淆矩阵的函数
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues, normalize=True, save_path=None):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)  # 每行归一化
        cm = np.nan_to_num(cm)  # 避免除0出现nan

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap=cmap,
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


# 定义计算混淆矩阵的函数
def get_confusion_matrix(model, data_loader, device, num_classes):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
    return cm


# 修改fit_one_cycle以返回训练和验证准确率
def fit_one_cycle_with_logging(epochs, max_lr, model, train_loader, val_loader,
                               grad_clip=None, weight_decay=None, opt_func=torch.optim.SGD, device='cuda'):
    torch.cuda.empty_cache()
    history = []

    # Set up optimizer
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                                steps_per_epoch=len(train_loader))

    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        correct = 0
        total = 0

        for batch in train_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = F.cross_entropy(outputs, labels)

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            train_losses.append(loss.item())
            loss.backward()

            # Gradient clipping
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()
            sched.step()

        # Calculate training accuracy
        train_acc = 100 * correct / total
        train_accuracies.append(train_acc)

        # Validation phase
        result = evaluate(model, val_loader, device)
        result['train_acc'] = train_acc
        val_accuracies.append(result['Acc'] * 100)

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {np.mean(train_losses):.4f}, "
              f"Train Acc: {train_acc:.2f}%, Val Loss: {result['Loss']:.4f}, "
              f"Val Acc: {result['Acc'] * 100:.2f}%")

        history.append(result)

    return history, train_accuracies, val_accuracies


# 训练原始模型并记录准确率
print("\nTraining Original Model...")
history, original_train_accs, original_val_accs = fit_one_cycle_with_logging(
    epochs, max_lr, model, train_dl, valid_dl,
    grad_clip=grad_clip,
    weight_decay=weight_decay,
    opt_func=opt_func, device=device
)

torch.save(model.state_dict(), "AllCNN_cic2017_ALL_CLASSES.pt")

# 生成并绘制原始模型的混淆矩阵
print("\nGenerating confusion matrix for original model...")
cm_original = get_confusion_matrix(model, valid_dl, device, num_classes)
cm_normalized = cm_original.astype('float') / cm_original.sum(axis=1, keepdims=True)
cm_normalized = np.nan_to_num(cm_normalized)  # 防止除0错误导致NaN
csv_save_path = 'original_confusion_matrix_normalized2.csv'
pd.DataFrame(cm_normalized, index=train_ds.classes, columns=train_ds.classes).to_csv(csv_save_path)
print(f"Confusion matrix saved to: {csv_save_path}")

# 绘图
plot_confusion_matrix(cm_original, train_ds.classes,
                      title='Original Model Confusion Matrix (Normalized)',
                      normalize=True,
                      save_path='original_confusion_matrix_normalized.png')
plot_confusion_matrix(cm_original, train_ds.classes, title='Original Model Confusion Matrix')

# %%
model.load_state_dict(torch.load("AllCNN_cic2017_ALL_CLASSES.pt"))


# %%
class Noise(nn.Module):
    def __init__(self, *dim):
        super().__init__()
        self.noise = torch.nn.Parameter(torch.randn(*dim), requires_grad=True)

    def forward(self):
        return self.noise


def fill_classwise_dict(dataset):
    num_classes = len(dataset.classes)
    classwise_dict = {i: [] for i in range(num_classes)}
    for img, label in dataset:
        label = label.item() if isinstance(label, torch.Tensor) else label
        classwise_dict[label].append((img, label))
    return classwise_dict


unknown_class = train_ds.unknown_class
classes_to_forget = [unknown_class,4,5]

classwise_train = fill_classwise_dict(train_ds)
classwise_test = fill_classwise_dict(valid_ds)

retain_samples = []
for label in classwise_train:
    if label not in classes_to_forget:
        retain_samples += classwise_train[label]


class ForgettingDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


forgotten_train_ds = ForgettingDataset(retain_samples)
forgotten_valid_ds = ForgettingDataset([
    sample for sample in valid_ds
    if (isinstance(sample[1], torch.Tensor) and sample[1].item() not in classes_to_forget)
       or (not isinstance(sample[1], torch.Tensor) and sample[1] not in classes_to_forget)
])

print("\nFinal dataset sizes:")
print(f"Training set: {len(forgotten_train_ds)}")
print(f"Validation set: {len(forgotten_valid_ds)}")

# %%
retain_classes = [cls for cls in range(num_classes) if cls not in classes_to_forget]
num_samples_per_class = 1000
retain_samples = []

for cls in retain_classes:
    class_samples = [sample for sample in forgotten_train_ds if sample[1] == cls]
    retain_samples += random.sample(class_samples, min(num_samples_per_class, len(class_samples)))

retain_valid = [sample for sample in forgotten_valid_ds]
forget_valid = [sample for sample in valid_ds if sample[1] in classes_to_forget]

forget_valid_dl = DataLoader(forget_valid, batch_size, num_workers=0, pin_memory=True)
retain_valid_dl = DataLoader(retain_valid, batch_size * 2, num_workers=0, pin_memory=True)
filtered_train_dl = DataLoader(retain_samples, batch_size, shuffle=True, num_workers=0)
forget_test_dl = DataLoader(forget_valid, batch_size, num_workers=0)

print(f"Retain samples: {len(retain_samples)}")
print(f"Retain validation: {len(retain_valid)}")
print(f"Forget validation: {len(forget_valid)}")
print(f"Filtered training set: {len(retain_samples)}")
print(f"Forget test set: {len(forget_valid)}")

# %%
model = AllCNN().float().to(device)
model.load_state_dict(torch.load("AllCNN_cic2017_ALL_CLASSES.pt"))

print("Performance of model on Forget Class")
forget_history = evaluate(model, forget_valid_dl, device=device, num_classes=num_classes)
print("Accuracy: {:.2f}%".format(forget_history["Acc"] * 100))
print("Loss: {:.4f}".format(forget_history["Loss"]))

print("Performance of model on Retain Class")
retain_history = evaluate(model, retain_valid_dl, device=device, num_classes=num_classes)
print("Accuracy: {:.2f}%".format(retain_history["Acc"] * 100))
print("Loss: {:.4f}".format(retain_history["Loss"]))

original_retain_accuracy = retain_history["Acc"] * 100

# %%
model = AllCNN().float().to(device)
model.load_state_dict(torch.load("AllCNN_cic2017_ALL_CLASSES.pt"))
teacher1 = AllCNN().float().to(device)
teacher2 = AllCNN().float().to(device)
teacher1.load_state_dict(torch.load("AllCNN_cic2017_ALL_CLASSES.pt"))

# %%
noises = {}
for cls in classes_to_forget:
    print("Optiming loss for class {}".format(cls))
    noises[cls] = Noise(batch_size, 1, 28, 28).cuda()
    opt = torch.optim.Adam(noises[cls].parameters(), lr=0.1)
    for epoch in range(5):
        total_loss = []
        for _ in range(8):
            inputs = noises[cls]()
            labels = torch.zeros(batch_size).cuda() + cls
            outputs = teacher1(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = -F.cross_entropy(outputs, labels.long()) + 0.1 * torch.mean(
                torch.sum(torch.square(inputs), [1, 2, 3]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss.append(loss.cpu().detach().numpy())
        print("Loss: {}".format(np.mean(total_loss)))

# %%
noisy_data = []
num_batches = 20
class_num = 0

for cls in classes_to_forget:
    for _ in range(num_batches):
        batch = noises[cls]().cpu().detach()
        for i in range(batch.size(0)):
            noisy_data.append((batch[i], torch.tensor(class_num)))

other_samples = [(x[0].cpu(), torch.tensor(x[1])) for x in retain_samples]
noisy_data += other_samples
noisy_loader = torch.utils.data.DataLoader(noisy_data, batch_size=256, shuffle=True)

optimizer = torch.optim.Adam(teacher1.parameters(), lr=0.02)

teacher1_train_accs = []
teacher1_val_accs = []
teacher1_forget_accs = []

print("\nTraining Teacher1 Model...")
for epoch in range(40):
    teacher1.train(True)
    running_loss = 0.0
    running_acc = 0
    total_samples = 0

    for inputs, labels in noisy_loader:
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = teacher1(inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.detach(), 1)
        running_acc += (labels == predicted).sum().item()
        total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_acc * 100 / total_samples
    teacher1_train_accs.append(epoch_acc)

    # Evaluate on validation set
    val_result = evaluate(teacher1, retain_valid_dl, device=device, num_classes=num_classes)
    val_acc = val_result['Acc'] * 100
    teacher1_val_accs.append(val_acc)

    # Evaluate on forget set
    forget_result = evaluate(teacher1, forget_valid_dl, device=device, num_classes=num_classes)
    forget_acc = forget_result['Acc'] * 100
    teacher1_forget_accs.append(forget_acc)

    print(f"Teacher1 Epoch {epoch + 1}: "
          f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, "
          f"Val Acc: {val_acc:.2f}%, Forget Acc: {forget_acc:.2f}%")

# %%
print("Performance of Standard Forget teacher1 on Forget Class")
history = [evaluate(teacher1, forget_valid_dl, device=device, num_classes=num_classes)]
print("Accuracy: {:.2f}%".format(history[0]["Acc"] * 100))
print("Loss: {:.4f}".format(history[0]["Loss"]))

print("Performance of Standard Forget teacher1 on Retain Class")
history = [evaluate(teacher1, retain_valid_dl, device=device, num_classes=num_classes)]
print("Accuracy: {:.2f}%".format(history[0]["Acc"] * 100))
print("Loss: {:.4f}".format(history[0]["Loss"]))

heal_loader = torch.utils.data.DataLoader(other_samples, batch_size=256, shuffle=True)

optimizer = torch.optim.Adam(teacher2.parameters(), lr=0.001)
teacher2_train_accs = []
teacher2_val_accs = []
teacher2_forget_accs = []

print("\nTraining Teacher2 Model...")
for epoch in range(40):
    teacher2.train()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for i, data in enumerate(heal_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = teacher2(inputs)

        if isinstance(outputs, tuple):
            outputs = outputs[0]

        if outputs.shape != (inputs.size(0), num_classes):
            raise ValueError(f"Invalid output shape: {outputs.shape}")

        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(outputs.detach(), dim=1)
        running_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct * 100 / total_samples
    teacher2_train_accs.append(epoch_acc)

    # Evaluate on validation set
    val_result = evaluate(teacher2, retain_valid_dl, device=device, num_classes=num_classes)
    val_acc = val_result['Acc'] * 100
    teacher2_val_accs.append(val_acc)

    # Evaluate on forget set
    forget_result = evaluate(teacher2, forget_valid_dl, device=device, num_classes=num_classes)
    forget_acc = forget_result['Acc'] * 100
    teacher2_forget_accs.append(forget_acc)

    print(f"Teacher2 Epoch {epoch + 1}: "
          f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, "
          f"Val Acc: {val_acc:.2f}%, Forget Acc: {forget_acc:.2f}%")

# %%
print("Performance of Standard Forget teacher2 on Forget Class")
history = [evaluate(teacher2, forget_valid_dl, device=device, num_classes=num_classes)]
print("Accuracy: {:.2f}%".format(history[0]["Acc"] * 100))
print("Loss: {:.4f}".format(history[0]["Loss"]))

print("Performance of Standard Forget teacher2 on Retain Class")
history = [evaluate(teacher2, retain_valid_dl, device=device, num_classes=num_classes)]
print("Accuracy: {:.2f}%".format(history[0]["Acc"] * 100))
print("Loss: {:.4f}".format(history[0]["Loss"]))

# %%
student = AllCNN().float().to(device)
optimizer = torch.optim.Adam(student.parameters(), lr=0.01)


def calculate_metrics(outputs, labels, num_classes):
    acc = accuracy(outputs, labels).item() * 100
    prec = precision(outputs, labels, num_classes)
    rec = recall(outputs, labels, num_classes)
    f1 = f1_score(outputs, labels, num_classes)
    return acc, prec, rec, f1


# 学生模型训练
temperature = 4.0
alpha = 0.7
student_train_accs = []
student_val_accs = []
student_forget_accs = []

print("\nTraining Student Model...")
for epoch in range(40):
    student.train()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for inputs, labels in filtered_train_dl:
        inputs, labels = inputs.cuda(), labels.cuda()

        # 教师模型预测
        with torch.no_grad():
            teacher1_logits = teacher1(inputs)[0] if isinstance(teacher1(inputs), tuple) else teacher1(inputs)
            teacher2_logits = teacher2(inputs)[0] if isinstance(teacher2(inputs), tuple) else teacher2(inputs)
            teachers_logits = (teacher1_logits + teacher2_logits) / 2

        # 学生模型预测
        student_logits = student(inputs)[0] if isinstance(student(inputs), tuple) else student(inputs)

        # 计算损失
        soft_loss = nn.KLDivLoss()(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teachers_logits / temperature, dim=1)
        ) * (temperature ** 2) * alpha

        hard_loss = F.cross_entropy(student_logits, labels) * (1 - alpha)
        loss = soft_loss + hard_loss

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(student_logits.detach(), 1)
        running_correct += (labels == predicted).sum().item()
        total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct * 100 / total_samples
    student_train_accs.append(epoch_acc)

    # Evaluate on validation set
    val_result = evaluate(student, retain_valid_dl, device=device, num_classes=num_classes)
    val_acc = val_result['Acc'] * 100
    student_val_accs.append(val_acc)

    # Evaluate on forget set
    forget_result = evaluate(student, forget_valid_dl, device=device, num_classes=num_classes)
    forget_acc = forget_result['Acc'] * 100
    student_forget_accs.append(forget_acc)

    print(f"Student Epoch {epoch + 1}: "
          f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, "
          f"Val Acc: {val_acc:.2f}%, Forget Acc: {forget_acc:.2f}%")

student_model_path = "student_model_cic2017.pt"
torch.save(student.state_dict(), student_model_path)
print(f"\n学生模型已保存至: {student_model_path}")

# === 获取 student 模型的混淆矩阵 ===
print("\nGenerating confusion matrix for student model...")

cm_student = get_confusion_matrix(student, valid_dl, device, num_classes)

# 归一化处理（按行）
cm_student_normalized = cm_student.astype('float') / cm_student.sum(axis=1, keepdims=True)
cm_student_normalized = np.nan_to_num(cm_student_normalized)

# 保存为 CSV 文件
student_csv_path = 'student_confusion_matrix_normalized2.csv'
pd.DataFrame(cm_student_normalized, index=train_ds.classes, columns=train_ds.classes).to_csv(student_csv_path)
print(f"Normalized student confusion matrix saved to: {student_csv_path}")

# 绘制并保存混淆矩阵图像
plot_confusion_matrix(cm_student,
                      classes=train_ds.classes,
                      title='Student Model Confusion Matrix (Normalized)',
                      normalize=True,
                      save_path='student_confusion_matrix_normalized.png')

# %%
# 填充最终的metrics DataFrame
max_epochs = max(
    len(original_train_accs),
    len(teacher1_train_accs),
    len(teacher2_train_accs),
    len(student_train_accs)
)

for epoch in range(max_epochs):
    metrics_df.loc[epoch] = {
        'epoch': epoch + 1,
        'original_model_train_acc': original_train_accs[epoch] if epoch < len(original_train_accs) else None,
        'teacher1_train_acc': teacher1_train_accs[epoch] if epoch < len(teacher1_train_accs) else None,
        'teacher2_train_acc': teacher2_train_accs[epoch] if epoch < len(teacher2_train_accs) else None,
        'student_train_acc': student_train_accs[epoch] if epoch < len(student_train_accs) else None,
        'original_model_val_acc': original_val_accs[epoch] if epoch < len(original_val_accs) else None,
        'teacher1_val_acc': teacher1_val_accs[epoch] if epoch < len(teacher1_val_accs) else None,
        'teacher2_val_acc': teacher2_val_accs[epoch] if epoch < len(teacher2_val_accs) else None,
        'student_val_acc': student_val_accs[epoch] if epoch < len(student_val_accs) else None,
        'teacher1_forget_acc': teacher1_forget_accs[epoch] if epoch < len(teacher1_forget_accs) else None,
        'teacher2_forget_acc': teacher2_forget_accs[epoch] if epoch < len(teacher2_forget_accs) else None,
        'student_forget_acc': student_forget_accs[epoch] if epoch < len(student_forget_accs) else None
    }

# 保存到CSV文件
metrics_df.to_csv('AllCNN-30.csv', index=False)
print("\n训练指标已保存到 training_metrics.csv")

# 打印最终的DataFrame
print("\n最终训练指标:")
print(metrics_df)