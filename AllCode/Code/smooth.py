import argparse
import torch 
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import os
import numpy as np
import time
import random

from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import cycle

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        
    def forward(self, x, target):
        if hasattr(x, 'logits'):
            x = x.logits
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

parser = argparse.ArgumentParser()
parser.add_argument("--model_names", type=str, default="efficientnet-b0", help="选择的模型名称")
parser.add_argument("--pre_trained", type=bool, default=False, help="是否使用预训练模型")
parser.add_argument("--classes_num", type=int, default=8, help="类别数量")
parser.add_argument("--dataset", type=str, default="AllCode/ABCD/Dataset", help="数据集路径")
parser.add_argument("--batch_size", type=int, default=128, help="批量大小")
parser.add_argument("--epoch", type=int, default=50, help="训练轮数")
parser.add_argument("--lr", type=float, default=0.01, help="学习率")
parser.add_argument("--momentum", type=float, default=0.9, help="动量")
parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减")
parser.add_argument("--seed", type=int, default=33, help="随机种子")
parser.add_argument("--gpu-id", type=int, default=0, help="GPU 编号")
parser.add_argument("--print_freq", type=int, default=1, help="打印频率")
parser.add_argument("--exp_postfix", type=str, default="seed33", help="实验后缀名称")
parser.add_argument("--txt_name", type=str, default="lr0.01_wd5e-4", help="日志文本文件名称")
parser.add_argument("--disable_tsne", type=str, default="False", help="是否禁用TSNE可视化")
args = parser.parse_args()

def seed_torch(seed=74):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
seed_torch(seed=args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

exp_name = args.exp_postfix
dataset_name = os.path.basename(args.dataset.rstrip('/\\'))
exp_path = "./report/{}/{}/{}".format(dataset_name, args.model_names, exp_name)
os.makedirs(exp_path, exist_ok=True)

transform_train = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.Resize([256, 256]),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize((0.2099, 0.2099, 0.2099), (0.1826, 0.1826, 0.1826)),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.2))
])
transform_test = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize((0.2099, 0.2099, 0.2099), (0.1826, 0.1826, 0.1826))
])

trainset = datasets.ImageFolder(root=os.path.join(args.dataset, 'train'), transform=transform_train)
valset = datasets.ImageFolder(root=os.path.join(args.dataset, 'val'), transform=transform_test)
testset = datasets.ImageFolder(root=os.path.join(args.dataset, 'test'), transform=transform_test)

train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True)
val_loader = DataLoader(valset, batch_size=args.batch_size, num_workers=8, shuffle=False, pin_memory=True)
test_loader = DataLoader(testset, batch_size=args.batch_size, num_workers=8, shuffle=False, pin_memory=True)

def train_one_epoch(model, optimizer, train_loader, scheduler):
    model.train()
    acc_recorder = AverageMeter()
    loss_recorder = AverageMeter()
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    criterion = LabelSmoothingCrossEntropy(smoothing=0)
    
    for inputs, targets in tqdm(train_loader, desc="Train"):
        if torch.cuda.is_available():
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            
        if scaler is not None:
            with torch.cuda.amp.autocast():
                out = model(inputs)
                loss = criterion(out, targets)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(inputs)
            loss = criterion(out, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()
            
        loss_recorder.update(loss.item(), n=inputs.size(0))
        
        if hasattr(out, 'logits'):
            out = out.logits
            
        acc = accuracy(out, targets)[0]
        acc_recorder.update(acc.item(), n=inputs.size(0))
    
    return loss_recorder.avg, acc_recorder.avg

def evaluation(model, test_loader):
    model.eval()
    acc_recorder = AverageMeter()
    loss_recorder = AverageMeter()
    with torch.no_grad():
        for img, label in tqdm(test_loader, desc="Eval"):
            if torch.cuda.is_available():
                img = img.cuda()
                label = label.cuda()
            out = model(img)
            
            if hasattr(out, 'logits'):
                out = out.logits
                
            loss = F.cross_entropy(out, label)
            loss_recorder.update(loss.item(), img.size(0))
            acc = accuracy(out, label)[0]
            acc_recorder.update(acc.item(), img.size(0))
    return loss_recorder.avg, acc_recorder.avg

def generate_evaluation_plots(model, test_loader, save_path):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    all_logits = []
    with torch.no_grad():
        for img, label in tqdm(test_loader, desc="Gen Plots"):
            if torch.cuda.is_available():
                img = img.cuda()
            out = model(img)
            
            if hasattr(out, 'logits'):
                out = out.logits
                
            all_logits.append(out.detach().cpu())
            prob = F.softmax(out, dim=1)
            all_probs.append(prob.detach().cpu())
            pred = torch.argmax(prob, dim=1)
            all_preds.append(pred.detach().cpu())
            all_labels.append(label)
    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_probs = torch.cat(all_probs).numpy()
    all_logits = torch.cat(all_logits).numpy()

    class_names = test_loader.dataset.classes
    n_classes = len(class_names)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(save_path, "confusion_matrix.png"))
    plt.close()

    from sklearn.preprocessing import label_binarize
    all_labels_bin = label_binarize(all_labels, classes=range(n_classes))
    plt.figure()
    colors_list = ['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple', 'brown', 'pink']
    for i, color in zip(range(n_classes), cycle(colors_list)):
        fpr, tpr, _ = roc_curve(all_labels_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label='{} (AUC = {:0.2f})'.format(class_names[i], roc_auc))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_path, "roc_curve.png"))
    plt.close()

    plt.figure()
    for i, color in zip(range(n_classes), cycle(colors_list)):
        precision, recall, _ = precision_recall_curve(all_labels_bin[:, i], all_probs[:, i])
        plt.plot(recall, precision, color=color, lw=2,
                 label='{}'.format(class_names[i]))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(save_path, "precision_recall_curve.png"))
    plt.close()

    if args.disable_tsne.lower() != "true":
        try:
            tsne = TSNE(n_components=2, random_state=42)
            tsne_results = tsne.fit_transform(all_logits)
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=all_labels, cmap='jet', alpha=0.7)
            cbar = plt.colorbar(scatter)
            cbar.set_ticks(np.arange(n_classes) + 0.5)
            cbar.set_ticklabels(class_names)
            plt.title("TSNE Visualization")
            plt.savefig(os.path.join(save_path, "tsne_visualization.png"))
            plt.close()
        except Exception as e:
            print(f"TSNE可视化生成失败: {str(e)}")
            print("跳过TSNE可视化...")
    else:
        print("TSNE可视化已禁用")

def train(model, optimizer, train_loader, val_loader, scheduler):
    since = time.time()
    best_acc = -1
    log_file = os.path.join(exp_path, "{}.txt".format(args.txt_name))
    f = open(log_file, "w")
    
    for epoch in range(args.epoch):
        train_loss, train_acc = train_one_epoch(model, optimizer, train_loader, scheduler)
        val_loss, val_acc = evaluation(model, val_loader)
        
        if val_acc > best_acc:
            best_acc = val_acc
            state_dict = {
                "epoch": epoch + 1, 
                "model": model.state_dict(), 
                "acc": val_acc,
                "classes": [c for c in train_loader.dataset.classes]
            }
            ckpt_path = os.path.join(exp_path, "ckpt", "best.pth")
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save(state_dict, ckpt_path)
        
        if not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()
            
        tags = ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'learning_rate']
        tb_writer.add_scalar(tags[0], train_loss, epoch + 1)
        tb_writer.add_scalar(tags[1], train_acc, epoch + 1)
        tb_writer.add_scalar(tags[2], val_loss, epoch + 1)
        tb_writer.add_scalar(tags[3], val_acc, epoch + 1)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]['lr'], epoch + 1)
        
        if (epoch + 1) % args.print_freq == 0:
            msg = "Epoch: {} Model: {} Train Loss: {:.2f} Acc: {:.2f} | Val Loss: {:.2f} Acc: {:.2f} | LR: {:.6f}\n".format(
                epoch + 1, args.model_names, train_loss, train_acc, val_loss, val_acc, optimizer.param_groups[0]['lr']
            )
            print(msg)
            f.write(msg)
            f.flush()
            
    msg_best = "Model: {} Best Val Acc: {:.2f}\n".format(args.model_names, best_acc)
    time_elapsed = "Total training time: {:.0f} seconds".format(time.time() - since)
    print(msg_best)
    f.write(msg_best)
    f.write(time_elapsed)
    f.close()

if __name__ == "__main__":
    tb_path = "runs/{}/{}/{}".format(dataset_name, args.model_names, exp_name)
    tb_writer = SummaryWriter(log_dir=tb_path)
    lr = args.lr
    model = model_dict[args.model_names](num_classes=args.classes_num, pretrained=args.pre_trained)
    if torch.cuda.is_available():
        model = model.cuda()
    
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=args.momentum,
        nesterov=True,
        weight_decay=args.weight_decay
    )
    
    from torch.optim.lr_scheduler import OneCycleLR
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epoch
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1000
    )
    
    train(model, optimizer, train_loader, val_loader, scheduler)
    
    test_loss, test_acc = evaluation(model, test_loader)
    print(f"Final Test Loss: {test_loss:.2f} Acc: {test_acc:.2f}")
    log_file = os.path.join(exp_path, "{}.txt".format(args.txt_name))
    with open(log_file, "a") as f:
        f.write(f"\nFinal Test Loss: {test_loss:.2f} Acc: {test_acc:.2f}\n")
    
    generate_evaluation_plots(model, test_loader, exp_path)