import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from model import model_dict

TEST_DIR = "/root/autodl-tmp/AllCode/ABCD/Dataset/test"
WEIGHT_PATH = "/root/autodl-tmp/report/Dataset/efficientnet-b0/seed33/ckpt/best.pth"
MODEL_NAME = "efficientnet-b0"
NUM_CLASSES = 8
BATCH_SIZE = 128
INPUT_SIZE = 224

def load_model():
    try:
        model = model_dict[MODEL_NAME](num_classes=NUM_CLASSES, pretrained=False)
    except KeyError:
        raise ValueError(f"模型 {MODEL_NAME} 未定义，请检查 model.py 中的 model_dict")

    checkpoint = torch.load(WEIGHT_PATH, map_location='cpu')
    state_dict = checkpoint.get('model', checkpoint)

    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()
    return model

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def evaluate_metrics():
    transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE + 32),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.2099, 0.2099, 0.2099), (0.1826, 0.1826, 0.1826)),
    ])

    test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    model = load_model().to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"\n模型参数总量: {total_params:,}（约 {total_params / 1e6:.2f} M）")
    print(f"可训练参数量: {trainable_params:,}（约 {trainable_params / 1e6:.2f} M）")

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)

            if hasattr(outputs, 'logits'):
                outputs = outputs.logits
            elif isinstance(outputs, tuple):
                outputs = outputs[0]

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    macro_precision = precision_score(all_labels, all_preds, average='macro')
    weighted_precision = precision_score(all_labels, all_preds, average='weighted')
    macro_recall = recall_score(all_labels, all_preds, average='macro')
    weighted_recall = recall_score(all_labels, all_preds, average='weighted')

    print("\n" + "=" * 40)
    print(f"模型 {MODEL_NAME} 测试结果：")
    print(f"宏平均 F1: {macro_f1:.4f}")
    print(f"加权 F1: {weighted_f1:.4f}")
    print(f"宏平均 Precision: {macro_precision:.4f}")
    print(f"加权 Precision: {weighted_precision:.4f}")
    print(f"宏平均 Recall: {macro_recall:.4f}")
    print(f"加权 Recall: {weighted_recall:.4f}")
    print("=" * 40)

if __name__ == '__main__':
    evaluate_metrics()