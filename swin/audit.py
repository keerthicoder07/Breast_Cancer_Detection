import torch
import torch.nn as nn
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import timm

# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "output_v5_pro_amp", "swin_convnext_v5_final.pth")
TEST_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "test_images"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['0_benign', '1_light_malignant', '2_heavy_malignant']

# ================= MODEL =================
class SwinConvNeXt_V6(nn.Module):
    def __init__(self, num_classes=3):
        super(SwinConvNeXt_V6, self).__init__()
        self.cnn = timm.create_model('convnext_tiny', pretrained=True, num_classes=0)
        self.swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0)
        self.fusion = nn.Sequential(
            nn.Linear(1536, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4) # Increased dropout for better generalization
        )
        self.classifier = nn.Linear(512, num_classes)
    def forward(self, x):
        c_feat = self.cnn(x)
        s_feat = self.swin(x)
        return self.classifier(self.fusion(torch.cat((c_feat, s_feat), dim=1)))

# ================= AUDIT LOGIC =================
def run_audit():
    model = SwinConvNeXt_V6().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=test_transform)
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    y_true, y_pred = [], []

    print("Running Full Clinical Audit...")
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    # 1. Classification Report
    print("\n" + "="*40)
    print("V6 CLINICAL AUDIT REPORT")
    print("="*40)
    print(classification_report(y_true, y_pred, target_names=CLASSES))

    # 2. Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title("V6 Confusion Matrix: Clinical Focus")
    plt.ylabel('Ground Truth (Actual)')
    plt.xlabel('AI Prediction')
    plt.savefig(os.path.join(BASE_DIR, "v6_confusion_matrix.png"))
    print(f"✅ Confusion matrix saved to {BASE_DIR}")
    plt.show()

if __name__ == "__main__":
    run_audit()