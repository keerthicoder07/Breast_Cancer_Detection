import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torch.amp import GradScaler, autocast
import timm
from tqdm import tqdm
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ================= CONFIGURATION =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
BATCH_SIZE = 8 
EPOCHS = 30
DATA_DIR = r"C:\Users\Lenovo\BreastCancer\final_dataset_10k_augmented"
OUTPUT_DIR = "./output_v6_clinical"

# ================= 1. THE CLINICAL MASK =================
class ClinicalMask(object):
    """Aggressively masks edges and corners where 'ghost' hotspots appear."""
    def __call__(self, tensor):
        h, w = tensor.shape[1], tensor.shape[2]
        # Mask the top 30% (Removes most background noise/labels)
        tensor[:, 0:int(h*0.30), :] = 0 
        # Mask 10% of left and right edges (Removes edge artifacts)
        tensor[:, :, 0:int(w*0.10)] = 0
        tensor[:, :, int(w*0.90):w] = 0
        return tensor

# ================= 2. MODEL =================
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
        combined = torch.cat((self.cnn(x), self.swin(x)), dim=1)
        return self.classifier(self.fusion(combined))

# ================= 3. TRAINING FUNCTION =================
def run_training():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ClinicalMask(), # Apply the edge-cleaner
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = ImageFolder(root=DATA_DIR, transform=transform)
    train_size = int(0.85 * len(dataset))
    train_ds, _ = random_split(dataset, [train_size, len(dataset)-train_size])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    model = SwinConvNeXt_V6().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=8e-6, weight_decay=0.02)
    criterion_ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler('cuda') 

    target_layers = [model.cnn.stages[-1].blocks[-1]]
    cam_engine = GradCAM(model=model, target_layers=target_layers)

    print(f"🚀 V6 CLINICAL TRAINING: Masking background distractors...")

    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for imgs, labels in loop:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            with autocast('cuda'):
                logits = model(imgs)
                loss_ce = criterion_ce(logits, labels)
                
                # Grad-CAM Attention Penalty
                targets = [ClassifierOutputTarget(lbl.item()) for lbl in labels]
                grayscale_cam = cam_engine(input_tensor=imgs, targets=targets, eigen_smooth=True)
                grayscale_cam = torch.from_numpy(grayscale_cam).to(DEVICE).requires_grad_(True)
                
                # Increased weight (2.0) to force focus away from background
                loss_attention = torch.mean(grayscale_cam * (1.0 - (imgs != 0).float()[:,0,:,:]))
                total_loss = loss_ce + (2.0 * loss_attention)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loop.set_postfix(ce=f"{loss_ce.item():.3f}", att=f"{loss_attention.item():.3f}")

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "swin_v6_clinical_final.pth"))

if __name__ == '__main__':
    run_training()