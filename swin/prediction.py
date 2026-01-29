import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
import cv2
import numpy as np
from PIL import Image
import timm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# ================= SETTINGS =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['0_benign', '1_light_malignant', '2_heavy_malignant']
IMG_SIZE = 224

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "output_v6_clinical", "swin_v6_clinical_final.pth")
TEST_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "test_images"))
RESULT_DIR = os.path.join(BASE_DIR, "v6_visual_audit")
os.makedirs(RESULT_DIR, exist_ok=True)

# ================= V6 ARCHITECTURE =================
class SwinConvNeXt_V6(nn.Module):
    def __init__(self, num_classes=3):
        super(SwinConvNeXt_V6, self).__init__()
        self.cnn = timm.create_model('convnext_tiny', pretrained=False, num_classes=0)
        self.swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=0)
        self.fusion = nn.Sequential(
            nn.Linear(1536, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        c_feat = self.cnn(x)
        s_feat = self.swin(x)
        return self.classifier(self.fusion(torch.cat((c_feat, s_feat), dim=1)))

# ================= INITIALIZE =================
model = SwinConvNeXt_V6().to(DEVICE)
if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: Model not found at {MODEL_PATH}. Wait for training to finish!")
    exit()

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Watch the last ConvNeXt block for attention maps
cam_engine = GradCAM(model=model, target_layers=[model.cnn.stages[-1].blocks[-1]])

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def run_predictions():
    print(f"{'Filename':<35} | {'True Value':<15} | {'Predicted':<15} | {'Confidence'}")
    print("-" * 90)

    for root, _, files in os.walk(TEST_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Extract True Label from parent folder name
                folder = os.path.basename(root).lower()
                true_label = "benign" if "benign" in folder else "light_malignant" if "light" in folder else "heavy_malignant" if "heavy" in folder else "Unknown"
                
                img_path = os.path.join(root, file)
                pil_img = Image.open(img_path).convert("RGB")
                input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

                # Predict
                with torch.no_grad():
                    logits = model(input_tensor)
                    probs = F.softmax(logits, dim=1)
                    conf, idx = torch.max(probs, 1)
                    pred_label = CLASSES[idx.item()]

                # Generate Grad-CAM for Visual Evidence
                targets = [ClassifierOutputTarget(idx.item())]
                grayscale_cam = cam_engine(input_tensor=input_tensor, targets=targets)[0, :]
                rgb_img = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
                visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                
                # Save combined image
                cv2.imwrite(os.path.join(RESULT_DIR, f"v6_{file}"), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
                
                print(f"{file:<35} | {true_label:<15} | {pred_label:<15} | {conf.item()*100:.2f}%")

if __name__ == "__main__":
    run_predictions()