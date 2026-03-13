import argparse
import cv2
import numpy as np
import torch
from torchvision import transforms
from pytorch_grad_cam import (
    GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, 
    AblationCAM, XGradCAM, EigenCAM, FullGrad
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from model.classifier import BirdDroneUAVClassifier

def main():
    parser = argparse.ArgumentParser(description='Grad-CAM Visualization')
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model weight')
    parser.add_argument('--model_type', type=str, default='convnext_tiny')
    parser.add_argument('--method', type=str, default='gradcam', 
                        choices=['gradcam', 'hirescam', 'gradcam++', 'xgradcam', 'eigencam'],
                        help='CAM method to use')
    parser.add_argument('--output', type=str, default='cam_result.jpg', help='Output path')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Checkpoint and Model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    num_classes = checkpoint['classifier_head.4.bias'].shape[0]
    model = BirdDroneUAVClassifier(model_type=args.model_type, num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint)
    model.to(device).eval()

    # Target layer for ConvNext or EfficientNet
    if 'convnext' in args.model_type:
        target_layers = [model.backbone.stages[-1]]
    else:
        target_layers = [model.backbone.blocks[-1]]

    # Select CAM Method
    methods = {
        "gradcam": GradCAM,
        "hirescam": HiResCAM,
        "gradcam++": GradCAMPlusPlus,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM
    }
    
    cam_class = methods.get(args.method, GradCAM)
    cam = cam_class(model=model, target_layers=target_layers)

    # Preprocess Image
    rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])(rgb_img).unsqueeze(0).to(device)

    # Generate CAM
    targets = None  # Uses the highest scoring category
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    # Visualization
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    cv2.imwrite(args.output, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print(f"CAM visualization saved to {args.output}")

if __name__ == '__main__':
    main()