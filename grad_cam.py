import argparse
import cv2
import numpy as np
import torch
import os
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
    parser.add_argument('--test_list', type=str, default='data/test.txt', help='Path to test list file')
    parser.add_argument('--data_dir', type=str, default='.', help='Base directory for data')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model weight')
    parser.add_argument('--model_type', type=str, default='convnext_tiny')
    parser.add_argument('--method', type=str, default='gradcam', 
                        choices=['gradcam', 'hirescam', 'gradcam++', 'xgradcam', 'eigencam'],
                        help='CAM method to use')
    parser.add_argument('--output_dir', type=str, default='cam_results', help='Output directory')
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

    if not os.path.exists(args.test_list):
        print(f"Error: test list {args.test_list} does not exist.")
        return

    with open(args.test_list, 'r') as f:
        lines = f.readlines()

    total_images = len(lines)
    print(f"Starting Grad-CAM generation for {total_images} images...")

    for i, line in enumerate(lines):
        parts = line.strip().split()
        if not parts:
            continue
        
        img_rel_path = parts[0]
        full_img_path = os.path.join(args.data_dir, img_rel_path)
        out_path = os.path.join(args.output_dir, img_rel_path)
        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # Preprocess Image
        rgb_img = cv2.imread(full_img_path, 1)
        if rgb_img is None:
            print(f"Warning: Could not read image {full_img_path}")
            continue
            
        rgb_img = rgb_img[:, :, ::-1]
        rgb_img = cv2.resize(rgb_img, (224, 224))
        rgb_img = np.float32(rgb_img) / 255
        input_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])(rgb_img).unsqueeze(0).to(device)

        # Get classification prediction
        with torch.no_grad():
            preds = model(input_tensor)
            pred_class = preds.argmax(dim=1).item()

        # Append prediction result as suffix to the output file name
        base_path, ext = os.path.splitext(out_path)
        out_path = f"{base_path}_pred_{pred_class}{ext}"

        # Generate CAM
        targets = None  # Uses the highest scoring category
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        # Visualization
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        # Convert original image back to uint8 BGR and concatenate horizontally
        orig_bgr = cv2.cvtColor(np.uint8(255 * rgb_img), cv2.COLOR_RGB2BGR)
        cam_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
        concat_img = np.concatenate((orig_bgr, cam_bgr), axis=1)
        
        cv2.imwrite(out_path, concat_img)

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{total_images} images.")

    print(f"All CAM visualizations saved to {args.output_dir}")

if __name__ == '__main__':
    main()