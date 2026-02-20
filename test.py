import torch
import argparse
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from model.classifier import BirdDroneUAVClassifier
from model.dataloader import CustomImageDataset

def test_model(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    print("Starting evaluation...")
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    accuracy = 100. * correct / total
    print(f'\nTest Results:')
    print(f'Accuracy: {correct}/{total} ({accuracy:.2f}%)\n')
    return accuracy

def main():
    parser = argparse.ArgumentParser(description='Bird, Drone, and UAV Classification Inference')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the dataset directory containing test.txt')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the trained model weight (.pth)')
    parser.add_argument('--model_type', type=str, default='convnext_tiny', 
                        choices=['convnext_tiny', 'convnext_small', 'convnext_base', 
                                 'efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l'],
                        help='Backbone model architecture')
    parser.add_argument('--batch_size', type=int, default=8, help='Input batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for data loading')
    
    args = parser.parse_args()

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Transforms (Must match training transforms)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load Test Dataset
    test_dataset = CustomImageDataset(
        txt_file=os.path.join(args.data_dir, 'test.txt'), 
        data_dir='.', 
        transform=transform
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )

    # Load Checkpoint and infer num_classes
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found at {args.checkpoint}")
        return

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    # Infer num_classes from the last layer of the classifier head
    num_classes = checkpoint['classifier_head.4.bias'].shape[0]

    # Initialize Model
    model = BirdDroneUAVClassifier(
        model_type=args.model_type,
        num_classes=num_classes,
        pretrained=False
    ).to(device)
    model.load_state_dict(checkpoint)

    # Run Evaluation
    test_model(model, device, test_loader)

if __name__ == '__main__':
    main()
