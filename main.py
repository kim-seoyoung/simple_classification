import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model.classifier import BirdDroneUAVClassifier
from model.dataloader import CustomImageDataset
import os

def train(model, device, train_loader, optimizer, criterion, epoch, model_type):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    best_acc = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % 5 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')            
            if not os.path.exists('checkpoints'):
                os.makedirs('checkpoints')
            
            if correct > best_acc:
                best_acc = correct
                torch.save(model.state_dict(), f'checkpoints/{model_type}_best.pth')

    
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    return train_loss, train_acc

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    test_loss /= len(test_loader)
    test_acc = 100. * correct / total

    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({test_acc:.2f}%)\n')
    return test_loss, test_acc

def main():
    # Argument Parser for parameter settings
    parser = argparse.ArgumentParser(description='Bird, Drone, and UAV Classification Training')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the dataset directory containing train.txt, val.txt, test.txt')
    parser.add_argument('--model_type', type=str, default='convnext_tiny', 
                        choices=['convnext_tiny', 'convnext_small', 'convnext_base', 
                                 'efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l'],
                        help='Backbone model architecture')
    parser.add_argument('--batch_size', type=int, default=8, help='Input batch size for training')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate in the head')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for data loading')
    parser.add_argument('--no_pretrained', action='store_false', dest='pretrained', help='Train from scratch')
    parser.set_defaults(pretrained=True)
    
    args = parser.parse_args()

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Augmentation and Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load Datasets
    train_dataset = CustomImageDataset(txt_file=os.path.join(args.data_dir, 'train.txt'), data_dir='.', transform=transform)
    # Assuming you have a val.txt as well, you can create a validation loader
    val_dataset = CustomImageDataset(txt_file=os.path.join(args.data_dir, 'val.txt'), data_dir='.', transform=transform)
    # test_dataset = CustomImageDataset(txt_file=os.path.join(args.data_dir, 'test.txt'), data_dir='.', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Get number of classes
    num_classes = 0
    with open(os.path.join(args.data_dir, 'train.txt'), 'r') as f:
        for line in f:
            label = int(line.strip().split()[1])
            if label > num_classes:
                num_classes = label
    num_classes += 1


    # Model, Criterion, and Optimizer
    model = BirdDroneUAVClassifier(
        model_type=args.model_type,
        num_classes=num_classes,
        dropout_rate=args.dropout,
        pretrained=args.pretrained
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training and Testing Loop
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(model, device, train_loader, optimizer, criterion, epoch, args.model_type)
        val_loss, val_acc = test(model, device, val_loader, criterion)
        
        print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n')

if __name__ == '__main__':
    main()