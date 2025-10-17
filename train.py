import os

import torch
from torchvision import transforms
from tqdm import tqdm
import dataset
from torch import nn
from torch.utils.data import DataLoader, random_split
from model.model import TumorSegmentationModel
from sklearn.metrics import accuracy_score


def get_train_test_loader(path_to_dataset, test_size = 0.3, transform = None):
    data = dataset.TumorDataset(path_to_dataset, transform = transform)
    test_size = int(len(data) * test_size)
    train_size = len(data) - test_size

    train_dataset, test_dataset = random_split(data, [train_size, test_size])

    return (DataLoader(train_dataset, batch_size=32, num_workers=4, pin_memory=True, drop_last=False, shuffle=True),
            DataLoader(test_dataset, batch_size=32, num_workers=4, pin_memory=True, drop_last=False, shuffle=False))


def train(epochs = 30, dir_checkpoints = './checkpoints', path_to_dataset = './datasets'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create folder checkpoint if it is not exist
    if dir_checkpoints != './checkpoints':
        os.mkdir(dir_checkpoints)

    # define model, loss function, optimization, etc...
    model = TumorSegmentationModel()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    start_epoch = 0
    best_accuracy = 0

    # handle checkpoint
    last_checkpoint = os.path.join(dir_checkpoints, 'last_checkpoint.pt')
    best_checkpoint = os.path.join(dir_checkpoints, 'best_checkpoint.pt')

    if os.path.exists(last_checkpoint):
        print(f'[INFO]: Found last checkpoint at {last_checkpoint}. Loading...')
        checkpoints = torch.load(last_checkpoint)
        model.load_state_dict(checkpoints['model_state'])
        optimizer.load_state_dict(checkpoints['optimizer_state'])
        start_epoch = checkpoints['epoch'] + 1
        best_accuracy = checkpoints['best_accuracy']
        print(f'[INFO]: Resuming from epoch {start_epoch}. Best accuracy: {best_accuracy}')

    transform = transforms.Compose([
        transforms.ToTensor(),
        # continue code here.
    ])

    train_loader, test_loader = get_train_test_loader(path_to_dataset, test_size=0.3, transform=transform)

    for epoch in range(start_epoch, epochs):
        model.train()
        progress_bar = tqdm(train_loader)

        for iteration, (image, labels) in enumerate(progress_bar):
            output = model(image.to(device))
            loss = criterion(output, labels.to(device))
            optimizer.zero_grad()
            loss.backward()

            progress_bar.set_description(
                f'[INFO]: Epoch: {epoch}, Iteration: {iteration / len(train_loader)}, Loss: {loss.item():.4f}'
            )

            model.eval()
            all_predictions = []
            all_labels = []

            with torch.no_grad():
                for (test_image, test_labels) in tqdm(test_loader):
                    output = model(test_image.to(device))
                    all_predictions.append(output.detach().cpu().numpy())
                    all_labels.append(test_labels.detach().cpu().numpy())

            accuracy = accuracy_score(all_predictions, all_labels)

            checkpoint = {
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'epoch': epoch,
                'best_accuracy': best_accuracy,
            }

            torch.save(checkpoint, last_checkpoint)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(checkpoint, best_checkpoint)
                print(f'[INFO]: New best checkpoint at epoch: {epoch + 1} with accuracy: {best_accuracy}')

if __name__ == '__main__':
    train()