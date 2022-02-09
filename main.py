# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import torch
from torch import nn
from torch.utils.data import DataLoader
from urbansounddata import UrbanSoundDataset
import torchaudio
from cnn import CNNNetwork
BATCH_SIZE = 128
EPOCHS=10
LEARNING_RATE=.001
ANNOTATIONS_FILE = "/Users/devarithish/Downloads/UrbanSound8K/metadata/UrbanSound8K.csv"
AUDIO_DIR = "/Users/devarithish/Downloads/UrbanSound8K/audio"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050
# Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.


# See PyCharm help at https://www.jetbrains.com/help/pycharm/









def train_one_epoch(model, data_loader,loss_fn,optimiser,device):
    for inputs,target in data_loader:
        inputs,target=inputs.to(device),target.to(device)
        predictions=model(inputs)
        loss=loss_fn(predictions,target)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    print(f"Loss:{loss.item()}")



def train(model, data_loader,loss_fn,optimiser,device,epochs):
    for i in range(epochs):
        print(f"Epoch{i+1}")
        train_one_epoch(model,data_loader,loss_fn,optimiser,device)
        print("---------------------------------")
    print("Training is done")




if __name__ == '__main__':
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device} device")
    mel_spectogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64,
    )
    usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectogram, SAMPLE_RATE, NUM_SAMPLES, device)


    train_data_loader = DataLoader(usd, batch_size=BATCH_SIZE)



    cnn = CNNNetwork().to(device)
    print(cnn)
    loss_fn=nn.CrossEntropyLoss()
    optimiser=torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

    train(cnn, train_data_loader, loss_fn, optimiser, device, 10)

    torch.save(cnn.state_dict(), "feedforwardnet.pth")
    print("Model trained and stored at feedforwardnet.pth")