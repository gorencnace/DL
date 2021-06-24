import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.functional import interpolate
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import pickle
from tqdm import tqdm
from TSCNN import CNN
from json import dumps

def show_data():
    y, sr = librosa.load('data/train_short_audio/acafly/XC6671.ogg')

    #def features(y, sr):
    plt.plot(y)
    plt.title('Signal')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.show()

    n_fft = 2048
    ft = np.abs(librosa.stft(y[:n_fft], hop_length = n_fft+1))
    plt.plot(ft)
    plt.title('Spectrum')
    plt.xlabel('Frequency Bin')
    plt.ylabel('Amplitude')
    plt.show()

    spectrogram = np.abs(librosa.stft(y))#, hop_length=512))
    spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)
    librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.show()

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)#, n_fft=2048, hop_length=1024)
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    librosa.display.specshow(mel_spec, y_axis='mel', fmax=8000, x_axis='time')
    plt.title('Mel Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.show()

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128)
    librosa.display.specshow(mfcc, x_axis='time')
    plt.title('MFCC Spectrogram')
    plt.colorbar()
    plt.show()

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
    plt.title('CHROMA')
    plt.colorbar()
    plt.figure(figsize=(1, 8))
    plt.show()

    S = np.abs(librosa.stft(y))
    spec_contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    librosa.display.specshow(spec_contrast, x_axis='time')
    plt.title('SPECTRAL CONTRAST')
    plt.colorbar()
    plt.show()

    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    librosa.display.specshow(tonnetz, y_axis='tonnetz', x_axis='time')
    plt.title('TONNETZ')
    plt.colorbar()
    plt.show()

    lm_tensor = torch.from_numpy(mel_spec)
    mfcc_tensor = torch.from_numpy(mfcc)
    chroma_tensor = torch.from_numpy(chroma)
    spec_contrast_tensor = torch.from_numpy(spec_contrast)
    tonnetz_tensor = torch.from_numpy(tonnetz)

    lmc = torch.cat((lm_tensor, chroma_tensor, spec_contrast_tensor, tonnetz_tensor))
    mc = torch.cat((mfcc_tensor, chroma_tensor, spec_contrast_tensor, tonnetz_tensor))
    x = 1



import pandas as pd
from pathlib import Path

class BirdsDataSet(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 datadir: Path,
                 data: dict,
                 period: int):
        self.df = df
        self.datadir = datadir
        self.period = period
        self.data = data
        self.labels = list(self.data.keys())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        sample = self.df.loc[idx, :]
        wav_name = sample["filename"]
        ebird_code = sample["primary_label"]

        bird = self.data[ebird_code][wav_name]

        lm_tensor = bird['lm']
        mfcc_tensor = bird['mfcc']
        chroma_tensor = bird['chroma']
        spec_contrast_tensor = bird['spec_contrast']
        tonnetz_tensor = bird['tonnetz']

        a = torch.cat((lm_tensor, chroma_tensor, spec_contrast_tensor, tonnetz_tensor))
        a = a.resize_((1, a.shape[0], a.shape[1]))
        a = interpolate(a, size=self.period)
        lmc = a.resize_((1, a.shape[1], a.shape[2])).float()

        a = torch.cat((mfcc_tensor, chroma_tensor, spec_contrast_tensor, tonnetz_tensor))
        a = a.resize_((1, a.shape[0], a.shape[1]))
        a = interpolate(a, size=self.period)
        mc = a.resize_((1, a.shape[1], a.shape[2])).float()

        return {
            "lmc": lmc,
            "mc": mc,
            "label": self.labels.index(ebird_code),
            "ebird_code": ebird_code,
            "wav_name": wav_name
        }


def train(rating, number):
    data = pd.read_csv('data/train_metadata.csv')
    datadir = Path("data/train_short_audio")
    d = pickle.load(open((f'birds_tensors_{rating}_{number}'), 'rb'))

    def drop_birds(df, data):
        d = []
        for idx, row in df.iterrows():
            if row['primary_label'] not in data.keys():
                d.append(idx)
            elif row['filename'] not in data[row['primary_label']].keys():
                d.append(idx)
        return df.drop(d)#.reset_index(drop=True)

    from sklearn.model_selection import train_test_split


    data = drop_birds(data, d)
    train, test = train_test_split(data, test_size=0.1)
    train.reset_index(inplace=True, drop=True)
    test.reset_index(inplace=True, drop=True)
    period = 2048*2
    train_set = BirdsDataSet(train, datadir, data=d, period=period)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, num_workers=0)
    test_set = BirdsDataSet(test, datadir, data=d, period=period)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, num_workers=0)

    epochs = 200

    lmc_net = CNN()
    lmc_net.cuda()
    mc_net = CNN()
    mc_net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer_lmc = optim.Adam(lmc_net.parameters(), lr=0.001)
    optimizer_mc = optim.Adam(mc_net.parameters(), lr=0.001)
    for epoch in range(epochs):
        with tqdm(total=len(train_set), desc='Epoch: ' + str(epoch) + "/" + str(epochs), unit='img') as prog_bar:
            for i, data in enumerate(train_loader):
                lmc = data['lmc'].cuda()
                mc = data['mc'].cuda()
                labels = data['label'].cuda()

                optimizer_lmc.zero_grad()
                outputs_lmc = lmc_net.forward(lmc)
                loss_lmc = criterion(outputs_lmc, labels)
                loss_lmc.backward()
                optimizer_lmc.step()

                optimizer_mc.zero_grad()
                outputs_mc = mc_net.forward(mc)
                loss_mc = criterion(outputs_mc, labels)
                loss_mc.backward()
                optimizer_mc.step()

                prog_bar.set_postfix(**{'loss': loss_lmc.data.cpu().detach().numpy() + loss_mc.data.cpu().detach().numpy()})
                prog_bar.update(32)
        if (epoch + 1) % 20 == 0:
            torch.save({'lmc_dict': lmc_net.state_dict(),
                        'mc_dict': mc_net.state_dict(),
                        'optimizer_lmc_dict': optimizer_lmc.state_dict(),
                        'optimizer_mc_dict': optimizer_mc.state_dict()},
                       'models/networks_3_51')

    correct = 0
    total = 0
    running_loss = 0.0
    lmc_net.eval()
    mc_net.eval()

    with torch.no_grad():
        for data in test_loader:
            lmc = data['lmc'].cuda()
            mc = data['mc'].cuda()
            labels = data['label'].cuda()

            outputs_lmc = lmc_net.predict(lmc)

            outputs_mc = mc_net.predict(mc)

            outputs_ds = outputs_mc * outputs_lmc

            loss = criterion(outputs_ds, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs_ds.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))
    print("Test loss: "+str(running_loss/(total/4)))


def test(rating, number):
    data = pd.read_csv('data/train_metadata.csv')
    datadir = Path("data/train_short_audio")
    d = pickle.load(open((f'birds_tensors_{rating}_{number}'), 'rb'))
    period = 2048 * 2

    def drop_birds(df, data):
        d = []
        for idx, row in df.iterrows():
            if row['primary_label'] not in data.keys():
                d.append(idx)
            elif row['filename'] not in data[row['primary_label']].keys():
                d.append(idx)
        return df.drop(d)  # .reset_index(drop=True)

    from sklearn.model_selection import train_test_split

    test = drop_birds(data, d)
    test_set = BirdsDataSet(test, datadir, data=d, period=period)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, num_workers=0)

    lmc_net = CNN().cuda()
    checkpoint = torch.load('models/networks_3_51')
    lmc_net.load_state_dict(checkpoint['lmc_dict'])
    lmc_net.eval()

    mc_net = CNN().cuda()
    mc_net.load_state_dict(checkpoint['mc_dict'])
    mc_net.eval()

    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    running_loss = 0.0
    wrong = list()
    with torch.no_grad():
        for data in test_loader:
            lmc = data['lmc'].cuda()
            mc = data['mc'].cuda()
            labels = data['label'].cuda()

            outputs_lmc = lmc_net.predict(lmc)

            outputs_mc = mc_net.predict(mc)

            outputs_ds = outputs_mc * outputs_lmc

            loss = criterion(outputs_ds, labels)

            running_loss += loss.item()

            probabilities, predicted = torch.max(outputs_ds.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for i, e in enumerate(zip(predicted, labels)):
                p, l = e
                if p != l:
                    a = {'label': data['ebird_code'][i], 'wav': data['wav_name'][i], 'prediction_label': list(d.keys())[p.item()], 'confidence_true': outputs_ds[i][l.item()].item(), 'confidence_false': probabilities[i].item()}
                    wrong.append(a)

    for w in wrong:
        print(dumps(w, indent=1))
    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))
    print("Test loss: "+str(running_loss/(total/4)))



def get_data(rating, number):
    train = pd.read_csv('data/train_metadata.csv')
    train_datadir = Path("data/train_short_audio")

    data = dict()
    with tqdm(total=len(train), unit='posnetek') as prog_bar:
        for idx in range(len(train)):
            sample = train.loc[idx, :]
            wav_name = sample["filename"]
            ebird_code = sample["primary_label"]
            rating = sample["rating"]
            if float(rating) >= rating:
                if ebird_code not in data.keys():
                    if len(data.keys()) > number:
                        continue
                    data[ebird_code] = dict()

                y, sr = librosa.load(train_datadir / ebird_code / wav_name)

                mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
                mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128)

                chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

                S = np.abs(librosa.stft(y))
                spec_contrast = librosa.feature.spectral_contrast(S=S, sr=sr)

                tonnetz = librosa.feature.tonnetz(y=y, sr=sr)

                lm_tensor = torch.from_numpy(mel_spec)
                mfcc_tensor = torch.from_numpy(mfcc)
                chroma_tensor = torch.from_numpy(chroma)
                spec_contrast_tensor = torch.from_numpy(spec_contrast)
                tonnetz_tensor = torch.from_numpy(tonnetz)

                #lmc = torch.cat((lm_tensor, chroma_tensor, spec_contrast_tensor, tonnetz_tensor))
                #a = a.resize_((1, a.shape[0], a.shape[1]))
                #a = interpolate(a, size=period)
                #lmc = a.resize_((a.shape[1], a.shape[2]))

                #mc = torch.cat((mfcc_tensor, chroma_tensor, spec_contrast_tensor, tonnetz_tensor))
                #a = a.resize_((1, a.shape[0], a.shape[1]))
                #a = interpolate(a, size=period)
                #mc = a.resize_((a.shape[1], a.shape[2]))

                data[ebird_code][wav_name] = {'lm': lm_tensor,
                                              'mfcc': mfcc_tensor,
                                              'chroma': chroma_tensor,
                                              'spec_contrast': spec_contrast_tensor,
                                              'tonnetz': tonnetz_tensor}

            prog_bar.update(1)

    pickle.dump(data, open(f'birds_tensors_3_10', 'wb'))

if __name__ == '__main__':
    #train(3, 10)
    #get_data(3, 10)
    test(3, 10)
    #show_data()