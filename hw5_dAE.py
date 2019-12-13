import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import time

start_load = time.perf_counter()
# Load MNIST data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,),(1,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
data_train = torch.utils.data.DataLoader(dataset=trainset, batch_size=64, shuffle=True)
data_test = torch.utils.data.DataLoader(dataset=testset, batch_size=64, shuffle=False)
end_load = time.perf_counter()
print('Image loaded, time consumed: {}'.format(end_load - start_load))

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 20),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(20, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = x.view(-1, 784)
        return self.decoder(self.encoder(x))

# create an autoencoder
AE = AutoEncoder()

# binary cross entropy
criterion = nn.BCELoss()

# Adam
optimizer = optim.Adam(AE.parameters(), lr = 0.0002, betas = (0.5, 0.999))

n_epoch = 10
loss_epoch = np.zeros(n_epoch)
print('===Training Start===')
start_train = time.perf_counter()
for epoch in range(n_epoch):
    for batch_idx, (x_train, y_train) in enumerate(data_train):
        # feedforward
        x_noised = (x_train + torch.rand(x_train.shape)) / 2 # add noise
        predict = AE(x_noised).view(x_train.shape)
        loss = criterion(predict, x_train)
        # backward propagate
        optimizer.zero_grad()
        loss.backward()
        # update weight
        optimizer.step()
        # record loss
        loss_epoch[epoch] += loss.item()

end_train = time.perf_counter()
print('===Training End===')
print('Time consumed: {}'.format(end_train - start_train))

# save model
torch.save(AE, 'hw5_dAE.pth')

plt.figure()
plt.plot(range(n_epoch), loss_epoch, 'b-', label = 'Auto Encoder')
plt.legend()
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.title('Denoising AE')

x_test = next(iter(data_test))[0]
plt.figure()
for i in range(5):
    # denoise
    x_noised = (x_test[i] + torch.rand(x_test[i].shape)) / 2
    denoise = AE(x_noised).detach().view(28,28)
    # plot
    plt.subplot(2,5,i+1)
    plt.suptitle('Noisy')
    plt.imshow(torchvision.utils.make_grid(x_noised).permute(1,2,0))
    plt.subplot(2,5,i+6)
    plt.suptitle('Denoised')
    plt.imshow(torchvision.utils.make_grid(denoise).permute(1,2,0))
    
plt.show()
