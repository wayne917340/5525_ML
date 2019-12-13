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
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
data_train = torch.utils.data.DataLoader(dataset=trainset, batch_size=100, shuffle=True)
data_test = torch.utils.data.DataLoader(dataset=testset, batch_size=100, shuffle=False)
end_load = time.perf_counter()
print('Image loaded, time consumed: {}'.format(end_load - start_load))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = x.view(-1, 784)
        return self.fc(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 784),
            nn.Tanh()
        )
    def forward(self, x):
        return self.fc(x)

# create networks D and G
D = Discriminator()
G = Generator()

# binary cross entropy
criterion = nn.BCELoss()

# ADAM
optimizer_D = optim.Adam(D.parameters(), lr = 0.0002, betas = (0.5,0.999))
optimizer_G = optim.Adam(G.parameters(), lr = 0.0002, betas = (0.5,0.999))

noise_test = torch.randn(16,128)

n_epoch = 50
loss_D = np.zeros(n_epoch)
loss_G = np.zeros(n_epoch)
print('===Training start===')
start_train = time.perf_counter()
for epoch in range(n_epoch):
    for batch_idx, (x_train, y_train) in enumerate(data_train):
        # Train discriminator
        # Train with real images
        optimizer_D.zero_grad()
        predict = D(x_train)
        loss_D_real = criterion(predict, torch.ones((100,1)))
        loss_D_real.backward()
        # Train with generated images
        noise = torch.randn(100, 128)
        generated_image = G(noise)
        predict = D(generated_image.detach())
        loss_D_generated = criterion(predict, torch.zeros((100,1)))
        loss_D_generated.backward()
        # Update discriminator
        optimizer_D.step()

        # Train generator
        optimizer_G.zero_grad()
        predict = D(generated_image)
        loss_G_generated = criterion(predict, torch.ones((100,1)))
        loss_G_generated.backward()
        # Update generator
        optimizer_G.step()

        # Record loss
        loss_D[epoch] += loss_D_real.item() + loss_D_generated.item()
        loss_G[epoch] += loss_G_generated.item()

    if epoch % 10 == 9:
        temp_train = time.perf_counter()
        print('Epoch: {}, time consumed: {}'.format(epoch+1, temp_train - start_train))
        plt.figure()
        for i in range(16):
            generated_image = G(noise_test[i]).detach().view(28,28)
            plt.subplot(4,4,i+1)
            plt.suptitle('GAN Epoch:{}'.format(epoch))
            plt.imshow(torchvision.utils.make_grid(generated_image).permute(1,2,0))

end_train = time.perf_counter()
print('===Training end===')
print('Time consumed: {}'.format(end_train - start_train))

# Save model
torch.save(D, 'hw5_gan_dis.pth')
torch.save(G, 'hw5_gan_gen.pth')

plt.figure()
plt.plot(range(n_epoch), loss_D, 'r-', label = 'Discriminator')
plt.plot(range(n_epoch), loss_G, 'b-', label = 'Generator')
plt.legend()
plt.xlabel('Number of Epoch')
plt.ylabel('Loss')
plt.title('GAN')
plt.show()
