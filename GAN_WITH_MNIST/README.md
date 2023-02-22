# GAN_WITH_MNIST

# GAN을 이용한 MNIST 패턴을 학습하여 글씨체 생성


## Motivation

- 포스코 아카데미 AI 교육에서 GAN에 대하여 이론 강의를 들을 수 있었고, 실전 성능을 확인해보기 위해서 손글씨 데이터를 활용하여 새로운 이미지를 생성하였습니다.

## Project summary

- GAN 학습 algorithm을 이해합니다.

- GAN을 Pytorch로 구현하여 MNIST dataset을 이용하여 학습 모델을 training 시킵니다.

- Testing을 통해, 10 epoch마다 학습된 모델이 이미지를 어떻게 생성확인합니다. 


## Goals

- GAN이 무엇이며 알고리즘과 모델이 어떤 과정을 통해 학습하는지 이해하려고 하였습니다. 

- GAN을 이용하면 Input Image에 대해 유사한 새로운 이미지를 생성할 수 있습니다. 

## Dataset
- MNIST 손글씨 숫자 이미지

## Getting start
### 1. colab에 접속

### 2. Data Loader 생성
```python
import torch
if torch.cuda.is_available() == True:
    device = 'cuda:0'
    print('현재 가상환경 GPU 사용 가능상태')
else:
    device = 'cpu'
    print('GPU 사용 불가능 상태')
```

```python
import torch
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Resize, Normalize, RandomHorizontalFlip, RandomCrop
import torchvision.datasets as datasets

batch_size = 100

# MNIST Dataset
transform = transforms.Compose([ToTensor(), Normalize(mean=(0.5,), std=(0.5,))]) # -1 ~ 1 사이로 정규화

train_dataset = datasets.MNIST(root='./', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./', train=False, transform=transform, download=False)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
```

### 3. Generator와 Discriminator 모델 구조 설계
```python
import torch.nn as nn
import torch.nn.functional as F
```
#### Generator
```python
class Generator(nn.Module):
    
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()       
        self.fc1 = nn.Linear(g_input_dim, 256) # 100 -> 256
        self.fc2 = nn.Linear(256, 512) # 256 -> 512
        self.fc3 = nn.Linear(512, 1024) # 512 -> 1024
        self.fc4 = nn.Linear(1024, g_output_dim) # 1024 -> 784
    
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x)) # Tanh : -1 ~ 1 사이로 데이터 생성
```

#### Discriminator
```python
class Discriminator(nn.Module):

    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024) # 784 -> 1024
        self.fc2 = nn.Linear(1024, 512) # 1024 -> 512
        self.fc3 = nn.Linear(512, 256) # 512 -> 256
        self.fc4 = nn.Linear(256, 1) # 256 -> 1
    
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))
```

### 4. 모델 선언
```python
z_dim = 100
mnist_dim = train_dataset.train_data.size(1) * train_dataset.train_data.size(2) # 28 * 28 = 784

G = Generator(g_input_dim = z_dim, g_output_dim = mnist_dim).to(device)
D = Discriminator(mnist_dim).to(device)
```
### 5. Optimizer, Loss Function 선언
#### loss
```python
criterion = nn.BCELoss()
```
#### optimizer
```python

lr = 0.0002
G_optimizer = torch.optim.Adam(G.parameters(), lr = lr)
D_optimizer = torch.optim.Adam(D.parameters(), lr = lr)
```
### 6. 학습 알고리즘 제작
#### Generator 
```def G_train(x):
    
    G.zero_grad()
    
    z = torch.randn(batch_size, z_dim).to(device) # 100 * 100 행렬을 -1 ~ 1 사이 수로 생성
    y = torch.ones(batch_size, 1).to(device) # 1로 가득찬 y를 생성

    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y) 

    # 역전파 + 경사하강
    G_loss.backward()
    G_optimizer.step()
        
    return G_loss.data.item()
```
#### Discriminator
```python
def D_train(x):
    
    D.zero_grad()

    x_real = x.view(-1, mnist_dim) # 실제 MNIST 데이터 1장 불러오기
    y_real = torch.ones(batch_size, 1) # 1로 가득찬 100개 리스트 생성(레이블 1 = 진짜)
    
    x_real, y_real = x_real.to(device), y_real.to(device) # gpu

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)

    z = torch.randn(batch_size, z_dim).to(device) # 랜덤 노이즈 생성(100 * 100)
    x_fake, y_fake = G(z), torch.zeros(batch_size, 1).to(device) # 1로 가득찬 100개 리스트 생성

    D_output = D(x_fake)
    D_fake_loss = criterion(D_output, y_fake)

    # 역전파 + 경사하강 시행
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()

    return  D_loss.data.item()
```
### 7. 1 epoch씩 학습, 10 epoch마다 결과 출력
```python
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import cv2

n_epoch = 200
cnt = 0
print_time = 10
for epoch in range(1, n_epoch+1):           
    D_losses, G_losses = [], []
    for batch_idx, (x, _) in enumerate(train_loader):
        D_losses.append(D_train(x))
        G_losses.append(G_train(x))

    print_dis_loss = round(float(torch.mean(torch.FloatTensor(G_losses))), 5)
    print_gen_loss = round(float(torch.mean(torch.FloatTensor(D_losses))), 5)
    print('[{}/{}]: loss_discre.: {}, loss_gen.: {}'.format(epoch, n_epoch, print_dis_loss, print_gen_loss))
    if epoch % print_time == 0:
      with torch.no_grad():
          test_z = torch.randn(batch_size, z_dim).to(device)
          generated = G(test_z)
          img_path = './GAN_MNIST' + str(epoch) + '.png'
          save_image(generated.view(generated.size(0), 1, 28, 28), img_path)
          img = cv2.imread(img_path)
          img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
          plt.figure(figsize=(10,10))
          plt.imshow(img)
          plt.title('Epoch: {}'.format(epoch))
          plt.show()
    cnt += 1
```

## Result
epoch 10
![image](https://github.com/0cars0903/GAN_WITH_MNIST/blob/main/GAN_MNIST_Samplie/GAN_MNIST10.png)

epoch 20
![image](https://github.com/0cars0903/GAN_WITH_MNIST/blob/main/GAN_MNIST_Samplie/GAN_MNIST20.png)

epoch 40
![image](https://github.com/0cars0903/GAN_WITH_MNIST/blob/main/GAN_MNIST_Samplie/GAN_MNIST40.png)

epoch 80
![image](https://github.com/0cars0903/GAN_WITH_MNIST/blob/main/GAN_MNIST_Samplie/GAN_MNIST80.png)

epoch 160
![image](https://github.com/0cars0903/GAN_WITH_MNIST/blob/main/GAN_MNIST_Samplie/GAN_MNIST60.png)

epoch 200
![image](https://github.com/0cars0903/GAN_WITH_MNIST/blob/main/GAN_MNIST_Samplie/GAN_MNIST200.png)
