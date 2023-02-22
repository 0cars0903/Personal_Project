# CycleGAN을 이용한 프로젝트

## CycleGAN 공부
### GAN이란?
* Data의 분포를 변형시켜서 원하는 데이터와 유사하게 만들어 낸다.(세상에 존재하지 않는 데이터 생성!)  
* G(generator) : input Data의 분포를 변형시켜서 Real Data와 유사한 Fake Data를 생성하는 model  
* D(Discriminator) : input Data가 Fake Data인지 Real Data인지 판별하는 model  
* G는 더욱 유사한 data를 생성하도록 학습하고, D는 가짜를 잘 찾아내도록 학습한다!  

### CycleGAN의 차이점?
* 이전 model인 pix2pix model의 단점인 paired data를 해결하였다.  
* X domain과 Y domain 간에 image를 변환시킨다. (사과를 오렌지처럼, 말을 얼룩말처럼)
* G는 X domain의 A image를 Y domain의 Feature와 유사하게 만들어 내기 위해 학습된다. 그러나 유사하게만 만들어 낸다면 기존의 image의 형태가 소실되는 문제가 발생한다. 이를 해결하기 위해서 순환구조의 생성모델을 사용하게 된다. image A에서 G를 통해 image B라는 Fake image를 생성한다. image B는 F를 통과하여 다시 image A로 돌아가는 데, 이때 원본 image A와 유사할수록 Y domain의 Feature만 잘 골라와서 변형되어졌다고 할 수 있다. 

## Project
벚꽃 이미지를 활용해서 거리 나무를 벚꽃 나무로 만들기

## Training
이번 프로젝트에서 사용한 CycleGAN model은 [원본 링크](https://github.com/aitorzip/PyTorch-CycleGAN.git)의 model을 사용하였다.

### Dataset
벚꽃 이미지는 Selenium을 이용하여 구글에서 '벚꽃 이미지'를 검색, 웹 크롤링을 통해 수집하였다.
거리 이미지는 
Train 00장, Test 00장
Data structure는 다음과 같다.:

    .
    ├── datasets                   
    |   ├── <dataset_name>         # i.e. brucewayne2batman
    |   |   ├── train              # Training
    |   |   |   ├── A              # Contains domain A images (i.e. Bruce Wayne)
    |   |   |   └── B              # Contains domain B images (i.e. Batman)
    |   |   └── test               # Testing
    |   |   |   ├── A              # Contains domain A images (i.e. Bruce Wayne)
    |   |   |   └── B              # Contains domain B images (i.e. Batman)

## Error Code
### 1. caught runtimeerror in dataloader worker process 3
opt.n_cpu -> 0 
```
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True), 
                        batch_size=opt.batchSize, shuffle=True, num_workers=0)
``` 
#### Dataload Multi-Processing? (https://jybaek.tistory.com/799) 
* dataloader에서 random crop, shuffle등은 cpu의 영역이다. 가용 cpu 스레드 개수를 n개를 주면, 그만큼 cpu의 스레드(일꾼) n가 일해서 dataload에서 걸리는 보틀넥 현상 방지
![image](https://user-images.githubusercontent.com/111993984/220262291-5a131145-e517-46cd-b699-c4ecd7141da4.png)

### 2. can't convert cuda:0 device type tensor to numpy.
gpu에 할당되어 있는 tensor를 numpy 배열로 변환할 때 생기는 에러

torch 라이브러리 내부의 _tensor.py에서 self.numpy()를 self.cpu().numpy()로 변경

## Train!
```
python train.py --n_epochs 50 --dataroot datasets/apple2orange/ --decay_epoch 25 --cuda
``` 
### Loss
Torch visdom을 활용하여 Loss 시각화
[http://localhost:8097/](http://localhost:8097/)

![Generator loss](https://github.com/ai-tor/PyTorch-CycleGAN/raw/master/output/loss_G.png)
![Discriminator loss](https://github.com/ai-tor/PyTorch-CycleGAN/raw/master/output/loss_D.png)
![Generator GAN loss](https://github.com/ai-tor/PyTorch-CycleGAN/raw/master/output/loss_G_GAN.png)
![Generator identity loss](https://github.com/ai-tor/PyTorch-CycleGAN/raw/master/output/loss_G_identity.png)
![Generator cycle loss](https://github.com/ai-tor/PyTorch-CycleGAN/raw/master/output/loss_G_cycle.png)

## Test
```
./test --dataroot datasets/<dataset_name>/ --cuda
```

![Real horse](https://github.com/ai-tor/PyTorch-CycleGAN/raw/master/output/real_A.jpg)
![Fake zebra](https://github.com/ai-tor/PyTorch-CycleGAN/raw/master/output/fake_B.png)
![Real zebra](https://github.com/ai-tor/PyTorch-CycleGAN/raw/master/output/real_B.jpg)
![Fake horse](https://github.com/ai-tor/PyTorch-CycleGAN/raw/master/output/fake_A.png)