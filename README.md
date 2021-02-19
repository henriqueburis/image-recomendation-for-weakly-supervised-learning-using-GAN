## Image Recomendation for Weakly Supervised Learning Using GAN

![N|Solid](https://github.com/henriqueburis/image-recomendation-for-weakly-supervised-learning-using-GAN/blob/main/a1443bca-3401-44d5-9350-ef50c393f129.jpg?raw=true)

![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)


## Features

  - [Features cifar100 SNGAN Resnet50](https://drive.google.com/file/d/1gFfK7lzOqzJgRlAV4U8tstpGvYMtv8TX/view?usp=sharing)
  - [Features cifar10 SNGAN Resnet50](https://drive.google.com/file/d/1t_URo0NqnOJqQeR4kl-gpABUnhr_7Mw5/view?usp=sharing)
  - [Features cifar100 original Resnet50](https://drive.google.com/file/d/1-09ebn0a-v-jTy4uS1MRBcl8BxyBKKwc/view?usp=sharing)
  - [Features cifar10 original Resnet50](https://drive.google.com/file/d/1WZDPLqeRjC6IOAJSDV0JgriurMscBvWZ/view?usp=sharing)


## Samples SNGAN

  - [samples cifar10 SNGAN 200k img](https://drive.google.com/file/d/1-8VoomUgJgKWv6PjcUESSx-IkNMuEHdD/view?usp=sharing)
  - [samples cifar100 SNGAN 200k img](https://drive.google.com/file/d/17jgEoXO7p1uCpE4ET_c_1EANKirGw_XJ/view?usp=sharing)

## Samples SNGAN

  - [samples cifar10 MSGAN 50k img correct_0 acc 80.46%  img](https://drive.google.com/file/d/1Iz9S5cAUyKvg-4OCPAGdPE9jflUq0EYs/view?usp=sharing)

## Interpolation

airplane <--> airplane    |  bird <--> bird  | deer <--> dog | horse <--> ship
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/henriqueburis/image-recomendation-for-weakly-supervised-learning-using-GAN/blob/main/airplane.gif) |  ![](https://github.com/henriqueburis/image-recomendation-for-weakly-supervised-learning-using-GAN/blob/main/bird.gif) |  ![](https://github.com/henriqueburis/image-recomendation-for-weakly-supervised-learning-using-GAN/blob/main/deer-dog.gif) |  ![](https://github.com/henriqueburis/image-recomendation-for-weakly-supervised-learning-using-GAN/blob/main/horse_ship.gif)

## Installation

## MSGAM

```sh
generation of cifar10 examples, 5k img by classe
python3 test_msgan_interpolation.py --num=100000  --resume ${00199.pth }
```

```sh
Interpolation
python3 test_msgan_interpolation.py --num=100000  --resume ${00199.pth }
```
