import torch
import torch.nn as nn
from options import MsganInterpolationOptions
from model import CDCGAN
from saver import *
import os
import random
import sys
import torchvision
import torchvision.transforms as transforms
from models import *
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from keras.utils import np_utils
from numpy import linspace
from numpy import asarray
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image


#label_name=[[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]
label_name=[]
test_class_images =[]
generate_class_images = []
#label_name=[[0]]

# uniform interpolation between two points in latent space
def interpolate_points(p1, p2, n_steps):
	# interpolate ratios between the points
	ratios = linspace(0, 1, num=n_steps)
	# linear interpolate vectors
	vectors = list()
	vectors.append(p1)
	#print(p1)
	for ratio in ratios:
		v = (1.0 - ratio) * p1 + ratio * p2
		vectors.append(v)
	vectors.append(p2)
	return asarray(vectors)

def main():
    # parse options
    parser = MsganInterpolationOptions()
    opts = parser.parse()

    if (opts.name == "CIFAR10"):
      n_classe = 10
      label_names=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    elif (opts.name == "CIFAR100"):
      n_classe = 100
      label_names=['beaver', 'dolphin', 'otter', 'seal', 'whale','aquarium' 'fish',
      'flatfish', 'ray', 'shark', 'trout','orchids', 'poppies', 'roses', 'sunflowers', 'tulips','bottles', 
      'bowls', 'cans', 'cups', 'plates','apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers',	'clock', 
      'computer keyboard', 'lamp', 'telephone', 'television','bed', 'chair', 'couch', 'table', 'wardrobe',	'bee', 
      'beetle', 'butterfly', 'caterpillar', 'cockroach','bear', 'leopard', 'lion', 'tiger', 'wolf','bridge', 'castle', 
      'house', 'road', 'skyscraper','cloud', 'forest', 'mountain', 'plain', 'sea','camel', 'cattle', 'chimpanzee', 
      'elephant', 'kangaroo','fox', 'porcupine', 'possum', 'raccoon', 'skunk','crab', 'lobster', 'snail', 'spider', 
      'worm','baby', 'boy', 'girl', 'man', 'woman','crocodile', 'dinosaur', 'lizard', 'snake', 'turtle','hamster', 'mouse', 
      'rabbit', 'shrew', 'squirrel','maple', 'oak', 'palm', 'pine', 'willow','bicycle', 'bus', 'motorcycle', 'pickup' 'truck', 
      'train','awn-mower', 'rocket', 'streetcar', 'tank', 'tractor']

    for it in range(n_classe):
      label_name.append([it])
      test_class_images.append(0)
      generate_class_images.append(0)

    # data loader
    #print('\n--- load label_name to tensor ---')
    targets = torch.tensor(label_name)

    # model
    print('\n--- load model ---')
    torch.manual_seed(0)
    #torch.cuda.manual_seed(0)

    model = CDCGAN(opts)
    model.eval()
    model.setgpu(opts.gpu)
    model.resume(opts.resume, train=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = ResNet18()
    net = net.to(device)

    checkpoint = torch.load(opts.cnnresume,map_location=device)
    net.load_state_dict(checkpoint['net'], strict=False)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print("ACC --- ",best_acc)
    net.eval()
    ## cnn cifar10 or cifar100 orinal 10ep or 15ep

    # directory
    result_dir = os.path.join(opts.result_dir, opts.name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    all_imgs = []
    all_class = []
    #count = 0
    progress = 100
    stop_generetor = 0
    # test
    print('\n--- Gerator z example ---')
    for label in targets:
        print("--------------------------")
        imgs = []
        namesimgs = []
        classe = []
        label_id = label[0].numpy()
        print("label_names ",label_names[label_id])
        test_class_images[label_id] += 1
        for idx2 in range(opts.num):
            generate_class_images[label_id] += 1
            if(progress == idx2):
              print("progress in ",idx2,".it","--- (Z) save", len(imgs))
              progress +=100
            if(len(imgs)==0):
              #print(stop_generetor,"Falha")
              stop_generetor +=1
            if(stop_generetor == 10):
              print("Stop (z), because no example was generat ")
              stop_generetor == 0
              break  
            with torch.no_grad():
                img = model.test_forward(label)
            ## chamar cnn input img feature and logit
            output = net(img)
            _, predicted = output.max(1)
            predicted = predicted.cpu()
            #print(predicted)
            correct = predicted.eq(label).sum().item()
            #print(correct)
            if(correct == 0): # outra variação pode ser entropia
              if(len(imgs) == 5000):
                #count = 0
                progress = 100
                print("Save 5000 img by class")
                break
              #count += 1
              imgs.append(img)
              all_imgs.append(img)
              classe.append(label_names[label_id])
              all_class.append(label_names[label_id])
              namesimgs.append('img_{}'.format(generate_class_images[label_id]))
        
        save_imgs(imgs, namesimgs, os.path.join(result_dir, '{}'.format(label_names[label_id])))
 
    # Interpolation
    print("\n--- Gerator linear Interpolation ---")
    print(len(all_imgs))
    if(len(all_imgs) > 50000):
      print("No examples to interpolation")
      sys.exit()
  
    n = random.randint(0, len(all_imgs))
    n_space = 200
    print(n)
    print(all_class[n],"<------->",all_class[n+n_space])
    n_steps = 98
    #n_steps = 8


    interpolated = interpolate_points(all_imgs[n],all_imgs[n + n_space], n_steps)
    print(all_class[n],"==>",len(interpolated) ,"Example")

    img_interpolated = []
    names_imgs = []
    
    #for inputs in interpolated:
    for idx, inputs in enumerate(interpolated):
      #inputs =  inputs.to(device)
      output = net(inputs)
      _, predicted = output.max(1)
      predicted = predicted.cpu()
      correct = predicted.eq(targets[0]).sum().item()
      if(correct == 0):
        img_interpolated.append(inputs)
        names_imgs.append('imgInterpolated_'+str(idx))
        image = inputs.data.cpu().numpy()[0]
        image = (1/(2*2.25)) * image + 0.5
        image = np.transpose(image, (1,2,0))
        plt.imshow(image)
        plt.savefig('/content/results/'+str(idx)+'.png')
    #save_imgs(img_interpolated, names_imgs, os.path.join(result_dir, '{}'.format('interpolated_'+label_names[0])))
        #break

    return

if __name__ == '__main__':
    main()
