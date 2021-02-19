import torch
from options import TestOptions
from model import CDCGAN
from saver import *
import os
import torchvision
import torchvision.transforms as transforms
from models import *
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from keras.utils import np_utils

label_names=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
label_name=[[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]
#label_name=[[9]]


def main():
    # parse options
    parser = TestOptions()
    opts = parser.parse()

    # data loader
    print('\n--- load label_name to tensor ---')
    targets = torch.tensor(label_name)

    # model
    print('\n--- load model ---')
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    model = CDCGAN(opts)
    model.eval()
    model.setgpu(opts.gpu)
    model.resume(opts.resume, train=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = ResNet18()
    net = net.to(device)

    checkpoint = torch.load('/content/drive/MyDrive/pytorch-cifar-master/ckpt.pth',map_location=device)
    net.load_state_dict(checkpoint['net'], strict=False)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print("ACC --- ",best_acc)
    #net=torch.load('/content/drive/MyDrive/pytorch-cifar-master/cifar10_MSGAN-resnet18-noDict.pth', device)
    net.eval()
    ## cnn cifar10 orinal 5ep

    # directory
    result_dir = os.path.join(opts.result_dir, opts.name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    all_imgs = []
    all_class = []
    count = 0
    progress = 100
    # test
    print('\n--- testing ---')
    test_class_images =[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    generate_class_images = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for label in targets:
        print("--------------------------")
        imgs = []
        names = []
        classe = []
        label_id = label[0].numpy()
        print("label_names ",label_names[label_id])
        test_class_images[label_id] += 1
        for idx2 in range(opts.num):
            generate_class_images[label_id] += 1
            if(progress == idx2):
              print("progress in ",idx2,".it","--- (Z) save", len(imgs))
              progress +=100
            with torch.no_grad():
                img = model.test_forward(label)
            ## chamar cnn input img feature and logit
            output = net(img)
            _, predicted = output.max(1)
            predicted = predicted.cpu()
            #print(predicted)
            correct = predicted.eq(label).sum().item()
            #print(correct)# outra variação pode ser entropia
            if(correct == 0):
              if(count == 5000):
                count = 0
                progress = 100
                print("Save 5000 img by class")
                break
              count += 1
              imgs.append(img)
              all_imgs.append(img)
              classe.append(label_names[label_id])
              all_class.append(label_names[label_id])
              names.append('img_{}'.format(generate_class_images[label_id]))
        
        save_imgs(imgs, names, os.path.join(result_dir, '{}'.format(label_names[label_id])))
 
    #Save
    print('\n--- Save result MSGAN ---')
    np.savez(os.path.join(result_dir, '{}'.format(label_names[label_id]))+'/save_MSGan', imgs=all_imgs, classe=all_class) 
    return

if __name__ == '__main__':
    main()
