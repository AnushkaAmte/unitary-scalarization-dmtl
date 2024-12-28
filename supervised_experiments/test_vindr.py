## Basic libraries
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glob import glob
from sklearn.model_selection import train_test_split
# import pydicom
from skimage.transform import resize

## Torchvision libraries
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# from torch.utils.tensorboard import SummaryWriter

## Image augmentations
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

## Helper libraries
#from engine import train_one_epoch, evaluate
import torch.utils as utils
#import transforms as T

from lung_images import LungImages
import argparse

 ## Parent path to data
path = '/data6/rajivporana_scratch/vindr_data/'

boxCSV = 'train.csv'
# dataCSV = 'stage_2_detailed_class_info.csv'

imageFolders = ['png_train/', 'png_test/']

## First class is background
classes = [-1, 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]

## Create the base model to be trained
def detectionModel(numClasses):
    ## Load a model pretrained resnet model to speed training time
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    ## Get number of input features for the classifier
    inFeatures = model.roi_heads.box_predictor.cls_score.in_features
    ## Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(inFeatures, numClasses) 

    return model

def global_transformer():
    return transforms.Compose([transforms.ToTensor()])

## Data transformations during training to reduce overfit
def createTransform(train):
    if train:
        return A.Compose([A.HorizontalFlip(0.5), ToTensorV2(p=1.0)], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    else:
        return A.Compose([ToTensorV2(p=1.0)], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def load_model(model, folder, net_basename, name):
    state = torch.load(f"{folder}{net_basename}_{name}_model.pkl")
    #state = torch.load(f"{folder}{net_basename}")
    #model.load_state_dict(state['rep'])
    filtered_state_dict = {k: v for k, v in state['model_rep'].items() if k in model.state_dict()}

# Load the filtered state dictionary into the model
    num_classes = 16  # Set this to the number of classes in your dataset
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(filtered_state_dict, strict=False)
    return model


def ols_score(fname, image_size, lung_segment_path, heatmap_hr,heatmap_lr, row, grad_type):
    lung_region = np.load(os.path.join(lung_segment_path,fname.replace(".png",".npy")))
    lung_region = resize(lung_region, (image_size, image_size))
        
    heatmap_hr = heatmap_hr.cpu().detach().numpy()
    heatmap_lr = heatmap_lr.cpu().detach().numpy()        
    if np.sum(lung_region) > 0:
        for threshold in [0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
            # print("Numerator :", np.sum((heatmap_hr>threshold)), np.sum((lung_region>0)) , np.sum((heatmap_hr>threshold)*(lung_region>0)))
            # print("Denominator :", np.sum(heatmap_hr>threshold)+0.00000001)
            dice_hr = (np.sum((heatmap_hr>threshold)*(lung_region>0))/(np.sum(heatmap_hr>threshold)+0.00000001))
            dice_lr = (np.sum((heatmap_lr>threshold)*(lung_region>0))/(np.sum(heatmap_lr>threshold)+0.00000001))
            row[grad_type+'HR_Score_' + str(threshold)] = dice_hr
            row[grad_type+'LR_Score_' + str(threshold)] = dice_lr
            print("Threshold:{}, hr:{}, lr:{}".format(threshold, dice_hr, dice_lr))
    return row      



def test(args):
     
    height = 256
    width = 256
    pathDir = "/data6/rajivporana_scratch/vindr_data/png_test"
    wh_pathDir = "/data6/rajivporana_scratch/vindr_data/wh_proper_train/"
    lung_segment_path = "/data6/rajivporana_scratch/vindr_bbox/dataset/png_test_lung_segment/"
    target_dir = "/data6/anushkapa_scratch/unitary-scalarization-dmtl/vindr/saved_results/"
    dataTest = LungImages(pathDir, wh_pathDir, height, width, 
            classes, transforms =  global_transformer())

    ## Split the dataset into train and test sets
    torch.manual_seed(1)

    dataTestLoader = torch.utils.data.DataLoader(dataTest, batch_size=1, shuffle=False, num_workers=4, collate_fn=None)
    print("Dataloader length :", len(dataTestLoader))

    ## Determine if we can use a GPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ## Initialize model
    numClasses = 1
    model = detectionModel(numClasses)

    model = load_model(model, args.model_folder, args.net_basename, args.model_name)
    model.to(device)

    #df = compute_rho(args, model, dataTestLoader, lung_segment_path, device=device)d
    df = pd.DataFrame()
    filename = args.net_basename + ".csv"  
    df.to_csv(os.path.join(target_dir, filename), index=False)                              


def main():
    parser = argparse.ArgumentParser(description='Multi-task Learning Trainer')
    parser.add_argument('--net_basename', type=str, default='vindr_obj_det_5', help='model name')
    parser.add_argument('--model_name', type=str, default='best', help='model name')
    parser.add_argument('--model_folder', type=str, default='/data6/anushkapa_scratch/unitary-scalarization-dmtl/vindr/saved_models/saved_models/', help='model folder')
    args = parser.parse_args()
    test(args)


main()

#python3 test_vindr.py --net_basename vindr_0-lr:0.01-wd:0.0005_spring-donkey-30 --model_name last