import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch
from imageio import imread
from lung_segmentation import segmentation_func,PretrainedUNet
from skimage.transform import resize
import numpy as np
from tqdm import tqdm
import pandas as pd
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu',
                        help='Torch device to use')
    parser.add_argument('--original_image_path', type=str,
            default='/data6/rajivporana_scratch/datasets/covid_binary_test/1_COVID-19/',
            help='Input image path')

    parser.add_argument('--output-dir', type=str, 
            default='/data6/anushkapa_scratch/unitary-scalarization-dmtl/heatmaps/csvfiles/grad_cam/',
            help='Output directory to save the result')
    parser.add_argument('--heatmap_dir', type=str,
            default='/scratch/mariamma/mtl/heatmap/Pne_Nor_Cov0.0_magic-smoke-320/',
            help='Heatmap directory')
    parser.add_argument('--model_cam_name_hr', type=str, default='original/', 
            help='model-cam-combination to evaluate ols')
    parser.add_argument('--model_cam_name_lr', type=str, default='whatsapp/', 
            help='model-cam-combination to evaluate ols')   
    parser.add_argument('--model_cam_name', type=str, default='', 
            help='model-cam-combination to evaluate ols')                        
    parser.add_argument('--image_size', type=int, default=64, help='image size')
    args = parser.parse_args()
    
    if args.device:
        print(f'Using device "{args.device}" for acceleration')
    else:
        print('Using CPU for computation')

    return args



def load_segmentation_model(model_path):  
    """
        Loading the lung segmentation Model
        
        inputs - 
            model_path: path to the .pt file

        output -
            segmentation_model: lung segmentation model loaded on the cuda
            device: torch.device("cpu") or torch.device("cuda")
    """
    
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    segmentation_model = torch.load(model_path)
    segmentation_model.to(device)
    segmentation_model.eval()
    return (segmentation_model, device)


def get_dataframe(args, segmentation_model, device):

    """
        Function to get dataframe of diceindex between hr and lr images
    """
    heatmap_hr_dir = os.path.join(args.heatmap_dir, args.model_cam_name_hr)
    heatmap_lr_dir = os.path.join(args.heatmap_dir, args.model_cam_name_lr)

    filenames = [f for f in os.listdir(args.original_image_path) if not f.startswith('.')]
    df = pd.DataFrame(columns = ['Image_Name', 'HR_Score_0.4', 'LR_Score_0.4',\
        'HR_Score_0.5', 'LR_Score_0.5', 'HR_Score_0.55', \
        'LR_Score_0.55', 'HR_Score_0.6', 'LR_Score_0.6', 'HR_Score_0.65', 'LR_Score_0.65', \
        'HR_Score_0.7', 'LR_Score_0.7', 'HR_Score_0.75', 'LR_Score_0.75'])

    iter_filenames = tqdm(enumerate(filenames))

    
    for idx, fname in iter_filenames:
        row = {}
        lung_region = segmentation_func(os.path.join(args.original_image_path,fname),segmentation_model,device)
        #print(type(lung_region[0]))
        #if isinstance(lung_region, tuple):
        #    lung_region = np.array(lung_region)
        lung_region = resize(lung_region[0], (args.image_size, args.image_size))
        
        image_name = fname
        if ".png" in image_name:
            image_name_pt = image_name.replace(".png", ".pt")
        elif ".jpg" in image_name:
            image_name_pt = image_name.replace(".jpg", ".pt")
        elif ".jpeg" in image_name:
            image_name_pt = image_name.replace(".jpeg", ".pt")
        elif ".JPG" in image_name:
            image_name_pt = image_name.replace(".JPG", ".pt")              
        heatmap_hr = torch.load(os.path.join(heatmap_hr_dir, image_name_pt))
        heatmap_hr = heatmap_hr.cpu().numpy()
        heatmap_lr = torch.load(os.path.join(heatmap_lr_dir, image_name_pt))
        heatmap_lr = heatmap_lr.cpu().numpy()
        
        row['Image_Name'] = fname
        for threshold in [0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
            #fix this to floats
            dice_hr = (np.sum((heatmap_hr>threshold)*(lung_region>0))/(np.sum(heatmap_hr>threshold)+0.0000001))
            dice_lr = (np.sum((heatmap_lr>threshold)*(lung_region>0))/(np.sum(heatmap_lr>threshold)+0.0000001))
            row['HR_Score_' + str(threshold)] = dice_hr
            row['LR_Score_' + str(threshold)] = dice_lr
        df = df.append(row, ignore_index = True)
        
    return df


def main(args):
    """main function"""    

    print("OLS Score of {}".format(args.model_cam_name))
    model_path = '/data6/anushkapa_scratch/lung_segmentation.pt'
    segmentation_model, device = load_segmentation_model(model_path) 
   
    df = get_dataframe(args, segmentation_model, device)
    
    df.to_csv(os.path.join(args.output_dir, "ols_"+args.model_cam_name+".csv"))


if __name__ == '__main__':
    args = get_args()
    main(args)

#python3 supervised_experiments/ols_score.py --heatmap_dir /data6/anushkapa_scratch/unitary-scalarization-dmtl/heatmaps/apricot_fire_8_norm_sq/Ate_Pne_Nor_Cov_baseline-lr:0.001-wd:0.0_apricot-fire-8 --model_cam_name Ate_Pne_Nor_Cov_baseline-lr:0.001-wd:0.0_apricot-fire-8