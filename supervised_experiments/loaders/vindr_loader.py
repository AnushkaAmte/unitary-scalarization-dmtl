import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class BBoxDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, img_size=(512, 512)):
       
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transforms.Compose([
                    transforms.Resize(size=img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]) if transform is None else transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # Get the image information from the CSV
        row = self.annotations.iloc[index]
        img_name = row['image_id'] + '.jpg'  # Assuming images are in jpg format
        img_path = os.path.join(self.root_dir, img_name)
        
        # Load the image
        image = Image.open(img_path).convert('RGB')

        # Get bounding box coordinates and class label
        bbox = torch.tensor([row['x_min'], row['y_min'], row['x_max'], row['y_max']], dtype=torch.float32)
        class_id = torch.tensor(row['class_id'], dtype=torch.long)
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Return image, bounding box, and class label
        return image, bbox, class_id
