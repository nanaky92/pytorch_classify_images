import numpy as np
import torch
from PIL import Image


def process_image(image_path, resize_size=256, crop_size=224):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a Numpy array
    '''
    image = Image.open(image_path)
    
    # Resize
    old_width, old_height = image.size
    min_dim = min(old_width, old_height)
    ratio = resize_size/min_dim
    image.thumbnail((ratio*old_width, ratio*old_height))
    
    # Crop
    old_width, old_height = image.size
    width_ref = (old_width-crop_size)/2
    heigth_ref = (old_height-crop_size)/2

    left, botton, right, top = width_ref, heigth_ref, crop_size+width_ref, crop_size+heigth_ref
    image = image.crop( ( left, botton, right, top ) )  
    
    # Translate to PyTorch model input
    np_image = np.array(image)
    np_image = np_image/255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    std_image = (np_image-mean)/std

    # PyTorch has the color dim the first, PIL the last
    std_image = std_image.transpose(2,0,1)

    return std_image


def predict_top_k_image(image_path, model, device='cpu', topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    np_image = process_image(image_path)
    torch_image = torch.from_numpy(np_image)
    torch_image = torch_image.unsqueeze_(0)
    torch_image = torch_image.to(device, dtype=torch.float)
    model.eval()
    model.to(device)
    with torch.no_grad():
        logps = model.forward(torch_image)
        ps = torch.exp(logps)
        
        top_k_ps, top_k_idxs = ps.topk(topk)
        
        top_k_ps = top_k_ps.to('cpu').data.numpy().tolist()[0]
        top_k_idxs = top_k_idxs.to('cpu').data.numpy().tolist()[0]

        idx_to_class = {v: k for k, v in model.class_to_idx.items()} 
        top_k_classes = [idx_to_class[top_k_idx] for top_k_idx in top_k_idxs]

        return top_k_ps, top_k_classes
