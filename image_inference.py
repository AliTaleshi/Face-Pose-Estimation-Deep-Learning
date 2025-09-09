import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os

import util.uitls as utils
import hopenet

def main():
    parser = argparse.ArgumentParser(description='Hopenet inference on images')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to the trained model .pkl file')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to the input image or directory containing images')
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='Directory to save output images')
    parser.add_argument('--num_bins', type=int, default=66,
                       help='Number of bins for classification (default: 66)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device id to use (default: 0)')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    cuda_available = torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if cuda_available else "cpu")
    print(f"Using device: {device}")

    print("Loading model...")
    model = hopenet.Hopenet(models.resnet.Bottleneck, [3, 4, 6, 3], args.num_bins)
    
    saved_state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(saved_state_dict)
    
    model.to(device)
    model.eval()

    transformations = transforms.Compose([
        transforms.Resize(240),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    idx_tensor = [idx for idx in range(args.num_bins)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(device)

    if os.path.isfile(args.image_path):
        image_paths = [args.image_path]
    else:
        image_paths = [os.path.join(args.image_path, f) for f in os.listdir(args.image_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Found {len(image_paths)} image(s) to process")

    for image_path in image_paths:
        print(f"Processing {image_path}...")
        
        orig_image = cv2.imread(image_path)
        if orig_image is None:
            print(f"Could not load image: {image_path}")
            continue
            
        rgb_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        input_tensor = transformations(pil_image)
        input_batch = input_tensor.unsqueeze(0)
        input_batch = input_batch.to(device)
        
        with torch.no_grad():
            yaw, pitch, roll = model(input_batch)
        
        yaw_pred = utils.softmax_temperature(yaw.data, 1)
        pitch_pred = utils.softmax_temperature(pitch.data, 1)
        roll_pred = utils.softmax_temperature(roll.data, 1)
        
        yaw_pred = torch.sum(yaw_pred * idx_tensor, 1).cpu().numpy() * 3 - 99
        pitch_pred = torch.sum(pitch_pred * idx_tensor, 1).cpu().numpy() * 3 - 99
        roll_pred = torch.sum(roll_pred * idx_tensor, 1).cpu().numpy() * 3 - 99
        
        print(f"Predicted angles - Yaw: {yaw_pred[0]:.2f}, Pitch: {pitch_pred[0]:.2f}, Roll: {roll_pred[0]:.2f}")
        
        center_x = orig_image.shape[1] // 2
        center_y = orig_image.shape[0] // 2
        output_image = utils.draw_axis(orig_image, yaw_pred[0], pitch_pred[0], roll_pred[0], center_x, center_y)
        
        text = f"Yaw: {yaw_pred[0]:.2f}, Pitch: {pitch_pred[0]:.2f}, Roll: {roll_pred[0]:.2f}"
        cv2.putText(output_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        filename = os.path.basename(image_path)
        output_path = os.path.join(args.output_dir, f"output_{filename}")
        cv2.imwrite(output_path, output_image)
        print(f"Saved result to {output_path}")

if __name__ == '__main__':
    main()
    