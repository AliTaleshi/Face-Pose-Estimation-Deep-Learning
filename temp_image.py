import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os
import mediapipe as mp

import util.uitls as utils
import hopenet

def detect_faces(image, confidence_threshold=0.5):
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=confidence_threshold
    ) as face_detection:
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_image)
        
        faces = []
        if results.detections:
            h, w = image.shape[:2]
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                margin_x = int(width * 0.2)
                margin_y = int(height * 0.2)
                
                x1 = max(0, x - margin_x)
                y1 = max(0, y - margin_y)
                x2 = min(w, x + width + margin_x)
                y2 = min(h, y + height + margin_y)
                
                faces.append([x1, y1, x2, y2])
        return faces

def crop_and_square(image, bbox):
    x1, y1, x2, y2 = bbox
    face = image[y1:y2, x1:x2]
    
    h, w = face.shape[:2]
    if h == 0 or w == 0:
        return None
    
    size = max(h, w)
    top = (size - h) // 2
    bottom = size - h - top
    left = (size - w) // 2
    right = size - w - left
    
    squared = cv2.copyMakeBorder(face, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
    return squared

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
        transforms.Resize(224),
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
            
        faces = detect_faces(orig_image)
        if len(faces) == 0:
            print(f"No faces detected in {image_path}")
            cv2.putText(orig_image, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            output_path = os.path.join(args.output_dir, f"output_{os.path.basename(image_path)}")
            cv2.imwrite(output_path, orig_image)
            continue

        output_image = orig_image.copy()

        for i, bbox in enumerate(faces):
            x1, y1, x2, y2 = bbox
            
            face_crop = crop_and_square(orig_image, bbox)
            if face_crop is None:
                continue
                
            rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            pil_face = Image.fromarray(rgb_face)
            
            input_tensor = transformations(pil_face)
            input_batch = input_tensor.unsqueeze(0).to(device)
            
            with torch.no_grad():
                yaw, pitch, roll = model(input_batch)
            
            yaw_pred = utils.softmax_temperature(yaw.data, 1)
            pitch_pred = utils.softmax_temperature(pitch.data, 1)
            roll_pred = utils.softmax_temperature(roll.data, 1)
            
            yaw_deg = torch.sum(yaw_pred * idx_tensor, 1).cpu().numpy() * 3 - 99
            pitch_deg = torch.sum(pitch_pred * idx_tensor, 1).cpu().numpy() * 3 - 99
            roll_deg = torch.sum(roll_pred * idx_tensor, 1).cpu().numpy() * 3 - 99
            
            print(f"Face {i+1}: Yaw: {yaw_deg[0]:.2f}, Pitch: {pitch_deg[0]:.2f}, Roll: {roll_deg[0]:.2f}")
            
            face_center_x = (x1 + x2) // 2
            face_center_y = (y1 + y2) // 2
            output_image = utils.draw_axis(output_image, yaw_deg[0], pitch_deg[0], roll_deg[0], face_center_x, face_center_y)
            
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            text = f"Yaw: {yaw_deg[0]:.1f} Pitch: {pitch_deg[0]:.1f} Roll: {roll_deg[0]:.1f}"
            cv2.putText(output_image, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        filename = os.path.basename(image_path)
        output_path = os.path.join(args.output_dir, f"output_{filename}")
        cv2.imwrite(output_path, output_image)
        print(f"Saved result to {output_path}")

if __name__ == '__main__':
    main()