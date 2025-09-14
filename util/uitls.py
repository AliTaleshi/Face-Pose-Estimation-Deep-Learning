import numpy as np
import torch
import os
import scipy.io as sio
import cv2
import math
from math import cos, sin

def softmax_temperature(tensor, temperature):
    result = torch.exp(tensor / temperature)
    result = torch.div(result, torch.sum(result, 1).unsqueeze(1).expand_as(result))
    return result

def get_pose_params_from_mat(mat_path):
    mat = sio.loadmat(mat_path)
    pre_pose_params = mat['Pose_Para'][0]
    pose_params = pre_pose_params[:5]
    return pose_params

def get_ypr_from_mat(mat_path):
    mat = sio.loadmat(mat_path)
    pre_pose_params = mat['Pose_Para'][0]
    pose_params = pre_pose_params[:3]
    return pose_params

def get_pt2d_from_mat(mat_path):
    # Get 2D landmarks
    mat = sio.loadmat(mat_path)
    pt2d = mat['pt2d']
    return pt2d

def mse_loss(input, target):
    return torch.sum(torch.abs(input.data - target.data) ** 2)

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img

import os
import numpy as np
import matplotlib.pyplot as plt

def compute_metrics(yaw_errors, pitch_errors, roll_errors):
    yaw_mae = np.mean(yaw_errors)
    pitch_mae = np.mean(pitch_errors)
    roll_mae = np.mean(roll_errors)
    total_mae = (yaw_mae + pitch_mae + roll_mae) / 3

    yaw_rmse = np.sqrt(np.mean(np.array(yaw_errors) ** 2))
    pitch_rmse = np.sqrt(np.mean(np.array(pitch_errors) ** 2))
    roll_rmse = np.sqrt(np.mean(np.array(roll_errors) ** 2))
    total_rmse = (yaw_rmse + pitch_rmse + roll_rmse) / 3

    yaw_std = np.std(yaw_errors)
    pitch_std = np.std(pitch_errors)
    roll_std = np.std(roll_errors)

    print("\n========== Performance Metrics ==========")
    print(f"MAE  (deg) - Yaw: {yaw_mae:.4f}, Pitch: {pitch_mae:.4f}, Roll: {roll_mae:.4f}, Total: {total_mae:.4f}")
    print(f"RMSE (deg) - Yaw: {yaw_rmse:.4f}, Pitch: {pitch_rmse:.4f}, Roll: {roll_rmse:.4f}, Total: {total_rmse:.4f}")
    print(f"STD  (deg) - Yaw: {yaw_std:.4f}, Pitch: {pitch_std:.4f}, Roll: {roll_std:.4f}")
    print("=========================================\n")

    return {
        "mae": {"yaw": yaw_mae, "pitch": pitch_mae, "roll": roll_mae, "total": total_mae},
        "rmse": {"yaw": yaw_rmse, "pitch": pitch_rmse, "roll": roll_rmse, "total": total_rmse},
        "std": {"yaw": yaw_std, "pitch": pitch_std, "roll": roll_std}
    }

def generate_plots(yaw_errors, pitch_errors, roll_errors,
                   yaw_labels, pitch_labels, roll_labels,
                   yaw_preds, pitch_preds, roll_preds,
                   output_dir):
    """Generate error histograms, cumulative error distribution, and scatter plots."""
    plots_path = os.path.join(output_dir, "plots")
    os.makedirs(plots_path, exist_ok=True)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.hist(yaw_errors, bins=50, color='skyblue', edgecolor='black')
    plt.title("Yaw Error Distribution"); plt.xlabel("Error (deg)"); plt.ylabel("Count")

    plt.subplot(1, 3, 2)
    plt.hist(pitch_errors, bins=50, color='lightgreen', edgecolor='black')
    plt.title("Pitch Error Distribution"); plt.xlabel("Error (deg)"); plt.ylabel("Count")

    plt.subplot(1, 3, 3)
    plt.hist(roll_errors, bins=50, color='salmon', edgecolor='black')
    plt.title("Roll Error Distribution"); plt.xlabel("Error (deg)"); plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig(os.path.join(plots_path, "error_histograms.png"))
    plt.close()

    thresholds = np.arange(0, 31, 1)
    def ced_curve(errors): return [np.mean(np.array(errors) < t) * 100 for t in thresholds]

    plt.figure()
    plt.plot(thresholds, ced_curve(yaw_errors), label="Yaw")
    plt.plot(thresholds, ced_curve(pitch_errors), label="Pitch")
    plt.plot(thresholds, ced_curve(roll_errors), label="Roll")
    plt.title("Cumulative Error Distribution")
    plt.xlabel("Error Threshold (deg)"); plt.ylabel("Samples (%)")
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(plots_path, "cumulative_error_distribution.png"))
    plt.close()

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.scatter(yaw_labels, yaw_preds, alpha=0.5, c="skyblue")
    plt.plot([-99, 99], [-99, 99], 'r--')
    plt.title("Yaw: Predicted vs True"); plt.xlabel("True (deg)"); plt.ylabel("Predicted (deg)")

    plt.subplot(1, 3, 2)
    plt.scatter(pitch_labels, pitch_preds, alpha=0.5, c="lightgreen")
    plt.plot([-99, 99], [-99, 99], 'r--')
    plt.title("Pitch: Predicted vs True"); plt.xlabel("True (deg)"); plt.ylabel("Predicted (deg)")

    plt.subplot(1, 3, 3)
    plt.scatter(roll_labels, roll_preds, alpha=0.5, c="salmon")
    plt.plot([-99, 99], [-99, 99], 'r--')
    plt.title("Roll: Predicted vs True"); plt.xlabel("True (deg)"); plt.ylabel("Predicted (deg)")

    plt.tight_layout()
    plt.savefig(os.path.join(plots_path, "pred_vs_true_scatter.png"))
    plt.close()

    print(f"Plots saved in {plots_path}")
