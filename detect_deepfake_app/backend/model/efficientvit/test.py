import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
import os
import cv2
import numpy as np
import torch
from torch import nn, einsum

from sklearn.metrics import f1_score
from albumentations import Compose, RandomBrightnessContrast, \
    HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur, Rotate

from transforms.albu import IsotropicResize

from utils import get_method, check_correct, resize, shuffle_dataset, get_n_params
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from efficient_vit import EfficientViT
from utils import transform_frame
import glob
from os import cpu_count
import json
from multiprocessing.pool import Pool
from progress.bar import Bar
import pandas as pd
from tqdm import tqdm
from multiprocessing import Manager
from utils import custom_round, custom_video_round

import yaml
import argparse


# MODELS_DIR = "models"
# BASE_DIR = '/content/drive/MyDrive/DeepFakes/Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection/'
# DATA_DIR = os.path.join(BASE_DIR, "dataset")
# TEST_DIR = os.path.join(DATA_DIR, "test_set")
# OUTPUT_DIR = os.path.join(MODELS_DIR, "tests")

# TEST_LABELS_PATH = os.path.join(BASE_DIR, "dataset/dfdc_test_labels.csv")



# if not os.path.exists(MODELS_DIR):
#     os.makedirs(MODELS_DIR)

# if not os.path.exists(OUTPUT_DIR):
#     os.makedirs(OUTPUT_DIR)

# def create_base_transform(size):
#     return Compose([
#         IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
#         PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
#     ])

# def save_roc_curves(correct_labels, preds, model_name, accuracy, loss, f1):
#   plt.figure(1)
#   plt.plot([0, 1], [0, 1], 'k--')

#   fpr, tpr, th = metrics.roc_curve(correct_labels, preds)

#   model_auc = auc(fpr, tpr)


#   plt.plot(fpr, tpr, label="Model_"+ model_name + ' (area = {:.3f})'.format(model_auc))

#   plt.xlabel('False positive rate')
#   plt.ylabel('True positive rate')
#   plt.title('ROC curve')
#   plt.legend(loc='best')
#   plt.savefig(os.path.join(OUTPUT_DIR, model_name +  "_" + opt.dataset + "_acc" + str(accuracy*100) + "_loss"+str(loss)+"_f1"+str(f1)+".jpg"))
#   plt.clf()

# def read_frames(video_path, videos):
    
#     # Get the video label based on dataset selected
#     method = get_method(video_path, DATA_DIR)
#     if "Original" in video_path:
#         label = 0.
#     elif method == "DFDC":
#         test_df = pd.DataFrame(pd.read_csv(TEST_LABELS_PATH))
#         video_folder_name = os.path.basename(video_path)
#         video_key = video_folder_name + ".mp4"
#         label = test_df.loc[test_df['filename'] == video_key]['label'].values[0]
#     else:
#         label = 1.
    

#     # Calculate the interval to extract the frames
#     frames_number = len(os.listdir(video_path))
#     frames_interval = int(frames_number / opt.frames_per_video)
#     frames_paths = os.listdir(video_path)
#     frames_paths_dict = {}

#     # Group the faces with the same index, reduce probabiity to skip some faces in the same video
#     for path in frames_paths:
#         for i in range(0,3):
#             if "_" + str(i) in path:
#                 if i not in frames_paths_dict.keys():
#                     frames_paths_dict[i] = [path]
#                 else:
#                     frames_paths_dict[i].append(path)

#     # Select only the frames at a certain interval
#     if frames_interval > 0:
#         for key in frames_paths_dict.keys():
#             if len(frames_paths_dict) > frames_interval:
#                 frames_paths_dict[key] = frames_paths_dict[key][::frames_interval]
            
#             frames_paths_dict[key] = frames_paths_dict[key][:opt.frames_per_video]

#     # Select N frames from the collected ones
#     video = {}
#     for key in frames_paths_dict.keys():
#         for index, frame_image in enumerate(frames_paths_dict[key]):
#             #image = np.asarray(resize(cv2.imread(os.path.join(video_path, frame_image)), IMAGE_SIZE))
#             transform = create_base_transform(config['model']['image-size'])
#             image = transform(image=cv2.imread(os.path.join(video_path, frame_image)))['image']
#             if len(image) > 0:
#                 if key in video:
#                     video[key].append(image)
#                 else:
#                     video[key] = [image]
#     videos.append((video, label, video_path))





# # Main body
# if __name__ == "__main__":
    
#     parser = argparse.ArgumentParser()
    
#     parser.add_argument('--workers', default=6, type=int,
#                         help='Number of data loader workers.')
#     parser.add_argument('--model_path', default='', type=str, metavar='PATH',
#                         help='Path to model checkpoint (default: none).')
#     parser.add_argument('--dataset', type=str, default='All', 
#                         help="Which dataset to use (Deepfakes|Face2Face|FaceShifter|FaceSwap|NeuralTextures|DFDC)")
#     parser.add_argument('--max_videos', type=int, default=-1, 
#                         help="Maximum number of videos to use for training (default: all).")
#     parser.add_argument('--config', type=str, 
#                         help="Which configuration to use. See into 'config' folder.")
#     parser.add_argument('--efficient_net', type=int, default=0, 
#                         help="Which EfficientNet version to use (0 or 7, default: 0)")
#     parser.add_argument('--frames_per_video', type=int, default=30, 
#                         help="How many equidistant frames for each video (default: 30)")
#     parser.add_argument('--batch_size', type=int, default=32, 
#                         help="Batch size (default: 32)")
    
#     opt = parser.parse_args()
#     print(opt)

#     with open(opt.config, 'r') as ymlfile:
#         config = yaml.safe_load(ymlfile)

#     if opt.efficient_net == 0:
#         channels = 1280
#     else:
#         channels = 2560

#     if os.path.exists(opt.model_path):
#         model = EfficientViT(config=config, channels=channels, selected_efficient_net = opt.efficient_net)
#         model.load_state_dict(torch.load(opt.model_path))
#         model.eval()
#         model = model.cuda()
#     else:
#         print("No model found.")
#         exit()

#     model_name = os.path.basename(opt.model_path)

#     if not os.path.exists(OUTPUT_DIR):
#         os.makedirs(OUTPUT_DIR)
        
    
#     preds = []
#     mgr = Manager()
#     paths = []
#     videos = mgr.list()

#     if opt.dataset != "All" and opt.dataset != "DFDC":
#         folders = ["Original", opt.dataset]
#     else:
#         folders = ["Original", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "GenAI"]

#     for folder in folders:
#         method_folder = os.path.join(TEST_DIR, folder)  
#         for index, video_folder in enumerate(os.listdir(method_folder)):
#             paths.append(os.path.join(method_folder, video_folder))
      
#     with Pool(processes=opt.workers) as p:
#         with tqdm(total=len(paths)) as pbar:
#             for v in p.imap_unordered(partial(read_frames, videos=videos),paths):
#                 pbar.update()

#     video_names = np.asarray([row[2] for row in videos])
#     correct_test_labels = np.asarray([row[1] for row in videos])
#     videos = np.asarray([row[0] for row in videos])
#     preds = []

#     bar = Bar('Predicting', max=len(videos))

#     f = open(opt.dataset + "_" + model_name + "_labels.txt", "w+")
#     for index, video in enumerate(videos):
#         video_faces_preds = []
#         video_name = video_names[index]
#         f.write(video_name)
#         for key in video:
#             faces_preds = []
#             video_faces = video[key]
#             for i in range(0, len(video_faces), opt.batch_size):
#                 faces = video_faces[i:i+opt.batch_size]
#                 # faces = torch.tensor(np.asarray(faces))
#                 image_size = config['model']['image-size']  # Lấy kích thước từ config.yaml, ví dụ: 224
#                 faces = [cv2.resize(face, (image_size, image_size)) for face in faces]
#                 faces = torch.tensor(np.array(faces))

#                 if faces.shape[0] == 0:
#                     continue
#                 faces = np.transpose(faces, (0, 3, 1, 2))
#                 faces = faces.cuda().float()
                
#                 pred = model(faces)
                
#                 scaled_pred = []
#                 for idx, p in enumerate(pred):
#                     scaled_pred.append(torch.sigmoid(p))
#                 faces_preds.extend(scaled_pred)
                
#             current_faces_pred = sum(faces_preds)/len(faces_preds)
#             face_pred = current_faces_pred.cpu().detach().numpy()[0]
#             f.write(" " + str(face_pred))
#             video_faces_preds.append(face_pred)
#         bar.next()
#         if len(video_faces_preds) > 1:
#             video_pred = custom_video_round(video_faces_preds)
#         else:
#             # video_pred = video_faces_preds[0]
#           if len(video_faces_preds) == 0:
#               # print(f"Warning: No faces predicted for video {video_path}, skipping.")
#               continue  # Bỏ qua video này
#           video_pred = video_faces_preds[0]
#         preds.append([video_pred])
        
#         f.write(" --> " + str(video_pred) + "(CORRECT: " + str(correct_test_labels[index]) + ")" +"\n")
        
#     f.close()
#     bar.finish()

#     loss_fn = torch.nn.BCEWithLogitsLoss()
#     tensor_labels = torch.tensor([[float(label)] for label in correct_test_labels])
#     tensor_preds = torch.tensor(preds)


#     loss = loss_fn(tensor_preds, tensor_labels).numpy()

#     #accuracy = accuracy_score(np.asarray(preds).round(), correct_test_labels)
#     accuracy = accuracy_score(custom_round(np.asarray(preds)), correct_test_labels)

#     f1 = f1_score(correct_test_labels, custom_round(np.asarray(preds)))
#     print(model_name, "Test Accuracy:", accuracy, "Loss:", loss, "F1", f1)
#     save_roc_curves(correct_test_labels, preds, model_name, accuracy, loss, f1)


# Đầu file giữ nguyên import như bạn đã viết
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import auc, accuracy_score, f1_score
import os
import cv2
import numpy as np
import torch
from albumentations import Compose, PadIfNeeded
from transforms.albu import IsotropicResize
from utils import get_method, custom_video_round
import pandas as pd
import yaml
import argparse
from tqdm import tqdm
from multiprocessing import Manager, Pool
from functools import partial
from efficientvit.efficient_vit import EfficientViT

# Cấu hình
MODELS_DIR = "models"
BASE_DIR = '/content/drive/MyDrive/DeepFakes/Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection/'
DATA_DIR = os.path.join(BASE_DIR, "dataset")
TEST_DIR = os.path.join(DATA_DIR, "test_set")
# OUTPUT_DIR = os.path.join(MODELS_DIR, "tests")
OUTPUT_DIR = "/content/drive/MyDrive/DeepFakes/Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection/efficient-vit/models (1)"
TEST_LABELS_PATH = os.path.join(BASE_DIR, "dataset/dfdc_test_labels.csv")

def create_base_transform(size):
    return Compose([
        IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
    ])

def save_roc_curves(correct_labels, preds, model_name, accuracy, loss, f1):
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    fpr, tpr, th = metrics.roc_curve(correct_labels, preds)
    model_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {model_auc:.3f})")
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_acc{accuracy*100:.2f}_loss{loss:.4f}_f1{f1:.4f}.jpg"))
    plt.clf()

def read_frames(video_path, videos):
    method = get_method(video_path, DATA_DIR)
    if "Original" in video_path:
        label = 0.
    elif method == "DFDC":
        test_df = pd.read_csv(TEST_LABELS_PATH)
        video_key = os.path.basename(video_path) + ".mp4"
        label = test_df.loc[test_df['filename'] == video_key]['label'].values[0]
    else:
        label = 1.

    frames_paths = os.listdir(video_path)
    frames_paths_dict = {}

    for path in frames_paths:
        for i in range(3):
            if f"_{i}" in path:
                frames_paths_dict.setdefault(i, []).append(path)

    opt_frames = opt.frames_per_video
    for k in frames_paths_dict:
        frames = frames_paths_dict[k]
        if len(frames) > opt_frames:
            frames_paths_dict[k] = frames[:opt_frames]

    video = {}
    for k in frames_paths_dict:
        images = []
        for f in frames_paths_dict[k]:
            image_path = os.path.join(video_path, f)
            image = cv2.imread(image_path)
            if image is None:
                continue
            transform = create_base_transform(config['model']['image-size'])
            image = transform(image=image)['image']
            images.append(image)
        if images:
            video[k] = images

    if video:
        videos.append((video, label, video_path))

# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='All')
    parser.add_argument('--max_videos', type=int, default=-1)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--efficient_net', type=int, default=0)
    parser.add_argument('--frames_per_video', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    opt = parser.parse_args()

    with open(opt.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)

    channels = 1280 if opt.efficient_net == 0 else 2560

    model = EfficientViT(config=config, channels=channels, selected_efficient_net=opt.efficient_net)
    model.load_state_dict(torch.load(opt.model_path))
    model.eval().cuda()

    folders = ["Original", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
    if opt.dataset != "All" and opt.dataset != "DFDC":
        folders = ["Original", opt.dataset]

    paths = []
    for folder in folders:
        for v in os.listdir(os.path.join(TEST_DIR, folder)):
            paths.append(os.path.join(TEST_DIR, folder, v))

    mgr = Manager()
    videos = mgr.list()
    with Pool(opt.workers) as pool:
        list(tqdm(pool.imap_unordered(partial(read_frames, videos=videos), paths), total=len(paths)))

    preds = []
    true_labels = []
    video_names = []

    for video_data, label, path in videos:
        video_faces_preds = []
        for key in video_data:
            faces = video_data[key]
            if not faces:
                continue
            image_size = config['model']['image-size']
            faces = [cv2.resize(face, (image_size, image_size)) for face in faces]
            faces = np.transpose(np.array(faces), (0, 3, 1, 2))
            faces_tensor = torch.tensor(faces).float().cuda()

            with torch.no_grad():
                logits = model(faces_tensor)
                scores = torch.sigmoid(logits).cpu().numpy()
                face_preds = scores.mean(axis=0)[0]
                video_faces_preds.append(face_preds)

        if not video_faces_preds:
            continue

        video_pred = custom_video_round(video_faces_preds) if len(video_faces_preds) > 1 else video_faces_preds[0]
        preds.append([video_pred])
        true_labels.append([label])
        video_names.append(path)

    preds = np.array(preds)
    true_labels = np.array(true_labels)

    # Tính loss, accuracy, F1
    loss_fn = nn.BCEWithLogitsLoss()
    loss = loss_fn(torch.tensor(preds), torch.tensor(true_labels)).item()
    acc = accuracy_score((preds > 0.5).astype(int), true_labels)
    f1 = f1_score(true_labels, (preds > 0.5).astype(int))

    print(f"✅ Done. Accuracy: {acc:.4f} | Loss: {loss:.4f} | F1: {f1:.4f}")
    save_roc_curves(true_labels, preds, os.path.basename(opt.model_path), acc, loss, f1)

    from sklearn.metrics import precision_score, recall_score
    import pandas as pd

    # Tính thêm chỉ số precision và recall
    precision = precision_score(true_labels, (preds > 0.5).astype(int))
    recall = recall_score(true_labels, (preds > 0.5).astype(int))

    print(f"Precision: {precision:.4f} | Recall: {recall:.4f}")

    # Lưu kết quả ra file Excel
    results_df = pd.DataFrame({
        "video_path": video_names,
        "true_label": [int(l[0]) for l in true_labels],
        "pred_score": [float(p[0]) for p in preds],
        "pred_label": [(1 if p[0] > 0.5 else 0) for p in preds],
    })

    # Thêm tổng quan vào cuối file
    summary_df = pd.DataFrame({
        "Accuracy": [acc],
        "Loss": [loss],
        "F1": [f1],
        "Precision": [precision],
        "Recall": [recall]
    })
    results_df = pd.concat([results_df, pd.DataFrame([{}]), summary_df], ignore_index=True)

    # Đảm bảo OUTPUT_DIR tồn tại
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    excel_path = os.path.join(OUTPUT_DIR, f"results_{os.path.basename(opt.model_path)}.xlsx")
    results_df.to_excel(excel_path, index=False)
    print(f"📁 Đã lưu kết quả vào: {excel_path}")