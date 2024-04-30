import os
from pathlib import Path
import random
import torch
from torch.utils.data import Dataset
import json
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import cv2
from mtcnn_cv2 import MTCNN


def get_a_face(path: Path, detector):
    ext = os.path.splitext(path)[-1]
    if ext in ('.mp4',):
        cap = cv2.VideoCapture(path)
        n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_pos = random.randint(0, n_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
    elif ext in ('.jpg', '.png'):
        frame = cv2.imread(path)
    else:
        raise ValueError("Wrong file format !")
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    bounding_boxes = detector.detect_faces(frame)
    
    if len(bounding_boxes) > 0:
        left, top, width, height = bounding_boxes[0]['box']
        return frame[top:top + height, left:left + width]
    else:
        return None


def add_corruption(corruption: str, severity: int):
    assert severity > 0 and severity < 6, "The severity should be 1 - 5 !"
    severity -= 1
    if corruption == 'contrast':
        contrast_factors = [0.85, 0.725, 0.6, 0.475, 0.35]
        return Contrast(contrast_factor=contrast_factors[severity])
    elif corruption == 'noise':
        std2 = [0.001, 0.002, 0.005, 0.01, 0.05]
        return GaussianNoise(std=std2[severity]**0.5)
    elif corruption == 'blur':
        kernel_sizes = [7, 9, 13, 17, 21]
        return transforms.GaussianBlur(kernel_size=kernel_sizes[severity], sigma=kernel_sizes[severity] / 6)
    elif corruption == 'pixelation':
        scaling_factors = [2, 3, 4, 5, 6]
        return Rescale(corruption='pixelation', scaling_factor=scaling_factors[severity])
    elif corruption == 'compression':
        scaling_factors = [2, 3, 4, 5, 6]
        return Rescale(corruption='compression', scaling_factor=scaling_factors[severity])
    else:
        raise NotImplementedError(f"{corruption} is not implemented !")
    
    
class Contrast(object):
    def __init__(self, contrast_factor: float=0.85) -> None:
        self.contrast_factor = contrast_factor
        
    def __call__(self, sample):
        return TF.adjust_contrast(sample, self.contrast_factor)
    
    
class GaussianNoise(object):
    def __init__(self, mean=0, std=0.1) -> None:
        self.std = std
        self.mean = mean
        
    def __call__(self, sample):
        return (sample + torch.randn(sample.shape) * self.std + self.mean).clamp(0, 1)
    
    
class Rescale(object):
    def __init__(self, corruption='pixelation', scaling_factor=2) -> None:
        assert corruption in ('pixelation', 'compression')
        self.scaling_factor = scaling_factor
        self.interpolation = transforms.InterpolationMode.NEAREST if corruption == 'pixelation' else transforms.InterpolationMode.BILINEAR
        
    def __call__(self, sample):
        c, h, w = sample.shape
        scale_down = TF.resize(sample, (h // self.scaling_factor, w // self.scaling_factor))
        return TF.resize(scale_down, (h, w), self.interpolation)


class ForgeryFaceDataset(Dataset):
    def __init__(self, dataset="FF++", mode="train", manipulation="all", quality="c23", transform=None) -> None:
        assert dataset in ("FF++", "CelebDF", "DFDC")
        assert mode in ("train", "val", "test")

        self.transform = transform

        self.detector = MTCNN()

        self.media = []
        self.label = []
        
        split_seed = random.Random(0)

        if dataset == "FF++":
            assert manipulation in ("all", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")
            assert quality in ("raw", "c23", "c40")
            
            dir="data/FF++"
            with open(os.path.join(dir, f"splits/{mode}.json"), "r") as f:
                data_ids = json.load(f)

            # 1 for real, 0 for fake
            for id1, id2 in data_ids:
                self.media.append(os.path.join(dir, "original_sequences/youtube", quality, "videos", f"{id1}.mp4"))
                self.media.append(os.path.join(dir, "original_sequences/youtube", quality, "videos", f"{id2}.mp4"))
                self.label += [1, 1]

                manipulations = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"] if manipulation == "all" else [manipulation]

                for m in manipulations:
                    self.media.append(os.path.join(dir, "manipulated_sequences", m, quality, "videos", f"{id1}_{id2}.mp4"))
                    self.media.append(os.path.join(dir, "manipulated_sequences", m, quality, "videos", f"{id2}_{id1}.mp4"))
                    self.label += [0, 0]

        elif dataset == "CelebDF":
            dir="data/Celeb-DF-v2"
            with open(os.path.join(dir, f"List_of_{'test' if mode == 'test' else 'train'}ing_videos.txt"), 'r') as t:
                for line in t:
                    label, file_name = line.split()
                    self.label.append(int(label))
                    self.media.append(os.path.join(dir, file_name))
                    
            if mode != 'test':
                indices = list(range(len(self.label)))
                split_seed.shuffle(indices)
                split_index = int(len(indices) * 0.8)
                subset_indices = indices[:split_index] if mode == 'train' else indices[split_index:]
                
                self.media = [self.media[i] for i in subset_indices]
                self.label = [self.label[i] for i in subset_indices]

        elif dataset == "DFDC":
            dir=os.path.join("data/DFDC", 'test' if mode == 'test' else 'train')
            for seq, lab in [("fake", 0), ("real", 1)]:
                media_dir = os.path.join(dir, seq)
                for vid in os.listdir(media_dir):
                    self.media.append(os.path.join(media_dir, vid))
                    self.label.append(lab)
                    
            if mode != 'test':
                indices = list(range(len(self.label)))
                split_seed.shuffle(indices)
                split_index = int(len(indices) * 0.8)
                subset_indices = indices[:split_index] if mode == 'train' else indices[split_index:]
                
                self.media = [self.media[i] for i in subset_indices]
                self.label = [self.label[i] for i in subset_indices]
        
        else:
            raise NotImplementedError(f"{dataset} is not implemented !")

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        face = get_a_face(self.media[index], self.detector)
        
        return (self.transform(face), torch.FloatTensor([self.label[index]])) if face is not None else self.__getitem__(random.randint(0, self.__len__() - 1))

