import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from skimage import io
from sklearn.model_selection import StratifiedKFold
import dlib
import pickle
from tqdm import tqdm

from util import load_pic

class MyData(Dataset):

    def __init__(self, args):
        self.raw_path = args.raw_dataset_path
        self.dataset = args.dataset
        self.fold_num = args.fold_num
        self.algorithm = args.algorithm
        self.dataset_path = os.path.join(args.raw_dataset_path, args.dataset)
        self.sample_path_list, self.labels = self.get_sample_path_list()
        self.fold_idx = self.make_kfold_idx(args.fold_num)
        self.sample_list = self.prepare_sample_list()


    def __getitem__(self, idx):
        sample_path, pic_class, label = self.sample_path_list[idx]
        data = self.sample_list[idx]
        if self.algorithm == 'dlib':
            return data, torch.tensor(label)
        else:
            img_tensor = transforms.ToTensor()
            return img_tensor(data), torch.tensor(label)
    
    def __len__(self):
        return len(self.sample_path_list)

    def get_sample_path_list(self):
        sample_list = []
        labels = []
        if self.dataset == "PIE_dataset":
            pass
        elif self.dataset == "faces94":
            pic_class_list = os.listdir(self.dataset_path)
            for i, pic_class in enumerate(pic_class_list):
                sub_pic_class_list = os.listdir(os.path.join(self.dataset_path, pic_class))
                for j, sub_pic_class in enumerate(sub_pic_class_list):
                    sample_dir = os.path.join(self.dataset_path, pic_class, sub_pic_class)
                    for sample_name in os.listdir(sample_dir):
                        if sample_name[-4:] != '.jpg':
                            continue
                        sample_path = os.path.join(sample_dir, sample_name)
                        class_idx = i*len(pic_class_list)+j
                        sample_list.append((sample_path, '_'.join([pic_class, sub_pic_class]), class_idx))
                        labels.append(class_idx)
        else:
            pic_class_list = os.listdir(self.dataset_path)
            for i, pic_class in enumerate(pic_class_list):
                sample_dir = os.path.join(self.dataset_path, pic_class)
                for sample_name in os.listdir(sample_dir):
                    if sample_name[-4:] != '.jpg':
                        continue
                    sample_path = os.path.join(sample_dir, sample_name)
                    sample_list.append((sample_path, pic_class, i))
                    labels.append(i)
        return sample_list, labels

    def prepare_sample_list(self):
        sample_list = []
        if self.algorithm == 'dlib':
            detector = dlib.get_frontal_face_detector()
            face_rec = dlib.face_recognition_model_v1("checkpoints/dlib_face_recognition_resnet_model_v1.dat")
            predictor = dlib.shape_predictor("checkpoints/shape_predictor_68_face_landmarks.dat")
            sample_dict = {}
            prepare_path = './prepare/dlib_{}.pkl'.format(self.dataset)
            if os.path.exists(prepare_path):
                sample_dict = pickle.load(open(prepare_path, 'rb'))

        for sample_path, pic_class, class_idx in tqdm(self.sample_path_list):
            if self.algorithm == 'dlib':
                if sample_path in sample_dict:
                    face_descriptor = sample_dict[sample_path]
                else:
                    img_rd = io.imread(sample_path)
                    img_gray = cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB)
                    try:
                        faces = detector(img_gray, 1)
                        shape = predictor(img_gray, faces[0])
                        face_descriptor = face_rec.compute_face_descriptor(img_gray, shape)
                    except:
                        face_descriptor = np.zeros((128,1))
                    sample_list.append(face_descriptor)
                    sample_dict[sample_path] = face_descriptor
            else:
                sample_list.append(load_pic(sample_path))
        
        if self.algorithm == 'dlib' and not os.path.exists(prepare_path):
            pickle.dump(sample_dict, open(prepare_path, 'wb+'))

        return sample_list

    def make_kfold_idx(self, fold_num):
        skf_train_test = StratifiedKFold(n_splits=fold_num, shuffle=True)
        train_test_idxs = []
        for train_idx, test_idx in skf_train_test.split(list(range(len(self.sample_path_list))), self.labels):
            train_test_idxs.append([train_idx, test_idx])
        return train_test_idxs

    def get_all_train_data(self, fold):
        train_idx, _ = self.fold_idx[fold]
        samples = []
        labels = []
        for idx in train_idx:
            sample_path, pic_class, label = self.sample_path_list[idx]
            pic_data = load_pic(sample_path)
            if pic_data is not None:
                samples.append(pic_data)
                labels.append(label)
            else:
                import pdb; pdb.set_trace()
        return np.array(samples), np.array(labels)

    def get_all_test_data(self, fold):
        _, test_idx = self.fold_idx[fold]
        samples = []
        labels = []
        for idx in test_idx:
            sample_path, pic_class, label = self.sample_path_list[idx]
            samples.append(load_pic(sample_path))
            labels.append(label)
        return np.array(samples), np.array(labels)
    
    def get_class_num(self):
        return max(self.labels) + 1
    
    def get_idx(self, fold):
        return self.fold_idx[fold]
    