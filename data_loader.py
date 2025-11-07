import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import glob
import imgaug.augmenters as iaa
from perlin import rand_perlin_2d_np

class MVTecDRAEMTestDataset(Dataset):

    def __init__(self, root_dir, resize_shape=None):
        self.root_dir = root_dir
        self.image_paths = sorted(glob.glob(root_dir+"/*/*.png"))
        self.resize_shape=resize_shape
        
        # Preload all images and masks during initialization
        print(f"Loading {len(self.image_paths)} test images into memory...")
        self.images = []
        self.masks = []
        self.has_anomalies = []
        
        for img_path in self.image_paths:
            dir_path, file_name = os.path.split(img_path)
            base_dir = os.path.basename(dir_path)
            
            if base_dir == 'good':
                image, mask = self.transform_image(img_path, None)
                has_anomaly = np.array([0], dtype=np.float32)
            else:
                mask_path = os.path.join(dir_path, '../../ground_truth/')
                mask_path = os.path.join(mask_path, base_dir)
                mask_file_name = file_name.split(".")[0]+"_mask.png"
                mask_path = os.path.join(mask_path, mask_file_name)
                image, mask = self.transform_image(img_path, mask_path)
                has_anomaly = np.array([1], dtype=np.float32)
            
            self.images.append(image)
            self.masks.append(mask)
            self.has_anomalies.append(has_anomaly)
        
        print(f"Finished loading test images into memory.")

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0],image.shape[1]))
        if self.resize_shape != None:
            image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
            mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0]))

        image = image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        return image, mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'image': self.images[idx], 'has_anomaly': self.has_anomalies[idx], 
                  'mask': self.masks[idx], 'idx': idx}

        return sample



class MVTecDRAEMTrainDataset(Dataset):

    def __init__(self, root_dir, anomaly_source_path, resize_shape=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.resize_shape=resize_shape

        self.image_paths = sorted(glob.glob(root_dir+"/*.png"))
        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path+"/*/*.jpg"))

        # Preload all training images and anomaly sources into memory
        print(f"Loading {len(self.image_paths)} training images into memory...")
        self.images = []
        for img_path in self.image_paths:
            image = cv2.imread(img_path)
            image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
            image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
            self.images.append(image)
        print(f"Finished loading training images.")
        
        print(f"Loading {len(self.anomaly_source_paths)} anomaly source images into memory...")
        self.anomaly_source_images = []
        for anomaly_path in self.anomaly_source_paths:
            anomaly_img = cv2.imread(anomaly_path)
            anomaly_img = cv2.resize(anomaly_img, dsize=(self.resize_shape[1], self.resize_shape[0]))
            self.anomaly_source_images.append(anomaly_img)
        print(f"Finished loading anomaly source images.")

        self.augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                      iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                      iaa.pillike.EnhanceSharpness(),
                      iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                      iaa.Solarize(0.5, threshold=(32,128)),
                      iaa.Posterize(),
                      iaa.Invert(),
                      iaa.pillike.Autocontrast(),
                      iaa.pillike.Equalize(),
                      iaa.Affine(rotate=(-45, 45))
                      ]

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])


    def __len__(self):
        return len(self.images)


    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug

    def augment_image(self, image, anomaly_source_img):
        aug = self.randAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0

        anomaly_img_augmented = aug(image=anomaly_source_img)
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

        beta = torch.rand(1).numpy()[0] * 0.8

        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
            perlin_thr)

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5:
            image = image.astype(np.float32)
            return image, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0],dtype=np.float32)
        else:
            augmented_image = augmented_image.astype(np.float32)
            msk = (perlin_thr).astype(np.float32)
            augmented_image = msk * augmented_image + (1-msk)*image
            has_anomaly = 1.0
            if np.sum(msk) == 0:
                has_anomaly=0.0
            return augmented_image, msk, np.array([has_anomaly],dtype=np.float32)

    def transform_image(self, image, anomaly_source_img):
        do_aug_orig = torch.rand(1).numpy()[0] > 0.7
        if do_aug_orig:
            image = self.rot(image=image)

        image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32)
        augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, anomaly_source_img)
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
        return image, augmented_image, anomaly_mask, has_anomaly

    def __getitem__(self, idx):
        idx = torch.randint(0, len(self.images), (1,)).item()
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_images), (1,)).item()
        image, augmented_image, anomaly_mask, has_anomaly = self.transform_image(
            self.images[idx].copy(),
            self.anomaly_source_images[anomaly_source_idx].copy()
        )
        sample = {'image': image, "anomaly_mask": anomaly_mask,
                  'augmented_image': augmented_image, 'has_anomaly': has_anomaly, 'idx': idx}

        return sample
