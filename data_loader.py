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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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

    def __init__(self, root_dir, anomaly_source_path, resize_shape=None, image_limit=None, texture_limit=None, blend_method='beta_uniform', use_image_placeholder=False, use_anomaly_placeholder=False):
        self.root_dir = root_dir
        self.resize_shape=resize_shape
        self.blend_method=blend_method
        self.use_image_placeholder=use_image_placeholder
        self.use_anomaly_placeholder=use_anomaly_placeholder

        self.image_paths = sorted(glob.glob(root_dir+"/*.png"))
        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path+"/*/*.jpg"))

        # Preload all training images and anomaly sources into memory
        self.images = []

        if self.use_image_placeholder:
            # Use a single gray placeholder image instead of loading actual images
            # Gray value of 0.5 (middle gray in normalized range [0, 1])
            gray_image = np.ones((self.resize_shape[0], self.resize_shape[1], 3), dtype=np.float32) * 0.5
            self.images = [gray_image]
            print(f"Using gray placeholder image (no actual images loaded).")
        else:
            # Load actual training images
            image_limit = min(image_limit, len(self.image_paths)) if image_limit is not None else len(self.image_paths)
            print(f"Loading {image_limit} training images into memory...")
            for img_path in self.image_paths[:image_limit]:
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
                image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
                self.images.append(image)
            print(f"Finished loading training images.")
        
        self.anomaly_source_images = []
        if self.use_anomaly_placeholder:
            # Use a single pure red image instead of loading actual anomaly textures
            # Pure red: RGB = (255, 0, 0) in uint8 range
            red_image = np.zeros((self.resize_shape[0], self.resize_shape[1], 3), dtype=np.uint8)
            red_image[:, :, 0] = 255  # Set red channel to maximum
            self.anomaly_source_images = [red_image]
            print(f"Using pure red anomaly placeholder (no actual anomaly textures loaded).")
        else:
            # Load actual anomaly texture images
            texture_limit = min(texture_limit, len(self.anomaly_source_paths)) if texture_limit is not None else len(self.anomaly_source_paths)
            print(f"Loading {texture_limit} anomaly source images into memory...")
            for anomaly_path in self.anomaly_source_paths[:texture_limit]:
                anomaly_img = cv2.imread(anomaly_path)
                anomaly_img = cv2.cvtColor(anomaly_img, cv2.COLOR_BGR2RGB)
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

    def _blend_uniform_beta(self, image, anomaly_texture, perlin_thr):
        """
        Original blending method: uses a single beta value for entire mask.

        Formula: augmented = image * (1 - mask) + (1 - beta) * texture + beta * image * mask

        Args:
            image: Original image (H, W, C)
            anomaly_texture: Preprocessed anomaly texture (H, W, C)
            perlin_thr: Binary mask (H, W, 1)

        Returns:
            Blended image (H, W, C), Beta map (H, W, 1)
        """
        beta = torch.rand(1).numpy()[0] * 0.8
        augmented_image = image * (1 - perlin_thr) + (1 - beta) * anomaly_texture + beta * image * perlin_thr

        # Create uniform beta map (constant value everywhere)
        beta_map = np.ones_like(perlin_thr) * beta

        return augmented_image, beta_map

    def _blend_perlin_beta(self, image, anomaly_texture, perlin_thr):
        """
        Perlin noise-based blending: uses spatially-varying beta values.

        Generates a Perlin noise map and scales it so that:
        - Minimum value = (1 - beta_sample)
        - Maximum value = 1.0

        This creates natural-looking variation in how much the original image shows through.

        Args:
            image: Original image (H, W, C)
            anomaly_texture: Preprocessed anomaly texture (H, W, C)
            perlin_thr: Binary mask (H, W, 1)

        Returns:
            Blended image (H, W, C), Beta map (H, W, 1)
        """
        # Sample a base beta value
        beta_sample = torch.rand(1).numpy()[0] * 0.8

        # Generate Perlin noise for spatially-varying beta
        perlin_scale = 6
        min_perlin_scale = 0
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        beta_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
        beta_noise = self.rot(image=beta_noise)

        # Normalize Perlin noise to [0, 1]
        beta_noise = (beta_noise - beta_noise.min()) / (beta_noise.max() - beta_noise.min() + 1e-8)

        beta_map = beta_noise * beta_sample
        beta_map = np.expand_dims(beta_map, axis=2)

        # Apply spatially-varying blending
        # In anomaly regions: blend = (1 - beta_map) * texture + beta_map * image
        # In normal regions: blend = image
        augmented_image = image * (1 - perlin_thr) + ((1 - beta_map) * anomaly_texture + beta_map * image) * perlin_thr

        return augmented_image, beta_map

    def augment_image(self, image, anomaly_source_img):
        aug = self.randAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0

        # Augment anomaly source texture
        anomaly_img_augmented = aug(image=anomaly_source_img)

        # Generate Perlin noise mask for anomaly region shape
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        # Prepare anomaly texture (normalized)
        anomaly_texture = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

        # Apply selected blending method
        if self.blend_method == 'beta_uniform':
            augmented_image, beta_map = self._blend_uniform_beta(image, anomaly_texture, perlin_thr)
        elif self.blend_method == 'beta_perlin':
            augmented_image, beta_map = self._blend_perlin_beta(image, anomaly_texture, perlin_thr)
        else:
            raise ValueError(f"Unknown blend_method: {self.blend_method}")

        # Randomly return images without anomalies (50% chance)
        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5:
            image = image.astype(np.float32)
            return image, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0],dtype=np.float32), np.zeros_like(perlin_thr, dtype=np.float32)
        else:
            augmented_image = augmented_image.astype(np.float32)
            msk = (perlin_thr).astype(np.float32)
            # Final blending to ensure clean boundaries
            augmented_image = msk * augmented_image + (1-msk)*image
            has_anomaly = 1.0
            if np.sum(msk) == 0:
                has_anomaly=0.0
            return augmented_image, msk, np.array([has_anomaly],dtype=np.float32), beta_map.astype(np.float32)

    def transform_image(self, image, anomaly_source_img):
        do_aug_orig = torch.rand(1).numpy()[0] > 0.7
        if do_aug_orig:
            image = self.rot(image=image)

        image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32)

        # Keep a copy of the anomaly source for visualization
        anomaly_source_normalized = anomaly_source_img.astype(np.float32) / 255.0

        augmented_image, anomaly_mask, has_anomaly, beta_map = self.augment_image(image, anomaly_source_img)
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
        beta_map = np.transpose(beta_map, (2, 0, 1))
        anomaly_source_normalized = np.transpose(anomaly_source_normalized, (2, 0, 1))
        return image, augmented_image, anomaly_mask, has_anomaly, beta_map, anomaly_source_normalized

    def __getitem__(self, idx):
        idx = torch.randint(0, len(self.images), (1,)).item()
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_images), (1,)).item()
        image, augmented_image, anomaly_mask, has_anomaly, beta_map, anomaly_source_img = self.transform_image(
            self.images[idx].copy(),
            self.anomaly_source_images[anomaly_source_idx].copy()
        )
        sample = {'image': image, "anomaly_mask": anomaly_mask,
                  'augmented_image': augmented_image, 'has_anomaly': has_anomaly, 'idx': idx, 'beta_map': beta_map,
                  'anomaly_source_img': anomaly_source_img}

        return sample
