import os.path
import cv2
import numpy as np
from PIL import Image
from .base_dataset import BaseDataset
import torchvision.transforms as transforms
from .image_folder import make_dataset
from .utils import MaskToTensor

class CrackDataset(BaseDataset):
    """A dataset class for crack dataset."""

    def __init__(self, args):
        """Initialize this dataset class.

        Parameters:
            args (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, args)
        self.phase = args.phase
        self.img_dir, self.lab_dir = self._resolve_split_dirs(args.dataset_path, args.phase)
        self.img_paths = make_dataset(self.img_dir)
        if len(self.img_paths) == 0:
            raise RuntimeError("Found 0 images in: {}".format(self.img_dir))
        self.img_transforms = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.5, 0.5, 0.5),
                                                                       (0.5, 0.5, 0.5))])
        self.lab_transform = MaskToTensor()

    @staticmethod
    def _resolve_split_dirs(dataset_path, phase):
        split_candidates = [
            (os.path.join(dataset_path, phase, 'images'), os.path.join(dataset_path, phase, 'masks')),
            (os.path.join(dataset_path, '{}_img'.format(phase)), os.path.join(dataset_path, '{}_lab'.format(phase))),
        ]
        for img_dir, lab_dir in split_candidates:
            if os.path.isdir(img_dir) and os.path.isdir(lab_dir):
                return img_dir, lab_dir
        tried = [
            "{} / {}".format(img_dir, lab_dir)
            for img_dir, lab_dir in split_candidates
        ]
        raise FileNotFoundError(
            "Could not find split '{}' under '{}'. Tried: {}".format(phase, dataset_path, ', '.join(tried))
        )

    def __getitem__(self, index):
        """
        Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            image (tensor) - - an image
            label (tensor) - - its corresponding segmentation
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        img_path = self.img_paths[index]
        lab_path = os.path.join(self.lab_dir, os.path.splitext(os.path.basename(img_path))[0] + '.png')

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError("Could not read image: {}".format(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        lab = cv2.imread(lab_path, cv2.IMREAD_UNCHANGED)
        if lab is None:
            raise FileNotFoundError("Could not read mask: {}".format(lab_path))

        if len(lab.shape) == 3:
            lab = cv2.cvtColor(lab, cv2.COLOR_BGR2GRAY)

        # adjust the image size
        w, h = self.args.load_width, self.args.load_height
        if w > 0 or h > 0:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
            lab = cv2.resize(lab, (w, h), interpolation=cv2.INTER_CUBIC)

        _, lab = cv2.threshold(lab, 127, 255, cv2.THRESH_BINARY)
        _, lab = cv2.threshold(lab, 127, 1, cv2.THRESH_BINARY)
        lab01 = lab  # 0/1 二值 numpy（512x512），供边界距离变换使用

        img = self.img_transforms(Image.fromarray(img.copy()))
        lab = self.lab_transform(lab.copy()).unsqueeze(0)
        sample = {'image': img, 'label': lab, 'A_paths': img_path, 'B_paths': lab_path}
        # 边界加权 BCE 的逐像素权重图：w = 1 + α·(dist < τ)。
        # α=0（默认）时不计算也不附加该键，样本结构与旧版完全一致。
        band = self._boundary_band(lab01, img_path)
        if band is not None:
            alpha = float(getattr(self.args, 'boundary_alpha', 0.0))
            w = 1.0 + alpha * band.astype(np.float32)
            sample['weight'] = w[None, ...]          # (1, H, W)，与 label 同形
        return sample

    def _boundary_band(self, lab01, img_path):
        """裂缝边界距离带：dist < τ 的前景像素为 1，其余为 0（uint8）。

        对二值前景做距离变换，dist 为每个前景像素到最近背景（即裂缝边界）
        的 L2 像素距离。结果按 (τ, 尺寸, phase) 离线缓存到
        <dataset_path>/boundary_cache/ 下，α 在加载时施加，缓存与 α 无关。
        α=0 时返回 None（不计算、不写缓存）。
        """
        alpha = float(getattr(self.args, 'boundary_alpha', 0.0))
        if alpha == 0.0:
            return None
        tau = float(getattr(self.args, 'boundary_tau', 3.0))
        cache_dir = os.path.join(
            self.args.dataset_path, 'boundary_cache',
            'tau{:g}_{}x{}'.format(tau, self.args.load_width, self.args.load_height),
            self.phase)
        name = os.path.splitext(os.path.basename(img_path))[0] + '.npy'
        cache_path = os.path.join(cache_dir, name)
        if os.path.isfile(cache_path):
            return np.load(cache_path)
        dist = cv2.distanceTransform(lab01.astype(np.uint8), cv2.DIST_L2, 3)
        band = ((dist < tau) & (lab01 > 0)).astype(np.uint8)
        os.makedirs(cache_dir, exist_ok=True)
        # 先写临时文件再原子改名，避免多 worker 并发写坏缓存。
        tmp_path = cache_path + '.tmp.npy'
        np.save(tmp_path, band)
        os.replace(tmp_path, cache_path)
        return band

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.img_paths)


