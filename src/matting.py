import os

import torch
import numpy as np

from transformers import AutoModelForImageSegmentation
from torchvision import transforms
from huggingface_hub import snapshot_download
from torchvision.transforms.functional import normalize
import torch.nn.functional as F



project_root = os.path.abspath("..")
models_root = os.path.join(project_root, "models")

class BaseMattingModel:

    def predict_mask(self, image):
        raise NotImplementedError


class BiRefNetMattingModel(BaseMattingModel):

    REPO_ID = "ZhengPeng7/BiRefNet"
    MODEL_DIR_NAME = "BiRefNet"


    def __init__(self, force_download=False, models_root='../models'):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = os.path.join(models_root, self.MODEL_DIR_NAME)

        self._prepare_local_model_dir(force_download=force_download)

        self.model = AutoModelForImageSegmentation.from_pretrained(
            self.model_dir,
            revision="refs/pr/9",
            trust_remote_code=True,
            local_files_only=not force_download
        )

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((1024, 1024)),  # пример 1024, 1024
            transforms.ToTensor(),  # делает /255 и (C,H,W)
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

    # Скачиваю модель и веса если не было
    def _prepare_local_model_dir(self, force_download=False):
        os.makedirs(self.model_dir, exist_ok=True)

        config_path = os.path.join(self.model_dir, "config.json")

        if force_download or not os.path.exists(config_path):
            snapshot_download(
                repo_id=self.REPO_ID,
                revision="refs/pr/9",
                local_dir=self.model_dir
            )


    def predict_mask(self, image):
        image_size = (image.shape[1], image.shape[0])
        tensor = self.transform(image)

        # добавляем batch: (1, 3, H, W)
        tensor = tensor.unsqueeze(0).to(self.device)
        tensor = tensor.to(dtype=next(self.model.parameters()).dtype)

        with torch.no_grad():
            preds = self.model(tensor)[-1].sigmoid().cpu()

        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)

        mask = np.array(mask, dtype=np.uint8)
        return mask


class RMBGMattingModel(BaseMattingModel):
    REPO_ID = "briaai/RMBG-1.4"
    MODEL_DIR_NAME = "RMBGMattingModel"

    def __init__(self, force_download=False, models_root='../models'):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = os.path.join(models_root, self.MODEL_DIR_NAME)

        self._prepare_local_model_dir(force_download=force_download)

        self.model = AutoModelForImageSegmentation.from_pretrained(
            self.model_dir,
            trust_remote_code=True,
            local_files_only=not force_download
        )

        self.model.to(self.device)
        self.model.eval()

    def _prepare_local_model_dir(self, force_download=False):
        os.makedirs(self.model_dir, exist_ok=True)

        config_path = os.path.join(self.model_dir, "config.json")

        if force_download or not os.path.exists(config_path):
            snapshot_download(
                repo_id=self.REPO_ID,
                local_dir=self.model_dir,
                local_dir_use_symlinks=False
            )

    def preprocess_image(self, im: np.ndarray, model_input_size: list) -> torch.Tensor:
        if len(im.shape) < 3:
            im = im[:, :, np.newaxis]
        # orig_im_size=im.shape[0:2]
        im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
        im_tensor = F.interpolate(torch.unsqueeze(im_tensor, 0), size=model_input_size, mode='bilinear')
        image = torch.divide(im_tensor, 255.0)
        image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
        return image

    def postprocess_image(self, result: torch.Tensor, im_size: list) -> np.ndarray:
        result = F.interpolate(result, size=im_size, mode='bilinear')
        result = result.squeeze().cpu().numpy()

        result = (result - result.min()) / (result.max() - result.min() + 1e-8)
        result = (result * 255).astype(np.uint8)

        return result

    def predict_mask(self, image):
        model_input_size = [1024, 1024]
        image_size = (image.shape[0], image.shape[1])

        preprocessed_image = self.preprocess_image(image, model_input_size).to(self.device)
        preprocessed_image = preprocessed_image.to(dtype=next(self.model.parameters()).dtype)

        with torch.no_grad():
            result = self.model(preprocessed_image)

        mask = self.postprocess_image(result[0][0], image_size)

        return mask
