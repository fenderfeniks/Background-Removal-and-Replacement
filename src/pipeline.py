import numpy as np

from src.matting import BiRefNetMattingModel, RMBGMattingModel
from src.ensemble import ensemble_masks
from src.compositing import (
    replace_background,
    create_solid_background,
    apply_mask,
    prepare_background_image,
)

class BackgroundRemovalPipeline:
    def __init__(
        self,
        use_birefnet=True,
        use_rmbg=True,
        birefnet_force_download=False,
        rmbg_force_download=False,
        models_root="../models",
    ):
        self.birefnet_model = None
        self.rmbg_model = None

        if use_birefnet:
            self.birefnet_model = BiRefNetMattingModel(
                force_download=birefnet_force_download,
                models_root=models_root,
            )

        if use_rmbg:
            self.rmbg_model = RMBGMattingModel(
                force_download=rmbg_force_download,
                models_root=models_root,
            )

    def get_mask(
        self,
        image: np.ndarray,
        model_type: str = "birefnet",
        ensemble_weights=(0.85, 0.15),
    ) -> np.ndarray:
        model_type = model_type.lower()

        if model_type == "birefnet":
            if self.birefnet_model is None:
                raise ValueError("BiRefNet model is not initialized.")
            return self.birefnet_model.predict_mask(image)

        if model_type == "rmbg":
            if self.rmbg_model is None:
                raise ValueError("RMBG model is not initialized.")
            return self.rmbg_model.predict_mask(image)

        if model_type == "ensemble":
            if self.birefnet_model is None or self.rmbg_model is None:
                raise ValueError("Both BiRefNet and RMBG must be initialized for ensemble mode.")

            w1, w2 = ensemble_weights
            mask1 = self.birefnet_model.predict_mask(image)
            mask2 = self.rmbg_model.predict_mask(image)
            return ensemble_masks(mask1, mask2, w1=w1, w2=w2)

        raise ValueError(f"Unsupported model_type: {model_type}")

    def run(
            self,
            image: np.ndarray,
            model_type: str = "birefnet",
            background_mode: str = "solid",
            background_color=(255, 255, 255),
            background_image=None,
            ensemble_weights=(0.85, 0.15),
    ):
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy.ndarray")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must have shape (H, W, 3)")

        mask = self.get_mask(
            image=image,
            model_type=model_type,
            ensemble_weights=ensemble_weights,
        )

        background_mode = background_mode.lower()

        if background_mode == "solid":
            background = create_solid_background(image.shape, color=background_color)
            result = replace_background(image, background, mask)

        elif background_mode == "transparent":
            result = apply_mask(image, mask)

        elif background_mode == "image":
            if background_image is not None and not isinstance(background_image, np.ndarray):
                raise TypeError("background_image must be a numpy.ndarray")

            background = prepare_background_image(background_image, image.shape)
            result = replace_background(image, background, mask)

        else:
            raise ValueError(f"Unsupported background_mode: {background_mode}")

        return {
            "mask": mask,
            "result": result,
        }
