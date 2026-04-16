import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from src.pipeline import BackgroundRemovalPipeline
from src.utils.io import load_image, save_image



def main():
    print("Loading image...")
    image = load_image("data/test.jpg")

    print("Initializing pipeline...")
    pipeline = BackgroundRemovalPipeline(models_root="models")

    print("Running pipeline...")
    output = pipeline.run(
        image=image,
        model_type="ensemble",
        background_mode="solid",
        background_color=(255, 255, 255),
        ensemble_weights=(0.85, 0.15),
    )

    mask = output["mask"]
    result = output["result"]

    print("Saving outputs...")
    save_image(mask, "data/output_mask.png")
    save_image(result, "data/output_result.png")

    print("Done: saved data/output_mask.png and data/output_result.png")


if __name__ == "__main__":
    main()