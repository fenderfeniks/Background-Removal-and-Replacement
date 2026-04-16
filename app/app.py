import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gradio as gr
import numpy as np
from PIL import Image

from src.pipeline import BackgroundRemovalPipeline

pipeline = BackgroundRemovalPipeline(models_root="models")


def process_image(image, model_type, bg_color, bg_mode, background_image):
    if image is None:
        return None, None

    image_np = np.array(image)

    color_rgb = (255, 255, 255)
    if bg_color:
        color_rgb = tuple(int(bg_color[i:i+2], 16) for i in (1, 3, 5))

    background_np = None
    if background_image is not None:
        background_np = np.array(background_image)

    output = pipeline.run(
        image=image_np,
        model_type=model_type.lower(),
        background_mode=bg_mode,
        background_color=color_rgb,
        background_image=background_np,
        ensemble_weights=(0.85, 0.15),
    )

    return Image.fromarray(output["mask"]), Image.fromarray(output["result"])


def toggle_background_inputs(bg_mode):
    if bg_mode == "solid":
        return (
            gr.update(visible=True),   # color picker
            gr.update(visible=False),  # bg image input
        )
    elif bg_mode == "image":
        return (
            gr.update(visible=False),
            gr.update(visible=True),
        )
    else:  # transparent
        return (
            gr.update(visible=False),
            gr.update(visible=False),
        )


with gr.Blocks() as demo:
    gr.Markdown("# Background Removal App")
    gr.Markdown("Remove and replace background using BiRefNet / RMBG / Ensemble")

    with gr.Row():
        input_image = gr.Image(type="pil", label="Input Image")

        with gr.Column():
            model_choice = gr.Radio(
                ["birefnet", "rmbg", "ensemble"],
                value="birefnet",
                label="Model"
            )

            bg_mode = gr.Radio(
                ["solid", "image", "transparent"],
                value="solid",
                label="Background Mode"
            )

            color_picker = gr.ColorPicker(
                value="#ffffff",
                label="Background Color",
                visible=True,
            )

            background_image_input = gr.Image(
                type="pil",
                label="Background Image",
                visible=False,
            )

            run_button = gr.Button("Run")

    with gr.Row():
        mask_output = gr.Image(label="Mask")
        result_output = gr.Image(label="Result")

    bg_mode.change(
        fn=toggle_background_inputs,
        inputs=bg_mode,
        outputs=[color_picker, background_image_input],
    )

    run_button.click(
        fn=process_image,
        inputs=[
            input_image,
            model_choice,
            color_picker,
            bg_mode,
            background_image_input,
        ],
        outputs=[mask_output, result_output],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        inbrowser=True,
        share=False,
        debug=True,
    )