import torch
from diffusers import AutoPipelineForText2Image
import diffusers
import streamlit as st
import os
import streamlit as st

LORA_WEIGHTS = os.environ.get("LORA_WEIGHTS", "onstage3890/maya_model_v1_lora") 
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32

@st.cache_resource
def load_model():
    from diffusers import AutoPipelineForText2Image
    import torch
    import os

    LORA_WEIGHTS = os.environ.get("LORA_WEIGHTS", "onstage3890/maya_model_v1_lora")

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipeline = AutoPipelineForText2Image.from_pretrained(
        "CompVis/stable-diffusion-v1-4", torch_dtype=dtype
    )
    pipeline.load_lora_weights(LORA_WEIGHTS, weight_name="pytorch_lora_weights.safetensors")
    pipeline.to(device)
    return pipeline

from PIL import ImageDraw, ImageFont

def add_text_to_image(image, text, text_color="white", outline_color="black",
                      font_size=50, border_width=2, font_path="arial.ttf"):
    font = ImageFont.truetype(font_path, size=font_size)
    draw = ImageDraw.Draw(image)
    width, height = image.size


    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

  
    x = (width - text_width) / 2
    y = (height - text_height) / 2

   
    draw.text((x, y), text, font=font, fill=text_color,
              stroke_width=border_width, stroke_fill=outline_color)


def generate_images(prompt, pipeline, n):
    
    prompts = [prompt] * n

   
    result = pipeline(prompt=prompts)

 
    return result.images

def generate_memes(prompt, text, pipeline, n):
   
    images = generate_images(prompt, pipeline, n)

   
    for image in images:
        add_text_to_image(image, text)

    return images   
def main():
    st.title("🐶 Meme Generator - Maya Edition")

    st.sidebar.header("🛠️ Settings")
    num_images = st.sidebar.number_input("Number of Images", min_value=1, max_value=4, value=1)
    prompt = st.sidebar.text_area("Prompt", value="A cute dog coding in Python")
    text = st.sidebar.text_area("Text on Image", value="AI is paw-some!")

    if st.sidebar.button("Generate Images"):
        if not prompt.strip() or not text.strip():
            st.error("Please enter both a prompt and text.")
        else:
            with st.spinner("Generating... Please wait."):
                pipeline = load_model()
                images = generate_memes(prompt, text, pipeline, num_images)
                for image in images:
                    st.image(image)

if __name__ == '__main__':
    main()
