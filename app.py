import os
import torch
import streamlit as st
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

st.set_page_config(page_title="Juggernaut XL v9 Image Generator", layout="centered")
st.title("🎨 Juggernaut XL v9 Image Generator")

HF_TOKEN = os.environ.get("HF_TOKEN", None)
MODEL_ID = "RunDiffusion/Juggernaut-XL-v9"

@st.cache_resource(show_spinner=False)
def load_pipeline():
    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            token=HF_TOKEN,
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="sde-dpmsolver++",
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            pipe = pipe.to(device)
            pipe.enable_xformers_memory_efficient_attention()
        else:
            pipe = pipe.to(device)
            pipe.enable_attention_slicing()
        return pipe, device
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None

with st.spinner("Loading Juggernaut XL v9 model (this may take a while on first run)..."):
    pipe, device = load_pipeline()

if pipe is None:
    st.stop()

st.success(f"Model loaded on **{device.upper()}**")

with st.form("generate_form"):
    prompt = st.text_area(
        "Enter your prompt",
        placeholder="A majestic lion standing on a rocky cliff at sunset, photorealistic, 8k",
        height=100,
    )
    negative_prompt = st.text_area(
        "Negative prompt (optional)",
        value="ugly, blurry, low quality, distorted, deformed, watermark, text",
        height=68,
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        steps = st.slider("Inference Steps", min_value=20, max_value=60, value=30, step=5)
    with col2:
        guidance = st.slider("Guidance Scale", min_value=1.0, max_value=15.0, value=7.0, step=0.5)
    with col3:
        seed = st.number_input("Seed (-1 = random)", min_value=-1, max_value=2**31 - 1, value=-1)
    submitted = st.form_submit_button("🖼️ Generate Image", use_container_width=True)

if submitted:
    if not prompt.strip():
        st.warning("Please enter a prompt before generating.")
    else:
        try:
            generator = None
            if seed != -1:
                generator = torch.Generator(device=device).manual_seed(int(seed))
            with st.spinner("Generating image..."):
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt if negative_prompt.strip() else None,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=generator,
                    height=1024,
                    width=1024,
                )
                image = result.images[0]
            st.image(image, caption="Generated Image", use_container_width=True)
            from io import BytesIO
            buf = BytesIO()
            image.save(buf, format="PNG")
            st.download_button(
                label="⬇️ Download Image",
                data=buf.getvalue(),
                file_name="generated_image.png",
                mime="image/png",
                use_container_width=True,
            )
        except torch.cuda.OutOfMemoryError:
            st.error("GPU out of memory. Try reducing inference steps or restarting the app.")
        except Exception as e:
            st.error(f"Image generation failed: {e}")
