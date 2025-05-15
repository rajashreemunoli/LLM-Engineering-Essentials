import os
import base64
from io import BytesIO
from PIL import Image
import gradio as gr
from openai import OpenAI
from IPython.display import Markdown  # for nice display, optional in Gradio

# Initialize NEBIUS client
client = OpenAI(
    base_url="https://api.studio.nebius.ai/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY"),
)

def generate_image_from_prompt(prompt: str):
    response = client.images.generate(
        model="black-forest-labs/flux-dev",
        prompt=prompt,
        response_format="b64_json",
        extra_body={
            "response_extension": "png",
            "width": 512,  # smaller for faster generation on web
            "height": 512,
            "num_inference_steps": 28,
            "negative_prompt": "",
            "seed": -1,
        },
    )
    image_b64 = response.data[0].b64_json
    return Image.open(BytesIO(base64.b64decode(image_b64)))

def describe_image(image: Image.Image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    image_bytes = buffered.getvalue()
    b64_image = base64.b64encode(image_bytes).decode()

    response = client.chat.completions.create(
        model="Qwen/Qwen2-VL-72B-Instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image as clearly and accurately as possible."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
                ],
            }
        ],
    )
    return response.choices[0].message.content.strip()

def broken_telephone_game(start_prompt, n_rounds=6):
    current_text = start_prompt
    is_text_step = True
    history = []  # store tuples ("image"/"text", content)

    for round_num in range(n_rounds):
        if is_text_step:
            # Text -> Image
            image = generate_image_from_prompt(current_text)
            history.append(("image", image))
            current_text = None
        else:
            # Image -> Text
            current_text = describe_image(history[-1][1])  # last image
            history.append(("text", current_text))
        is_text_step = not is_text_step

    return history

def gradio_interface(start_prompt, rounds):
    rounds = int(rounds)
    history = broken_telephone_game(start_prompt, rounds)
    outputs = []
    for i, (typ, content) in enumerate(history):
        if typ == "image":
            outputs.append((f"Round {i+1} Image", content))
        else:
            outputs.append((f"Round {i+1} Text", content))
    return outputs

with gr.Blocks() as demo:
    gr.Markdown("# Broken Telephone with NEBIUS Models")
    prompt_input = gr.Textbox(label="Start Prompt", value="A cat playing chess with a robot")
    rounds_input = gr.Slider(minimum=2, maximum=10, step=2, value=6, label="Total Rounds (text or image steps)")
    run_button = gr.Button("Play Game")
    output_gallery = gr.Gallery(label="Rounds Output").style(grid=[2], height="auto")

    def run_game(start_prompt, rounds):
        history = broken_telephone_game(start_prompt, rounds)
        # Format for gallery: images show as is, text wrapped in an image
        formatted = []
        for typ, content in history:
            if typ == "image":
                formatted.append(content)
            else:
                # Render text as image for gallery display
                img = Image.new("RGB", (512, 128), color="white")
                from PIL import ImageDraw, ImageFont
                draw = ImageDraw.Draw(img)
                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except:
                    font = ImageFont.load_default()
                draw.multiline_text((10,10), content, fill="black", font=font)
                formatted.append(img)
        return formatted

    run_button.click(run_game, inputs=[prompt_input, rounds_input], outputs=output_gallery)

if __name__ == "__main__":
    demo.launch()