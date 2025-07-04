# captcha_generator.py
import random
import string
from PIL import Image, ImageDraw, ImageFont
import os

CAPTCHA_LENGTH = 4
CHAR_SET = string.ascii_uppercase + string.digits  # A-Z + 0-9
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # Linux下字体路径，可换成本地路径

def generate_captcha_text():
    return ''.join(random.choices(CHAR_SET, k=CAPTCHA_LENGTH))

def generate_captcha_image(text, width=160, height=60):
    image = Image.new('RGB', (width, height), (255, 255, 255))
    font = ImageFont.truetype(FONT_PATH, 40)
    draw = ImageDraw.Draw(image)
    
    for i, char in enumerate(text):
        draw.text((10 + i * 35, 10), char, font=font, fill=(0, 0, 0))
    
    return image

def save_captcha_dataset(output_dir='captcha_dataset', num_samples=1000):
    os.makedirs(output_dir, exist_ok=True)
    for i in range(num_samples):
        text = generate_captcha_text()
        image = generate_captcha_image(text)
        image.save(os.path.join(output_dir, f"{text}_{i}.png"))