import os
os.environ['MKL_ENABLE_INSTRUCTIONS'] = 'AVX'

import os
import random
import json
from PIL import Image, ImageEnhance
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_image(image_path):
    try:
        with Image.open(image_path) as img:
            return img.convert("RGBA")
    except IOError:
        print(f"Unable to open image: {image_path}")
        return None

def is_image_white(image):
    arr = np.array(image)
    return np.all(arr[:, :, :3] >= 250)

def make_image_translucent(image, opacity):
    translucent_image = Image.new("RGBA", image.size)
    translucent_image.paste(image, (0, 0), image.split()[3].point(lambda p: p * opacity // 255))
    return translucent_image

def apply_translucent_overlay(large_image_path, small_image, small_image_path, save_dir, iterations=5, opacity_values=[0.25, 0.5, 0.75]):
    if small_image is None:
        return
    with Image.open(large_image_path) as img:
        large_image = img.convert("RGBA")

        if large_image.width < small_image.width or large_image.height < small_image.height:
            print(f"Large image {large_image_path} is smaller than small image {small_image_path}, skipping.")
            return

        small_image_label = os.path.splitext(os.path.basename(small_image_path))[0]
        for opacity in opacity_values:
            translucent_large_image = make_image_translucent(large_image, int(opacity * 255))
            for j in range(iterations):
                x = random.randint(0, translucent_large_image.width - small_image.width)
                y = random.randint(0, translucent_large_image.height - small_image.height)
                crop = translucent_large_image.crop((x, y, x + small_image.width, y + small_image.height))
                enhancer = ImageEnhance.Color(crop)
                crop = enhancer.enhance(random.uniform(0.5, 1.5))

                if not is_image_white(crop):
                    combined_image = Image.alpha_composite(small_image, crop)

                    """# Data augmentation
                    data_augmentation = ImageDataGenerator(
                        rotation_range=20,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        shear_range=0.2,
                        zoom_range=0.2,
                        horizontal_flip=True,
                        fill_mode='nearest'
                    )"""
                    # Data augmentation with NO transformations
                    data_augmentation = ImageDataGenerator()

                    combined_image_array = np.expand_dims(np.array(combined_image), axis=0)
                    it = data_augmentation.flow(combined_image_array, batch_size=1)
                    augmented_image = next(it)
                    augmented_image_pil = Image.fromarray(augmented_image[0].astype('uint8'), 'RGBA')

                    variant_dir = os.path.join(save_dir, small_image_label, f"opacity_{int(opacity*100)}", str(j))
                    os.makedirs(variant_dir, exist_ok=True)
                    filename = f"{small_image_label}.png"
                    augmented_image_pil.save(os.path.join(variant_dir, filename))
                    print(f"Saved: {os.path.join(variant_dir, filename)}")

                    metadata = {
                        'filename': filename,
                        'label': small_image_label,
                        'opacity': opacity,
                        'crop_position': (x, y)
                    }
                    with open(os.path.join(variant_dir, f"{small_image_label}.json"), 'w') as f:
                        json.dump(metadata, f)

def process_folder(small_images_folder, large_image_path, save_dir, opacity_values=[0.25, 0.5, 0.75]):
    os.makedirs(save_dir, exist_ok=True)
    for image_name in os.listdir(small_images_folder):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            small_image_path = os.path.join(small_images_folder, image_name)
            small_image = load_image(small_image_path)
            if small_image:
                apply_translucent_overlay(large_image_path, small_image, small_image_path, save_dir, opacity_values=opacity_values)

def process_all_folders(top_folder, large_image_path, base_save_dir, opacity_values=[0.25, 0.5, 0.75]):
    os.makedirs(base_save_dir, exist_ok=True)
    for folder_name in os.listdir(top_folder):
        folder_path = os.path.join(top_folder, folder_name)
        if os.path.isdir(folder_path):
            save_dir = os.path.join(base_save_dir, folder_name)
            os.makedirs(save_dir, exist_ok=True)
            process_folder(folder_path, large_image_path, save_dir, opacity_values=opacity_values)

if __name__ == "__main__":
    large_image_path = 'images/o44115g2.tif'
    base_save_dir = 'augmented_images'
    # top_folder = 'images/output/removed_text'
    top_folder = 'example'
    opacity_values = [0.25, 0.5, 0.75]  # Adjust the opacity values as needed
    process_folder(top_folder, large_image_path, base_save_dir, opacity_values)
    