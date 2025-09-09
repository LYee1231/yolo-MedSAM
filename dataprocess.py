from PIL import Image
import os
import re

def fill_white_circle(input_path, output_path):
    img = Image.open(input_path).convert('RGB')
    pixels = img.load()
    width, height = img.size
    white = (255, 255, 255)
    for y in range(height):
        white_positions = []
        for x in range(width):
            if pixels[x, y] == white:
                white_positions.append(x)
        if len(white_positions) >= 2:
            start = white_positions[0]
            end = white_positions[-1]
            for x in range(start, end + 1):
                pixels[x, y] = white

    img.save(output_path)


def batch_process_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    processed_count = 0
    pattern = re.compile(r'_\d*HC_')

    for filename in os.listdir(input_folder):
        if pattern.search(filename) and filename.lower().endswith('.png'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            print(f"Processing: {filename}")
            fill_white_circle(input_path, output_path)
            processed_count += 1

    print(f"Done! Processed {processed_count} images in total.")


input_folder = "C:/Users/lyjco/Desktop/final/1327317/training_set"
output_folder = "C:/Users/lyjco/Desktop/final/1327317/1"

batch_process_images(input_folder, output_folder)
