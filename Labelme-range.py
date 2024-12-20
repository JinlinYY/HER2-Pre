import json
import os
from shapely.geometry import Polygon
from shapely.affinity import scale
from PIL import Image, ImageDraw

# Function to expand the contour
def expand_contour(points, expansion_factor):
    polygon = Polygon(points)
    scaled_polygon = scale(polygon, xfact=expansion_factor, yfact=expansion_factor, origin='center')
    return list(scaled_polygon.exterior.coords)

# Paths
input_folder = 'HER2/HER2_low/HER2_low_breast'  # Set the path to your input folder containing JSON and image files
output_folder = 'HER2-18/HER2_low-18/HER2_low_breast-18'  # Set the path to your output folder

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Expansion factor
expansion_factor = 1.8  # Set your desired expansion factor here

# Process each JSON file in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.json'):
        input_json_file = os.path.join(input_folder, filename)
        image_filename = filename.replace('.json', '.bmp')  # Assuming the image files have the same base name as JSON files
        input_image_file = os.path.join(input_folder, image_filename)

        # Load the LabelMe JSON file
        with open(input_json_file, 'r') as file:
            data = json.load(file)

        # Expand each shape's points
        for shape in data['shapes']:
            points = shape['points']
            expanded_points = expand_contour(points, expansion_factor)
            shape['points'] = expanded_points

        # Save the updated JSON to the new folder with the same filename
        output_json_file = os.path.join(output_folder, os.path.basename(input_json_file))
        with open(output_json_file, 'w') as file:
            json.dump(data, file, indent=4)

        # Check if the corresponding image file exists
        if os.path.exists(input_image_file):
            # Load the original image
            image = Image.open(input_image_file)
            draw = ImageDraw.Draw(image)

            # Draw expanded contours on the image
            for shape in data['shapes']:
                points = shape['points']
                draw.polygon(points, outline='red')  # Draw expanded contour in red

            # Save the image with drawn contours to the new folder with the same filename
            output_image_file = os.path.join(output_folder, os.path.basename(input_image_file))
            image.save(output_image_file)

print(f"Expanded contours and images saved to {output_folder}")
