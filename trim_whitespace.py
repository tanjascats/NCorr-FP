from PIL import Image


def trim_whitespace(image_path):
    """
    Removes white space from the edges of a PNG image.
    """
    image = Image.open(image_path)
    # Convert image to RGBA if it is not already
    if image.mode != 'RGBA':
        image = image.convert('RGBA')

    # Get image data
    image_data = image.getdata()

    # Find the extreme corners that are not white
    non_white_pixels = [(x, y) for y in range(image.height) for x in range(image.width) if
                        image_data[y * image.width + x] != (255, 255, 255, 255)]
    if not non_white_pixels:
        return image  # Return original image if all pixels are white

    # Get the bounding box of non-white pixels
    x_list, y_list = zip(*non_white_pixels)
    box = (min(x_list), min(y_list), max(x_list) + 1, max(y_list) + 1)

    # Crop and return the image
    cropped_image = image.crop(box)
    return cropped_image


# Usage
input_image_path = 'NCorrFP/analysis/figures/pairwise_hist_adult_mss.png'
output_image_path = 'NCorrFP/analysis/figures/pairwise_hist_adult_mss.png'

trimmed_image = trim_whitespace(input_image_path)
trimmed_image.save(output_image_path)
