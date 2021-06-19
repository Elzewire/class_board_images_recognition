from PIL import Image, ExifTags


def get_meta(image):
    img = Image.open(image)
    exif = {
        ExifTags.TAGS[k]: v
        for k, v in img._getexif().items()
        if k in ExifTags.TAGS
    }

    return exif