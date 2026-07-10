# coding=utf-8
from pathlib import Path
from typing import Any

from PIL import ExifTags, Image

from archaeo import logger
from archaeo.io.docs import LocalFileMetadata

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass


def get_image_metadata(file_path: str | Path) -> LocalFileMetadata:
    def ratio_to_float(value: Any) -> float:
        if hasattr(value, "numerator") and hasattr(value, "denominator"):
            return value.numerator / value.denominator
        return float(value)

    def dms_to_decimal(value: Any) -> float:
        degrees = ratio_to_float(value[0])
        minutes = ratio_to_float(value[1])
        seconds = ratio_to_float(value[2])
        return degrees + minutes / 60 + seconds / 3600

    def parse_gps(gps_info: dict) -> dict[str, float]:
        gps_tags = {
            ExifTags.GPSTAGS.get(key, key): val
            for key, val in gps_info.items()
        }

        result: dict[str, float] = {}

        lat = gps_tags.get("GPSLatitude")
        lat_ref = gps_tags.get("GPSLatitudeRef")

        lon = gps_tags.get("GPSLongitude")
        lon_ref = gps_tags.get("GPSLongitudeRef")

        if lat and lat_ref:
            latitude = dms_to_decimal(lat)
            if str(lat_ref).upper() == "S":
                latitude = -latitude
            result["latitude"] = latitude

        if lon and lon_ref:
            longitude = dms_to_decimal(lon)
            if str(lon_ref).upper() == "W":
                longitude = -longitude
            result["longitude"] = longitude

        return result

    try:
        with Image.open(file_path) as img:
            metadata: dict[str, Any] = {
                "width": img.width,
                "height": img.height,
                "format": img.format,
                "mode": img.mode,
            }

            exif = img.getexif()

            if exif:
                for tag_id, value in exif.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)

                    if tag == "Make":
                        metadata["camera_make"] = str(value)

                    elif tag == "Model":
                        metadata["camera_model"] = str(value)

                    elif tag == "LensModel":
                        metadata["lens_model"] = str(value)

                    elif tag == "DateTimeOriginal":
                        metadata["datetime_original"] = str(value)

                    elif tag == "DateTimeDigitized":
                        metadata["datetime_digitized"] = str(value)

                    elif tag == "DateTime":
                        metadata["datetime"] = str(value)

                    elif tag == "Orientation":
                        metadata["orientation"] = value

                    elif tag == "GPSInfo":
                        gps = parse_gps(value)
                        if gps:
                            metadata.update(gps)

            return LocalFileMetadata(metadata=metadata)

    except Exception as e:
        logger.warning("get image metadata failed: %s, file=%s", e, file_path)
        return LocalFileMetadata()
