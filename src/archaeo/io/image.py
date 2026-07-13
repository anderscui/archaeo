# coding=utf-8
from pathlib import Path
from typing import Any

from PIL import ExifTags, Image

from archaeo import logger
from archaeo.io.docs import LocalFileMetadata
from archaeo.io.files import get_absolute_path

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass


def get_image_metadata(file_path: str | Path) -> LocalFileMetadata:
    def ratio_to_float(value: Any) -> float:
        if hasattr(value, "numerator") and hasattr(value, "denominator"):
            denominator = value.denominator
            if denominator == 0:
                raise ValueError("EXIF rational denominator is zero")
            return value.numerator / denominator

        return float(value)

    def dms_to_decimal(value: Any) -> float:
        if not isinstance(value, (tuple, list)) or len(value) != 3:
            raise ValueError(f"invalid GPS DMS value: {value!r}")

        degrees = ratio_to_float(value[0])
        minutes = ratio_to_float(value[1])
        seconds = ratio_to_float(value[2])
        return degrees + minutes / 60 + seconds / 3600

    def normalize_gps_ref(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("ascii", errors="ignore").upper()
        return str(value).upper()

    def parse_gps(gps_info: dict) -> dict[str, float]:
        if not isinstance(gps_info, dict):
            logger.warning(f'invalid gps info: {gps_info}')
            return {}

        gps_tags = {
            ExifTags.GPSTAGS.get(key, key): val
            for key, val in gps_info.items()
        }

        try:
            lat = gps_tags.get("GPSLatitude")
            lat_ref = gps_tags.get("GPSLatitudeRef")

            lon = gps_tags.get("GPSLongitude")
            lon_ref = gps_tags.get("GPSLongitudeRef")
            if any(val is None for val in (lat, lat_ref, lon, lon_ref)):
                return {}

            lat_ref = normalize_gps_ref(lat_ref)
            lon_ref = normalize_gps_ref(lon_ref)
            if lat_ref not in {"N", "S"}:
                raise ValueError(f"invalid GPS latitude ref: {lat_ref!r}")
            if lon_ref not in {"E", "W"}:
                raise ValueError(f"invalid GPS longitude ref: {lon_ref!r}")

            latitude = dms_to_decimal(lat)
            longitude = dms_to_decimal(lon)
            if not 0 <= latitude <= 90:
                raise ValueError(f"invalid latitude: {latitude}")
            if not 0 <= longitude <= 180:
                raise ValueError(f"invalid longitude: {longitude}")

            if lat_ref == "S":
                latitude = -latitude

            if lon_ref == "W":
                longitude = -longitude

            return {
                "latitude": latitude,
                "longitude": longitude,
            }

        except (TypeError, ValueError, ZeroDivisionError) as exc:
            logger.warning(f'parse GPS info error: {exc}, gps={gps_tags}')
            return {}

    file_path = get_absolute_path(file_path)
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

                    elif tag == "DateTime":
                        metadata["datetime"] = str(value)

                    elif tag == "Orientation":
                        metadata["orientation"] = value

                gps_info = exif.get_ifd(ExifTags.IFD.GPSInfo)
                gps = parse_gps(gps_info)
                if gps:
                    metadata.update(gps)

                exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
                for tag_id, value in exif_ifd.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)

                    if tag == "LensModel":
                        metadata["lens_model"] = str(value)

                    elif tag == "DateTimeOriginal":
                        metadata["datetime_original"] = str(value)

                    elif tag == "DateTimeDigitized":
                        metadata["datetime_digitized"] = str(value)

            return LocalFileMetadata(metadata=metadata)

    except Exception as e:
        logger.warning("get image metadata failed: %s, file=%s", e, file_path)
        return LocalFileMetadata()


if __name__ == '__main__':
    file = '~/Downloads/427D92B6-C697-43CC-B4E1-F542E2BD84C9_1_105_c.jpeg'
    print(get_image_metadata(file))
