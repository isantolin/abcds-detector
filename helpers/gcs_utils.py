import json

from google.cloud import storage

from configuration import Configuration


def get_annotation_uri(config: Configuration, video_uri: str) -> str:
    """Helper to translate video to annotation uri."""
    return video_uri.replace("gs://", config.annotation_path).replace(".", "_") + "/"

def get_reduced_uri(config: Configuration, video_uri: str) -> str:
    """Helper to translate video to reduced video uri."""
    return get_annotation_uri(config, video_uri) + "reduced_1st_5_secs.mp4"

def get_blob(uri: str):
    """Return GCS blob object from full uri."""
    bucket, path = uri.replace("gs://", "").split("/", 1)
    return storage.Client().get_bucket(bucket).get_blob(path)

def upload_blob(uri: str, file_path: str):
    """Uploads GCS blob object from file."""
    bucket, path = uri.replace("gs://", "").split("/", 1)
    storage.Client().get_bucket(bucket).blob(path).upload_from_filename(file_path)

def load_annotation_blob(annotation_uri: str):
    """Loads an annotation blob to json"""
    blob = get_blob(annotation_uri)
    if not blob:
        return {}
    return json.loads(blob.download_as_string()).get("annotation_results", [{}])[0]

def get_video_name_from_uri(uri: str):
    """Gets the video name from the video uri"""
    return uri.split("/")[-1] if uri else ""
