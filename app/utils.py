import hashlib
import pandas as pd
import os


def validate_file(file_obj):
    # file_obj is a Streamlit UploadedFile
    if not file_obj.name.endswith('.log'):
        return False, "Only .log files are supported"

    # Optional: check if file is empty
    content = file_obj.read()
    if not content:
        return False, "The file is empty"
    
    file_obj.seek(0)  # Reset read pointer after reading
    return True, "File is valid"


# Compute a consistent hash from file contents
def compute_file_hash(file) -> str:
    file.seek(0)
    content = file.read()
    file.seek(0)
    return hashlib.sha256(content).hexdigest()

# Validate the uploaded file type and size
def is_valid_log_file(file) -> bool:
    filename = file.name
    allowed_extensions = (".log", ".txt")
    max_size_mb = 10

    return (
        filename.lower().endswith(allowed_extensions)
        and (file.size / (1024 * 1024)) <= max_size_mb
    )

# Get readable size
def readable_file_size(file_size_bytes: int) -> str:
    if file_size_bytes < 1024:
        return f"{file_size_bytes} B"
    elif file_size_bytes < 1024 ** 2:
        return f"{file_size_bytes / 1024:.2f} KB"
    else:
        return f"{file_size_bytes / (1024 ** 2):.2f} MB"
