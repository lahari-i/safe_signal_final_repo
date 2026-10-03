import os
import zipfile
from huggingface_hub import hf_hub_download


def download_dataset():

    print("=" * 60)
    print("Downloading WiFall Dataset")
    print("=" * 60)

    zip_path = hf_hub_download(
        repo_id="RS2002/WiFall",
        repo_type="dataset",
        filename="WiFall.zip"
    )

    print("\nDownload Complete")
    print(zip_path)

    return zip_path


def extract_dataset(zip_path):

    print("\nExtracting dataset...")

    os.makedirs("dataset", exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall("dataset")

    print("Dataset extracted successfully.")
    print(os.path.abspath("dataset"))


def main():

    zip_path = download_dataset()

    extract_dataset(zip_path)


if __name__ == "__main__":
    main()