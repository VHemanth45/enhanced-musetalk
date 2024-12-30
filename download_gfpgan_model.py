import requests
import os

# URL to the GFPGAN model file
url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth" #-P experiments/pretrained_models"
output_dir = "./external_models/GFPGAN/experiments/pretrained_models/"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
output_path = os.path.join(output_dir, "GFPGANv1.3.pth")

# Download the file
print("Downloading GFPGAN model...")
response = requests.get(url, stream=True)
if response.status_code == 200:
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)
    print(f"Model downloaded successfully to {output_path}")
else:
    print(f"Failed to download the model. HTTP Status Code: {response.status_code}")
