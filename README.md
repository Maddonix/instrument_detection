# Instrument Detection
## Requirements
Python 3.9

## Setup
Open the directory in a terminal and install the dependencies via:
  pip install -r requirements.txt
  
Download the model checkpoint at: https://fex.ukw.de/public/download-shares/3JAU5PUTUGAWjjM8LPNOKDzYYVq55LNW
Save the file in the directory as "model.cpkt"
  
## Usage
To predict the provided sample images open a terminal and run the script with:
  python instrument_detection.py
  
To predict other images, remove the provided sample images and place your own images in the folder "image_folder"
  
## Limitations
The script will only accept ".jpg" files. For demonstration purposes, this example script does not utilize the systems GPU. Therefore, prediction will be slow.
