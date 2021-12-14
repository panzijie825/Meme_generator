# Meme generator website
Source code repository for vision and language course project. 
Website modified from the demo webpage of vislang.ai. Meme captioning module refers:https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning

## Overview
This webpage is implemented using Flask. The backend implements an encoder-decoder and bert

## Requirements
- Setup a conda environment and install some prerequisite packages. See the full dependencies in environment.yml.
```bash
conda create -n vislang python=3.7    # Create a virtual environment
source activate vislang         	    # Activate virtual environment
conda install whoosh flask  # Install dependencies
conda install simpletransformers==0.9.1
conda install torch torchvision
conda install transformers==2.2.0
conda install seqeval
conda install tensorboardx
conda install matplotlib
pip3 install --user simpletransformers==0.9.1
```

##Dataset and trained models
The dataset(images and image-to-caption csv) and trained models can be downloaded from here:
https://drive.google.com/drive/folders/1HhHfHV9q7TxfEyWDIhRjsAVGCgPALrn8?usp=sharing

## Running the website
In order to test the website only the following commands need to be run for the backend.
```bash
source activate vislang
export FLASK_ENV=development
export FLASK_APP=main.py
flask run
```

## Usage
1. Run the website backend.
2. Open http://127.0.0.1:5000/ in the browser.
3. Enter the text and click submit to see the generated result.


