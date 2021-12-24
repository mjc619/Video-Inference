# Easy Video Inference

This repository is an inference repo similar to that of the ESRGAN inference repository, but for various video machine learning models. The idea is to allow anyone to easily run various models on video without having to worry about different repo setups. PRs welcome.

## Currently supported architectures

- SOFVSR ([traiNNer](https://github.com/victorca25/traiNNer/) Version)
  - Original SOFVSR SR net
  - RRDB SR net
- RIFE ([traiNNer](https://github.com/victorca25/traiNNer/) Version)
  - Supports regular and 'HD' RIFE models
  - Original RIFE models will need to be converted to BasicSR's single-model .pth file (conversion script located in utils folder)
- [TecoGAN-Pytorch](https://github.com/skycrapers/TecoGAN-PyTorch)

## Additional features

- Automatic scale, number of frames, number of channels, and SR architecture detection
- Automatic 'HD' RIFE model detection
- Automatic beginning and end frame padding so all frames get included in output
- Direct video input and output through ffmpeg
- FP16 support for faster inference on RTX cards
- EXR file support with PyEXR (https://github.com/tvogels/pyexr)

## Using this repo

Requirements: `numpy, opencv-python, pytorch, progressbar2`

Optional requirements: `ffmpeg-python` to use video input/output (requires ffmpeg to be installed)

### Obtaining models

#### SOFVSR

- [Game Upscale Wiki Model Database](https://upscale.wiki/wiki/Model_Database#SOFVSR_.28.22vicGAN.22.29_Models)

#### RIFE

- Converted .pth files: [1.3](https://mega.nz/file/DhBWgRYQ#hLkR4Eiks6s3ZvwLCl4eA57J3baR0eDXjyaV9yzmTeM) | [1.4](https://u.pcloud.link/publink/show?code=XZR9gLXZWREwfp3svoRW1WNKY0H5bFxaufkk) | [1.5 (HD)](https://u.pcloud.link/publink/show?code=XZeXKLXZdqXM0uCIGvH7IFyg0sSwC7dl2y2X)
- Model conversion script located in utils folder

#### TecoGAN

- [BD degradation](https://drive.google.com/file/d/13FPxKE6q7tuRrfhTE7GB040jBeURBj58/view) | [BI degradation](https://drive.google.com/file/d/1ie1F7wJcO4mhNWK8nPX7F0LgOoPzCwEu/view)

### Upscaling exported frames

- Place exported video frames in the `input` folder
- Place model in the `models` folder
- Example: `python run.py ./models/video_model.pth`

### Upscaling video files

- Place model in the `models` folder
- Set `--input` to your input video
- Set `--output` to your output video
- Example: `python run.py ./models/video_model.pth --input "./input/input_video.mp4" --output "./output/output_video.mp4"`

## Extra flags

- `--input`: Specifies input directory or file
- `--output`: Specifies output directory or file
- `--denoise`: Denoises the chroma layer
- `--chop_forward`: Splits tensors to avoid out-of-memory errors
- `--crf`: The crf (quality) of the output video when using video input/output. Defaults to 0 (lossless)
- `--exp`: RIFE exponential interpolation amount
- `--fp16`: Speedup on RTX cards using HalfTensors
- `--exr`: output fp32 images in .exr format (will be affected by fp16)

## Planned architecture support

- EDVR (modified)
- RRN
- Updated RIFE models
- Deep Video Deinterlacing

## Planned additional features

- More FFMPEG options
- Model chaining
- Will probably modify this repository to also run image models such as ESRGAN
