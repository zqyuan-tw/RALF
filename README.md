# Face Forgery Detection via Reconstructing Authentic-like Frequency

Welcome to the official repository of Face Forgery Detection via Reconstructing Authentic-like Frequency.

## Preparation

1. To accelerate the training process, it is strongly recommended to crop faces from each image or frame of the video and save them as individual image files in advance.

1. Download the pretrained weight of Xception.
    ```shell
    wget http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth
    ```

1. Install the required packages.
    ```shell
    pip install -r requirements.txt
    ```

1. Download the following datasets or prepare your own dataset and modify `./ffdataset.py` if necessary:
    - [FF++](https://github.com/ondyari/FaceForensics)
    - [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics)
    - [DFDC](https://ai.facebook.com/datasets/dfdc/)

## Quick start

```shell
python3 train.py --log_dir LOG_DIR
```
Make sure to replace `LOG_DIR` with the desired directory path.

<!-- ## Citations

If you use this repository in your research or work, please cite it as:
```tex
``` -->

## Contributing

Your contributions are highly appreciated! If you have any suggestions, optimizations, or improvements for this project, please feel free to open an issue or submit a pull request. We welcome any feedback or ideas to make this project better.

## Acknowledgments

This work was supported in part by the Cybersecurity Center of Excellence (CCoE) program under National Science and Technology Council (NSTC), Taiwan.