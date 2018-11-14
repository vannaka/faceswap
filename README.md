# FaceSwap

This python script will take two images, with a single head in each, and take the face of one and put it on the head of the other.

## Environment Setup

To run the script you'll need to install a few things. If you already have a working python 3 install then skip to step 3.

### 1. Install Python
  * Download Anaconda for Python 3.7 [here](https://www.anaconda.com/download/).
  * Make sure to chose the option to add to PATH during installation.

### 2. Create and Activate an Environment
  * Open the command prompt and execute the following commands:

    ```shell
    conda createe --name opencv-env python=3.7
    activate opencv-env
    ```

### 3. Install required packages
  * Continuing from the above prompt, execute the following commands:

    ```shell
    pip install numpy scipy matplotlib scikit-learn jupyter
    pip install opencv-contrib-python
    pip install cmake
    pip install dlib
    ```

### 4. Run the Script
  * The script is run like so:

    ```shell
    ./faceswap.py <head image> <face image>
    ```

    If successful, a file `output.jpg` will be produced with the facial features
    from `<head image>` replaced with the facial features from `<face image>`.

