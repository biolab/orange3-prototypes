Orange3 Prototypes
==================
Prototype Orange widgets. Only for the brave.

Installing
----------
This add-on requires Orange3 and Python 3.4 or newer. To install
it, run:

    # Clone the repository and move into it
    git clone https://github.com/biolab/orange3-prototypes.git
    cd orange3-prototypes

    # Setup the add-on
    pip install .

To install Orange in editable/development mode, run

    pip install -e .

Alternatively, you can install the add-on from PyPI:

    pip install orange3-prototypes


### OpenCV dependency

To access Face Detector and Webcam widgets, you need OpenCV library.

#### Windows

Download the required [OpenCV] package. Make sure you download the package for your version of Python and OS.

[OpenCV]: http://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv

Then go to the folder containing the downloaded file and open the terminal. Run (insert the right file name after install):

    pip install <opencv_python‑3.2.0‑cp36‑cp36m‑win_amd64.whl>

#### MacOS

In the terminal, run:

    /Applications/Orange3.app/Contents/MacOS/pip install opencv-python
