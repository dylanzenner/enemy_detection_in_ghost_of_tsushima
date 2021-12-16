[![CodeFactor](https://www.codefactor.io/repository/github/dylanzenner/enemy_detection_in_ghost_of_tsushima/badge)](https://www.codefactor.io/repository/github/dylanzenner/enemy_detection_in_ghost_of_tsushima)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/dylanzenner/enemy_detection_in_ghost_of_tsushima)
![GitHub last commit](https://img.shields.io/github/last-commit/dylanzenner/enemy_detection_in_ghost_of_tsushima)

# Enemy Detection in Ghost of Tsushima

A cascade classifier built using OpenCV for detecting enemy NPCs in the 2020 action-adventure game *Ghost of Tsushima* by Sucker Punch Productions.


# Steps for replication

### Step 1 Environment Setup:

You will need the following before starting this project:
-   A PlayStation 5 (or any console with *Ghost of Tsushima* in stalled)
-   PS Remote Play (or the equivalent software for your console) installed on the machine you will be creating the project on
-   PS Remote Play (or the equivalent software for your console) opened up with *Ghost of Tsushima* running in a window of your choosing. You can use dual monitors or you can size the window to your liking.


### Step 2 Data Collection:
Before we start the data collection we will need to set up our development environment. I am using Pycharm but you can use whatever you are comfortable with
-   The first thing we need to do is create a function which will allow us to take screen shots of the game while we are playing it. The images captured will be used to train the cascade classifier. You can find the function in the src directory inside of the collect_data.py file. 
```{python}
def window_capture():
    """
    Captures the monitor window (area to be specified by user). We are using this in place of a camera because we are
    trying to capture the video on the screen.
    :return: An output image to be used for imshow in the while loop
    """
    with mss() as sct:
        monitor = {"top": 27, "left": 0, "width": 960, "height": 547}
        img = sct.grab(monitor)
        img = np.array(img, dtype=np.uint8)
        np.flip(img[:, :, :3])
    return img
```
As stated in the function doc string it captures a screenshot of a specified window. You will have to adjust the ```monitor = {"top": 27, "left": 0, "width": 960, "height": 547}``` to a window that fits the area of the window that *Ghost of Tsushima* is running in.
-   After you have a window adjusted to your needs you'll have to create the folders ```positive_images``` and ```negative_images``` in your project directory.
-   Then you can run the file ```collect_data.py``` in the src directory.
-   This will allow you to play the game while also taking screenshots of images with enemies (or whatever it is you are trying to detect) in them and save them to either ```positive_images``` (contains images with enemies) or ```negative_images``` (contains images with no enemies).
-   Once you have collected a sufficient amount of data (I collected 1000 images for both folders) move on to step 3.




