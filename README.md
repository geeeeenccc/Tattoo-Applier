# Tattoo-Applier

This project demonstrates how to apply a tattoo image on a person's face using computer vision techniques. The program detects faces and eyes in real-time using Haar cascade classifiers and overlays a tattoo image on the detected eye regions.

## Requirements

- Python 3.x
- PySimpleGUI
- OpenCV
- NumPy

## Project Structure

The project consists of the following files:

- `left_eye.py`: The main Python script that runs the tattoo image applier on the detected eye regions.
- `haarcascades/haarcascade_frontalface_default.xml`: XML file describing the Haar cascade classifier for detecting faces.
- `haarcascades/haarcascade_lefteye_2splits.xml`: XML file describing the Haar cascade classifier for detecting eyes.
- `images/Love-tattoo.png`, `images/mustache.png`, `images/Tribal-Arm-Tattoo.png`, `images/wave-tattoo.png`: Image file of the tattoo to be applied.

## Usage


To run the project, make sure you have all the requirements installed. Then, execute the `left_eye.py` script. The GUI window will open, displaying the live video stream from your default camera.

The GUI provides the following options:

- **None**: No additional effects are applied.
- **Blur**: Applies a blur effect to the video stream. Use the slider to control the intensity of the blur.
- **Hue**: Adjusts the hue of the video stream. Use the slider to control the hue shift.
- **Enhance**: Enhances the video stream using contrast-limited adaptive histogram equalization (CLAHE). Use the slider to control the enhancement level.
- **Change Image**: Changes a tattoo to apply. Click a button to change the tattoo image.

You can choose an option and adjust the corresponding sliders to see the effect on the video stream. The GUI window also displays the current frame with the mustache overlay applied to the detected eyes.

To exit the application, click the "Exit" button or close the GUI window.

**Note:** Before running the tattoo applier program, make sure you have a camera connected to your computer. The program relies on the webcam feed for real-time face and eye detection.

## Notes

- The project uses Haar cascade classifiers to detect faces and eyes in the video stream.
- The overlay image (`Love-tattoo.png`) is loaded and resized based on the detected eye region.
- Additional image processing effects like blur, hue adjustment, and contrast enhancement can be applied to the video stream.

## Conclusion

The tattoo applier project demonstrates how to use computer vision techniques to apply a tattoo image on a person's face in real-time. It can serve as a starting point for more advanced projects involving facial image processing and augmentation.
