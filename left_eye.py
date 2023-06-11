import PySimpleGUI as sg
import cv2
import numpy as np

def main():
    
    # XML files describing our haar cascade classifiers
    faceCascadeFilePath = "haarcascades/haarcascade_frontalface_default.xml"
    noseCascadeFilePath = "haarcascades/haarcascade_lefteye_2splits.xml"
    
    # Build our cv2 Cascade Classifiers
    faceCascade = cv2.CascadeClassifier(faceCascadeFilePath)
    eyeCascade = cv2.CascadeClassifier(noseCascadeFilePath)
    
    # Load and configure image (.png with alpha transparency)
    image_files = [
       "images/Love-tattoo.png",
       "images/wave-tattoo.png",
       "images/Tribal-Arm-Tattoo.png"
    ]
    
    # Load and configure image (.png with alpha transparency)
    imgTattoo = cv2.imread(image_files[0], -1)
    orig_mask = imgTattoo[:, :, 3]
    orig_mask_inv = cv2.bitwise_not(orig_mask)
    imgTattoo = imgTattoo[:, :, 0:3]
    origTattooHeight, origTattooWidth = imgTattoo.shape[:2]
    
    sg.theme("LightGreen")

    # Define the window layout
    layout = [
        [sg.Text("Tattoo APPlier", size=(60, 1), justification="center")],
        [sg.Image(filename="", key="-IMAGE-")],
        [sg.Radio("None", "Radio", True, size=(10, 1))],
        [
            sg.Radio("blur", "Radio", size=(10, 1), key="-BLUR-"),
            sg.Slider(
                (1, 11),
                1,
                1,
                orientation="h",
                size=(40, 15),
                key="-BLUR SLIDER-",
            ),
        ],
        [
            sg.Radio("hue", "Radio", size=(10, 1), key="-HUE-"),
            sg.Slider(
                (0, 225),
                0,
                1,
                orientation="h",
                size=(40, 15),
                key="-HUE SLIDER-",
            ),
        ],
        [
            sg.Radio("enhance", "Radio", size=(10, 1), key="-ENHANCE-"),
            sg.Slider(
                (1, 255),
                128,
                1,
                orientation="h",
                size=(40, 15),
                key="-ENHANCE SLIDER-",
            ),
        ],
        [sg.Button("Change Image", size=(12, 1)), sg.Button("Exit",  button_color=('red'), size=(10, 1))],
    ]

    # Create the window and show it without the plot
    window = sg.Window("OpenCV Integration", layout, location=(800, 400))

    cap = cv2.VideoCapture(0)
    selected_image_index = 0
    
    while True:
        event, values = window.read(timeout=20)
        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        if event == "Change Image":
            selected_image_index = (selected_image_index + 1) % len(image_files)
            imgTattoo = cv2.imread(image_files[selected_image_index], -1)
            orig_mask = imgTattoo[:, :, 3]
            orig_mask_inv = cv2.bitwise_not(orig_mask)
            imgTattoo = imgTattoo[:, :, 0:3]
            origTattooHeight, origTattooWidth = imgTattoo.shape[:2]
            
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Iterate over each face found
        for (x, y, w, h) in faces:
            # Un-comment the next line for debug (draw box around all faces)
            # cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            eyes = eyeCascade.detectMultiScale(roi_gray)

            for (ex, ey, ew, eh) in eyes:
                # Un-comment the next line for debug (draw box around the nose)
                # cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)

                # Calculate the position for the mustache
                tattooWidth = 3 * ew
                tattooHeight = tattooWidth * origTattooHeight / origTattooWidth

                y_offset = int(eh * 0.9)  # Adjust this value to control the vertical offset

                x1 = int(ex - (tattooWidth / 4))
                x2 = int(ex + ew + (tattooWidth / 4))
                y1 = int(ey + eh / 2 - (tattooHeight / 2) - y_offset)
                y2 = int(ey + eh / 2 + (tattooHeight / 2) - y_offset)

                if x1 < 0:
                    x1 = 0
                if y1 < 0:
                    y1 = 0
                if x2 > w:
                    x2 = w
                if y2 > h:
                    y2 = h

                # Re-calculate the width and height of the mustache image
                tattooWidth = x2 - x1
                tattooHeight = y2 - y1

                # Re-size the original image and the masks to the mustache sizes
                # calculated above
                tattoo = cv2.resize(imgTattoo, (tattooWidth, tattooHeight), interpolation=cv2.INTER_AREA)
                mask = cv2.resize(orig_mask, (tattooWidth, tattooHeight), interpolation=cv2.INTER_AREA)
                mask_inv = cv2.resize(orig_mask_inv, (tattooWidth, tattooHeight), interpolation=cv2.INTER_AREA)

                # Convert the mask and roi to the appropriate data type
                mask = mask.astype(np.uint8)
                mask_inv = mask_inv.astype(np.uint8)
                roi = roi_color[y1:y2, x1:x2].astype(np.uint8)

                roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
                roi_fg = cv2.bitwise_and(tattoo, tattoo, mask=mask)
                dst = cv2.add(roi_bg, roi_fg)
                roi_color[y1:y2, x1:x2] = dst

                break

        if values["-BLUR-"]:
            frame = cv2.GaussianBlur(frame, (21, 21), values["-BLUR SLIDER-"])
        elif values["-HUE-"]:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            frame[:, :, 0] += int(values["-HUE SLIDER-"])
            frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
        elif values["-ENHANCE-"]:
            enh_val = values["-ENHANCE SLIDER-"] / 40
            clahe = cv2.createCLAHE(clipLimit=enh_val, tileGridSize=(8, 8))
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        imgbytes = cv2.imencode(".png", frame)[1].tobytes()
        window["-IMAGE-"].update(data=imgbytes)

    cap.release()
    cv2.destroyAllWindows()
    window.close()


main()
