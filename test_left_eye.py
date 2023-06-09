import cv2
import pytest
from left_eye import main

def test_main_function_runs_without_error():
    # Ensure that the main function runs without raising any exceptions
    main()

def test_capture_returns_frame():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    assert ret is True

# @pytest.fixture(scope="module")
# def window():
#     import PySimpleGUI as sg

#     layout = [
#         [sg.Text("Tattoo APPlier", size=(60, 1), justification="center")],
#         [sg.Button("Exit", size=(10, 1))],
#     ]

#     window = sg.Window("OpenCV Integration Test", layout)
#     yield window
#     window.close()

# def test_window_creation(window):
#     assert window is not None

# def test_window_closing(window):
#     assert window.finalize() is None
