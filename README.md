This project uses my MacBook's webcam to detect if others are using my computer.

trainMe.py
To start, it needs to be trained with a video using the webcam where frames are labeled and encoded into a histogram of oriented gradients model.

recog.py
This uses the webcam and displays a live feed that draws a bounding box around faces along with their label. If an unrecognized face is detected the computer
will say 'intruder' through the webcam.
