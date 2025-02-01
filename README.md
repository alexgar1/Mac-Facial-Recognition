# Facial Recognition Security

This script uses your computers webcam to identify if unauthorized users have logged in using a Histogram of Oriented Gradients model trained on your face. 

The script will alert if only an unidentified face is seen ie. if the owner logs on with someone else next to them it will not send an alert. Using shell script this can be added as a log in trigger to your Mac OS machine. Further modifications could send a text message alert if only an unidenitified face is seen.

## Usage

trainMe.py
To identify who the owner of the computer is take a video with the webcam to get all angles of your face. Use this video with trainMe.py to build the HOG model.

recog.py
This uses the webcam and displays a live feed that draws a bounding box around faces along with their label. If an unrecognized face is detected the computer
will say 'intruder' through the webcam.
