import cv2
import numpy as np

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
def open_video(filename = ""):
	if filename == "":
		return cv2.VideoCapture(0)
	
	return cv2.VideoCapture(filename)

# Create a numpy array with all frames from a file
def getFramesFromFile(filename):

	videoCapture = cv2.VideoCapture(filename)

	# Check if camera opened successfully
	if (videoCapture.isOpened()== False): 
		print("Error opening video stream or file")

	frames_count = 0
	frames = []

	# Read until video is completed
	while(videoCapture.isOpened()):
	
		# Capture frame-by-frame
		ret, frame = videoCapture.read()
		
		if ret == True:
			frames_count += 1
			frames.append(frame)
		
		# Break the loop
		else: 
			break
 
	# When everything done, release the video capture object
	videoCapture.release()

	return frames_count, np.array(frames)
	
# Main body
VIDEO_FILENAME = 'calle1.mp4'

frames_count, frames = getFramesFromFile(VIDEO_FILENAME)

print(frames_count)

