import cv2
import numpy as np

import matplotlib.pyplot as plt


# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('calle1.mp4')
 
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

num_frames = 0
frames = []

# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:

    num_frames += 1
	
    frames.append(frame)
	
    # Display the resulting frame
    #cv2.imshow('Frame',frame)
 
    # Press Q on keyboard to  exit
    #if cv2.waitKey(25) & 0xFF == ord('q'):
    #  break
 
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
#cv2.destroyAllWindows()

frames = np.array(frames)

print(frames.shape)

# reescalado imagenes
imgScale = 0.2
frame1 = cv2.resize(frames[0],(int(frames[0].shape[1]*imgScale),int(frames[0].shape[0]*imgScale)))
frame2 = cv2.resize(frames[20],(int(frames[20].shape[1]*imgScale),int(frames[20].shape[0]*imgScale)))

# conversion a escala de grises
frame1_gs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
frame2_gs = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

auto_corr = cv2.absdiff(frame1_gs, frame2_gs)
cv2.imshow('Dif', auto_corr)

hist,bins = np.histogram(auto_corr.ravel(),256,[0,256])
print(hist)


plt.hist(auto_corr.ravel(),256,[0,256])
plt.show()

opening = cv2.morphologyEx(auto_corr, cv2.MORPH_OPEN, np.ones((2,2),np.uint8))
cv2.imshow('open', opening)

opening2 = cv2.morphologyEx(opening, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
cv2.imshow('open2', opening2)

ret,thresh1 = cv2.threshold(opening,10,255,cv2.THRESH_BINARY)
cv2.imshow('thres', thresh1)

erode = cv2.erode(auto_corr,np.ones((2,2),np.uint8),1)
cv2.imshow('erd', erode)

erode2 = cv2.erode(erode,np.ones((2,2),np.uint8),1)
cv2.imshow('erd2', erode2)


dilation = cv2.dilate(erode,np.ones((2,2),np.uint8),1)
cv2.imshow('dil', dilation)

erode3 = cv2.erode(auto_corr,np.ones((3,3),np.uint8),1)
cv2.imshow('erd3', erode3)

dilation2 = cv2.dilate(erode3,np.ones((2,2),np.uint8),1)
cv2.imshow('dil2', dilation2)


ret,thresh2 = cv2.threshold(dilation2,4,255,cv2.THRESH_BINARY)
cv2.imshow('thres2', thresh2)

opening3 = cv2.morphologyEx(thresh2, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
cv2.imshow('open3', opening3)
opening3 = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
cv2.imshow('open3', opening3)


print(frame1_gs.shape)

# correlaciones cruzadas
VENTANA = 10
LEN = 1 + 2 * VENTANA

cross_corr = np.zeros((LEN, LEN))
for y in range(-VENTANA, VENTANA + 1):
	for x in range(-VENTANA, VENTANA + 1):
		f1 = frame1_gs[ VENTANA : frame1_gs.shape[0] - VENTANA, VENTANA : frame1_gs.shape[1] - VENTANA]
		f2 = frame2_gs[ y + VENTANA : frame2_gs.shape[0] + y - VENTANA, x + VENTANA : frame2_gs.shape[1] + x - VENTANA]
		cross_corr[y + VENTANA, x + VENTANA] = np.sum(f1 * f2)

cross_corr_max_x = cross_corr.argmax() % LEN
cross_corr_max_y = cross_corr.argmax() // LEN

#print(cross_corr/cross_corr.max() )
##print(cross_corr.max())
#print(cross_corr.argmax())
#print("max:")
#print(cross_corr_max_y, cross_corr_max_x)

frame_diffa = cv2.absdiff(frame1_gs[VENTANA : frame1_gs.shape[0] - VENTANA, VENTANA : frame1_gs.shape[1] - VENTANA], frame2_gs[VENTANA : frame1_gs.shape[0] - VENTANA, VENTANA : frame1_gs.shape[1] - VENTANA])

f1 = frame1_gs[VENTANA : frame1_gs.shape[0] - VENTANA, VENTANA : frame1_gs.shape[1] - VENTANA]
f2 = frame2_gs[cross_corr_max_y : frame2_gs.shape[0] + cross_corr_max_y - 2 * VENTANA, cross_corr_max_x : frame2_gs.shape[1] + cross_corr_max_x - 2 * VENTANA]
frame_diffb = cv2.absdiff(f1, f2)

#cv2.imshow('Diffa',frame_diffa)
#cv2.imshow('Diffb',frame_diffb)

#print(frame1_gs[:,0])
#print(frame2_gs[0,:])
#for y in range(2 * VENTANA + 1):
#for y in [4,9,10,11]:
	#for x in range(2 * VENTANA + 1):
#		x=10
#		f2 = frame2_gs[y : frame2_gs.shape[0] + y - 2 * VENTANA, x : frame2_gs.shape[1] + x - 2 * VENTANA]
#		frame_diff2 = cv2.absdiff(f1, f2)
#		print(y,x)
#		cv2.imshow('Frame' ,frame_diff2)
#		cv2.waitKey(0)

#
#cv2.imshow('Frame',frame1_gs)
#cv2.imshow('Frame1',frame2_gs)
#cv2.imshow('Diff3',frame_diff2)


# Press Q on keyboard to  exit
cv2.waitKey(0)
cv2.destroyAllWindows() 