import cv2
import os, os.path
import glob
directory_save_frames = "pictures"
if not os.path.exists(directory_save_frames):
    os.makedirs(directory_save_frames)
video_file = glob.glob('*.mp4')[0]
vidcap = cv2.VideoCapture(video_file)
success, image = vidcap.read()
i = 0
success = True
while success:
    image = cv2.resize(image, (64,64))
    if i < 10:
        cv2.imwrite(directory_save_frames + "/frame00"+str(i)+".png", image)
    elif i >= 10 and i < 100:
        cv2.imwrite(directory_save_frames + "/frame0"+str(i)+".png", image)
    elif i >= 100 and i < 110:
        cv2.imwrite(directory_save_frames + "/frame10"+str(i-100)+".png", image)
    elif i >= 110 and i < 200:
        cv2.imwrite(directory_save_frames + "/frame1"+str(i-100)+".png", image)
    elif i >= 200 and i < 210:
        cv2.imwrite(directory_save_frames + "/frame20"+str(i-200)+".png", image)
    elif i >= 210 and i < 300:
        cv2.imwrite(directory_save_frames + "/frame2"+str(i-200)+".png", image)
    success,image = vidcap.read()
    i += 1
    
