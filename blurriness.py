import glob
import cv2
import pandas as pd

directories_frames = ["naive_time_series/predictions/",
                      "optimal_time_series/predictions/",
                      "lstm1/predictions/",
                      "lstm2/predictions/"]
for directory_frames in directories_frames:
    frames = glob.glob(directory_frames + '*.png')
    result = []
    for frame in frames:
        image = cv2.imread(frame)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()
        result.append(fm)
        
    blurriness = ["Blurry" if i < 100 else "Not blurry" for i in result]
    df = pd.DataFrame()
    df["Variance of Laplace"] = result
    df["Blurriness"] = blurriness
    df.index = [i[-12:-4] for i in frames]
    df.to_csv(directory_frames + "blurriness.csv",
                                  sep=',',
                                  encoding='utf-8')
    