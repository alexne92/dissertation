# Video Prediction Using Machine Learning Techniques

This is a part of the code that was implemented for the MSc Project with the topic ***"Video Prediction Using Machine Learning Techniques"***.

The .mp4 file, which is included in the file, is the video/dataset.\
Using the get_frames.py script a new folder is created, which contains the frames of the video.

It is not required any argument to the code for execution by the user.\
Thus, the implementation of the code is easy and straightforward.

The next script for execution is the main.py file, where the full model is located.\
This script constructs the files, where the results are included.

Blurriness.py is the script that measures the blurriness of the prediction, hence, it should be executed after the main.py file.\
Its results can be found in the folders of each predictive method.

Moreover, the grid_search_eps.py file performs the grid search in order to find the optimal value of a, which is mentioned in the dissertation. Note that the a parameter is named eps in the script.\
Only after executing the grid_search_eps.py file, it is possible to run the grid_search_epsilon_graph.py file, as the results of the first script are saved epsilon_grid_search folder.
