This is a project helping people building a personal PC.


# Pipeline
The pipeline is contained in the folder cpu_gpu_fps_predictor.

## test data
cpu_gpu_fps_predictor/tested_data contains the data to train the model.
It is stored in sqlite3 database. 


## model
cpu_gpu_fps_predictor/model.ipynb is the jupyternote book file that construct the model to predict FPS based on GPU, CPU, and Game. The basic method is tensor decomposition and tensorflow is used to implement this model.

## prediction data
prediction data is contained in the folder cpu_gpu_fps_predictor/prediction_data

## frontend.py
frontend.py is the API written with Streamlit for PC gamers to interact with the prediciton data to find the best GPU-CPU combo under the budget they have in mind.

http://18.212.243.178:8501
