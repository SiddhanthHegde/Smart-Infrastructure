# Smart-Infrastructure

The challenge of measuring and monitoring bridge behaviour and analyzing structural deterioration over time was solved mainly using several sensors placed at several key positions on the bridge. The inputs from these are processed and analyzed in order to understand the behaviour of the bridge. The main sensors whose data is being utilized in this tasks are:

- There are 20 sensors of the type accelerometers placed on the bridge. 
- Each accelerometer provides 3 channels of data measuring acceleration in axes orthogonal to each other.
- One air temperature sensor located near the bridge.

Note that all sensors do not have the same orientation in space and may be rotated relative to each other in both the horizontal and vertical plane. Channels 1 (axis a), 2 (axis b) and 3 (axis c) may thus represent different directions for different sensors, i.e. point in different directions, that could even be opposite to each other.
The accelerometers deliver data in 64 Hz, while temperature measurements are updated every 10 minutes.

## Task 1: Data Compression 

- Since there are 20 accelerometers and 3 channels for each of them, we have 60 values/dimensions recorded for each point in time [A1_a, A1_b, A1_c, A2_a, A2_b, A2_c … A20_a, A20_b, A20_c]. These number of dimensions can place a toll on bandwidth as well as storage, especially as the number of sensors and sampling frequency increase. The objective of the task is to reduce the number of dimensions from 60 while ensuring that at least 90 percent of the original data is retained. Let these new reduced dimensions be referred to as K1, K2 ... KN, where N<60.
- The solution should consist of two parts, an encoder/compressor, and a decoder/un-compressor. Let X1, X2, X3...X60, be the values for 60 dimensions at a point of time t. The encoder should take as input these 60 dimensions and encode them into K1, K2 ... KN, where N<60. The decoder should be able take as input, these K1, K2 ... KN and use it to recreate/predict the original values/dimensions, let these decoded values be X1*, X2*, … X60*. The data loss would be measured between the original values X1, X2 … X60 and the encoded-decoded result X1*, X2*, … X60*.

### Metrics:

<img src = "https://s3-ap-southeast-1.amazonaws.com/he-public-data/metric%20for%20data%20loss8387117.png">

#### Solution: Auto Encoders with Stacked LSTMs

Using LSTM as the main building block for the encoding technique and storing the information in a single neuron compressed most of the data. Decoding was done using stacked transpose convolutions to get back the data using MAE as the loss function. Achieved the difference score of around 4% for a subset of data in the dataset 1.

Code for Task 1: 
- To train, run Task 1/train.py file 
- Run compute() function in Task 1/test.py with the appropriate csv_path and model checkpoint path  
- Run compute_error() function with predictions and ground truth to find the difference percentage

## Task 2: Sensor Network Robustness
- The objective of the task is to handle situations of sensor-fallouts, by predicting the values of those sensors using the data of other sensors. The task is to develop 20 Models M1 to M20 that can predict the values of A1 to A20, respectively. For example, Model M1 should predict the value of A1 using the values of A2 to A20, M2 should predict the value of A2 and so on. 
- FYI: These individual models M1 to M20 can contain more models within them if the algorithm needs. For example, to predict the value of A1, M1 needs to predict the value of A1_a, A1_b and A1_c. So, M1 can have different component models within it, dedicated to predicting each of the axes, ie M1_a for A1_a, M1_b for A1_b, M1_c for A1_c.
- The corresponding temperature data will also be provided, and can be used, if the candidate wants. Note that the sensors may be subject to (internal and individual) temperature bias, giving a constant change of values in each channel, and may also rotate slightly as the structure deforms upon change in temperature. 

#### Solution: 1d Convolution predictor

Using stacked 1d convolutions and data of two sensors as the input, trained the model to predict the data of the third sensor. Achieved an RMSE of around 0.7 with this network.

Code for Task 2:
- Run Task 2/train.py file to train the network and predict the value
- Code for a single model M1 is created, multiple models can be created using a loop and with different number of convolution layers
