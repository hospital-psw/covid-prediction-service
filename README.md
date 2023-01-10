# covid-prediction-service

## Requirements
<p>The requirements can be found inside of the requirements.txt file<br>
To install them just run <b><br>
pip install -r requirements.txt</b><br></p>

## Running the Server
<p>The server runs on the port 6900<br>
To run the server just type<br>
<b>python -m server.flask_server</b><br></p>

## Server Endpoints
<p>The server has 3 endpoints</p>
<p>The first one is a GET on / which is a sanity check for wether the server works</p>
<p>The second one is also a GET on <b>/model/train</b> it trains and saves the neural network, the network will be saved as trained_model/model.pickle</p>
<p>The last endpoint is a POST on <b>/model/predict</b> which takes a JSON with the patients symptoms and returns a prediction response JSON with the prediction, confidence and prediction name as string.</p>
