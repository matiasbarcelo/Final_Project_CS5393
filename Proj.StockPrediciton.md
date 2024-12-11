
# Recurrent Neural Networks and Stock Price Prediciton

First, let's start with the LSTM model:

```python
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Fetch stock data
ticker = "AAPL"
data = yf.download(ticker, start="2010-01-01", end="2023-11-13")
df = data[['Close']].reset_index()

# Prepare data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# Create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), 0])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)

seq_length = 60
X, y = create_sequences(scaled_data, seq_length)

# Split data
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape input for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=32, epochs=100, validation_split=0.1, verbose=1)

# Make predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate RMSE
rmse = np.sqrt(np.mean((predictions - y_test)**2))
print(f"RMSE: {rmse}")
```

Now, let's modify the code to use GRUs instead of LSTMs:

```python
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam

# Fetch stock data
ticker = "AAPL"
data = yf.download(ticker, start="2010-01-01", end="2023-11-13")
df = data[['Close']].reset_index()

# Prepare data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# Create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), 0])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)

seq_length = 60
X, y = create_sequences(scaled_data, seq_length)

# Split data
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape input for GRU
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build GRU model
model = Sequential([
    GRU(50, return_sequences=True, input_shape=(seq_length, 1)),
    GRU(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=32, epochs=100, validation_split=0.1, verbose=1)

# Make predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate RMSE
rmse = np.sqrt(np.mean((predictions - y_test)**2))
print(f"RMSE: {rmse}")
```

These code examples demonstrate how to use LSTM and GRU models for stock price prediction. Both models use a similar structure with two layers of LSTM/GRU units followed by dense layers. The main difference is the use of LSTM layers in the first example and GRU layers in the second.

Key points to note:

1. The code uses the `yfinance` library to fetch historical stock data for Apple (AAPL).
2. Data is preprocessed using MinMaxScaler to normalize the values.
3. Sequences of 60 days are created as input features.
4. The data is split into training and testing sets.
5. The models are compiled using the Adam optimizer and mean squared error loss function.
6. After training, predictions are made on the test set, and the Root Mean Square Error (RMSE) is calculated to evaluate the model's performance.

Remember that these are basic examples and may not be suitable for real-world trading without further refinement and validation. Stock price prediction is a complex task influenced by many factors, and past performance does not guarantee future results.

Citations:
[1] https://github.com/YC-Coder-Chen/GRU-stock-price-prediction
[2] http://paper.ijcsns.org/07_book/201901/20190126.pdf
[3] https://sist.sathyabama.ac.in/sist_naac/documents/1.3.4/1922-b.sc-cs-batchno-24.pdf
[4] https://www.youtube.com/watch?v=WcJ6Ojfvla4
[5] https://sunscrapers.com/blog/deep-learning-for-nlp-an-overview/
[6] https://slds-lmu.github.io/seminar_nlp_ss20/recurrent-neural-networks-and-their-applications-in-nlp.html
[7] https://www.ibm.com/topics/recurrent-neural-networks
[8] https://developer.nvidia.com/discover/recurrent-neural-network


*****  Add factors

To modify the GRU model to include additional features like volume changes, you can make the following adjustments:

1. Expand the input features:
   Instead of just using the closing price, include additional features like volume changes in your input data. For example:

   ```python
   features = ['Close', 'Volume']
   df = data[features].values
   ```

2. Normalize the additional features:
   Apply normalization to all input features, including volume:

   ```python
   scaler = MinMaxScaler()
   scaled_data = scaler.fit_transform(df)
   ```

3. Modify the input shape:
   Update the input shape of your GRU model to account for the additional features:

   ```python
   input_shape = (seq_length, len(features))
   ```

4. Adjust the model architecture:
   Modify your GRU model to handle the new input shape:

   ```python
   model = Sequential([
       GRU(50, return_sequences=True, input_shape=input_shape),
       GRU(50, return_sequences=False),
       Dense(25),
       Dense(1)
   ])
   ```

5. Update the sequence creation function:
   Modify your `create_sequences` function to handle multiple features:

   ```python
   def create_sequences(data, seq_length):
       X, y = [], []
       for i in range(len(data) - seq_length):
           X.append(data[i:(i + seq_length), :])
           y.append(data[i + seq_length, 0])  # Assuming you're predicting the closing price
       return np.array(X), np.array(y)
   ```

6. Consider feature engineering:
   You might want to engineer new features based on volume, such as volume change percentage or moving averages of volume[1].

By incorporating these changes, your GRU model will be able to utilize volume changes and other additional features in its predictions. This can potentially improve the model's performance by providing more contextual information about stock price movements[2].

Remember to experiment with different combinations of features and model architectures to find the optimal setup for your specific prediction task. Additionally, you may want to consider using techniques like feature importance analysis to determine which features contribute most significantly to the predictions[5].

Citations:
[1] http://paper.ijcsns.org/07_book/201901/20190126.pdf
[2] https://www.datacamp.com/tutorial/tutorial-for-recurrent-neural-network
[3] https://slds-lmu.github.io/seminar_nlp_ss20/recurrent-neural-networks-and-their-applications-in-nlp.html
[4] https://pmc.ncbi.nlm.nih.gov/articles/PMC9844920/
[5] http://www.jatit.org/volumes/Vol101No5/21Vol101No5.pdf
[6] https://github.com/YC-Coder-Chen/GRU-stock-price-prediction
[7] https://www.ibm.com/topics/recurrent-neural-networks
[8] https://sunscrapers.com/blog/deep-learning-for-nlp-an-overview/



======================

To modify the GRU architecture to handle additional features like volume data in stock price prediction, you can follow these steps:

## Modifications Needed

1. **Input Feature Expansion**:
   - Include additional features such as volume data alongside the closing prices in your input dataset.
   - For example, if you are currently using only the closing prices, you can expand your dataset to include both closing prices and volume.

2. **Data Normalization**:
   - Normalize all input features to ensure they are on a similar scale. This is crucial for effective training of neural networks.
   - Use techniques like Min-Max Scaling or Standardization for this purpose.

3. **Adjust Input Shape**:
   - Modify the input shape of your GRU model to accommodate the additional features.
   - If you have $$n$$ features (e.g., closing price and volume), ensure that your input shape reflects this: `(seq_length, n_features)`.

4. **Model Architecture Update**:
   - Update the GRU model architecture to handle the new input shape.
   - Ensure that the GRU layers are configured to process multi-dimensional input data.

5. **Sequence Creation**:
   - Adjust your sequence creation logic to handle multiple features. Each sequence should now include all relevant features for each time step.

## Example Code

Here's an example of how you might modify the GRU model to include volume data:

```python
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam

# Fetch stock data
ticker = "AAPL"
data = yf.download(ticker, start="2010-01-01", end="2023-11-13")
df = data[['Close', 'Volume']].reset_index()

# Prepare data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Close', 'Volume']])

# Create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), :])
        y.append(data[i + seq_length, 0])  # Predicting the closing price
    return np.array(X), np.array(y)

seq_length = 60
X, y = create_sequences(scaled_data, seq_length)

# Split data
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build GRU model
model = Sequential([
    GRU(50, return_sequences=True, input_shape=(seq_length, 2)),  # Adjusted input shape for two features
    GRU(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=32, epochs=100, validation_split=0.1, verbose=1)

# Make predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(np.concatenate((predictions, np.zeros_like(predictions)), axis=1))[:, 0]
y_test = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros_like(y_test.reshape(-1, 1))), axis=1))[:, 0]

# Calculate RMSE
rmse = np.sqrt(np.mean((predictions - y_test)**2))
print(f"RMSE: {rmse}")
```

### Key Changes Explained

- **Input Features**: The model now uses both 'Close' and 'Volume' as features.
- **Normalization**: Both features are normalized together.
- **Input Shape**: The GRU layers are set up to accept inputs with two features per time step.
- **Sequence Creation**: Sequences now include both features at each time step.

These modifications allow the GRU model to leverage additional information from volume changes alongside price data, potentially improving prediction accuracy by capturing more complex patterns in the stock market data.

Citations:
[1] https://www.scaler.com/topics/deep-learning/gru-network/
[2] https://www.activeloop.ai/resources/glossary/gated-recurrent-units-gru/
[3] https://slds-lmu.github.io/seminar_nlp_ss20/recurrent-neural-networks-and-their-applications-in-nlp.html
[4] https://pmc.ncbi.nlm.nih.gov/articles/PMC9844920/
[5] http://www.jatit.org/volumes/Vol101No5/21Vol101No5.pdf
[6] https://www.ibm.com/topics/recurrent-neural-networks
[7] https://www.datacamp.com/tutorial/tutorial-for-recurrent-neural-network
[8] http://paper.ijcsns.org/07_book/201901/20190126.pdf

++++++++++++++++++++++++++++++++

Yes, you can use different activation functions in GRUs to better handle volume data or other specific characteristics of your dataset. Here’s how you can approach it:

## Choosing Activation Functions

1. **Standard Activation Functions**:
   - **Sigmoid**: Commonly used for the gates in GRUs to control information flow, as it outputs values between 0 and 1, effectively deciding what information to keep or discard.
   - **Tanh**: Typically used for the candidate hidden state because it outputs values between -1 and 1, allowing for both positive and negative updates.

2. **Alternative Activation Functions**:
   - **ReLU (Rectified Linear Unit)**: Useful for preventing saturation issues that can occur with sigmoid and tanh, especially with large input values. It can be beneficial if your data has non-negative characteristics.
   - **Leaky ReLU**: A variation of ReLU that allows a small gradient when the unit is not active, which can help mitigate the dying ReLU problem.
   - **PReLU (Parametric ReLU)** and **ELU (Exponential Linear Unit)**: These can offer more flexibility and potentially better performance by learning the activation function parameters or providing smoother outputs.

3. **Considerations for Volume Data**:
   - Volume data might have different statistical properties compared to price data, such as larger ranges or different distributions. Activation functions like ReLU or ELU could be more suitable if volume data tends to have larger positive values.
   - Experimenting with different activation functions for the hidden states or candidate states can help capture complex patterns in volume data.

## Implementing Different Activation Functions in GRUs

Here's an example of how you might modify a GRU model to use different activation functions:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import relu

# Build GRU model with custom activation functions
model = Sequential([
    GRU(50, return_sequences=True, input_shape=(seq_length, n_features),
        activation='relu', recurrent_activation='sigmoid'),  # Using ReLU for hidden state
    GRU(50, return_sequences=False,
        activation='relu', recurrent_activation='sigmoid'),
    Dense(25, activation='relu'),
    Dense(1)  # Output layer
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
```

## Key Points

- **Activation Function Selection**: Choose based on the data characteristics and the specific task requirements. For instance, if avoiding saturation is crucial due to large input values, ReLU might be preferable.
- **Experimentation**: There is no one-size-fits-all solution; experimenting with different activation functions and evaluating their impact on model performance is essential.
- **Hyperparameter Tuning**: Consider using hyperparameter tuning techniques to systematically explore different configurations and find the best-performing setup.

By carefully selecting and experimenting with different activation functions, you can potentially improve your GRU model's ability to handle volume data effectively.

Citations:
[1] https://stackoverflow.com/questions/63125782/why-shouldnt-we-use-multiple-activation-functions-in-the-same-layer
[2] https://www.linkedin.com/advice/3/how-can-you-choose-best-activation-function-gated-dsofc
[3] https://www.scaler.com/topics/deep-learning/gru-network/
[4] https://www.reddit.com/r/MachineLearning/comments/fikvm7/d_is_there_ever_a_reason_to_use_multiple/
[5] http://paper.ijcsns.org/07_book/201901/20190126.pdf
[6] https://www.datacamp.com/tutorial/tutorial-for-recurrent-neural-network
[7] http://www.jatit.org/volumes/Vol101No5/21Vol101No5.pdf
[8] https://www.ibm.com/topics/recurrent-neural-networks