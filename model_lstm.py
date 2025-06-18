import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Zaman serini buraya yapıştır
data = np.array([9.73495035e-11, 4.40713091e-11, 3.11573094e-11, 7.74761436e-11,
       9.32647640e-11, 2.72605358e-11, 2.40183116e-11, 1.04877114e-10,
       5.20342658e-11, 2.59628517e-11, 3.72879644e-11, 1.01202720e-10])  # buraya no2_series.values verisini kopyala

# Normalize et
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data.reshape(-1, 1))

# Veri penceresi fonksiyonu
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

# Pencere boyutu ve tensorlar
window_size = 4
X, y = create_sequences(data_scaled, window_size)
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float()
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


num_epochs = 200
X = X.view(X.shape[0], X.shape[1], 1)

for epoch in range(num_epochs):
    outputs = model(X)
    loss = criterion(outputs, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item():.4f}")

predicted = model(X).detach().numpy()
predicted_rescaled = scaler.inverse_transform(predicted)
actual_rescaled = scaler.inverse_transform(y.reshape(-1, 1))

plt.figure(figsize=(10, 5))
plt.plot(actual_rescaled, label='Gerçek NO₂')
plt.plot(predicted_rescaled, label='Tahmin Edilen NO₂')
plt.legend()
plt.title("NO₂ Tahmini - PyTorch LSTM")
plt.show()

