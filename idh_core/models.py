import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        attention_scores = self.attention(x)
        attention_weights = torch.softmax(attention_scores, dim=1)
        weighted_output = torch.sum(x * attention_weights, dim=1)
        return weighted_output, attention_weights


class LSTMWithAttention(nn.Module, BaseEstimator, ClassifierMixin):
    def __init__(self, input_size, hidden_size, output_size=2, epochs=50, lr=1e-3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.epochs = epochs
        self.lr = lr
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention = AttentionLayer(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out)
        return self.sigmoid(self.fc(attn_out))

    def fit(self, X, y):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        for _ in range(self.epochs):
            outputs = self.forward(X)
            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return self

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        outputs = self.forward(X)
        return torch.argmax(outputs, dim=1).cpu().numpy()


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        outputs, _ = self.lstm(x)
        last_hidden = self.dropout(outputs[:, -1, :])
        return self.sigmoid(self.fc(last_hidden))


class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=2, num_layers=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden=None):
        batch_size = x.shape[0]
        if hidden is None:
            hidden = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
        outputs, _ = self.gru(x, hidden)
        outputs = self.dropout(outputs[:, -1, :])
        return self.sigmoid(self.fc(outputs))


class BiLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=2, num_layers=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.randn(self.num_layers * 2, batch_size, self.hidden_size, device=x.device)
        c0 = torch.randn(self.num_layers * 2, batch_size, self.hidden_size, device=x.device)
        outputs, _ = self.lstm(x, (h0, c0))
        outputs = self.dropout(outputs[:, -1, :])
        return self.softmax(self.fc(outputs))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        positions = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=2, nhead=2, num_layers=2):
        super().__init__()
        self.project = nn.Linear(input_size, hidden_size)
        self.position = PositionalEncoding(hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.project(x)
        x = self.position(x)
        x = self.encoder(x)
        x = x[:, -1, :]
        return self.softmax(self.fc(x))


class MetaModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class TimeSeriesDiscriminator(nn.Module):
    def __init__(self, seq_len, num_features):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * num_features, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.model(x)


class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim, num_classes, seq_len, num_features, embedding_dim=8):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, seq_len * num_features),
        )

    def forward(self, z, labels):
        label_embedding = self.embedding(labels)
        x = torch.cat([z, label_embedding], dim=1)
        x = self.net(x)
        return x.view(z.size(0), self.seq_len, self.num_features)


def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    alpha = torch.rand(real_samples.size(0), 1, 1, device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    fake = torch.ones(d_interpolates.size(), requires_grad=False, device=device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

class LSTM_CNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_conv_filters, length):
        super(LSTM_CNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.conv1 = nn.Conv1d(hidden_size, num_conv_filters, kernel_size=1)
        self.fc1 = nn.Linear(num_conv_filters * length, 128)  
        self.fc2 = nn.Linear(128, output_size)  

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.permute(0, 2, 1)
        cnn_out = self.conv1(lstm_out)
        cnn_out = cnn_out.view(cnn_out.size(0), -1)
        fc1_out = torch.relu(self.fc1(cnn_out)) 
        output = self.fc2(fc1_out)
        
        return output
