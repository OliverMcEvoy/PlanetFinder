import torch

class LinearDropout(torch.nn.Module):
    def __init__(self, input_size, output_size, p=0.3):
        super().__init__()
        self.norm = torch.nn.BatchNorm1d(input_size)
        self.linear = torch.nn.Linear(input_size, output_size)
        if p < 1:
            self.dropout = torch.nn.Dropout(p)
        else:
            self.dropout = torch.nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x

class RNN(torch.nn.Module):
    def __init__(self, hidden_size=64, output_size=8, num_layers=1, dropout=0):
        super().__init__()
        self.rnn = torch.nn.RNN(1, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(hidden_size, output_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out, _ = self.rnn(x)
        logits = self.linear(out[:,-1])
        periods = self.relu(logits)
        return periods
    
class LSTM(torch.nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=8, num_layers=2, dropout=0.5):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, output_size)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        logits = self.linear(out[:,-1])
        periods = logits
        return periods
    
class Encoder(torch.nn.Module):
    def __init__(self, hidden_size=64, num_layers=1, dropout=0):
        super().__init__()
        self.lstm = torch.nn.LSTM(1, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        #layer = torch.nn.TransformerEncoderLayer(hidden_size, 4, dim_feedforward=1024, dropout=0.1, batch_first=True)
        #encoder = torch.nn.TransformerEncoder(layer, 2)
        
    def forward(self, x):
        x, states = self.lstm(x)
        x = x[:,-1]
        x = x.unsqueeze(1)
        return x, states
    
class Decoder(torch.nn.Module):
    def __init__(self, hidden_size=128, num_layers=1, dropout=0, output_size=8):
        super().__init__()
        self.output_size = output_size
        self.lstm = torch.nn.LSTM(1, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x, states):
        x = self.linear(x)

        for i in range(self.output_size):
            x, states = self.lstm(x, states)
            x = self.linear(x)
            x = self.relu(x)
            
            if i == 0:
                y = x.squeeze(1)
            else:
                y = torch.cat((y, x.squeeze(1)), dim=1)

        return y
    
class SequenceModel(torch.nn.Module):
    def __init__(self, hidden_size=64, output_size=8):
        super().__init__()
        self.encoder = Encoder(hidden_size=hidden_size)
        self.decoder = Decoder(hidden_size=hidden_size, output_size=output_size)

    def forward(self, x):
        x, states = self.encoder(x)
        x = self.decoder(x, states)
        return x
    
class AttentionLayer(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = torch.nn.Linear(hidden_size, 1)
        self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(self, x):
        weights = self.softmax(self.attention(x))
        return torch.sum(weights * x, dim=1)

class TransitModel(torch.nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=10, num_layers=4):
        super().__init__()
        
        # Bidirectional LSTM
        self.lstm = torch.nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers=num_layers, 
            dropout=0.1, 
            batch_first=True
        )
        
        # Double hidden size due to bidirectional
        self.attention = AttentionLayer(hidden_size)
        
        # Deep fully connected layers with residual connections
        self.fc_layers = torch.nn.ModuleList([
            LinearDropout(hidden_size, hidden_size),
            torch.nn.ReLU(),
            LinearDropout(hidden_size, hidden_size // 2),
            torch.nn.ReLU(),
            LinearDropout(hidden_size // 2, hidden_size // 2),
            torch.nn.ReLU(),
        ])
        
        self.layer_norm = torch.nn.LayerNorm(hidden_size // 2)
        self.final_fc = torch.nn.Linear(hidden_size // 2, output_size)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        # LSTM layer
        lstm_out, _ = self.lstm(x)
        
        # Attention
        context = self.attention(lstm_out)
        
        # Process through FC layers with residual connections
        x = context
        residual = x
        
        for i, layer in enumerate(self.fc_layers):
            x = layer(x)
            if i % 2 == 1 and x.shape == residual.shape:  # Apply residual after every 2 layers if shapes match
                x = x + residual
                residual = x
        
        x = self.layer_norm(x)
        logits = self.final_fc(x)
        periods = self.relu(logits)

        return periods