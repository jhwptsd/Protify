import math

import torch
import torch.nn as nn


# Converter class - essentially just a Transformer model
class Converter(nn.Module):
    def __init__(self, max_seq_len=150, d_model=64, nhead=4, num_layers=3, dim_feedforward=128, dropout=0.1):
        super(Converter, self).__init__()

        self.d_model = d_model

        self.input_embedding = nn.Linear(4, d_model)

        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_seq_len)

        self.transformer = nn.Transformer(d_model=d_model,
                                    nhead=nhead,
                                    dim_feedforward=dim_feedforward,
                                    num_encoder_layers=num_layers, num_decoder_layers=num_layers,
                                    dropout=dropout,
                                    batch_first=True)

        self.output_linear = nn.Linear(d_model, 20)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x, src_key_padding_mask=None, sm=False):
        # x shape: (batch_size, seq_len, 4)
        # src_key_padding_mask shape: (batch_size, seq_len), True at padded positions
        # (see padding_mask_from_lengths)
        x = self.input_embedding(x)  # Now: (batch_size, seq_len, d_model)

        x = self.pos_encoder(x)

        # The same mask applies to the encoder input, the decoder input (also x) and the memory
        x = self.transformer(x, x,
                             src_key_padding_mask=src_key_padding_mask,
                             tgt_key_padding_mask=src_key_padding_mask,
                             memory_key_padding_mask=src_key_padding_mask)

        x = self.output_linear(x)  # Now: (batch_size, seq_len, 20)
        x = self.softmax(x)

        if sm:
            return x

        # Convert softmaxxed matrices into one-dimensional indeces
        with torch.no_grad():
            out = torch.argmax(x, dim=-1).cpu().tolist()
        return out

# Classic positional encoder - good stuff!
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.max_len = max_len
        self.register_buffer('pe', self.build_table(d_model, max_len))

    @staticmethod
    def build_table(d_model, max_len):
        # Shape (1, max_len, d_model): broadcasts over the batch for batch_first inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].size(1)])
        return pe.unsqueeze(0)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        if x.size(1) > self.max_len:
            raise ValueError(f"Sequence length {x.size(1)} exceeds max_seq_len {self.max_len}")
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

def padding_mask_from_lengths(lengths, max_len=None, device=None):
    # Boolean mask of shape (batch_size, max_len) that is True at padded positions
    lengths = torch.as_tensor(lengths, device=device)
    if max_len is None:
        max_len = int(lengths.max())
    return torch.arange(max_len, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)

def load_converter(path, device="cpu", max_seq_len=None):
    # Load a Converter saved with torch.save(model, path).
    # Checkpoints written before the positional-encoding fix stored the table with shape
    # (max_len, 1, d_model), which silently gave every position the same encoding.
    # Rebuild it in the right shape so old pickles at least run; they should be retrained.
    model = torch.load(path, map_location=device)
    for module in model.modules():
        if isinstance(module, PositionalEncoding) and module.pe.dim() == 3 and module.pe.size(0) != 1:
            module.pe = PositionalEncoding.build_table(model.d_model, module.pe.size(0)).to(module.pe.device)
            module.max_len = module.pe.size(1)
    if max_seq_len is not None and model.pos_encoder.max_len < max_seq_len:
        raise ValueError(f"Checkpoint supports sequences up to {model.pos_encoder.max_len}, "
                         f"but max_seq_len={max_seq_len} was requested")
    return model.to(device)
