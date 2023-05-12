import torch
import torch.nn as nn

class BLSTM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model_type = "BLSTM"

        self.seq_len = self.args.blstm_input_length
        dropout = self.args.blstm_dropout
        hidden_size = self.args.blstm_hidden_size
        num_layers = self.args.blstm_num_layers
        output_dim = self.args.output_dim

        self.padding_value = 0
        input_size = self.args.whisper_dim

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.mlp = nn.Linear(hidden_size * 2, output_dim)

    def encode(self, whisper):
        """
        Args:
            whisper: Tensor, shape [real_whisper_len]
        Returns:
            input: Tensor, shape [seq_len]
            mask: Tensor, shape [seq_len]
        """
        input = torch.zeros(
            (self.seq_len, self.args.whisper_dim),
            device=self.args.device,
            dtype=torch.float,
        )

        mask = torch.ones(self.seq_len, device=self.args.device, dtype=torch.long)

        whisper = whisper[: self.seq_len]
        input[: len(whisper)] = torch.as_tensor(whisper, device=self.args.device)
        padding_len = self.seq_len - len(whisper)

        if padding_len > 0:
            input[-padding_len:] = self.padding_value
            mask[-padding_len:] = 0

        return input, mask

    def forward(self, idxs, dataset):
        """
        Args:
            idxs: Tensor, shape [batch_size]
            dataset: DatasetLoader
        Returns:
            output: Tensor, shape [batch_size, seq_len, output_dim]
            masks: Tensor, shape [batch_size, seq_len, 1]
        """
        inputs = [self.encode(dataset.whisper[idx.item()]) for idx in idxs]

        # (bs, seq_len, whisper_dim)
        input = torch.stack([i[0] for i in inputs])
        # (bs, seq_len, 1)
        masks = torch.stack([i[1] for i in inputs])[:, :, None]

        # (bs, seq_len, whisper_dim)
        output, _ = self.lstm(input)
        # (bs, seq_len, output_dim)
        output = self.mlp(output)
        return output, masks
