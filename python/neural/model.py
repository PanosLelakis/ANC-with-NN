import torch
import torch.nn as nn

from neural.features import (
    get_stft_params,
    signal_to_stft_channels,
    stft_channels_to_signal,
    apply_frame_delay
)

def parse_conv_channels(text):
    # Parse conv channels

    # Initialize channels list
    channels = []

    # Split text by commas and convert to integers
    for item in str(text).split(","):
        item = item.strip()

        if item:
            channels.append(int(item))

    # Return conv channels
    return channels

class SimpleCRN(nn.Module): # nn.Module means PyTorch model (fck my life)
    # Build encoder (2 layers)
    # Build lstm (1 layer)
    # Build decoder
    # Apply skip connections
    # Convert x(t) to STFT channels
    # Predict Y real-imag channels
    # Convert predicted Y to y(t)
    
    def __init__(self, fs, conv_channels, lstm_hidden, lstm_layers, delay_m):
        super().__init__()

        # Store model settings
        self.fs = int(fs)
        self.conv_channels = [int(x) for x in conv_channels] # conv channels to int list
        self.lstm_hidden = int(lstm_hidden)
        self.lstm_layers = int(lstm_layers)
        self.delay_m = int(delay_m)

        # Get STFT parameters
        params = get_stft_params(self.fs)

        # Store frequency bins
        self.freq_bins = int(params["n_fft"] // 2 + 1)

        # Build encoder
        self.encoder = self.build_encoder()
        
        # Build LSTM
        self.lstm, self.lstm_projection = self.build_lstm(
            encoded_channels = self.conv_channels[-1],
            encoded_freq_bins = self.freq_bins
        )
        
        # Build decoder
        self.decoder = self.build_decoder()
    
    def build_encoder(self):
        # Build encoder (list of conv layers)
        encoder = nn.ModuleList()

        in_channels = 2 # channel 0 = real STFT, channel 1 = imag STFT

        for out_channels in self.conv_channels:
            # Whole encoder block into one layer (conv2d + batchnorm + elu)
            layer = nn.Sequential( # nn.Sequential to combine multiple layers into one
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1), # Convolutional layer
                nn.BatchNorm2d(out_channels), # Batch normalization
                nn.ELU() # Activation function (Exponential Linear Unit)
            )

            # Add layer to encoder
            encoder.append(layer)
            
            # Next layer input channels = current layer output channels
            in_channels = out_channels

        # Return encoder
        return encoder

    def build_lstm(self, encoded_channels, encoded_freq_bins):
        # LSTM input size = conv output channels * freq bins
        lstm_input_size = encoded_channels * encoded_freq_bins

        self.lstm_input_size = lstm_input_size # Store for later use

        # Build LSTM layer
        lstm = nn.LSTM(
            input_size = lstm_input_size,
            hidden_size = self.lstm_hidden,
            num_layers = self.lstm_layers,
            batch_first = True # [batch, frames, features]
        )

        # LSTM projection layer to map hidden state back to input size
        lstm_projection = nn.Linear( # Linear layer
            self.lstm_hidden,
            lstm_input_size
        )

        # Return LSTM and projection layer
        return lstm, lstm_projection

    def build_decoder(self):
        # List of decoder layers
        decoder = nn.ModuleList()

        # Decoder output channels = reversed conv channels + 2 (for real and imag)
        decoder_out_channels = list(reversed(self.conv_channels[:-1])) + [2]
        
        # Decoder starts from last encoder channel
        decoder_in_channels = self.conv_channels[-1]

        # Build decoder (list of convtranspose layers)
        for out_channels in decoder_out_channels:
            # Whole decoder block into one layer (convtranspose2d)
            if out_channels == 2: # Last layer (output layer) does not have batchnorm and activation
                layer = nn.ConvTranspose2d(
                    decoder_in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            else:
                layer = nn.Sequential(
                    nn.ConvTranspose2d(
                        decoder_in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ELU()
                )
                
            # Add layer to decoder
            decoder.append(layer)
            
            # Next layer input channels = current layer output channels
            decoder_in_channels = out_channels

        # Return decoder
        return decoder

    def run_lstm(self, x):
        # LSTM expects input shape [batch, frames, features]

        batch = x.shape[0] # batch size
        channels = x.shape[1] # channels (real + imag)
        freq_bins = x.shape[2] # frequency bins
        frames = x.shape[3] # time frames

        # Permute tensor to [batch, frames, channels, freq_bins]
        x = x.permute(0, 3, 1, 2).contiguous()

        # Put channels and freq_bins into one dimension for LSTM input
        x = x.reshape(batch, frames, channels * freq_bins)

        # Run LSTM
        x, _ = self.lstm(x) # _ is the hidden state (not used)

        # Project LSTM output back to original input size
        x = self.lstm_projection(x)

        # Reshape back to original dimensions
        x = x.reshape(batch, frames, channels, freq_bins)
        
        # Permute back to original shape [batch, channels, freq_bins, frames]
        x = x.permute(0, 2, 3, 1).contiguous()

        # Return tensor
        return x
    
    def forward(self, x):
        # Store length of the input time-domain signal (number of samples)
        signal_length = x.shape[1]

        # Convert time-domain signal to STFT channels
        z = signal_to_stft_channels(x, self.fs)

        # Apply frame delay to STFT channels if specified
        z = apply_frame_delay(z, self.delay_m)

        # Create skip connections list
        skips = []

        # Iterate through encoder layers and store skip connections
        for layer in self.encoder:
            z = layer(z) # Apply encoder layer
            skips.append(z) # Store skip connection

        # Run LSTM on the encoded features
        z = self.run_lstm(z)

        # Iterate through decoder layers and apply skip connections in reverse order
        for layer_index, layer in enumerate(self.decoder):
            skip_index = len(skips) - 1 - layer_index
            z = z + skips[skip_index]
            z = layer(z) # Apply decoder layer

        # Convert predicted STFT channels back to time-domain signal
        y = stft_channels_to_signal(
            Y_channels=z,
            length=signal_length,
            fs=self.fs
        )

        # Return time-domain signal (NN output)
        return y

class FullCRN(nn.Module):
    # Same as SimpleCRN but with more layers
    # 5 conv layers
    # 2 lstm layers
    def __init__(self):
        super().__init__()
        raise NotImplementedError("Use at ur own risk ...")

def build_model(config):
    # Parse model settings
    fs = int(config.get("target_fs", 16000))
    conv_layers = int(config.get("conv_layers", 2))
    conv_channels = parse_conv_channels(config.get("conv_channels", "16,32"))
    lstm_layers = int(config.get("lstm_layers", 1))
    lstm_hidden = int(config.get("lstm_hidden", 128))
    delay_m = int(config.get("delay_m", 0))

    # Create model
    model = SimpleCRN(
        fs=fs,
        conv_channels=conv_channels,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        delay_m=delay_m
    )

    # Return model
    return model