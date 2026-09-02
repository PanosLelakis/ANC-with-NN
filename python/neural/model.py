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
    
    def __init__(
        self,
        fs,
        conv_channels,
        lstm_hidden,
        lstm_layers,
        delay_m,  # Frame prediction delay
        window_type,  # STFT window
        frame_ms,  # STFT frame duration
        hop_ms  # STFT frame shift
    ):
        super().__init__()

        # Store model settings
        self.fs = int(fs)
        self.conv_channels = [int(x) for x in conv_channels] # conv channels to int list
        self.lstm_hidden = int(lstm_hidden)
        self.lstm_layers = int(lstm_layers)
        self.delay_m = int(delay_m)
        # Store STFT window
        self.window_type = str(
            window_type
        ).lower()

        self.frame_ms = int(frame_ms)  # Store STFT frame duration
        self.hop_ms = int(hop_ms)  # Store STFT frame shift

        # Get STFT parameters
        params = get_stft_params(  # Read model STFT settings
            self.fs,  # Sampling rate
            frame_ms=self.frame_ms,  # Model frame duration
            hop_ms=self.hop_ms  # Model frame shift
        )

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
                #nn.GroupNorm(1, out_channels), # Group normalization (1 group = layer normalization)
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
                    #nn.GroupNorm(1, out_channels),
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

    def forward_stft(self, z):
        # Store encoder outputs
        skips = []

        # Run encoder
        for layer in self.encoder:
            z = layer(z)
            skips.append(z)

        # Run LSTM
        z = self.run_lstm(z)

        # Run decoder
        for layer_index, layer in enumerate(self.decoder):
            skip_index = len(skips) - 1 - layer_index
            z = z + skips[skip_index]
            z = layer(z)

        # Return predicted STFT channels
        return z
    
    def forward(
        self,
        x,
        return_stft=False
    ):
        # Store length of the input time-domain signal (number of samples)
        signal_length = x.shape[1]

        # Convert time-domain signal to STFT channels
        z = signal_to_stft_channels(  # Convert waveform to STFT
            x,  # Input waveform
            self.fs,  # Sampling rate
            window_type=self.window_type,  # Model window
            frame_ms=self.frame_ms,  # Model frame duration
            hop_ms=self.hop_ms  # Model frame shift
        )

        # Apply frame delay to STFT channels if specified
        z = apply_frame_delay(z, self.delay_m)

        # Run Neural Network core
        z = self.forward_stft(z)

        # Convert predicted STFT channels back to time-domain signal
        y = stft_channels_to_signal(  # Reconstruct controller waveform
            Y_channels=z,  # Predicted STFT
            length=signal_length,  # Original waveform length
            fs=self.fs,  # Sampling rate
            window_type=self.window_type,  # Model window
            frame_ms=self.frame_ms,  # Model frame duration
            hop_ms=self.hop_ms  # Model frame shift
        )

        # Return STFT when requested
        if return_stft:
            return y, z

        # Return time-domain signal
        return y

class GroupedLSTM(nn.Module):  # Two-layer grouped LSTM from the reference CRN
    def __init__(  # Build grouped recurrent module
        self,  # Current module
        feature_size=1024,  # Total recurrent feature size
        groups=2  # Number of recurrent groups
    ):
        super().__init__()  # Initialize PyTorch module

        self.feature_size = int(feature_size)  # Store total feature size
        self.groups = int(groups)  # Store group count
        self.group_size = self.feature_size // self.groups  # Compute features per group

        if self.feature_size % self.groups != 0:  # Check valid equal grouping
            raise ValueError("Grouped LSTM feature size must be divisible by groups")  # Reject invalid grouping

        self.layer_1 = nn.ModuleList([  # Build first grouped LSTM layer
            nn.LSTM(  # Build one independent LSTM group
                input_size=self.group_size,  # Group input size
                hidden_size=self.group_size,  # Group hidden size
                num_layers=1,  # One recurrent layer per group
                batch_first=True  # Use batch-time-feature format
            )
            for _ in range(self.groups)  # Create one LSTM per group
        ])

        self.layer_2 = nn.ModuleList([  # Build second grouped LSTM layer
            nn.LSTM(  # Build one independent LSTM group
                input_size=self.group_size,  # Group input size
                hidden_size=self.group_size,  # Group hidden size
                num_layers=1,  # One recurrent layer per group
                batch_first=True  # Use batch-time-feature format
            )
            for _ in range(self.groups)  # Create one LSTM per group
        ])

    def run_group_layer(  # Run one grouped recurrent layer
        self,  # Current module
        x,  # Input sequence
        layers  # LSTM groups
    ):
        input_groups = torch.chunk(  # Split features into independent groups
            x,  # Input sequence
            self.groups,  # Number of groups
            dim=-1  # Split feature dimension
        )

        outputs = []  # Store group outputs

        for input_group, lstm in zip(input_groups, layers):  # Process each group independently
            output_group, _ = lstm(input_group)  # Run one grouped LSTM
            outputs.append(output_group)  # Store group output

        return torch.cat(  # Join grouped outputs
            outputs,  # Group output tensors
            dim=-1  # Join feature dimension
        )

    def rearrange_features(  # Perform parameter-free representation rearrangement
        self,  # Current module
        x  # Grouped LSTM output
    ):
        batch = x.shape[0]  # Read batch size
        frames = x.shape[1]  # Read time-frame count

        x = x.reshape(  # Add explicit group dimension
            batch,  # Batch dimension
            frames,  # Time dimension
            self.groups,  # Group dimension
            self.group_size  # Features inside each group
        )

        x = x.transpose(  # Mix information between groups
            2,  # Group dimension
            3  # Within-group feature dimension
        ).contiguous()  # Store transposed tensor contiguously

        x = x.reshape(  # Restore original feature dimension
            batch,  # Batch dimension
            frames,  # Time dimension
            self.feature_size  # Total recurrent feature size
        )

        return x  # Return rearranged representation

    def forward(  # Run two grouped LSTM layers
        self,  # Current module
        x  # Input sequence
    ):
        x = self.run_group_layer(x, self.layer_1)  # Run first grouped LSTM
        x = self.rearrange_features(x)  # Rearrange groups between recurrent layers
        x = self.run_group_layer(x, self.layer_2)  # Run second grouped LSTM

        return x  # Return recurrent representation

class DeepANCOriginalCRN(nn.Module):  # Reference Deep ANC CRN architecture
    def __init__(  # Build original CRN
        self,  # Current module
        fs,  # Sampling rate
        delay_m,  # Prediction delay
        window_type,  # STFT window
        frame_ms,  # STFT frame duration
        hop_ms  # STFT frame shift
    ):
        super().__init__()  # Initialize PyTorch module

        self.fs = int(fs)  # Store sampling rate
        self.delay_m = int(delay_m)  # Store frame delay
        self.window_type = str(window_type).lower()  # Store STFT window
        self.frame_ms = int(frame_ms)  # Store frame duration
        self.hop_ms = int(hop_ms)  # Store frame shift

        self.conv_channels = [16, 32, 64, 128, 256]  # Store original encoder channels
        self.lstm_hidden = 1024  # Store total grouped LSTM width
        self.lstm_layers = 2  # Store recurrent layer count
        self.lstm_groups = 2  # Store recurrent group count

        params = get_stft_params(  # Read model STFT settings
            self.fs,  # Sampling rate
            frame_ms=self.frame_ms,  # Frame duration
            hop_ms=self.hop_ms  # Frame shift
        )

        self.freq_bins = int(  # Compute STFT frequency bins
            params["n_fft"] // 2 + 1  # Real-valued FFT frequency count
        )

        if self.freq_bins != 161:  # Verify reference input geometry
            raise ValueError(  # Reject incompatible STFT settings
                f"Deep ANC Original requires 161 frequency bins, got {self.freq_bins}"  # Explain mismatch
            )

        self.encoder = self.build_encoder()  # Build shared encoder

        self.grouped_lstm = GroupedLSTM(  # Build shared grouped LSTM bottleneck
            feature_size=1024,  # Use 256 channels times 4 frequencies
            groups=2  # Use reference group count
        )

        self.real_decoder = self.build_decoder()  # Build real-spectrum decoder
        self.imag_decoder = self.build_decoder()  # Build imaginary-spectrum decoder

    def build_encoder(self):  # Build five-layer encoder
        encoder = nn.ModuleList()  # Store encoder blocks
        in_channels = 2  # Start from real and imaginary STFT channels

        for out_channels in self.conv_channels:  # Build reference channel sequence
            layer = nn.Sequential(  # Build one encoder block
                nn.Conv2d(  # Downsample only the frequency dimension
                    in_channels=in_channels,  # Current feature maps
                    out_channels=out_channels,  # Next feature maps
                    kernel_size=(1, 3),  # Time-frequency kernel
                    stride=(1, 2),  # Keep time and halve frequency
                    padding=0  # Use valid convolution
                ),
                nn.BatchNorm2d(out_channels),  # Normalize convolution output
                nn.ELU()  # Apply reference activation
            )

            encoder.append(layer)  # Store encoder block
            in_channels = out_channels  # Update next input channels

        return encoder  # Return encoder blocks

    def build_decoder(self):  # Build one real or imaginary decoder
        decoder = nn.ModuleList()  # Store decoder blocks

        decoder_specs = [  # Store exact decoder geometry
            (512, 128, 0),  # 4 frequency bins to 9
            (256, 64, 0),  # 9 frequency bins to 19
            (128, 32, 0),  # 19 frequency bins to 39
            (64, 16, 1),  # 39 frequency bins to 80
            (32, 1, 0)  # 80 frequency bins to 161
        ]

        for layer_index, (in_channels, out_channels, output_padding) in enumerate(decoder_specs):  # Build every decoder stage
            is_output_layer = (  # Detect final spectrogram layer
                layer_index == len(decoder_specs) - 1  # Check last decoder stage
            )

            deconvolution = nn.ConvTranspose2d(  # Upsample frequency dimension
                in_channels=in_channels,  # Channels after skip concatenation
                out_channels=out_channels,  # Decoder output channels
                kernel_size=(1, 3),  # Time-frequency kernel
                stride=(1, 2),  # Keep time and double frequency
                padding=0,  # Use reference valid geometry
                output_padding=(0, output_padding)  # Match exact published frequency sizes
            )

            if is_output_layer:  # Keep output activation linear
                decoder.append(deconvolution)  # Store linear output layer
            else:  # Build hidden decoder block
                decoder.append(  # Store hidden decoder block
                    nn.Sequential(  # Combine decoder operations
                        deconvolution,  # Apply transposed convolution
                        nn.BatchNorm2d(out_channels),  # Normalize decoder features
                        nn.ELU()  # Apply reference activation
                    )
                )

        return decoder  # Return decoder blocks

    def run_lstm(self, x):  # Run grouped LSTM bottleneck
        batch = x.shape[0]  # Read batch size
        channels = x.shape[1]  # Read encoded channels
        frames = x.shape[2]  # Read time-frame count
        freq_bins = x.shape[3]  # Read encoded frequency count

        x = x.permute(  # Move time before flattened features
            0,  # Batch dimension
            2,  # Time dimension
            1,  # Channel dimension
            3  # Frequency dimension
        ).contiguous()  # Store tensor contiguously

        x = x.reshape(  # Flatten channel-frequency representation
            batch,  # Batch dimension
            frames,  # Time dimension
            channels * freq_bins  # 256 times 4 equals 1024 features
        )

        x = self.grouped_lstm(x)  # Model temporal dependencies

        x = x.reshape(  # Restore encoder representation
            batch,  # Batch dimension
            frames,  # Time dimension
            channels,  # 256 encoded channels
            freq_bins  # 4 encoded frequencies
        )

        x = x.permute(  # Restore convolution layout
            0,  # Batch dimension
            2,  # Channel dimension
            1,  # Time dimension
            3  # Frequency dimension
        ).contiguous()  # Store tensor contiguously

        return x  # Return recurrent features

    def run_decoder(  # Run one of the two independent decoders
        self,  # Current module
        x,  # Shared LSTM representation
        skips,  # Encoder skip connections
        decoder  # Selected decoder
    ):
        for layer_index, layer in enumerate(decoder):  # Run decoder from deep to shallow
            skip_index = len(skips) - 1 - layer_index  # Select matching encoder output
            skip = skips[skip_index]  # Read matching skip tensor

            x = torch.cat(  # Concatenate decoder and encoder features
                [x, skip],  # Decoder features and skip features
                dim=1  # Concatenate channel dimension
            )

            x = layer(x)  # Upsample current decoder representation

        return x  # Return one predicted spectral component

    def forward_stft(self, z):  # Run CRN directly on STFT channels
        if z.shape[2] != self.freq_bins:  # Check external STFT frequency count
            raise ValueError(  # Stop on wrong frame configuration
                f"Expected {self.freq_bins} frequency bins, got {z.shape[2]}"  # Explain mismatch
            )

        z = z.permute(  # Convert current B-C-F-T format to paper B-C-T-F format
            0,  # Batch dimension
            1,  # Real-imaginary channel dimension
            3,  # Time dimension
            2  # Frequency dimension
        ).contiguous()  # Store tensor contiguously

        skips = []  # Store encoder outputs

        for layer in self.encoder:  # Run all five encoder blocks
            z = layer(z)  # Downsample frequency representation
            skips.append(z)  # Store encoder output for skip connection

        z = self.run_lstm(z)  # Run shared two-layer grouped LSTM

        real = self.run_decoder(  # Predict real spectrum
            z,  # Shared recurrent representation
            skips,  # Shared encoder skip tensors
            self.real_decoder  # Real decoder parameters
        )

        imag = self.run_decoder(  # Predict imaginary spectrum
            z,  # Shared recurrent representation
            skips,  # Shared encoder skip tensors
            self.imag_decoder  # Imaginary decoder parameters
        )

        z = torch.cat(  # Join real and imaginary predictions
            [real, imag],  # Two one-channel decoder outputs
            dim=1  # Build two output channels
        )

        z = z.permute(  # Restore project B-C-F-T format
            0,  # Batch dimension
            1,  # Real-imaginary channel dimension
            3,  # Frequency dimension
            2  # Time dimension
        ).contiguous()  # Store tensor contiguously

        return z  # Return predicted complex STFT channels

    def forward(  # Run complete waveform controller
        self,  # Current module
        x,  # Input waveform
        return_stft=False  # Optional STFT return
    ):
        signal_length = x.shape[1]  # Store input waveform length

        z = signal_to_stft_channels(  # Convert reference waveform to STFT
            x,  # Reference waveform
            self.fs,  # Sampling rate
            window_type=self.window_type,  # Hamming window
            frame_ms=self.frame_ms,  # 20-ms frame
            hop_ms=self.hop_ms  # 10-ms shift
        )

        z = apply_frame_delay(  # Apply configured prediction delay
            z,  # STFT input
            self.delay_m  # Delay in frames
        )

        z = self.forward_stft(z)  # Predict real and imaginary canceling spectrum

        y = stft_channels_to_signal(  # Reconstruct canceling waveform
            Y_channels=z,  # Predicted complex spectrum
            length=signal_length,  # Original signal length
            fs=self.fs,  # Sampling rate
            window_type=self.window_type,  # Hamming window
            frame_ms=self.frame_ms,  # 20-ms frame
            hop_ms=self.hop_ms  # 10-ms shift
        )

        if return_stft:  # Return STFT when requested
            return y, z  # Return waveform and complex spectrum

        return y  # Return canceling waveform

def build_model(config):  # Build model selected by checkpoint configuration
    architecture = str(  # Read architecture identifier
        config.get(
            "architecture",
            "simple_crn"
        )
    ).lower()

    fs = int(  # Read sampling rate
        config.get(
            "target_fs",
            16000
        )
    )

    delay_m = int(  # Read frame delay
        config.get(
            "delay_m",
            0
        )
    )

    if architecture == "deep_anc_original":  # Build reference Deep ANC CRN
        window_type = str(  # Read original STFT window
            config.get(
                "window_type",
                "hamming"
            )
        ).lower()

        frame_ms = int(  # Read original frame duration
            config.get(
                "frame_ms",
                20
            )
        )

        hop_ms = int(  # Read original frame shift
            config.get(
                "hop_ms",
                10
            )
        )

        return DeepANCOriginalCRN(  # Create original Deep ANC model
            fs=fs,  # Sampling rate
            delay_m=delay_m,  # Frame prediction delay
            window_type=window_type,  # Hamming window
            frame_ms=frame_ms,  # 20-ms frame
            hop_ms=hop_ms  # 10-ms shift
        )

    if architecture == "simple_crn":  # Build existing simplified model
        conv_channels = parse_conv_channels(  # Read simplified encoder channels
            config.get(
                "conv_channels",
                "16,32"
            )
        )

        lstm_layers = int(  # Read simplified LSTM depth
            config.get(
                "lstm_layers",
                1
            )
        )

        lstm_hidden = int(  # Read simplified LSTM width
            config.get(
                "lstm_hidden",
                128
            )
        )

        window_type = str(  # Read saved simplified-model window
            config.get(
                "window_type",
                "rectangular"
            )
        ).lower()

        frame_ms = int(  # Preserve old 32-ms checkpoints
            config.get(
                "frame_ms",
                32
            )
        )

        hop_ms = int(  # Preserve old 10-ms shift
            config.get(
                "hop_ms",
                10
            )
        )

        return SimpleCRN(  # Create existing simplified model
            fs=fs,  # Sampling rate
            conv_channels=conv_channels,  # Encoder channels
            lstm_hidden=lstm_hidden,  # LSTM width
            lstm_layers=lstm_layers,  # LSTM depth
            delay_m=delay_m,  # Frame delay
            window_type=window_type,  # Saved window
            frame_ms=frame_ms,  # Saved frame duration
            hop_ms=hop_ms  # Saved frame shift
        )

    raise ValueError(  # Reject unknown model names
        f"Unknown Neural Network architecture: {architecture}"  # Show invalid identifier
    )