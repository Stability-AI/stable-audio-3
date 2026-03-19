import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio.prototype.transforms import ChromaScale
from torchaudio.transforms import MelSpectrogram
from . import hpss

class FeatureExtractor(nn.Module):
    """
    Unified audio feature extractor that computes spectral, rhythm, and basic audio features.
    Integrates with TorchAudio for optimized audio feature extraction.
    
    Parameters
    ----------
    sample_rate : int
        Sample rate of the audio signal
    n_fft : int
        FFT window size
    hop_length : int
        Number of samples between successive frames
    win_length : int or None
        Window length for STFT. If None, defaults to n_fft.
    n_mels : int
        Number of mel bands
    center : bool
        If True, signals are padded
    window : str
        Window type for analysis
    frames_per_chunk : int
        Number of frames per chunk for chunking operations
    n_chroma : int
        Number of chroma bins
    n_contrast_bands : int
        Number of frequency bands for spectral contrast
    fmin : float
        Minimum frequency for spectral contrast
    roll_percent : float
        Roll-off percentage
    poly_order : int
        Order of the polynomial for poly features (0 energy, 1 tilt, 2 curvature)
    tempo_min : float
        Minimum permissible tempo value in BPM
    tempo_max : float
        Maximum permissible tempo value in BPM
    prior_mean : float or None
        Mean of the normal distribution prior on tempo (in BPM)
    prior_std : float or None
        Standard deviation of the normal distribution prior on tempo
    """
    def __init__(
        self,
        sample_rate=22050,
        n_fft=2048,
        hop_length=512,
        win_length=None,
        n_mels=128,
        center=True,
        window="hann",
        n_chroma=12,
        n_contrast_bands=6,
        fmin=200.0,
        roll_percent=0.85,
        poly_order=2,
        tempo_min=30,
        tempo_max=300,
        prior_mean=120,
        prior_std=1.0,
        use_hpss=False,
        device=None
    ):
        super(FeatureExtractor, self).__init__()
        self.all_features = ['spectrogram',
                              'spectral_centroid', 
                              'spectral_bandwidth', 
                              'spectral_contrast', 
                              'spectral_rolloff',
                              'spectral_flatness', 
                              'rms', 
                              'poly_features', 
                              'mel_spectrogram', 
                              'chroma',
                              'onset_strength', 
                              'plp', 
                              'fourier_tempogram', 
                              'loudness'
                            ]
        
        n_freqs = n_fft // 2 + 1
        # Basic parameters
        self.sr = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft
        self.n_mels = n_mels
        self.center = center
        self.window_type = window
        self.n_chroma = n_chroma
        self.hpss = hpss if use_hpss else None
        
        # Spectral feature parameters
        self.n_contrast_bands = n_contrast_bands
        self.fmin = fmin
        self.roll_percent = roll_percent
        self.poly_order = poly_order
        
        # Rhythm feature parameters
        self.tempo_min = tempo_min
        self.tempo_max = tempo_max
        self.prior_mean = prior_mean
        self.prior_std = prior_std

        self.device = device
        
        # Create window for STFT
        win = self.get_window(window, self.win_length, fftbins=True)
        self.register_buffer("window", win)
        self.register_buffer("tempogram_window", win)
        
        # Register frequencies
        freqs = torch.fft.rfftfreq(n_fft, 1.0/self.sr)
        self.register_buffer("freqs", freqs)

        # Register tempo frequencies
        tempo_frequencies = self._fourier_tempo_frequencies()
        self.register_buffer("tempo_frequencies", tempo_frequencies)
        
        # Initialize TorchAudio transforms - we'll use parts of these for processing
        # For mel spectrogram: use the filter bank with our spectrogram
        self.mel_transform = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=self.win_length,
            hop_length=hop_length,
            center=center,
            n_mels=n_mels,
            f_min=0.0,  # Minimum frequency (Hz)
            f_max=sample_rate/2,  # Maximum frequency (Hz)
            power=2.0  # Power spectrogram
        )
        
        # For chroma features
        self.chroma_transform = ChromaScale(
            sample_rate=sample_rate,
            n_freqs=n_freqs,
            n_chroma=12,
            tuning=0.0,
            ctroct=5.0,
            octwidth=2.0,
            norm=2,
            base_c=True
        )

        # Precompute frequency bands for spectral contrast
        self._precompute_contrast_bands()
    
    def get_window(self, window_type, win_length, fftbins=True, device=None):
        """Create a window"""
        if isinstance(window_type, str):
            import numpy as np
            from scipy import signal
            
            window = signal.get_window(window_type, win_length, fftbins=fftbins)
            tensor = torch.from_numpy(window).float()
            if device is not None:
                tensor = tensor.to(device)
            return tensor
        elif isinstance(window_type, torch.Tensor):
            return window_type
        else:
            raise ValueError(f"Unsupported window type: {window_type}")

    def _precompute_contrast_bands(self):
        """Precompute frequency bands for spectral contrast."""
        # Implement your contrast bands calculation here
        pass

    def _fourier_tempo_frequencies(self):
        """Calculate Fourier tempo frequencies."""
        n_frames = 100  # A placeholder value, adjust as needed
        return torch.fft.rfftfreq(n_frames, self.hop_length / self.sr)  # Remove device parameter

    def stft(self, audio):
        """
        Compute the Short-Time Fourier Transform, ensuring proper handling of padding
        and gradient flow.
        """
        # Handle different input dimensions
        orig_dim = audio.dim()
        
        if orig_dim == 1:
            # Single audio signal: [samples] -> [1, samples]
            audio = audio.unsqueeze(0)  # Add batch dimension
        elif orig_dim == 3:
            # If input is [batch, channels, samples], reshape to [batch*channels, samples]
            batch_size, channels, samples = audio.shape
            audio = audio.reshape(-1, samples)
        elif orig_dim != 2:
            raise ValueError(f"Expected audio with 1, 2 or 3 dimensions, got {orig_dim}")
        
        if self.center:
            padding = (self.n_fft) // 2
            audio = F.pad(audio, (padding, padding), mode='reflect')
        
        # Compute STFT
        stft_matrix = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=True,
            normalized=False,
            onesided=True,
        )
        
        # Reshape back to original dimensions if input was 3D
        if orig_dim == 3:
            # stft_matrix shape: [batch*channels, freq_bins, time_frames]
            freq_bins, time_frames = stft_matrix.shape[1], stft_matrix.shape[2]
            stft_matrix = stft_matrix.reshape(batch_size, channels, freq_bins, time_frames)
        
        return stft_matrix
    
    def spectrogram(self, audio):
        """Compute magnitude spectrogram from audio."""
        stft = self.stft(audio)
        return torch.abs(stft.real).squeeze(1)

    def mel_spectrogram(self, audio=None, spectrogram=None):
        """
        Compute mel spectrogram from audio or spectrogram using TorchAudio.
        Prioritizes using pre-computed spectrogram for efficiency.
        """
        if spectrogram is None and audio is None:
            raise ValueError("Either audio or spectrogram must be provided")
            
        # Compute spectrogram if not provided
        if spectrogram is None:
            spectrogram = self.spectrogram(audio)
        
        # Remember original dimensions
        orig_dim = spectrogram.dim()
        
        # Ensure we have 3D tensor [batch, freq, time]
        if orig_dim == 2:  # [freq, time]
            spectrogram = spectrogram.unsqueeze(0)  # Add batch dimension
            
        # Apply mel filter bank (using power spectrogram)
        # TorchAudio mel_scale expects power spectrogram as input
        mel_spec = self.mel_transform.mel_scale(spectrogram.pow(2))
        
        # Restore original dimensions if needed
        if orig_dim == 2:
            mel_spec = mel_spec.squeeze(0)
            
        return mel_spec
    
    def chroma(self, audio=None, spectrogram=None):
        """
        Compute chroma features from spectrogram using TorchAudio's ChromaScale transform,
        with post-processing to match librosa's smooth, normalized output.
        """
        if spectrogram is None and audio is None:
            raise ValueError("Either audio or spectrogram must be provided")
            
        # Compute spectrogram if not provided
        if spectrogram is None:
            spectrogram = self.spectrogram(audio)
        
        # Remember original dimensions
        orig_dim = spectrogram.dim()

        if orig_dim == 2:  # [freq, time]
            spectrogram = spectrogram.unsqueeze(0)  # Add batch dimension
        
        # Use TorchAudio's chroma transform
        chroma_features = self.chroma_transform(spectrogram)

        # Use a small convolution to smooth across time
        kernel_size = 3
        padding = kernel_size // 2

        if chroma_features.dim() == 4 and chroma_features.shape[1] == 1:
            chroma_features = chroma_features.squeeze(1)
        batch_size, n_chroma, n_time = chroma_features.shape
        chroma_smooth = chroma_features.reshape(batch_size * n_chroma, 1, n_time)
        
        # Apply 1D convolution for temporal smoothing
        smooth_kernel = torch.ones(1, 1, kernel_size, device=chroma_features.device) / kernel_size
        chroma_smooth = F.conv1d(chroma_smooth, smooth_kernel, padding=padding)
        
        # Reshape back
        chroma_smooth = chroma_smooth.reshape(batch_size, n_chroma, n_time)
        
        # Normalize to [0, 1] range (librosa-style normalization)
        chroma_smooth = F.relu(chroma_smooth)
        
        # Normalize each frame to sum to 1 (like librosa's default)
        chroma_sum = torch.sum(chroma_smooth, dim=1, keepdim=True)
        chroma_sum = torch.clamp(chroma_sum, min=1e-8)  # Avoid division by zero
        chroma_normalized = chroma_smooth / chroma_sum
        
        
        # Apply small amount of regularization to match librosa's smoothness
        epsilon = 1e-8
        chroma_final = chroma_normalized + epsilon
        
        # Re-normalize after adding epsilon
        chroma_sum = torch.sum(chroma_final, dim=1, keepdim=True)
        chroma_final = chroma_final / chroma_sum
        
        # Restore original dimensions if needed
        if orig_dim == 2:
            chroma_final = chroma_final.squeeze(0)
            
        return chroma_final
    
    def onset_strength(self, audio=None, mel_spectrogram=None, lag=1, max_size=1, 
                          detrend=False, aggregate=torch.mean):
        """Compute onset strength envelope from audio or mel spectrogram.
        
        Parameters:
        -----------
        audio : torch.Tensor, optional
            Audio time series [batch, 1, time]
        mel_spectrogram : torch.Tensor, optional
            Pre-computed mel spectrogram [batch, n_mels, time]
        lag : int
            Time lag for computing differences
        max_size : int
            Size of the local max filter along frequency axis
        detrend : bool
            Whether to filter the onset strength to remove the DC component
        aggregate : callable
            Function to aggregate across frequency bands (default: torch.mean)
            
        Returns:
        --------
        onset_env : torch.Tensor
            Onset strength envelope [batch, 1, time]
        """
        if mel_spectrogram is None:
            if audio is None:
                raise ValueError("Either audio or mel_spectrogram must be provided")
            mel_spectrogram = self.mel_spectrogram(audio=audio)
        
        # Convert to dB scale
        mel_db = torch.log10(torch.clamp(mel_spectrogram, min=1e-10)) * 10.0
        
        # Create reference spectrum using max filtering
        if max_size > 1:
            # Implement max filtering along frequency axis
            # This is similar to scipy.ndimage.maximum_filter1d in librosa
            ref = torch.nn.functional.max_pool1d(
                mel_db.transpose(1, 2),  # [batch, time, n_mels]
                kernel_size=max_size,
                stride=1,
                padding=max_size // 2
            ).transpose(1, 2)  # [batch, n_mels, time]
        else:
            ref = mel_db
            
        # Compute difference with specified lag
        if lag > 0:
            onset_env = mel_db[:, :, lag:] - ref[:, :, :-lag]
            # Pad to original length
            padding = torch.zeros_like(onset_env[:, :, :lag])
            onset_env = torch.cat([padding, onset_env], dim=2)
        else:
            onset_env = torch.zeros_like(mel_db)
        
        # Keep only positive differences (increasing amplitude)
        onset_env = torch.relu(onset_env)
        
        # Aggregate across frequency bands
        if aggregate is torch.mean:
            onset_env = onset_env.mean(dim=1, keepdim=True)
        elif aggregate is torch.median:
            onset_env = onset_env.median(dim=1, keepdim=True)[0]  # [0] to get values, not indices
        elif aggregate is torch.max:
            onset_env = onset_env.max(dim=1, keepdim=True)[0]
        elif callable(aggregate):
            onset_env = aggregate(onset_env, dim=1, keepdim=True)
        
        # Apply detrending if requested (remove DC component)
        if detrend:
            # Approximate first-order IIR filter y[n] = x[n] - 0.99*x[n-1]
            # This is similar to scipy.signal.lfilter([1.0, -1.0], [1.0, -0.99], ...)
            filtered = torch.zeros_like(onset_env)
            filtered[:, :, 0] = onset_env[:, :, 0]
            for i in range(1, onset_env.shape[2]):
                filtered[:, :, i] = onset_env[:, :, i] - 0.99 * onset_env[:, :, i-1]
            onset_env = filtered
        
        return onset_env
    
    def get_window(self, window_type, win_length, fftbins=True):
        """Create a window"""
        if isinstance(window_type, str):
            # PyTorch doesn't have as many window types as scipy, so use numpy for consistency
            import numpy as np
            from scipy import signal
            
            # Get window from scipy
            window = signal.get_window(window_type, win_length, fftbins=fftbins)
            # Convert to torch tensor
            return torch.from_numpy(window).float().to(self.device)
        elif isinstance(window_type, torch.Tensor):
            return window_type
        else:
            raise ValueError(f"Unsupported window type: {window_type}")
    
    def frame(self, signal, frame_length, hop_length=1):
        """
        Memory-efficient and differentiable implementation for slicing signal into overlapping frames.
        """
        batch_size, signal_length = signal.shape
        
        # Calculate number of frames
        n_frames = 1 + (signal_length - frame_length) // hop_length
        
        # Define maximum frames to process at once
        max_frames_per_batch = min(10000, n_frames)
        
        # Preallocate output tensor
        result = torch.zeros(batch_size, n_frames, frame_length, device=signal.device)
        
        for start_frame in range(0, n_frames, max_frames_per_batch):
            end_frame = min(start_frame + max_frames_per_batch, n_frames)
            
            # Create frame indices for this batch - use unsqueeze for more explicit dimensions
            # Avoid direct indexing when possible
            indices = torch.arange(0, frame_length, device=signal.device).unsqueeze(0) + \
                    (torch.arange(start_frame, end_frame, device=signal.device) * hop_length).unsqueeze(1)
            
            # Use differentiable clamping instead of direct indexing
            indices = torch.clamp(indices, max=signal_length-1)
            
            # Extract frames using gather operation which is differentiable
            for b in range(batch_size):
                # Use .gather which maintains gradient flow
                signal_expanded = signal[b].unsqueeze(0).expand(end_frame - start_frame, -1)
                frames = torch.gather(signal_expanded, 1, indices)
                result[b, start_frame:end_frame] = frames
                
        return result
    
    # Rhythm Feature Methods
    def normalize(self, x, norm=2, axis=0, threshold=None, fill=None):
        """Normalize a tensor along a particular axis with differentiable operations."""
        # Handle different norm types
        if norm not in [1, 2, float('inf'), -float('inf')]:
            raise ValueError(f"Unsupported norm: {norm}")
        
        # Clone the input to avoid in-place modifications
        x_normalized = x.clone()
        
        # Calculate the norm along the specified axis
        if norm == float('inf'):
            # L-infinity norm (maximum absolute value)
            values = torch.amax(torch.abs(x), dim=axis, keepdim=True)
        elif norm == -float('inf'):
            # L-negative infinity norm (minimum absolute value)
            values = torch.amin(torch.abs(x), dim=axis, keepdim=True)
        elif norm == 1:
            # L1 norm (sum of absolute values)
            values = torch.sum(torch.abs(x), dim=axis, keepdim=True)
        else:  # norm == 2
            # L2 norm (Euclidean norm)
            values = torch.sqrt(torch.sum(torch.abs(x)**2, dim=axis, keepdim=True))
        
        # Set minimum threshold for normalization
        if threshold is None:
            threshold = torch.finfo(x.dtype).eps
        
        # Find indices where norm is below threshold - use soft masking
        # Create a soft mask that's 1 where values >= threshold and 0 otherwise
        # Using sigmoid with high temperature for an approximation of step function
        mask = torch.sigmoid((values - threshold) * 1000)
        
        # Compute inverse values with stability factor
        inverse_values = 1.0 / (values + 1e-7)
        
        # Apply normalization with soft masking
        x_normalized = x_normalized * (inverse_values * mask)
        
        # Handle fill value for small norms if requested
        if fill is not None:
            # Soft blend between normalized values and fill
            x_normalized = x_normalized * mask + fill * (1 - mask)
        
        return x_normalized


    def fourier_tempogram(self, onset_envelope):
        """Compute Fourier tempogram from onset strength envelope.
        
        Args:
            onset_envelope: Tensor of shape [batch_size, n_frames] or [extra_batch, batch_size, n_frames]
                            or [batch_size, 1, n_frames]
        
        Returns:
            Tensor of shape [batch_size, n_freq, n_frames] or [extra_batch, batch_size, n_freq, n_frames]
        """
        # Handle singleton dimension if present (e.g., [batch, 1, n_frames])
        if onset_envelope.dim() >= 3 and onset_envelope.shape[-2] == 1:
            onset_envelope = onset_envelope.squeeze(-2)
        
        # Handle extra batch dimension if present
        input_dims = onset_envelope.dim()
        if input_dims == 3:
            extra_batch, batch_size, n_frames = onset_envelope.shape
            # Reshape to [extra_batch*batch_size, n_frames] for processing
            onset_envelope = onset_envelope.reshape(-1, n_frames)
        else:
            extra_batch = None
            batch_size, n_frames = onset_envelope.shape
        
        # Center the windows
        if self.center:
            pad = self.win_length // 2
            onset_padded = F.pad(onset_envelope, (pad, pad), mode='constant')
        else:
            onset_padded = onset_envelope
        
        # Frame the onset envelope
        onset_frames = self.frame(onset_padded, self.win_length, hop_length=1)
        
        # Truncate to original length if centered
        if self.center:
            onset_frames = onset_frames[:, :n_frames, :]
        
        # Apply window
        window = self.tempogram_window.unsqueeze(0).unsqueeze(0)
        windowed_frames = onset_frames * window
        
        # Compute STFT (Fourier tempogram)
        ftgram = torch.fft.rfft(windowed_frames, dim=2)
        
        # Transpose to get [batch_size, n_freq, time]
        ftgram = ftgram.transpose(1, 2)
        
        # Restore extra batch dimension if it was present
        if extra_batch is not None:
            n_freq = ftgram.shape[1]
            ftgram = ftgram.reshape(extra_batch, batch_size, n_freq, n_frames)
        
        return ftgram

    def _fourier_tempo_frequencies(self):
        """Compute tempo frequencies for the Fourier tempogram.
        """
        frequencies = torch.fft.rfftfreq(self.win_length, d=1.0/self.sr, device=self.device)
        tempo = 60 * frequencies / self.hop_length
        return tempo

    def plp(self, onset_envelope):
        """Compute Predominant Local Pulse (PLP) feature with differentiable operations."""
        # Handle extra batch dimension
        input_dims = onset_envelope.dim()
        original_shape = onset_envelope.shape
        
        if input_dims == 3:
            extra_batch, batch_size, n_frames = original_shape
            # Reshape to [extra_batch*batch_size, n_frames]
            onset_envelope = onset_envelope.reshape(-1, n_frames)
        else:
            extra_batch = None
        
        # Get the fourier tempogram
        ftgram = self.fourier_tempogram(onset_envelope)
        
        # Pin to the feasible tempo range - use soft masking with sigmoid
        if hasattr(self, 'tempo_frequencies'):
            if self.tempo_min is not None:
                soft_mask = torch.sigmoid((self.tempo_frequencies - self.tempo_min) * 100)
                ftgram = ftgram * soft_mask.unsqueeze(0).unsqueeze(-1)
            
            if self.tempo_max is not None:
                soft_mask = torch.sigmoid((self.tempo_max - self.tempo_frequencies) * 100)
                ftgram = ftgram * soft_mask.unsqueeze(0).unsqueeze(-1)
        
        # Apply prior and keep only the peaks
        ftmag = torch.log1p(1e6 * torch.abs(ftgram))
        
        # Apply prior if specified
        if self.prior_mean is not None and self.prior_std is not None and hasattr(self, 'tempo_frequencies'):
            prior = -0.5 * ((self.tempo_frequencies - self.prior_mean) / self.prior_std) ** 2
            ftmag = ftmag + prior.unsqueeze(0).unsqueeze(-1)
        
        # Get peak values along frequency axis - use softmax to get a differentiable approximation
        # This is an approximation of the max operation
        temperature = 50.0  # High temperature for sharper peaks
        weights = torch.softmax(ftmag * temperature, dim=1)
        peak_values = torch.sum(ftmag * weights, dim=1, keepdim=True)
        
        # Create a soft mask instead of hard thresholding
        mask = torch.sigmoid((ftmag - peak_values) * 100)
        ftgram = ftgram * mask
        
        # Normalize to keep only phase information - with stability factor
        ftgram_max = torch.sqrt(torch.sum(torch.abs(ftgram)**2, dim=1, keepdim=True))
        ftgram = ftgram / (1e-8 + ftgram_max)
        
        # Invert the Fourier tempogram to get the pulse
        ftgram_t = ftgram.transpose(1, 2)
        pulse_frames = torch.fft.irfft(ftgram_t, n=self.win_length, dim=2)
        
        # Extract the center of each frame - use differentiable indexing
        center_idx = self.win_length // 2
        pulse = pulse_frames[:, :, center_idx]
        
        # Use ReLU for positive part rather than manually zeroing
        pulse = F.relu(pulse)
        
        # Normalize the pulse - with stability factor
        max_pulse = torch.max(pulse, dim=1, keepdim=True)[0]
        pulse = pulse / (max_pulse + 1e-8)
        
        # Restore extra batch dimension if present
        if extra_batch is not None:
            pulse = pulse.reshape(extra_batch, batch_size, -1)
        
        return pulse

    # Spectral Feature Methods
    def _precompute_contrast_bands(self):
        """Precompute frequency bands for spectral contrast."""
        octa = torch.zeros(self.n_contrast_bands + 2)
        octa[1:] = self.fmin * (2.0 ** torch.arange(0, self.n_contrast_bands + 1))
        
        if torch.any(octa[:-1] >= 0.5 * self.sr):
            raise ValueError("Frequency band exceeds Nyquist. Reduce either fmin or n_bands.")
        
        bands = []
        for k in range(self.n_contrast_bands + 1):
            f_low, f_high = octa[k], octa[k+1]
            band = (self.freqs >= f_low) & (self.freqs <= f_high)
            bands.append(band)
            
        self.register_buffer("contrast_bands", torch.stack(bands))

    # Example fix for a spectral feature method that might break gradients
    def spectral_centroid(self, S):
        """Compute spectral centroid with differentiable operations."""
        # Ensure S is non-negative with a small epsilon to avoid zero division
        S = F.relu(S) + 1e-10
        
        # Get original shape and handle arbitrary batch dimensions
        orig_shape = S.shape
        
        # Make sure we have at least a 3D tensor [batch, freq, time]
        if S.dim() == 2:
            S = S.unsqueeze(0)  # Add batch dimension
            orig_shape = S.shape
        
        n_freqs, n_frames = orig_shape[-2:]
        
        # Reshape to [-1, n_freqs, n_frames] to handle arbitrary batch dimensions
        S_reshaped = S.reshape(-1, n_freqs, n_frames)
        batch_size = S_reshaped.shape[0]
        
        # Expand freq dimensions for broadcasting
        freqs_expanded = self.freqs.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, n_frames)
        
        # Normalize S along frequency axis with stability factor
        S_sum = torch.sum(S_reshaped, dim=1, keepdim=True)
        S_sum = torch.clamp(S_sum, min=1e-10)
        S_norm = S_reshaped / S_sum
        
        # Compute centroid as weighted average of frequencies
        centroid = torch.sum(freqs_expanded * S_norm, dim=1, keepdim=True)
        
        # Reshape back to original batch dimensions with centroid values
        if len(orig_shape) > 3:
            output_shape = list(orig_shape[:-2]) + [1, n_frames]
            centroid = centroid.reshape(output_shape)
        
        return centroid

    def spectral_bandwidth(self, S, centroid=None, p=2, norm=True, freq=None):
        """Compute p'th-order spectral bandwidth.
        Args:
            S: Spectrogram of shape [..., n_freqs, n_frames]
            centroid: Optional precomputed centroid of shape [..., 1, n_frames]
            p: Order of the bandwidth
            norm: If True, normalize the spectrogram before computing bandwidth
            freq: Optional frequency array. If None, uses self.freqs
        Returns:
            Bandwidth of shape [..., 1, n_frames]
        """
        # Ensure S is non-negative
        if torch.any(S < 0):
            raise ValueError("Spectral bandwidth is only defined with non-negative energies")
        
        # Get original shape and handle arbitrary batch dimensions
        orig_shape = S.shape
        n_freqs, n_frames = orig_shape[-2:]
        
        # Reshape to [-1, n_freqs, n_frames] to handle arbitrary batch dimensions
        S_reshaped = S.reshape(-1, n_freqs, n_frames)
        batch_size = S_reshaped.shape[0]
        
        # Use provided frequencies or default ones
        if freq is None:
            freqs = self.freqs
        else:
            freqs = freq
        
        # Compute centroid if not provided
        if centroid is None:
            centroid = self.spectral_centroid(S_reshaped)
        else:
            # Ensure centroid has correct shape
            centroid = centroid.reshape(batch_size, 1, n_frames)
        
        # Handle frequency dimensions correctly
        if freqs.dim() == 1:
            # Create the outer subtraction similar to np.subtract.outer
            # This expands frequencies to match with each centroid value
            # Shape: [batch_size, n_freqs, n_frames]
            freqs_expanded = freqs.unsqueeze(0).unsqueeze(-1).expand(batch_size, n_freqs, n_frames)
            deviation = torch.abs(freqs_expanded - centroid)
        else:
            # If freqs is already multi-dimensional, just compute the difference
            deviation = torch.abs(freqs - centroid)
        
        # Apply normalization only if requested
        if norm:
            # Normalize S along frequency axis to sum to 1 for each frame
            S_norm = torch.nn.functional.normalize(S_reshaped, p=1, dim=1)
        else:
            S_norm = S_reshaped
        
        # Compute bandwidth
        bw = torch.sum(S_norm * (deviation**p), dim=1, keepdim=True)**(1.0/p)
        
        # Reshape back to original batch dimensions
        bw = bw.reshape(*orig_shape[:-2], 1, n_frames)
        
        return bw

    def spectral_contrast(self, S, contrast_quantile=0.02):
        """Compute spectral contrast for each frame.
        
        Args:
            S: Spectrogram of shape [..., n_freqs, n_frames]
            contrast_quantile: Quantile for valley/peak selection
        
        Returns:
            Contrast of shape [..., n_contrast_bands+1, n_frames]
        """
        # Ensure S is non-negative
        if torch.any(S < 0):
            raise ValueError("Spectral contrast is only defined with non-negative energies")
        
        # Get original shape and handle arbitrary batch dimensions
        orig_shape = S.shape
        n_freqs, n_frames = orig_shape[-2:]
        
        # Reshape to [-1, n_freqs, n_frames] to handle arbitrary batch dimensions
        S_reshaped = S.reshape(-1, n_freqs, n_frames)
        batch_size = S_reshaped.shape[0]
        
        # Initialize outputs
        valley = torch.zeros((batch_size, self.n_contrast_bands + 1, n_frames), device=S.device)
        peak = torch.zeros_like(valley)
        
        # Process each band - this loop is over a small constant (bands), not data dimensions
        for k in range(self.n_contrast_bands + 1):
            band = self.contrast_bands[k]
            # Use advanced indexing to get the sub-band - vectorized
            sub_band = S_reshaped[:, band, :]
            
            # Skip last bin for all but the last band
            if k < self.n_contrast_bands and sub_band.shape[1] > 1:
                sub_band = sub_band[:, :-1, :]
            
            if sub_band.shape[1] > 0:  # Check that the sub-band is not empty
                # Always take at least one bin from each side
                idx = max(int(contrast_quantile * band.sum().item()), 1)
                idx = min(idx, sub_band.shape[1])  # Don't exceed band size
                
                if idx > 0:
                    # Sort along frequency axis - vectorized
                    sorted_band, _ = torch.sort(sub_band, dim=-2)
                    
                    # Compute mean of valleys and peaks - vectorized
                    if sorted_band.shape[1] >= idx:
                        valley[:, k, :] = torch.mean(sorted_band[:, :idx, :], dim=-2)
                        peak[:, k, :] = torch.mean(sorted_band[:, -idx:, :], dim=-2)
        
        # Compute contrast in dB scale - vectorized
        peak_db = 20 * torch.log10(torch.clamp(peak, min=1e-10))
        valley_db = 20 * torch.log10(torch.clamp(valley, min=1e-10))
        contrast = peak_db - valley_db
        
        # Reshape back to original batch dimensions
        contrast = contrast.reshape(*orig_shape[:-2], self.n_contrast_bands + 1, n_frames)
            
        return contrast

    def spectral_rolloff(self, S):
        """Compute roll-off frequency for each frame in a differentiable manner.
        
        Args:
            S: Spectrogram of shape [..., n_freqs, n_frames]
        
        Returns:
            Rolloff of shape [..., 1, n_frames]
        """
        # Ensure S is non-negative
        if torch.any(S < 0):
            raise ValueError("Spectral rolloff is only defined with non-negative energies")
        
        # Get original shape and handle arbitrary batch dimensions
        orig_shape = S.shape
        n_freqs, n_frames = orig_shape[-2:]
        
        # Reshape to [-1, n_freqs, n_frames] to handle arbitrary batch dimensions
        S_reshaped = S.reshape(-1, n_freqs, n_frames)
        batch_size = S_reshaped.shape[0]
        
        # Compute cumulative sum along frequency axis - vectorized
        total_energy = torch.cumsum(S_reshaped, dim=-2)
        
        # Calculate threshold energy (roll_percent of total energy) - vectorized
        threshold = self.roll_percent * total_energy[:, -1, :]
        threshold = threshold.unsqueeze(1)  # Add frequency dimension
        
        # Instead of finding exact indices where threshold is crossed (which breaks gradients),
        # create a soft mask that transitions at the threshold
        # First, create a mask that's 1 where energy is below threshold, 0 where above
        mask = (total_energy < threshold).float()
        
        # Create frequency range tensor - expand to match batch and frame dimensions
        freq_range = self.freqs.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, n_frames)
        
        # Apply soft weighting to frequencies
        # This will weight each frequency by how much it contributes to the rolloff
        # Higher temperature makes the transition sharper
        temperature = 50.0
        weights = torch.softmax(temperature * (threshold - total_energy), dim=1)
        
        # Compute weighted sum of frequencies to get rolloff
        rolloff = torch.sum(weights * freq_range, dim=1, keepdim=True)
        
        # Reshape back to original batch dimensions
        rolloff = rolloff.reshape(*orig_shape[:-2], 1, n_frames)
        
        return rolloff

    def spectral_flatness(self, S):
        """Compute spectral flatness for each frame.
        
        Args:
            S: Spectrogram of shape [..., n_freqs, n_frames]
        
        Returns:
            Flatness of shape [..., 1, n_frames]
        """
        # Ensure S is non-negative
        if torch.any(S < 0):
            raise ValueError("Spectral flatness is only defined with non-negative energies")
        
        # Get original shape and handle arbitrary batch dimensions
        orig_shape = S.shape
        n_freqs, n_frames = orig_shape[-2:]
        
        # Reshape to [-1, n_freqs, n_frames] to handle arbitrary batch dimensions
        S_reshaped = S.reshape(-1, n_freqs, n_frames)
        
        # Apply minimum threshold - vectorized
        S_thresh = torch.clamp(S_reshaped, min=1e-10)
        
        # Compute geometric mean (exp of mean of log) - vectorized
        log_S = torch.log(S_thresh)
        gmean = torch.exp(torch.mean(log_S, dim=-2, keepdim=True))
        
        # Compute arithmetic mean - vectorized
        amean = torch.mean(S_thresh, dim=-2, keepdim=True)
        
        # Spectral flatness is ratio of geometric to arithmetic mean - vectorized
        flatness = gmean / amean
        
        # Reshape back to original batch dimensions
        flatness = flatness.reshape(*orig_shape[:-2], 1, n_frames)
        
        return flatness

    def rms(self, audio, win_length: int = None, hop_length: int = None):
        """Compute root-mean-square (RMS) value for each frame.
        
        Args:
            audio: Audio signal of shape [..., n_samples] or [..., 1, n_samples]
        
        Returns:
            RMS of shape [..., 1, n_frames]
        """
        
        # Handle case where audio has shape [..., 1, n_samples]
        if audio.dim() >= 2 and audio.shape[-2] == 1:
            # Remove singleton dimension for processing
            audio = audio.squeeze(-2)
        
        # Get original shape and handle arbitrary batch dimensions
        orig_shape = audio.shape
        
        # Flatten all but the last dimension to handle arbitrary batch dimensions
        audio_flat = audio.reshape(-1, orig_shape[-1])
        batch_size = audio_flat.shape[0]
        
        # Center the signal and frame it - vectorized
        padding = self.n_fft // 2
        audio_padded = F.pad(audio_flat, (padding, padding), mode='reflect')
        
        win_length = win_length if win_length else self.win_length
        hop_length = hop_length if hop_length else self.hop_length
        # Manual framing to avoid issues with F.unfold
        n_frames = (audio_padded.shape[1] - win_length) // hop_length + 1
        frames = torch.zeros(batch_size, n_frames, win_length, device=audio.device)
        
        for i in range(n_frames):
            start = i * hop_length
            end = start + win_length
            frames[:, i, :] = audio_padded[:, start:end]
        
        # Compute RMS for each frame - vectorized
        rms = torch.sqrt(torch.mean(frames**2, dim=-1, keepdim=True))
        
        # Reshape back to match original batch dimensions with added frame dimension
        output_shape = list(orig_shape[:-1]) + [1, n_frames]
        rms = rms.reshape(output_shape).unsqueeze(1)
        
        return rms
    
    def highpass_filter(self, audio, cutoff: float = 100.0):
        """Simple 1st-order high-pass filter approximation (not full K-weighting)"""
        rc = 1.0 / (2 * 3.1416 * cutoff)
        dt = 1.0 / self.sr
        alpha = rc / (rc + dt)

        y = torch.zeros_like(audio)
        y[:, 0] = audio[:, 0]
        for i in range(1, audio.shape[1]):
            y[:, i] = alpha * (y[:, i-1] + audio[:, i] - audio[:, i-1])
        return y
    
    def loudness(self, audio, win_length_ms=400, hop_length_ms=100):
        # lufs like loudness with approximate perceptual filter, frame-based RMS, and log10 of power
        # audio: (B, T)
        audio = self.highpass_filter(audio)
        win_length = int(win_length_ms / 1000 * self.sr)
        hop_length = int(hop_length_ms / 1000 * self.sr)
        rms = self.rms(audio, win_length=win_length, hop_length=hop_length)
        power = rms**2 + 1e-12  # avoid log(0)
        loudness = 10 * torch.log10(power)
        return loudness

    def poly_features(self, S=None, order=None):
        """Get coefficients of fitting an nth-order polynomial to the columns of a spectrogram.
        
        Args:
            S: Spectrogram of shape [..., n_freqs, n_frames]
            order: Order of polynomial fit
        
        Returns:
            Coefficients of shape [..., order+1, n_frames]
        """
        if order is None:
            order = self.poly_order   
        
        if S is None:
            raise ValueError("Spectrogram S must be provided")
        
        # Get original shape and handle arbitrary batch dimensions
        orig_shape = S.shape
        n_freqs, n_frames = orig_shape[-2:]
        
        # Reshape to [-1, n_freqs, n_frames] to handle arbitrary batch dimensions
        S_reshaped = S.reshape(-1, n_freqs, n_frames)
        batch_size = S_reshaped.shape[0]
        
        # Create normalized frequency array (0 to 1) for better numerical stability
        x = torch.linspace(0, 1, n_freqs, device=S.device)
        
        # Build design matrix for polynomial regression (Vandermonde matrix)
        # Each column is x^i for i from 0 to order
        design_matrix = torch.zeros((n_freqs, order + 1), device=S.device)
        for i in range(order + 1):
            design_matrix[:, i] = x ** i
        
        # Compute pseudoinverse once for efficiency
        # Result will be [order+1, n_freqs]
        pinv = torch.linalg.pinv(design_matrix)
        
        # Initialize coefficients tensor
        coefficients = torch.zeros(batch_size, order+1, n_frames, device=S.device)
        
        # Compute polynomial coefficients for each batch and frame
        for b in range(batch_size):
            # Process all frames at once for this batch for efficiency
            # S_reshaped[b] has shape [n_freqs, n_frames]
            # pinv has shape [order+1, n_freqs]
            # Result will have shape [order+1, n_frames]
            coefficients[b] = torch.matmul(pinv, S_reshaped[b])
        
        # Reshape back to original batch dimensions
        coefficients = coefficients.reshape(*orig_shape[:-2], order+1, n_frames)
        
        return coefficients

    def forward(self, audio, features=None, device=None):
        """Extract all requested features from audio.
        
        Args:
            audio: Audio signal of shape [..., n_samples] or [..., 1, n_samples]
            features: List of feature names to extract. If None, extract all available features.
        
        Returns:
            Dictionary of features
        """

        device = device
        
        # Move all registered buffers and transforms to the same device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
            self.window = self.window.to(device)
            self.tempogram_window = self.tempogram_window.to(device)
            self.tempo_frequencies = self.tempo_frequencies.to(device)
            self.mel_transform = self.mel_transform.to(device)
            self.chroma_transform = self.chroma_transform.to(device)
        
        # ... rest of the existing forward method remains unchanged ...
        # Default to extracting all features if none specified
        if features is None:
            features = self.all_features

        # Initialize results dictionary
        results = {}
        # Precompute spectrogram
        spectrogram = self.spectrogram(audio)
        results['spectrogram'] = spectrogram
        if self.hpss:
            if spectrogram.dim() == 3: 
                spectrogram = spectrogram.unsqueeze(1)
            spectrogram = self.hpss(spectrogram)[0] # use harmonic components
        # Precompute Mel spectrogram if needed
        mel_spectrogram = None
        if any(feat in features for feat in ['mel_spectrogram', 'chroma']):
            mel_spectrogram = self.mel_spectrogram(spectrogram=spectrogram)
            results['mel_spectrogram'] = mel_spectrogram
            if 'chroma' in features:
                results['chroma'] = self.chroma(spectrogram=spectrogram)
        
        # Precompute onset envelope if needed
        if any(feat in features for feat in ['onset_strength', 'plp', 'fourier_tempogram']):
            onset_envelope = self.onset_strength(audio=audio, mel_spectrogram=mel_spectrogram)
            results['onset_strength'] = onset_envelope


        # Extract selected spectral features
        if 'spectral_centroid' in features:
            results['spectral_centroid'] = self.spectral_centroid(spectrogram)

        if 'spectral_bandwidth' in features:
            centroid = results.get('spectral_centroid', None)
            results['spectral_bandwidth'] = self.spectral_bandwidth(spectrogram, centroid)

        if 'spectral_contrast' in features:
            results['spectral_contrast'] = self.spectral_contrast(spectrogram)

        if 'spectral_rolloff' in features:
            results['spectral_rolloff'] = self.spectral_rolloff(spectrogram)

        if 'spectral_flatness' in features:
            results['spectral_flatness'] = self.spectral_flatness(spectrogram)

        if 'rms' in features:
            results['rms'] = self.rms(audio)

        if 'loudness' in features:
            results['loudness'] = self.loudness(audio)

        if 'poly_features' in features:
            results['poly_features'] = self.poly_features(S=spectrogram)

        if 'plp' in features:
            results['plp'] = self.plp(onset_envelope)

        if 'fourier_tempogram' in features:
            results['fourier_tempogram'] = torch.abs(self.fourier_tempogram(onset_envelope).real)

        return results  

def median_filter_1d(input_tensor, kernel_size=10):
        """
        Apply a 1D median filter across the time axis (frames) of a 3D tensor.
        Shape: (batch, features, frames)
        """
        assert input_tensor.dim() == 3, "Input must be 3D (B, F, T)"
        assert kernel_size % 2 == 1, "Kernel size must be odd"

        padding = kernel_size // 2
        # Pad the time axis (last dimension)
        padded = torch.nn.functional.pad(input_tensor, (padding, padding), mode='reflect')

        # Unfold the frames dimension
        unfolded = padded.unfold(dimension=2, size=kernel_size, step=1)  # shape: (B, F, T, k)
        median = unfolded.median(dim=-1)[0]  # take median over the kernel window

        return median

def count_channels(features: list, n_fft=2048, n_mels=128, n_chroma=12, n_contrast_bands=6, poly_order=2, tempo_bins=384):
    ''' function to return the number of features'''
    counts = {
        'spectrogram': (n_fft//2) + 1,  # Can vary depending on FFT size
        'spectral_centroid': 1,
        'spectral_bandwidth': 1,
        'spectral_contrast': n_contrast_bands,
        'spectral_rolloff': 1,
        'spectral_flatness': 1,
        'rms': 1,
        'poly_features': poly_order + 1,  # Energy + tilt + curvature
        'mel_spectrogram': n_mels,
        'chroma': n_chroma,
        'onset_strength': 1,
        'plp': 1,
        'fourier_tempogram': tempo_bins,
        'loudness': 1
    }
    return sum(v for k, v in counts.items() if k in features)

