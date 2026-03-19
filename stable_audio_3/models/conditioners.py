#Heavily influenced by https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/modules/conditioners.py

import torch
import logging, warnings
import string
import typing as tp
import gc
import random
from enum import Enum


class PaddingMode(str, Enum):
    """Enum for handling padding in text conditioner embeddings."""
    NONE = "none"       # No padding handling (raw embeddings with pad token)
    ZERO = "zero"       # Zero out padding positions (default)
    LEARNED = "learned" # Use learned padding embedding

from .adp import NumberEmbedder
from ..inference.utils import set_audio_channels
from .factory import create_pretransform_from_config
from .pretransforms import Pretransform
from ..models.utils import copy_state_dict
from .utils import load_ckpt_state_dict, enable_torch_compile
from .transformer import AbsolutePositionalEmbedding

from torch import nn
from .mir import FeatureExtractor, count_channels
import soxr
from einops import rearrange
from typing import Optional, Union
from torch.nn import functional as F

class Conditioner(nn.Module):
    def __init__(
            self,
            dim: int,
            output_dim: int,
            project_out: bool = False,
            padding_mode: str = "zero"
            ):

        super().__init__()

        self.dim = dim
        self.output_dim = output_dim
        self.padding_mode = padding_mode
        self.proj_out = nn.Linear(dim, output_dim) if (dim != output_dim or project_out) else nn.Identity()

        # Learned padding embedding (only created if needed)
        if padding_mode == "learned" or padding_mode == PaddingMode.LEARNED:
            self.padding_embedding = nn.Parameter(torch.randn(output_dim) * 0.02)

    def apply_padding(self, embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply padding handling based on padding_mode.

        Args:
            embeddings: [batch, seq_len, dim] - the embeddings to process
            attention_mask: [batch, seq_len] bool/int, True/1 = valid token

        Returns:
            embeddings with padding handled according to mode
        """
        mode = self.padding_mode
        if isinstance(mode, str):
            mode = PaddingMode(mode)

        if mode == PaddingMode.NONE:
            return embeddings
        elif mode == PaddingMode.ZERO:
            return embeddings * attention_mask.unsqueeze(-1).float()
        elif mode == PaddingMode.LEARNED:
            mask_expanded = attention_mask.unsqueeze(-1).bool()
            return torch.where(
                mask_expanded,
                embeddings,
                self.padding_embedding.unsqueeze(0).unsqueeze(0).expand_as(embeddings)
            )
        else:
            raise ValueError(f"Unknown padding mode: {mode}")

    def forward(self, x: tp.Any) -> tp.Any:
        raise NotImplementedError()
    
class IntConditioner(Conditioner):
    def __init__(self, 
                output_dim: int,
                min_val: int=0,
                max_val: int=512
                ):
        super().__init__(output_dim, output_dim)

        self.min_val = min_val
        self.max_val = max_val
        self.int_embedder = nn.Embedding(max_val - min_val + 1, output_dim).requires_grad_(True)

    def forward(self, ints: tp.List[int], device=None) -> tp.Any:
            
            #self.int_embedder.to(device)
    
            ints = torch.tensor(ints).to(device)
            ints = ints.clamp(self.min_val, self.max_val)
    
            int_embeds = self.int_embedder(ints).unsqueeze(1)
    
            return [int_embeds, torch.ones(int_embeds.shape[0], 1).to(device)]

class NumberConditioner(Conditioner):
    '''
        Conditioner that takes a list of floats, normalizes them for a given range, and returns a list of embeddings
    '''
    def __init__(self, 
                output_dim: int,
                min_val: float=0,
                max_val: float=1,
                fourier_features_type : tp.Literal["learned", "expo"] = "learned"
                ):
        super().__init__(output_dim, output_dim)

        self.min_val = min_val
        self.max_val = max_val

        self.embedder = NumberEmbedder(features=output_dim, fourier_features_type=fourier_features_type)

    def forward(self, floats: tp.List[float], device=None) -> tp.Any:
            self.embedder.to(device)
            # Cast the inputs to floats
            floats = [float(x) for x in floats]

            floats = torch.tensor(floats).to(device)

            floats = floats.clamp(self.min_val, self.max_val)
    
            normalized_floats = (floats - self.min_val) / (self.max_val - self.min_val)

            # Cast floats to same type as embedder
            embedder_dtype = next(self.embedder.parameters()).dtype
            normalized_floats = normalized_floats.to(embedder_dtype)

            float_embeds = self.embedder(normalized_floats).unsqueeze(1)
    
            return [float_embeds, torch.ones(float_embeds.shape[0], 1).to(device)]

class ListConditioner(Conditioner):
    def __init__(self, 
                output_dim: int,
                options: tp.List[str]
                ):
        super().__init__(output_dim, output_dim)

        self.options = options
        self.embedder = nn.Embedding(len(options)+1, output_dim).requires_grad_(True)

    def forward(self, texts: tp.List[str], device=None) -> tp.Any:
        self.embedder.to(device)
        # Cast the inputs to floats, handling the case where the input is not in the options
        ints = [self.options.index(x) + 1 if x in self.options else 0 for x in texts]

        ints = torch.tensor(ints).to(device) # shape [batch_size]

        int_embeds = self.embedder(ints).unsqueeze(1) # shape [batch_size, 1, output_dim]

        return [int_embeds, torch.ones(int_embeds.shape[0], 1).to(device)]
            

class SATCLAPTextConditioner(Conditioner):
    def __init__(self, 
                clap_model, 
                output_dim: int, 
                project_out: bool = False,
                use_text_features = False,
                feature_layer_ix: int = -2,
                **kwargs):

        super().__init__(clap_model.text_branch.embed_dim, output_dim, project_out=project_out)

        self.model = clap_model
        self.use_text_features = use_text_features
        self.feature_layer_ix = feature_layer_ix

        self.model.requires_grad_(False)
        self.model.eval()

        del self.model.pretransform
        del self.model.audio_branch

    def forward(self, texts: tp.List[str], device: tp.Any = "cuda") -> tp.Any:
        self.model.to(device)

        if self.use_text_features:
            if len(texts) == 1:
                text_features, text_attention_mask = self.model.text_branch.get_text_features([texts[0], ""], layer_ix=self.feature_layer_ix)
                text_features = text_features[:1, ...]
                text_attention_mask = text_attention_mask[:1, ...]
            else:
                text_features, text_attention_mask = self.model.text_branch.get_text_features(texts, layer_ix=self.feature_layer_ix)

            # Cast text feature to same type as proj_out, unless proj_out is Identity
            if not isinstance(self.proj_out, nn.Identity):
                proj_out_dtype = next(self.proj_out.parameters()).dtype
                text_features = text_features.to(proj_out_dtype)                        

            return [self.proj_out(text_features), text_attention_mask]

        # Fix for CLAP bug when only one text is passed
        if len(texts) == 1:
            text_embedding = self.model.get_text_embedding([texts[0], ""])[:1, ...]
        else:
            text_embedding = self.model.get_text_embedding(texts)

        text_embedding = text_embedding.unsqueeze(1).to(device)

        # Cast text embedding to same type as proj_out, unless proj_out is Identity
        if not isinstance(self.proj_out, nn.Identity):
            proj_out_dtype = next(self.proj_out.parameters()).dtype
            text_embedding = text_embedding.to(proj_out_dtype)

        return [self.proj_out(text_embedding), torch.ones(text_embedding.shape[0], 1).to(device)]

class SATCLAPAudioConditioner(Conditioner):
    def __init__(self, 
                clap_model, 
                output_dim: int, 
                project_out: bool = False,
                **kwargs):

        super().__init__(clap_model.joint_embed_dim, output_dim, project_out=project_out)

        self.model = clap_model

        self.model.requires_grad_(False)
        self.model.eval()

        del self.model.text_branch

    def forward(self, latents: tp.Union[torch.Tensor, tp.List[torch.Tensor], tp.Tuple[torch.Tensor]], device: tp.Any = "cuda") -> tp.Any:
        self.model.to(device)

        if isinstance(latents, list) or isinstance(latents, tuple):
            latents = torch.stack(latents, dim=0)

        latents = latents.to(device)

        audio_embedding = self.model.get_audio_embedding(latents)

        # Cast text embedding to same type as proj_out, unless proj_out is Identity
        if not isinstance(self.proj_out, nn.Identity):
            proj_out_dtype = next(self.proj_out.parameters()).dtype
            audio_embedding = audio_embedding.to(proj_out_dtype)

        audio_embedding = audio_embedding.unsqueeze(1).to(device)

        return [self.proj_out(audio_embedding), torch.ones(audio_embedding.shape[0], 1).to(device)]

def clap_load_state_dict(clap_ckpt_path, clap_model):
    state_dict = torch.load(clap_ckpt_path, map_location="cpu", weights_only=False)["state_dict"]

    # Remove "module." from state dict keys
    state_dict = {k[7:]: v for k, v in state_dict.items()}

    # Fix for transformers library
    removed_keys = ["text_branch.embeddings.position_ids"]
    for removed_key in removed_keys:
        if removed_key in state_dict:
            del state_dict[removed_key]

    clap_model.load_state_dict(state_dict)

class CLAPTextConditioner(Conditioner):
    def __init__(self,
                 output_dim: int,
                 clap_ckpt_path,
                 use_text_features = False,
                 feature_layer_ix: int = -1,
                 audio_model_type="HTSAT-base",
                 enable_fusion=True,
                 project_out: bool = False,
                 finetune: bool = False,
                 padding_mode: str = "none"):
        super().__init__(768 if use_text_features else 512, output_dim, project_out=project_out, padding_mode=padding_mode)

        self.use_text_features = use_text_features
        self.feature_layer_ix = feature_layer_ix
        self.finetune = finetune

        # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                import laion_clap
                
                model = laion_clap.CLAP_Module(enable_fusion=enable_fusion, amodel=audio_model_type, device='cpu')

                if self.finetune:
                    self.model = model
                else: 
                    self.__dict__["model"] = model

                clap_load_state_dict(clap_ckpt_path, self.model.model)

                if self.finetune:
                    self.model.model.text_branch.requires_grad_(True)
                    self.model.model.text_branch.train()
                else:
                    self.model.model.text_branch.requires_grad_(False)
                    self.model.model.text_branch.eval()

            finally:
                logging.disable(previous_level)

        del self.model.model.audio_branch

        gc.collect()
        torch.cuda.empty_cache()

    def get_clap_features(self, prompts, layer_ix=-2, device: tp.Any = "cuda"):
        prompt_tokens = self.model.tokenizer(prompts)
        attention_mask = prompt_tokens["attention_mask"].to(device=device, non_blocking=True)
        prompt_features = self.model.model.text_branch(
            input_ids=prompt_tokens["input_ids"].to(device=device, non_blocking=True),
            attention_mask=attention_mask,
            output_hidden_states=True
        )["hidden_states"][layer_ix]

        return prompt_features, attention_mask

    def forward(self, texts: tp.List[str], device: tp.Any = "cuda") -> tp.Any:
        self.model.to(device)

        if self.use_text_features:
            if len(texts) == 1:
                text_features, text_attention_mask = self.get_clap_features([texts[0], ""], layer_ix=self.feature_layer_ix, device=device)
                text_features = text_features[:1, ...]
                text_attention_mask = text_attention_mask[:1, ...]
            else:
                text_features, text_attention_mask = self.get_clap_features(texts, layer_ix=self.feature_layer_ix, device=device)

            # Cast text feature to same type as proj_out, unless proj_out is Identity
            if not isinstance(self.proj_out, nn.Identity):
                proj_out_dtype = next(self.proj_out.parameters()).dtype
                text_features = text_features.to(proj_out_dtype)

            text_features = self.proj_out(text_features)
            text_features = self.apply_padding(text_features, text_attention_mask)

            return [text_features, text_attention_mask]

        # Fix for CLAP bug when only one text is passed
        if len(texts) == 1:
            text_embedding = self.model.get_text_embedding([texts[0], ""], use_tensor=True)[:1, ...]
        else:
            text_embedding = self.model.get_text_embedding(texts, use_tensor=True)

        text_embedding = text_embedding.unsqueeze(1).to(device)

        # Cast text embedding to same type as proj_out, unless proj_out is Identity
        if not isinstance(self.proj_out, nn.Identity):
            proj_out_dtype = next(self.proj_out.parameters()).dtype
            text_embedding = text_embedding.to(proj_out_dtype)

        return [self.proj_out(text_embedding), torch.ones(text_embedding.shape[0], 1).to(device)]

class CLAPAudioConditioner(Conditioner):
    def __init__(self, 
                 output_dim: int, 
                 clap_ckpt_path,
                 audio_model_type="HTSAT-base", 
                 enable_fusion=True,
                 project_out: bool = False):
        super().__init__(512, output_dim, project_out=project_out)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                import laion_clap
                
                model = laion_clap.CLAP_Module(enable_fusion=enable_fusion, amodel=audio_model_type, device='cpu')

                if self.finetune:
                    self.model = model
                else: 
                    self.__dict__["model"] = model

                clap_load_state_dict(clap_ckpt_path, self.model.model)

                if self.finetune:
                    self.model.model.audio_branch.requires_grad_(True)
                    self.model.model.audio_branch.train()
                else:
                    self.model.model.audio_branch.requires_grad_(False)
                    self.model.model.audio_branch.eval()

            finally:
                logging.disable(previous_level)

        del self.model.model.text_branch

        gc.collect()
        torch.cuda.empty_cache()

    def forward(self, audios: tp.Union[torch.Tensor, tp.List[torch.Tensor], tp.Tuple[torch.Tensor]] , device: tp.Any = "cuda") -> tp.Any:

        self.model.to(device)

        if isinstance(audios, list) or isinstance(audios, tuple):
            audios = torch.cat(audios, dim=0)

        # Convert to mono
        mono_audios = audios.mean(dim=1)

        with torch.cuda.amp.autocast(enabled=False):
            audio_embedding = self.model.get_audio_embedding_from_data(mono_audios.float(), use_tensor=True)

        audio_embedding = audio_embedding.unsqueeze(1).to(device)

        # Cast audio embedding to same type as proj_out, unless proj_out is Identity

        if not isinstance(self.proj_out, nn.Identity):
            proj_out_dtype = next(self.proj_out.parameters()).dtype
            audio_embedding = audio_embedding.to(proj_out_dtype)

        return [self.proj_out(audio_embedding), torch.ones(audio_embedding.shape[0], 1).to(device)]

class AudioEmbeddingConditioner(Conditioner):
    def __init__(self,
                dim: int = None,
                output_dim: int = None,
                project_out: bool = False,
                n_frames: int = None,
                mono_out: bool = True,
                sample_rate_in: int = 44100,
                sample_rate_out: int = 22050,
                bottleneck: dict = None,
                median_filter: bool = False
                ):
        """
        Base class for audio embedding conditioners.
        
        Args:
            dim (int): Input dimension of the audio features
            output_dim (int): Output dimension (if None, same as input dim)
            project_out (bool): Whether to project output to output_dim even if dim == output_dim
            n_frames (int): Number of frames to produce in the output
            mono_out (bool): Whether to convert stereo input to mono
            sample_rate_in (int): Sample rate of input audio
            sample_rate_out (int): Sample rate to convert audio to
            bottleneck (dict): Configuration for bottleneck compression
            median_filter (bool): Whether to apply a median filter to the features
        """
        super().__init__(dim, output_dim, project_out)
        from stable_audio_tools.models.transforms import dct, dct_2d
        from stable_audio_tools.models.factory import create_bottleneck_from_config
        from stable_audio_tools.models.mir import median_filter_1d
        # apply learned bottleneck compression based on type
        
        self.bottleneck = create_bottleneck_from_config(bottleneck) if bottleneck else None
        self.median_filter = median_filter_1d if median_filter else None
        
        self.n_frames = n_frames
        self.dct = dct
        self.dct_2d = dct_2d
        self.resample = None
        if sample_rate_in != sample_rate_out:
            import torchaudio
            self.resample = torchaudio.transforms.Resample(orig_freq=sample_rate_in, new_freq=sample_rate_out)
        self.mono_out = mono_out

    def preprocess(self, audios: tp.Union[torch.Tensor, tp.List[torch.Tensor], tp.Tuple[torch.Tensor]], device: tp.Any = "cuda") -> tp.Any:
        """
        Preprocess audio inputs for feature extraction.
        
        Args:
            audios: Input audio tensor(s) [batch, channels, samples]
            device: Target device for computation
            
        Returns:
            Preprocessed audio tensor
        """
        if isinstance(audios, list) or isinstance(audios, tuple):
            audios = torch.cat(audios, dim=0).float()
        if self.mono_out:
            # Convert to mono
            audios = audios.mean(dim=1).unsqueeze(1)
        if self.resample:
            # Resample audio if necessary
            audios = self.resample(audios)
        # autocast
        with torch.cuda.amp.autocast(enabled=False):
            # Convert to half-precision if necessary
            audios = audios.half()
        
        return audios.to(device)
    
    def feature_stats(self, features, compression_type):
        """
        create stats for features
        
        Args:
            features: Input features tensor
            compression_type: Type of pooling ('median', 'mean', 'max', 'min', 'distribution')
            
        Returns:
            Feature stats over the time dimension
        """
        if compression_type == 'median':
            pooled = torch.median(features, dim=2).values
        elif compression_type == 'mean':
            pooled = torch.mean(features, dim=2)
        elif compression_type == 'max':
            pooled = torch.max(features, dim=2).values
        elif compression_type == 'min':
            pooled = torch.min(features, dim=2).values
        elif compression_type == 'distribution':
            pooled = torch.cat([torch.mean(features, dim=2), torch.std(features, dim=2)], dim=1)

        return pooled.unsqueeze(2)  # Add frame dimension back
    
    def compress_frames(self, features: tp.Any, device: tp.Any = "cuda", compression_type: str = "dct", n_components: int = 2, drop_C0=False) -> tp.Any:
        """
        Apply time-based compression methods to features.
        
        Args:
            features: Input features tensor
            device: Target device for computation
            compression_type: Type of compression ('dct', 'dct_2d', 'median', 'mean', 'max', 'auto', 'interpolation')
            n_components: Number of components to keep for DCT compression
            
        Returns:
            Compressed features
        """
        
        first_component = 1 if drop_C0 else 0
        # Ensure features meet minimum dimension requirements
        if features.shape[-1] < self.n_frames:
            pad = self.n_frames - features.shape[-1]
            r = 0 if pad % 2 == 0 else 1
            features = F.pad(features, (pad // 2, pad // 2 + r), 'constant', 0)
            assert features.shape[-1] == self.n_frames, f"expected len {self.n_frames} but got {features.shape[-1]}"
        
        # Apply the requested compression type
        if compression_type in ['median', 'mean', 'max', 'min', 'distribution']:
            compressed = self.feature_stats(features, compression_type)
        elif compression_type == 'dct_2d':
            # 2D jpeg-style compression over channels and frames
            compressed = self.dct_2d(features, norm='ortho')
            compressed = compressed[:,:n_components,:self.n_frames].to(device)
        elif compression_type == 'dct':
            # short chunked 1D dct over frames at latent rate
            chunk_size = features.shape[-1] // self.n_frames
            total_samples = self.n_frames * chunk_size
            chunked_features = features[..., :total_samples].reshape(features.shape[0], features.shape[1], self.n_frames, chunk_size)
            compressed = self.dct(chunked_features, norm='ortho')
            compressed = compressed[...,:self.n_frames,first_component:n_components+first_component]
            compressed = rearrange(compressed, 'b c t d -> b (c d) t')
        elif compression_type == 'interpolation':
            # interpolate the audio embedding per channel over time to match the n_frames
            compressed = torch.nn.functional.interpolate(features, size=self.n_frames, mode='linear', align_corners=False)
        else:
            raise ValueError(f"Unknown compression type: {compression_type}")
                
        return compressed

    def encode_bottleneck(self, features):
        """
        Apply bottleneck encoding to features.
        """
        features_discrete = self.bottleneck.encode(features, return_info=False)
        return features_discrete


class MIRConditioner(AudioEmbeddingConditioner):
    """MIR conditioner for extracting audio features 
    using the custom differentiable MIR library.
    
    Args:
        output_dim (int): Output dimensionality of the extracted features.
        sample_rate (int): Target sample rate for input audio.
        project_out (bool): Whether to project output to output_dim even if dim == output_dim.
        features (list): List of features to extract. All will be extracted by default.
        compression_type (str): Type of compression to use (None, 'dct', 'dct_2d', 'median', 'mean', 'max', 'autopool').
        bottleneck (str): Type of bottleneck compression to use (None, 'dct', 'dct_2d', 'median', 'mean', 'max', 'autopool').
        n_fft (int): Number of FFT points for STFT.
        n_mels (int): Number of Mel bands for Mel spectrogram.
        n_chroma (int): Number of chroma bands for chroma features.
        n_contrast_bands (int): Number of contrast bands for contrast features.
        poly_order (int): Polynomial order for polyphonic features.
        n_components (int): Number of components for DCT compression.
        embedding (bool): Whether to use embedding for features.
        n_frames (int): Number of frames for output features.
        use_hpss (bool): Whether to use harmonic-percussive source separation.

    """
    def __init__(
            self,
            output_dim: int = None,
            sample_rate: int = 44100,
            project_out: bool = False,
            features: tp.List[str] = None,
            compression_type: str = None,
            bottleneck: str = None,
            n_fft: int = 2048,
            n_mels: int = 128,
            n_chroma: int = 12,
            n_contrast_bands: int = 6,
            poly_order: int = 2,
            n_components: int = 2,
            embedding: bool = False,
            n_frames: int = 1024,
            use_hpss: bool = False,
            drop_C0: bool = False

    ):
        dim = count_channels(features,
                            n_fft=n_fft,
                            n_mels=n_mels,
                            n_chroma=n_chroma,
                            n_contrast_bands=n_contrast_bands,
                            poly_order=poly_order
                            )
        output_dim = output_dim if output_dim else dim
        if compression_type == 'dct':
            dim = dim * n_components
        super().__init__(dim=dim,
                         output_dim=output_dim,
                         project_out=project_out,
                         n_frames=1024,
                         sample_rate_in=sample_rate, 
                         sample_rate_out=22050
                        )
        self.extractor = FeatureExtractor(sample_rate=22050,
                                            n_fft=2048,
                                            hop_length=512,
                                            win_length=None,
                                            n_mels=n_mels,
                                            center=True,
                                            window="hann",
                                            n_chroma=n_chroma,
                                            n_contrast_bands=n_contrast_bands,
                                            fmin=200.0,
                                            roll_percent=0.85,
                                            poly_order=poly_order,
                                            tempo_min=30,
                                            tempo_max=300,
                                            prior_mean=120,
                                            prior_std=1.0,
                                            use_hpss=use_hpss
                                        )
        self.features = features
        self.output_dim = output_dim
        self.sample_rate = sample_rate
        self.compression_type = compression_type
        self.bottleneck = bottleneck
        self.embedding = embedding
        self.n_frames = n_frames
        self.n_components = n_components
        self.drop_C0 = drop_C0
        

    def forward(self, audios: tp.Union[torch.Tensor, tp.List[torch.Tensor], tp.Tuple[torch.Tensor]] , device: tp.Union[torch.device, str]):
        # Preprocess audio
        mono_audios = self.preprocess(audios, device)
        # Process each audio in the batch separately
        all_features = []
        with torch.cuda.amp.autocast(enabled=False):
            for single_audio in mono_audios:
                
                # Extract features for this audio
                results = self.extractor.forward(single_audio, features=self.features, device=device)
                
                # Collect features for this audio
                audio_features = []
                for feature in self.features:
                    audio_features.append(results[feature])
                # Concatenate features for this audio
                audio_features = torch.cat(audio_features, dim=1)
                all_features.append(audio_features)
            
            # Stack all features from the batch
            features = torch.cat(all_features, dim=0)
            
            # Compress features
            if self.compression_type is not None:
                features = self.compress_frames(features,
                                                device,
                                                compression_type=self.compression_type,
                                                n_components=self.n_components,
                                                drop_C0=self.drop_C0
                                                )
            # Median Filter                                    
            if self.median_filter:
                features = self.median_filter_1d(features, kernel_size=3)
            
            # Create discrete feature embedding
            if self.embedding:
                # Quantize to tenths
                quantized = torch.round(features * 10).to(torch.long)  # (B, C, T)

                # Unique & inverse indices (flattened)
                uniq, inverse = torch.unique(quantized, return_inverse=True)
                embedding_layer = nn.Embedding(len(uniq), 1).to(features.device)

                # Map values
                embedded_flat = embedding_layer(inverse)  # shape: (B * C * T, 1)
                features = embedded_flat.view_as(quantized).float()  # shape: (B, C, T)
            # Apply RVQ bottleneck
            if self.bottleneck:
                features = self.encode_bottleneck(features)
            features_shape = features.shape

        return features.to(device).half(), torch.ones(features_shape).to(device).half()

class BeatThisConditioner(AudioEmbeddingConditioner):
    """Beat extraction conditioner using the BeatThis model.
    
    Args:
        output_dim (int): Output dimensionality of the extracted features.
        sample_rate (int): Target sample rate for input audio.
        downbeats (bool): Whether to include downbeats in the output.
        beats (bool): Whether to include beats in the output.
        compression_type (str): Type of compression to use (None, 'dct', 'dct_2d', 'median', 'mean', 'max', 'autopool').
        n_components (int): Number of components to keep for DCT compression.
        bottleneck (dict): Configuration for bottleneck compression.
    """
    def __init__(self,
                 n_frames: int = 1024,
                 sample_rate: int = 44100,
                 downbeats: bool = True,
                 beats: bool = True,
                 compression_type: tp.Optional[str] = None,
                 n_components: int = 2,
                 project_out: bool = False,
                 bottleneck = None,
                 output_dim = None):
        dim = 0
        if downbeats:
            dim += 1
        if beats:
            dim += 1
        output_dim = output_dim if output_dim else dim
        super().__init__(
            dim=1024,  # Default dimension for features
            output_dim=output_dim,
            project_out=project_out,
            mono_out=True,
            sample_rate_in=sample_rate,
            sample_rate_out=22050  # BeatThis requires 22050Hz
        )
        
        from stable_audio_tools.models import beat_this
        from beat_this.inference import split_predict_aggregate
        from beat_this.preprocessing import LogMelSpect
        self.output_dim = output_dim
        self.model = beat_this.BeatThis().requires_grad_(False).eval()
        self.split_predict_aggregate = split_predict_aggregate
        self.spect = LogMelSpect(device="cuda")
        self.downbeats = downbeats
        self.beats = beats
        self.compression_type = compression_type
        self.n_components = n_components
        self.n_frames = n_frames

        # Initialize bottleneck if provided
        if bottleneck:
            from stable_audio_tools.models.factory import create_bottleneck_from_config
            self.bottleneck = create_bottleneck_from_config(bottleneck)
        else:
            self.bottleneck = None

    def extract(self, x: torch.Tensor, device='cuda') -> torch.Tensor:
        """Extract beat features from preprocessed audio tensor."""
        b = x.shape[0]  # Batch size
        
        # Convert to spectrograms
        spectrograms = self.spect(x)
        
        # Process each spectrogram in the batch
        features = []
        for i in range(b):
            spect_i = spectrograms[i:i+1]  # Get single spectrogram with batch dim
            
            with torch.inference_mode():
                # Use split_predict_aggregate for long spectrograms
                predictions = self.split_predict_aggregate(
                    spect=spect_i.reshape(-1, 128),
                    chunk_size=1500,
                    overlap_mode="keep_first",
                    border_size=6,
                    model=self.model,
                )
                
                # Collect required beat features
                _predictions = []
                if self.beats:
                    _predictions.append(predictions["beat"])
                if self.downbeats:
                    _predictions.append(predictions["downbeat"])
                
                if _predictions:
                    # Stack instead of concatenate to maintain dimensions
                    batch_features = torch.stack(_predictions, dim=0)
                    features.append(batch_features)
        
        # Calculate number of channels
        channels = 0
        if self.beats:
            channels += 1
        if self.downbeats:
            channels += 1
        
        if not features:  # Handle case with no features
            return torch.zeros((b, channels, self.n_frames), device=device)
        
        # Stack the features along batch dimension instead of concatenating
        features = torch.cat(features, dim=0).to(device)
        
        # Check if the feature shape matches expected output shape
        if features.shape[0] != b:
            # Ensure we have the correct batch size
            features = features.view(b, channels, -1)
        
        return features

    def forward(self, audios: tp.Union[torch.Tensor, tp.List[torch.Tensor], tp.Tuple[torch.Tensor]], 
                device: tp.Union[torch.device, str], 
                return_mask: bool = True,
                sigmoid: bool = False,
                **kwargs) -> tp.Any:
        """Process audio inputs to extract beat features.
        
        Args:
            audios: Input audio tensor(s) [batch, channels, samples]
            device: Target device for computation
            return_mask: Whether to return attention mask
            sigmoid: Whether to apply sigmoid to the output features
            
        Returns:
            List containing [features, mask] if return_mask=True, otherwise features
        """
        # Preprocess audio (converts to mono, resamples, etc.)
        x = self.preprocess(audios, device)
        
        # Extract beat features
        features = self.extract(x, device)
        
        # Resize features to match n_frames
        if features.shape[-1] < self.n_frames:
            # Upsample using linear interpolation
            features = F.interpolate(features, size=self.n_frames, mode='linear')
        elif features.shape[-1] > self.n_frames:
            # Handle downsampling based on compression_type
            if self.compression_type:
                # Apply chosen compression method
                features = self.compress_frames(
                    features, 
                    device=device,
                    compression_type=self.compression_type, 
                    n_components=self.n_components
                )
        
        # Apply sigmoid if requested
        if sigmoid:
            features = torch.sigmoid(features)
        
        # Apply bottleneck if provided
        if self.bottleneck:
            features, bottleneck_info = self.bottleneck.encode(features)
        
        # Convert to half-precision for efficiency
        features = features.half()
        
        # Return with mask if requested
        if return_mask:
            mask = torch.ones(features.size(0), 1, device=device).half()
            return [features, mask]
        else:
            return features


class ContentVecConditioner(AudioEmbeddingConditioner):
    def __init__(self, 
                 output_dim: int = 768, 
                 sample_rate: int = 44100,
                 hidden_layer: int = 9,
                 project_out: bool = False,
                 n_frames: int = 1024):
        super().__init__(
            dim=768,  # Default dimension for features
            output_dim=output_dim,
            project_out=project_out,
            mono_out=True,
            n_frames=n_frames,
            sample_rate_in=sample_rate,
            sample_rate_out=16000 # ContentVec default is 16kHz
        )
        from transformers import AutoModel
        self.hidden_layer = hidden_layer
        self.model = None

        # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)

        # Also suppress transformers-specific logging and progress bars
        transformers_log_level = None
        transformers_disable_progress_bar = None
        try:
            import transformers
            transformers_log_level = transformers.logging.get_verbosity()
            transformers.logging.set_verbosity_error()
            # Disable progress bars
            try:
                from transformers.utils import is_progress_bar_enabled
                transformers_disable_progress_bar = not is_progress_bar_enabled()
                transformers.utils.logging.disable_progress_bar()
            except (ImportError, AttributeError) as e:
                # Progress bar control not available in this transformers version
                logging.debug(f"Could not disable transformers progress bar: {e}")
        except (ImportError, AttributeError) as e:
            # Transformers not available or version mismatch
            logging.debug(f"Could not configure transformers logging: {e}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                from transformers import AutoModel
                self.model = AutoModel.from_pretrained("lengyue233/content-vec-best")
                self.model.eval()

            finally:
                logging.disable(previous_level)
                if transformers_log_level is not None:
                    try:
                        transformers.logging.set_verbosity(transformers_log_level)
                    except (AttributeError, Exception) as e:
                        logging.debug(f"Could not restore transformers log level: {e}")
                if transformers_disable_progress_bar is not None and not transformers_disable_progress_bar:
                    try:
                        transformers.utils.logging.enable_progress_bar()
                    except (AttributeError, Exception) as e:
                        logging.debug(f"Could not re-enable transformers progress bar: {e}")

        gc.collect()
        torch.cuda.empty_cache()

    def forward(self, audios: tp.Union[torch.Tensor, tp.List[torch.Tensor], tp.Tuple[torch.Tensor]] , device: tp.Union[torch.device, str]) -> torch.Tensor:
        self.model.requires_grad_(False)
        self.model.to(device)

        # Ensure audio is mono and has the correct dimensions
        mono_audios = self.preprocess(audios)
        
        # Convert audio to float32 to match model weights
        mono_audios = mono_audios.float()  # Add this line to convert back to float32
        
        # Ensure the shape is [batch_size, time_steps] by removing any extra dimensions
        mono_audios = mono_audios.squeeze(1)  # Remove the extra channel dimension if it exists

        with torch.no_grad():
            # Call the model with the correct output_hidden_states parameter
            outputs = self.model(mono_audios, output_hidden_states=True)
            
            if self.hidden_layer == -1:
                # Get the last hidden state
                audio_embedding = outputs.last_hidden_state
            elif self.hidden_layer > 0:
                # Access the correct hidden state
                hidden_states = outputs.hidden_states
                # Make sure index is in bounds
                layer_idx = min(self.hidden_layer, len(hidden_states) - 1)
                audio_embedding = hidden_states[layer_idx]
        

        # Rearrange the audio embedding dimensions
        audio_embedding = rearrange(audio_embedding, 'b t c -> b c t')
        
        # Handle frame count adjustment
        if audio_embedding.shape[2] != self.n_frames:
            audio_embedding = self.compress_frames(
                audio_embedding, 
                device=device, 
                compression_type='interpolation', 
                n_components=self.n_frames
            )
        
        # Convert back to half precision for consistency with other conditioners
        audio_embedding = audio_embedding.half()
        
        # Return the embedding and a mask
        return audio_embedding.to(device), torch.ones(audio_embedding.shape[0], 1, device=device).half()

class T5Conditioner(Conditioner):

    T5_MODELS = ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b",
              "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large",
              "google/flan-t5-xl", "google/flan-t5-xxl", "google/t5-v1_1-xl", "google/t5-v1_1-xxl"]
    
    T5_MODEL_DIMS = {
        "t5-small": 512,
        "t5-base": 768,
        "t5-large": 1024,
        "t5-3b": 1024,
        "t5-11b": 1024,
        "google/t5-v1_1-xl": 2048,
        "google/t5-v1_1-xxl": 4096,
        "google/flan-t5-small": 512,
        "google/flan-t5-base": 768,
        "google/flan-t5-large": 1024,
        "google/flan-t5-3b": 1024,
        "google/flan-t5-11b": 1024,
        "google/flan-t5-xl": 2048,
        "google/flan-t5-xxl": 4096,
    }

    def __init__(
            self,
            output_dim: int,
            t5_model_name: str = "t5-base",
            max_length: str = 128,
            enable_grad: bool = False,
            project_out: bool = False,
            padding_mode: str = "zero"
    ):
        assert t5_model_name in self.T5_MODELS, f"Unknown T5 model name: {t5_model_name}"
        super().__init__(self.T5_MODEL_DIMS[t5_model_name], output_dim, project_out=project_out, padding_mode=padding_mode)

        import os

        self.max_length = max_length
        self.enable_grad = enable_grad

        # Set environment variables to disable progress bars BEFORE importing transformers
        prev_hf_hub = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
        prev_transformers = os.environ.get("TRANSFORMERS_VERBOSITY")
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"

        # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                from transformers import T5EncoderModel, AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
                model = T5EncoderModel.from_pretrained(t5_model_name).train(enable_grad).requires_grad_(enable_grad).to(torch.float16)

            finally:
                logging.disable(previous_level)
                # Restore environment variables
                if prev_hf_hub is None:
                    os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
                else:
                    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = prev_hf_hub
                if prev_transformers is None:
                    os.environ.pop("TRANSFORMERS_VERBOSITY", None)
                else:
                    os.environ["TRANSFORMERS_VERBOSITY"] = prev_transformers

        if self.enable_grad:
            self.model = model
        else:
            self.__dict__["model"] = model


    def forward(self, texts: tp.List[str], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        
        self.model.to(device)
        self.proj_out.to(device)

        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device).to(torch.bool)

        self.model.eval()
            
        with torch.cuda.amp.autocast(dtype=torch.float16) and torch.set_grad_enabled(self.enable_grad):
            embeddings = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            )["last_hidden_state"]    

        # Cast embeddings to same type as proj_out, unless proj_out is Identity
        if not isinstance(self.proj_out, nn.Identity):
            proj_out_dtype = next(self.proj_out.parameters()).dtype
            embeddings = embeddings.to(proj_out_dtype)

        embeddings = self.proj_out(embeddings)
        embeddings = self.apply_padding(embeddings, attention_mask)

        return embeddings, attention_mask

class T5GemmaConditioner(Conditioner):

    T5GEMMA_MODELS = ["google/t5gemma-b-b-ul2"]

    T5GEMMA_MODEL_DIMS = {
        "google/t5gemma-b-b-ul2": 768,
    }

    def __init__(
            self,
            output_dim: int,
            model_name: str = "google/t5gemma-b-b-ul2",
            max_length: str = 128,
            enable_grad: bool = False,
            project_out: bool = False,
            padding_mode: str = "zero"
    ):
        assert model_name in self.T5GEMMA_MODELS, f"Unknown T5 model name: {model_name}"
        super().__init__(self.T5GEMMA_MODEL_DIMS[model_name], output_dim, project_out=project_out, padding_mode=padding_mode)

        import os

        self.max_length = max_length
        self.enable_grad = enable_grad

        # Set environment variables to disable progress bars BEFORE importing transformers
        # This is the most reliable way to suppress HuggingFace progress bars
        prev_hf_hub = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
        prev_transformers = os.environ.get("TRANSFORMERS_VERBOSITY")
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"

        # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                from transformers import T5GemmaEncoderModel, AutoTokenizer, AutoConfig
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                config = AutoConfig.from_pretrained(model_name)
                config.is_encoder_decoder = False
                model = T5GemmaEncoderModel.from_pretrained(model_name, config=config).train(enable_grad).requires_grad_(enable_grad)

            finally:
                logging.disable(previous_level)
                # Restore environment variables
                if prev_hf_hub is None:
                    os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
                else:
                    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = prev_hf_hub
                if prev_transformers is None:
                    os.environ.pop("TRANSFORMERS_VERBOSITY", None)
                else:
                    os.environ["TRANSFORMERS_VERBOSITY"] = prev_transformers

        # Compile the model to reduce CPU-GPU kernel launch overhead,
        # which is sensitive to CPU contention from DataLoader workers
        if enable_torch_compile:
            model = torch.compile(model)

        if self.enable_grad:
            self.model = model
        else:
            self.__dict__["model"] = model

        self._device_initialized = False

    def forward(self, inputs: tp.Union[tp.List[str], tp.List[tp.Dict[str, torch.Tensor]]], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:

        # Only move to device once (avoid overhead on every forward call)
        if not self._device_initialized:
            self.model.to(device)
            self.proj_out.to(device)
            self.model.eval()
            self._device_initialized = True

        # Handle pre-tokenized inputs (dicts with input_ids/attention_mask from DataLoader workers)
        # or raw strings (from demo generation / inference)
        if isinstance(inputs[0], dict):
            input_ids = torch.stack([x["input_ids"] for x in inputs]).to(device, non_blocking=True)
            attention_mask = torch.stack([x["attention_mask"] for x in inputs]).to(device, non_blocking=True).to(torch.bool)
        else:
            encoded = self.tokenizer(
                inputs,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device, non_blocking=True)
            attention_mask = encoded["attention_mask"].to(device, non_blocking=True).to(torch.bool)

        with torch.no_grad():
            embeddings = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            )["last_hidden_state"]

        # Cast embeddings to same type as proj_out, unless proj_out is Identity
        if not isinstance(self.proj_out, nn.Identity):
            proj_out_dtype = next(self.proj_out.parameters()).dtype
            embeddings = embeddings.to(proj_out_dtype)

        embeddings = self.proj_out(embeddings)
        embeddings = self.apply_padding(embeddings, attention_mask)

        return embeddings, attention_mask

class CausalLMConditioner(Conditioner):

    MODELS = ["google/gemma-2-2b"]
    
    MODEL_DIMS = {
        "google/gemma-2-2b": 2304
    }

    def __init__(
            self,
            output_dim: int,
            model_name: str = "google/gemma-2-2b",
            max_length: str = 128,
            enable_grad: bool = False,
            project_out: bool = False,
            learned_scale: bool = True,
            padding_mode: str = "zero"
    ):
        assert model_name in self.MODELS, f"Unknown model name: {model_name}"
        super().__init__(self.MODEL_DIMS[model_name], output_dim, project_out=project_out, padding_mode=padding_mode)

        import os
        from .blocks import RMSNorm

        self.max_length = max_length
        self.enable_grad = enable_grad

        # Set environment variables to disable progress bars BEFORE importing transformers
        prev_hf_hub = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
        prev_transformers = os.environ.get("TRANSFORMERS_VERBOSITY")
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"

        # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(model_name).train(enable_grad).requires_grad_(enable_grad)

            finally:
                logging.disable(previous_level)
                # Restore environment variables
                if prev_hf_hub is None:
                    os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
                else:
                    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = prev_hf_hub
                if prev_transformers is None:
                    os.environ.pop("TRANSFORMERS_VERBOSITY", None)
                else:
                    os.environ["TRANSFORMERS_VERBOSITY"] = prev_transformers

        if self.enable_grad:
            self.model = model
        else:
            self.__dict__["model"] = model

        self.norm = RMSNorm(self.dim)

        self.learned_scale = learned_scale

        if self.learned_scale:
            self.scale = nn.Parameter(torch.tensor(.01))


    def forward(self, texts: tp.List[str], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        
        self.model.to(device)
        self.proj_out.to(device)

        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device).to(torch.bool)

        self.model.eval()
            
        with torch.set_grad_enabled(self.enable_grad):
            embeddings = self.model(
                input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False
            )["hidden_states"][-1]

        # Cast embeddings to same type as proj_out, unless proj_out is Identity
        if not isinstance(self.proj_out, nn.Identity):
            proj_out_dtype = next(self.proj_out.parameters()).dtype
            embeddings = embeddings.to(proj_out_dtype)

        embeddings = self.norm(embeddings)

        if self.learned_scale:
            embeddings = embeddings * self.scale

        embeddings = self.proj_out(embeddings)
        embeddings = self.apply_padding(embeddings, attention_mask)

        return embeddings, attention_mask

class PhonemeConditioner(Conditioner):
    """
    A conditioner that turns text into phonemes and embeds them using a lookup table
    Only works for English text

    Args:
        output_dim: the dimension of the output embeddings
        max_length: the maximum number of phonemes to embed
        project_out: whether to add another linear projection to the output embeddings
    """

    def __init__(
            self,
            output_dim: int,
            max_length: int = 1024,
            project_out: bool = False,
    ):
        super().__init__(output_dim, output_dim, project_out=project_out)
        
        from g2p_en import G2p

        self.max_length = max_length

        self.g2p = G2p()

        # Reserving 0 for padding, 1 for ignored
        self.phoneme_embedder = nn.Embedding(len(self.g2p.phonemes) + 2, output_dim)

    def forward(self, texts: tp.List[str], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        
        self.phoneme_embedder.to(device)
        self.proj_out.to(device)

        batch_phonemes = [self.g2p(text) for text in texts] # shape [batch_size, length]
        
        phoneme_ignore = [" ", *string.punctuation]

        # Remove ignored phonemes and cut to max length
        batch_phonemes = [[p if p not in phoneme_ignore else "_" for p in phonemes] for phonemes in batch_phonemes]

        # Convert to ids
        phoneme_ids = [[self.g2p.p2idx[p] + 2 if p in self.g2p.p2idx else 1 for p in phonemes] for phonemes in batch_phonemes]

        #Pad to match longest and make a mask tensor for the padding
        longest = max([len(ids) for ids in phoneme_ids])
        phoneme_ids = [ids + [0] * (longest - len(ids)) for ids in phoneme_ids]
        
        phoneme_ids = torch.tensor(phoneme_ids).to(device)

        # Convert to embeddings
        phoneme_embeds = self.phoneme_embedder(phoneme_ids)
        
        phoneme_embeds = self.proj_out(phoneme_embeds)

        return phoneme_embeds, torch.ones(phoneme_embeds.shape[0], phoneme_embeds.shape[1]).to(device)
  
class TokenizerLUTConditioner(Conditioner):
    """
    A conditioner that embeds text using a lookup table on a pretrained tokenizer's vocabulary

    Args:
        tokenizer_name: the name of the tokenizer from the Hugging Face transformers library
        output_dim: the dimension of the output embeddings
        max_length: the maximum length of the text to embed
        project_out: whether to add another linear projection to the output embeddings
    """

    def __init__(
            self,
            tokenizer_name: str, # Name of a tokenizer from the Hugging Face transformers library
            output_dim: int,
            max_length: int = 1024,
            use_abs_pos_emb = False,
            project_out: bool = False,
            special_tokens: tp.List[str] = []
    ):
        super().__init__(output_dim, output_dim, project_out=project_out)

        from transformers import AutoTokenizer

        # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)

        # Also suppress transformers-specific logging and progress bars
        transformers_log_level = None
        transformers_disable_progress_bar = None
        try:
            import transformers
            transformers_log_level = transformers.logging.get_verbosity()
            transformers.logging.set_verbosity_error()
            # Disable progress bars
            try:
                from transformers.utils import is_progress_bar_enabled
                transformers_disable_progress_bar = not is_progress_bar_enabled()
                transformers.utils.logging.disable_progress_bar()
            except (ImportError, AttributeError) as e:
                # Progress bar control not available in this transformers version
                logging.debug(f"Could not disable transformers progress bar: {e}")
        except (ImportError, AttributeError) as e:
            # Transformers not available or version mismatch
            logging.debug(f"Could not configure transformers logging: {e}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

            finally:
                logging.disable(previous_level)
                if transformers_log_level is not None:
                    try:
                        transformers.logging.set_verbosity(transformers_log_level)
                    except (AttributeError, Exception) as e:
                        logging.debug(f"Could not restore transformers log level: {e}")
                if transformers_disable_progress_bar is not None and not transformers_disable_progress_bar:
                    try:
                        transformers.utils.logging.enable_progress_bar()
                    except (AttributeError, Exception) as e:
                        logging.debug(f"Could not re-enable transformers progress bar: {e}")

        # Add special tokens
        if len(special_tokens) > 0:
            self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

        self.max_length = max_length

        self.token_embedder = nn.Embedding(len(self.tokenizer), output_dim)

        self.abs_pos_emb = None

        if use_abs_pos_emb:
            self.abs_pos_emb = AbsolutePositionalEmbedding(output_dim, max_length)

    def forward(self, texts: tp.List[str], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        self.proj_out.to(device)

        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device).to(torch.bool)
    
        embeddings = self.token_embedder(input_ids)
            
        embeddings = self.proj_out(embeddings)

        embeddings = embeddings * attention_mask.unsqueeze(-1).float()

        if self.abs_pos_emb is not None:
            embeddings = embeddings + self.abs_pos_emb(embeddings)

        return embeddings, attention_mask

class PretransformConditioner(Conditioner):
    """
    A conditioner that uses a pretransform's encoder for conditioning

    Args:
        pretransform: an instantiated pretransform to use for conditioning
        output_dim: the dimension of the output embeddings
    """
    def __init__(self, pretransform: Pretransform, output_dim: int, save_pretransform: bool = False):
        super().__init__(pretransform.encoded_channels, output_dim)


        if not save_pretransform:
            self.__dict__["pretransform"] = pretransform
        else:
            self.pretransform = pretransform
        

    def forward(self, audio: tp.Union[torch.Tensor, tp.List[torch.Tensor], tp.Tuple[torch.Tensor]], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:

        self.pretransform.to(device)
        self.proj_out.to(device)

        if isinstance(audio, list) or isinstance(audio, tuple):
            audio = torch.stack(audio, dim=0)

        # Add batch dimension if needed
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)

        # Convert audio to pretransform input channels
        audio = set_audio_channels(audio, self.pretransform.io_channels)

        audio = audio.to(device)
        
        latents = self.pretransform.encode(audio)

        latents = self.proj_out(latents)

        return [latents, torch.ones(latents.shape[0], latents.shape[2]).to(latents.device)]

class SourceMixConditioner(Conditioner):
    """
    A conditioner that mixes projected audio embeddings from multiple sources

    Args:
        pretransform: an instantiated pretransform to use for conditioning
        output_dim: the dimension of the output embeddings
        source_keys: a list of keys for the potential sources in the metadata

    """
    def __init__(
        self, 
        pretransform: Pretransform, 
        output_dim: int, 
        save_pretransform: bool = False, 
        source_keys: tp.List[str] = [], 
        pre_encoded: bool = False, 
        allow_null_source=False,
        source_length=None
    ):
        super().__init__(pretransform.encoded_channels, output_dim)

        if not save_pretransform:
            self.__dict__["pretransform"] = pretransform
        else:
            self.pretransform = pretransform

        self.source_keys = source_keys

        self.source_heads = nn.ModuleList([nn.Conv1d(pretransform.encoded_channels, output_dim, kernel_size=1) for _ in source_keys])        

        self.pre_encoded = pre_encoded

        self.allow_null_source = allow_null_source

        if self.allow_null_source:
            self.null_source = nn.Parameter(torch.randn(output_dim, 1))

            assert source_length is not None, "Source length must be specified if allowing null sources"

            self.source_length = source_length

    def forward(self, sources: tp.List[tp.Dict[str, torch.Tensor]], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:

        self.pretransform.to(device)
        self.proj_out.to(device)

        dtype = next(self.proj_out.parameters()).dtype

        # Output has to be the batch of summed projections
        # Input is per-batch-item list of source audio

        mixes = []

        num_null_sources = 0
        for source_dict in sources: # Iterate over batch items

            mix = None

            for key_ix, key in enumerate(self.source_keys): # Iterate over potential sources
                if key in source_dict:

                    source = source_dict[key]

                    if not self.pre_encoded:
                        assert source.dim() == 2, f"Source audio must be shape [channels, samples], got shape: {source.shape}"
                        audio = set_audio_channels(source.unsqueeze(0), self.pretransform.io_channels)

                        audio = audio.to(device)
                        latents = self.pretransform.encode(audio).squeeze(0)
                    else:
                        latents = source.to(device)           

                    latents = latents.to(dtype)

                    if mix is None:
                        mix = self.source_heads[key_ix](latents)
                    else:
                        mix += self.source_heads[key_ix](latents)
            
            if mix is not None:
                mixes.append(mix)
            else:
                if self.allow_null_source:
                    mixes.append(self.null_source.repeat(1, self.source_length))
                else:
                    raise ValueError("No sources found for mix")

        mixes = torch.stack(mixes, dim=0)

        return [mixes, torch.ones(mixes.shape[0], mixes.shape[2]).to(mixes.device)]


class MultiConditioner(nn.Module):
    """
    A module that applies multiple conditioners to an input dictionary based on the keys

    Args:
        conditioners: a dictionary of conditioners with keys corresponding to the keys of the conditioning input dictionary (e.g. "prompt")
        default_keys: a dictionary of default keys to use if the key is not in the input dictionary (e.g. {"prompt_t5": "prompt"})
    """
    def __init__(self, conditioners: tp.Dict[str, Conditioner], default_keys: tp.Dict[str, str] = {}, pre_encoded_keys: tp.List[str] = []):
        super().__init__()

        self.conditioners = nn.ModuleDict(conditioners)
        self.default_keys = default_keys
        self.pre_encoded_keys = pre_encoded_keys

    def forward(self, batch_metadata: tp.List[tp.Dict[str, tp.Any]], device: tp.Union[torch.device, str]) -> tp.Dict[str, tp.Any]:
        output = {}

        for key, conditioner in self.conditioners.items():
            condition_key = key

            conditioner_inputs = []

            for x in batch_metadata:

                if condition_key not in x:
                    if condition_key in self.default_keys:
                        condition_key = self.default_keys[condition_key]
                    else:
                        raise ValueError(f"Conditioner key {condition_key} not found in batch metadata")

                #Unwrap the condition info if it's a single-element list or tuple, this is to support collation functions that wrap everything in a list
                if isinstance(x[condition_key], list) or isinstance(x[condition_key], tuple) and len(x[condition_key]) == 1:
                    conditioner_input = x[condition_key][0]
                    
                else:
                    conditioner_input = x[condition_key]

                conditioner_inputs.append(conditioner_input)

            if key in self.pre_encoded_keys:
                output[key] = [torch.stack(conditioner_inputs, dim=0).to(device), None]
            else:
                output[key] = conditioner(conditioner_inputs, device)

        return output
    
def create_multi_conditioner_from_conditioning_config(config: tp.Dict[str, tp.Any], pretransform=None) -> MultiConditioner:
    """
    Create a MultiConditioner from a conditioning config dictionary

    Args:
        config: the conditioning config dictionary
        device: the device to put the conditioners on
    """
    conditioners = {}
    cond_dim = config["cond_dim"]
    
    default_keys = config.get("default_keys", {})

    pre_encoded_keys = config.get("pre_encoded_keys", [])

    for conditioner_info in config["configs"]:
        id = conditioner_info["id"]

        conditioner_type = conditioner_info["type"]

        conditioner_config = {"output_dim": cond_dim}
        
        conditioner_config.update(conditioner_info["config"])

        if conditioner_type == "t5":
            conditioners[id] = T5Conditioner(**conditioner_config)
        elif conditioner_type == "t5gemma":
            conditioners[id] = T5GemmaConditioner(**conditioner_config)
        elif conditioner_type == "causal_lm":
            conditioners[id] = CausalLMConditioner(**conditioner_config)
        elif conditioner_type == "clap_text":
            conditioners[id] = CLAPTextConditioner(**conditioner_config)
        elif conditioner_type == "clap_audio":
            conditioners[id] = CLAPAudioConditioner(**conditioner_config)
        elif conditioner_type == "int":
            conditioners[id] = IntConditioner(**conditioner_config)
        elif conditioner_type == "number":
            conditioners[id] = NumberConditioner(**conditioner_config)
        elif conditioner_type == "list":
            conditioners[id] = ListConditioner(**conditioner_config)
        elif conditioner_type == "phoneme":
            conditioners[id] = PhonemeConditioner(**conditioner_config)
        elif conditioner_type == "lut":
            conditioners[id] = TokenizerLUTConditioner(**conditioner_config)
        elif conditioner_type == "beat_this":
            conditioners[id] = BeatThisConditioner(**conditioner_config)
        elif conditioner_type == "content_vec":
            conditioners[id] = ContentVecConditioner(**conditioner_config)
        elif conditioner_type == "mir":
            conditioners[id] = MIRConditioner(**conditioner_config)
        elif conditioner_type == "sat_clap_text":
            from .clap import create_clap_from_config

            use_model_pretransform = conditioner_config.pop("use_model_pretransform", False)

            clap_model = create_clap_from_config(conditioner_config, pretransform=pretransform if use_model_pretransform else None)

            clap_ckpt_path = conditioner_config.get("ckpt_path", None)

            if clap_ckpt_path is not None:
                copy_state_dict(clap_model, load_ckpt_state_dict(clap_ckpt_path))

                # Ensure that loading the checkpoint doesn't overwrite the model's pretransform
                if use_model_pretransform:
                    clap_model.pretransform = pretransform

            conditioners[id] = SATCLAPTextConditioner(clap_model, **conditioner_config)

        elif conditioner_type == "sat_clap_audio":
            from .clap import create_clap_from_config

            sample_rate = conditioner_config.get("sample_rate", None)
            assert sample_rate is not None, "Sample rate must be specified for SAT-CLAP conditioners"

            use_model_pretransform = conditioner_config.pop("use_model_pretransform", False)

            clap_model = create_clap_from_config(conditioner_config, pretransform=pretransform if use_model_pretransform else None)

            clap_ckpt_path = conditioner_config.get("ckpt_path", None)

            if clap_ckpt_path is not None:
                copy_state_dict(clap_model, load_ckpt_state_dict(clap_ckpt_path))

                # Ensure that loading the checkpoint doesn't overwrite the model's pretransform
                if use_model_pretransform:
                    clap_model.pretransform = pretransform

            conditioners[id] = SATCLAPAudioConditioner(clap_model, **conditioner_config)

        elif conditioner_type == "pretransform":
            sample_rate = conditioner_config.pop("sample_rate", None)
            assert sample_rate is not None, "Sample rate must be specified for pretransform conditioners"

            use_model_pretransform = conditioner_config.pop("use_model_pretransform", False)

            if not use_model_pretransform:
                cond_pretransform = create_pretransform_from_config(conditioner_config.pop("pretransform_config"), sample_rate=sample_rate)
            else:
                assert pretransform is not None, "Model pretransform must be specified for pretransform conditioners"
                cond_pretransform = pretransform

            if conditioner_config.get("pretransform_ckpt_path", None) is not None:
                cond_pretransform.load_state_dict(load_ckpt_state_dict(conditioner_config.pop("pretransform_ckpt_path")))

            conditioners[id] = PretransformConditioner(cond_pretransform, **conditioner_config)
        elif conditioner_type == "source_mix":
            sample_rate = conditioner_config.pop("sample_rate", None)
            assert sample_rate is not None, "Sample rate must be specified for source_mix conditioners"

            use_model_pretransform = conditioner_config.pop("use_model_pretransform", False)

            if not use_model_pretransform:
                cond_pretransform = create_pretransform_from_config(conditioner_config.pop("pretransform_config"), sample_rate=sample_rate)
            else:
                assert pretransform is not None, "Model pretransform must be specified for source_mix conditioners if use_model_pretransform is True"
                cond_pretransform = pretransform

            if conditioner_config.get("pretransform_ckpt_path", None) is not None:
                cond_pretransform.load_state_dict(load_ckpt_state_dict(conditioner_config.pop("pretransform_ckpt_path")))

            conditioners[id] = SourceMixConditioner(cond_pretransform, **conditioner_config)
        else:
            raise ValueError(f"Unknown conditioner type: {conditioner_type}")

    return MultiConditioner(conditioners, default_keys=default_keys, pre_encoded_keys=pre_encoded_keys)