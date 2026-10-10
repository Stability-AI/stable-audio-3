"""Fixed shapes of the SA3-medium DiT's inputs. Used by the verification harnesses."""

SAMPLE_RATE = 44100
SAMPLES_PER_LATENT = 4096        # one latent frame = 4096 audio samples
IO_CHANNELS = 256                # latent channels in/out of the DiT
T5_TOKENS = 256                  # T5Gemma sequence length
T5_HIDDEN_DIM = 768
LOCAL_ADD_COND_DIM = 257         # per-frame additive conditioning width
