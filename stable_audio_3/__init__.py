from .models.factory import create_model_from_config, create_model_from_config_path
from stable_audio_3.pipeline import StableAudioPipeline

class StableAudio:
    @staticmethod
    def generate(prompt, duration=30, **kwargs):
        pipe = StableAudioPipeline.from_pretrained(...)
        return pipe.generate(prompt, duration=duration, **kwargs)