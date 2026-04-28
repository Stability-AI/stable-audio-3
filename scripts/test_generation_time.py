from stable_audio_3 import StableAudioPipeline
import time
import statistics
import torch

pipe = StableAudioPipeline.from_pretrained("small", "mps")

# Warm-up (not timed)
for _ in range(3):
    pipe.generate(prompt="120 BPM house loop", duration=30, steps=8)

lengths = [10, 30, 60, 120]
for l in lengths:
    times = []
    for i in range(10):
        start = time.perf_counter()
        audio = pipe.generate(
            prompt="120 BPM house loop",
            negative_prompt="poor quality",
            duration=l,
            steps=8,  # default
            cfg_scale=1,  # default
            seed=-1,  # default
            batch_size=1,  # default
        )
        torch.mps.synchronize() if torch.backends.mps.is_available() else None
        times.append(time.perf_counter() - start)
    print(f"\nDuration: {l} seconds")
    print(f"Mean:   {statistics.mean(times):.3f}s")
    print(f"Median: {statistics.median(times):.3f}s")
    print(f"StdDev: {statistics.stdev(times):.3f}s")
    print(f"Min:    {min(times):.3f}s")
    print(f"Max:    {max(times):.3f}s")
