import os
import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model  # Import apply_model
from pathlib import Path
import requests
from torch.amp import autocast  # Updated import

# Define the raw URL for the MP3 file and the local filename
url = 'https://raw.githubusercontent.com/markbmullins/transcriber/main/song.mp3'
local_audio_path = 'song.mp3'

# Check if the audio file already exists
if not os.path.isfile(local_audio_path):
    print(f"Downloading '{local_audio_path}' from '{url}'...")
    response = requests.get(url)
    with open(local_audio_path, 'wb') as f:
        f.write(response.content)
else:
    print(f"File '{local_audio_path}' already exists. Skipping download.")

# Load the audio file using torchaudio
waveform, sample_rate = torchaudio.load(local_audio_path)
waveform = waveform.to(torch.float32)

# Check for CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print("Using CPU")

# Initialize the model (e.g., htdemucs_ft for better drums separation)
model = get_model('htdemucs_ft')
model.to(device)

# Ensure waveform is on the same device as the model
waveform = waveform.to(device)

# Normalize the waveform
waveform = (waveform - waveform.mean()) / waveform.std()

# Prepare the mixture tensor by adding a batch dimension
mix = waveform.unsqueeze(0)  # Shape: [batch_size=1, channels, samples]

print("Separating track...")

with torch.no_grad():
    if device.type == 'cuda':
        with autocast('cuda'):
            estimates = apply_model(
                model, mix, device=device, split=True, overlap=0.1
            )
    else:
        estimates = apply_model(
            model, mix, device=device, split=True, overlap=0.1
        )

# Remove batch dimension
sources = estimates[0]  # Shape: [num_sources, channels, samples]

# Denormalize the sources
sources = sources * waveform.std() + waveform.mean()

# Map source indices to source names
source_indices = {i: name for i, name in enumerate(model.sources)}

# Save the separated sources
print("Saving sources...")
output_dir = Path('./output/song')
output_dir.mkdir(parents=True, exist_ok=True)

# Apply post-processing to drums
import torchaudio.functional as F

drums_index = [i for i, name in source_indices.items() if name == 'drums'][0]
drums_audio = sources[drums_index]

# Apply high-pass filter to drums
drums_audio = F.highpass_biquad(drums_audio, sample_rate, cutoff_freq=100)

# Replace the drums source with the processed audio
sources[drums_index] = drums_audio

# Save each source
for i in range(sources.shape[0]):
    source_name = source_indices[i]
    source_audio = sources[i]
    output_path = output_dir / f'{source_name}.wav'
    torchaudio.save(str(output_path), source_audio.cpu(), sample_rate)
    print(f"Saved {source_name} to {output_path}")

print("Stems have been separated and saved in the './output/song/' directory.")
