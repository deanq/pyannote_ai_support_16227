import modal

# Setup a custom image with your wheel and dependencies
stub = modal.Stub("pyannote-diarization")

image = modal.Image.debian_slim().apt_install(
    "ffmpeg", "curl", "libssl-dev", "python3-pip"
).copy_local_file("pyannote_ai-0.7.0-cp310-abi3-manylinux_2_28_x86_64.whl", "/root/pyannote_ai-0.7.0-cp310-abi3-manylinux_2_28_x86_64.whl").run_commands(
    "pip3 install torch torchaudio modal",
    "pip3 install /root/pyannote_ai-0.7.0-cp310-abi3-manylinux_2_28_x86_64.whl",
)

# Mount the .mp3 file to access it in the container
volume = modal.Mount.from_local_file("test_5h.mp3", remote_path="/root/test_5h.mp3")

@stub.function(image=image, mounts=[volume], timeout=600)
def run_diarization():
    from pyannote_ai import Pipeline

    path = "/root/test_5h.mp3"
    pipeline = Pipeline("test", batch_size=8)
    
    result = pipeline.diarize(path)
    print(result)
    return result
