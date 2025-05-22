import runpod
import asyncio
from pyannote_ai import Pipeline
from concurrent.futures import ProcessPoolExecutor
import os

# Process‐pool to dedicate its own GIL
_executor = ProcessPoolExecutor()

# Load the Pipeline once per container (model download + init is expensive!)
batch_size = int(os.environ.get("BATCH_SIZE", 2))
_pipeline = Pipeline("pyannote/speaker-diarization", batch_size=batch_size, debug=True)


def _blocking_diarize(path: str):
    # This entire call now runs in its own Python process (own GIL)
    print("Using existing pipeline object")
    print("Diarizing audio file")
    return _pipeline.diarize(path)


async def handler(event):
    path = event.get("input", {}).get("url")
    loop = asyncio.get_running_loop()

    try:
        # offload to child process
        result = await loop.run_in_executor(_executor, _blocking_diarize, path)
        print("Diarization complete:", result)
        return result

    except Exception as e:
        # log for debugging
        print(f"[ERROR] Inference failed for {path!r}: {e}")
        # return a dict with an "error" key so RunPod will treat it as a handled error
        return {"error": str(e)}


# Start the Serverless function when the script is run
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
