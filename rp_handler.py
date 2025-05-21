import runpod
import asyncio
from pyannote_ai import Pipeline
from concurrent.futures import ProcessPoolExecutor

# Process‐pool to dedicate its own GIL
_executor = ProcessPoolExecutor()

# Load the Pipeline once per container (model download + init is expensive!)
_pipeline = Pipeline("pyannote/speaker-diarization", batch_size=8, debug=True)

def _blocking_diarize(path: str):
    # This entire call now runs in its own Python process (own GIL)
    print("Using existing pipeline object")
    print("Diarizing audio file")
    return _pipeline.diarize(path)

async def _handler_async(event):
    # pull out the URL
    path = event.get("input", {}).get("url")

    # dispatch the blocking work into the process‐pool
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(_executor, _blocking_diarize, path)

    print("Diarization complete")
    print(result)
    return result

def handler(event):
    # wrap our async logic so runpod.serverless.start can call it synchronously
    return asyncio.run(_handler_async(event))

# Start the Serverless function when the script is run
if __name__ == '__main__':
    runpod.serverless.start({'handler': handler })