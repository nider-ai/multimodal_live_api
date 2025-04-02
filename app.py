from flask import Flask, render_template, Response
import asyncio
from gemini_live import AudioLoop, FORMAT, CHANNELS, RECEIVE_SAMPLE_RATE
import threading
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
audio_loop = None


def run_gemini():
    """Run Gemini Live in a separate thread"""
    global audio_loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    audio_loop = AudioLoop()
    logger.info("Starting Gemini Live...")
    loop.run_until_complete(audio_loop.run())


def init_gemini():
    global audio_loop
    if audio_loop is None:
        # Start Gemini Live in a separate thread
        thread = threading.Thread(target=run_gemini)
        thread.daemon = True
        thread.start()
        logger.info("Gemini Live thread started")


def audio_stream_generator():
    """Generator function to stream audio data"""
    init_gemini()  # Start Gemini Live
    logger.info("Audio stream generator started")

    last_chunk_time = None
    empty_count = 0

    while True:
        if audio_loop and audio_loop.audio_in_queue:
            try:
                data = audio_loop.audio_in_queue.get_nowait()
                if data:
                    logger.info(f"Sending audio chunk of size: {len(data)} bytes")
                    last_chunk_time = asyncio.get_event_loop().time()
                    empty_count = 0
                    yield data
                else:
                    # Empty data handling
                    empty_count += 1
                    yield b""
            except asyncio.QueueEmpty:
                # Check if we're at the end of a response (small gap)
                current_time = asyncio.get_event_loop().time()
                if last_chunk_time and (current_time - last_chunk_time < 1.0):
                    # We're possibly between chunks, add a small delay
                    yield b""
                elif empty_count > 30:  # Reset after too many empty chunks
                    last_chunk_time = None
                    empty_count = 0
                    yield b""
                else:
                    empty_count += 1
                    yield b""
        else:
            yield b""
            # Small delay to prevent CPU overuse
            time.sleep(0.01)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/audio-config")
def audio_config():
    return {"sampleRate": RECEIVE_SAMPLE_RATE, "channels": CHANNELS, "format": "pcm"}


@app.route("/audio")
def audio():
    logger.info("New audio stream connection")
    return Response(
        audio_stream_generator(),
        mimetype="audio/pcm",
        headers={
            "Content-Type": "audio/pcm",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)
