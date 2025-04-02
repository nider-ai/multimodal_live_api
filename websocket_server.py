import asyncio
import websockets
import logging
from gemini_live import AudioLoop
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

audio_loop = None


async def handle_websocket(websocket):
    global audio_loop
    logger.info("New WebSocket connection")

    try:
        if not audio_loop:
            audio_loop = AudioLoop()
            asyncio.create_task(audio_loop.run())
            logger.info("Started Gemini Live")

        last_data_time = None
        empty_count = 0

        while True:
            if audio_loop.audio_in_queue:
                try:
                    data = audio_loop.audio_in_queue.get_nowait()
                    if data:
                        # Convert binary PCM to base64
                        base64_data = base64.b64encode(data).decode("utf-8")
                        await websocket.send(base64_data)
                        logger.info(f"Sent audio chunk: {len(data)} bytes")
                        last_data_time = asyncio.get_event_loop().time()
                        empty_count = 0
                    else:
                        # Handle empty data
                        empty_count += 1
                        await asyncio.sleep(0.01)
                except asyncio.QueueEmpty:
                    # Check if we're possibly at the end of a response
                    current_time = asyncio.get_event_loop().time()
                    if last_data_time and (current_time - last_data_time < 1.0):
                        # Short pause, possibly between chunks
                        await asyncio.sleep(0.02)  # Slightly longer pause
                    else:
                        # Normal empty queue handling
                        await asyncio.sleep(0.01)
                        if empty_count > 50:  # Reset after too many empty reads
                            last_data_time = None
                            empty_count = 0
            else:
                await asyncio.sleep(0.01)

    except websockets.exceptions.ConnectionClosed:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"Error in WebSocket handler: {e}")


async def main():
    async with websockets.serve(handle_websocket, "localhost", 8765):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
