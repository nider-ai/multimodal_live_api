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

        while True:
            if audio_loop.audio_in_queue:
                try:
                    data = audio_loop.audio_in_queue.get_nowait()
                    if data:
                        # Convert binary PCM to base64
                        base64_data = base64.b64encode(data).decode("utf-8")
                        await websocket.send(base64_data)
                        logger.info(f"Sent audio chunk of size: {len(data)} bytes")
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.01)  # Small delay to prevent busy waiting
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
