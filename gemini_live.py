# -*- coding: utf-8 -*-
# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
## Setup

To install the dependencies for this script, run:

``` 
pip install google-genai opencv-python pyaudio pillow mss
```

Before running this script, ensure the `GOOGLE_API_KEY` environment
variable is set to the api-key you obtained from Google AI Studio.

Important: **Use headphones**. This script uses the system default audio
input and output, which often won't include echo cancellation. So to prevent
the model from interrupting itself it is important that you use headphones. 

## Run

To run the script:

```
python live_api_starter.py
```

The script takes a video-mode flag `--mode`, this can be "camera", "screen", or "none".
The default is "camera". To share your screen run:

```
python live_api_starter.py --mode screen
```
"""

import asyncio
import base64
from datetime import datetime
import io
import os
import sys
import traceback

import cv2
import pyaudio
import PIL.Image
import mss

import argparse

from google import genai
from dotenv import load_dotenv
from google.genai import types

from scheduler import schedule_meeting

load_dotenv()

if sys.version_info < (3, 11, 0):
    import taskgroup, exceptiongroup

    asyncio.TaskGroup = taskgroup.TaskGroup
    asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "models/gemini-2.0-flash-exp"

DEFAULT_MODE = "none"


client = genai.Client(http_options={"api_version": "v1alpha"})

pya = pyaudio.PyAudio()


class AudioLoop:
    """
    A class that handles real-time audio, video, and text interaction with the Gemini model.

    This class manages audio input/output streams, video capture (camera or screen),
    and text communication with the Gemini model. It supports different video modes
    and handles the asynchronous communication between various components.

    Attributes:
        video_mode (str): The video capture mode ('camera', 'screen', or 'none')
        audio_in_queue (asyncio.Queue): Queue for incoming audio data
        out_queue (asyncio.Queue): Queue for outgoing data (audio/video/text)
        session: Active session with the Gemini model
    """

    def __init__(self, video_mode=DEFAULT_MODE):
        """
        Initialize the AudioLoop with specified video mode.

        Args:
            video_mode (str): The video capture mode to use ('camera', 'screen', or 'none')
        """
        self.video_mode = video_mode

        self.audio_in_queue = None
        self.out_queue = None

        self.session = None

        self.send_text_task = None
        self.receive_audio_task = None
        self.play_audio_task = None

    async def send_text(self):
        """Handle text input from the user and send it to the model."""
        while True:
            text = await asyncio.to_thread(
                input,
                "message > ",
            )
            if text.lower() == "q":
                break
            await self.session.send(input=text or ".", end_of_turn=True)

    def _get_frame(self, cap):
        """
        Capture and process a single frame from the camera.

        Args:
            cap: OpenCV video capture object

        Returns:
            dict: Processed frame data with mime type and base64 encoded image
        """
        # Read the frame
        ret, frame = cap.read()
        if not ret:
            return None

        # Convert BGR to RGB color space
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)
        img.thumbnail([1024, 1024])

        # Convert to JPEG format
        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        mime_type = "image/jpeg"
        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_frames(self):
        """Continuously capture frames from the camera and add them to the output queue."""
        cap = await asyncio.to_thread(cv2.VideoCapture, 0)

        while True:
            frame = await asyncio.to_thread(self._get_frame, cap)
            if frame is None:
                break

            await asyncio.sleep(1.0)
            await self.out_queue.put(frame)

        cap.release()

    def _get_screen(self):
        """
        Capture and process a screenshot of the screen.

        Returns:
            dict: Processed screenshot data with mime type and base64 encoded image
        """
        sct = mss.mss()
        monitor = sct.monitors[0]
        i = sct.grab(monitor)

        # Convert screenshot to JPEG format
        mime_type = "image/jpeg"
        image_bytes = mss.tools.to_png(i.rgb, i.size)
        img = PIL.Image.open(io.BytesIO(image_bytes))

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_screen(self):
        """Continuously capture screenshots and add them to the output queue."""
        while True:
            frame = await asyncio.to_thread(self._get_screen)
            if frame is None:
                break

            await asyncio.sleep(1.0)
            await self.out_queue.put(frame)

    async def send_realtime(self):
        """Send data from the output queue to the model in real-time."""
        while True:
            msg = await self.out_queue.get()
            await self.session.send(input=msg)

    async def listen_audio(self):
        """
        Capture audio from the microphone and add it to the output queue.
        Handles audio input stream setup and continuous reading.
        """
        # Setup audio input stream
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )

        # Configure overflow handling based on debug mode
        kwargs = {"exception_on_overflow": False} if __debug__ else {}

        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def receive_audio(self):
        """
        Handle incoming responses from the model, including audio, text,
        and tool calls.
        """
        while True:
            turn = self.session.receive()
            async for response in turn:
                if data := response.data:
                    self.audio_in_queue.put_nowait(data)
                    continue

                if text := response.text:
                    print(text, end="")
                    continue

                server_content = response.server_content
                if server_content is not None:
                    self.handle_server_content(server_content)
                    continue

                tool_call = response.tool_call
                if tool_call is not None:
                    await self.handle_tool_call(self.session, tool_call)

            # Clear audio queue on model interruption
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()

    async def play_audio(self):
        """Play received audio through the system's audio output."""
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)

    async def handle_tool_call(self, session, tool_call):
        """
        Handle tool calls from the model, specifically for scheduling meetings.

        Args:
            session: Active model session
            tool_call: Tool call object from the model
        """
        for fc in tool_call.function_calls:
            if fc.name == "schedule_meeting":
                meeting = schedule_meeting(
                    user_email=fc.args["user_email"],
                    summary=fc.args["summary"],
                    start_time=fc.args["start_time"],
                )
            tool_response = types.LiveClientToolResponse(
                function_responses=[
                    types.FunctionResponse(
                        name=fc.name,
                        id=fc.id,
                        response={"result": meeting},
                    )
                ]
            )

        print("\n>>> ", tool_response)
        await session.send(input=tool_response)

    def handle_server_content(self, server_content):
        model_turn = server_content.model_turn
        if model_turn:
            for part in model_turn.parts:
                executable_code = part.executable_code
                if executable_code is not None:
                    print("-------------------------------")
                    print(f"Python code:\n{executable_code.code}")
                    print("-------------------------------")

                code_execution_result = part.code_execution_result
                if code_execution_result is not None:
                    print("-------------------------------")
                    print(f"Output:\n{code_execution_result.output}")
                    print("-------------------------------")

        grounding_metadata = getattr(server_content, "grounding_metadata", None)
        if grounding_metadata is not None:
            print("-------------------------------")
            print(grounding_metadata.search_entry_point.rendered_content)
            print("-------------------------------")

        return

    async def run(self):
        """
        Main execution method that sets up and runs all the async tasks
        for handling audio, video, and model interaction.
        """
        # Define tool schema for meeting scheduling
        schedule_meeting = {
            "name": "schedule_meeting",
            "description": """Schedule a meeting on the user's calendar. 
            Creates a Google Calendar event.""",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "user_email": {
                        "type": "STRING",
                        "description": "Email of the user scheduling the meeting",
                    },
                    "start_time": {
                        "type": "STRING",
                        "description": """ISO format datetime string for meeting 
                        start time. Include timezone, which is London.""",
                    },
                    "summary": {
                        "type": "STRING",
                        "description": "Meeting summary/title",
                    },
                    "description": {
                        "type": "STRING",
                        "description": "Meeting description",
                    },
                },
                "required": ["user_email", "start_time", "summary"],
            },
        }

        # Configure model tools and response settings
        tools = [
            {
                "function_declarations": [schedule_meeting],
                "google_search": {},
                "code_execution": {},
            }
        ]
        current_date = datetime.now().strftime("%Y-%m-%d")
        system_instruction = f"You are a helpful assistant that can schedule meetings on the user's calendar, execute code and search the web. Today is {current_date}."
        CONFIG = {
            "system_instruction": system_instruction,
            "tools": tools,
            "generation_config": {"response_modalities": ["AUDIO"]},
        }

        try:
            # Set up main session and task group
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

                # Create all necessary async tasks
                send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())

                # Initialize video tasks based on mode
                if self.video_mode == "camera":
                    tg.create_task(self.get_frames())
                elif self.video_mode == "screen":
                    tg.create_task(self.get_screen())

                tg.create_task(self.receive_audio())
                # Uncomment to play audio natively without frontend
                # tg.create_task(self.play_audio())

                await send_text_task
                raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            pass
        except ExceptionGroup as EG:
            self.audio_stream.close()
            traceback.print_exception(EG)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        help="pixels to stream from",
        choices=["camera", "screen", "none"],
    )
    args = parser.parse_args()
    main = AudioLoop(video_mode=args.mode)
    asyncio.run(main.run())
