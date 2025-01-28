# Gemini Live Audio Web Interface

This project provides a web interface for Gemini's audio responses, allowing you to interact with the Gemini AI model through voice and share the audio output through your browser (useful for screen sharing and video calls).

## Prerequisites

- Python 3.11 or higher
- A Google API key for Gemini
- A modern web browser
- pip (Python package installer)
- (Optional) Google Cloud service account for calendar scheduling

## Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory and add your Google API key:
```bash
echo "GOOGLE_API_KEY=your-api-key-here" > .env
```

5. (Optional) To enable calendar scheduling:
   - Create a service account in Google Cloud Console
   - Download the service account key as `service_account.json` and place it in the project root
   - Share your Google Calendar with the service account email (give it edit permissions)
   - The service account email will look like: `your-name@your-project.iam.gserviceaccount.com`

## Running Options

### Option 1: Web Interface (for sharing audio in calls)

You need to run two components:

1. Start the WebSocket server (handles audio streaming):
```bash
python websocket_server.py
```

2. In a new terminal, start the Flask web server:
```bash
python app.py
```

3. Open your web browser and navigate to:
```
http://localhost:5000
```

4. Click the "Start Audio" button to begin streaming.

5. In your terminal where `websocket_server.py` is running, type your messages after the "message >" prompt to interact with Gemini.

### Option 2: Direct Terminal Usage (with interruptions support)

If you want to use Gemini Live directly in the terminal with support for interruptions:

1. In `gemini_live.py`, uncomment the line `# tg.create_task(self.play_audio())` in the `run()` method.

2. Run the script directly:
```bash
python gemini_live.py --mode=none
```

This mode will play audio directly through your system's audio output and supports interrupting Gemini while it's speaking.

## Features

### Audio Interaction
- Real-time voice responses from Gemini
- Support for interrupting Gemini while speaking
- Audio sharing through browser for video calls

### Calendar Scheduling
If you've set up the service account:
- Ask Gemini to schedule meetings on your calendar
- Supports natural language requests like "Schedule a meeting with John tomorrow at 2 PM"
- Automatically creates Google Calendar events

## Sharing Audio in Video Calls

To share the audio in video calls (e.g., Google Meet):

1. When sharing your screen, make sure to select the browser tab with the Gemini Live interface
2. Check the "Share audio" option in your screen sharing settings
3. The audio from Gemini will now be shared through your browser

## Important Notes

- When using the web interface, the audio is played through your browser, not directly through your system's audio output
- Make sure your browser's audio permissions are enabled for the site
- For best results, use headphones to prevent audio feedback
- If you don't hear any audio, check that:
  - Your browser's audio is not muted
  - The correct audio output device is selected
  - You've clicked the "Start Audio" button
  - Both servers (WebSocket and Flask) are running

## Troubleshooting

If you encounter issues:

1. Check the terminal outputs for both servers for any error messages
2. Make sure your GOOGLE_API_KEY is correctly set in the .env file
3. Check your browser's console for any JavaScript errors
4. Try refreshing the page and restarting both servers
5. Ensure all required packages are installed correctly

For calendar scheduling issues:
- Verify that `service_account.json` is present in the project root
- Check that you've shared your calendar with the service account email
- Ensure the service account has edit permissions on your calendar

## Files Overview

- `websocket_server.py`: Handles real-time audio streaming via WebSocket
- `app.py`: Flask server for serving the web interface
- `gemini_live.py`: Main Gemini interaction logic
- `templates/index.html`: Web interface with audio playback functionality
- `service_account.json`: (Optional) Google Cloud service account credentials for calendar access 