import logging
from googleapiclient.discovery import build
from datetime import datetime, timedelta
from google.oauth2.service_account import Credentials

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

SCOPES = [
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/calendar.readonly",
]

# Define tool schema for meeting scheduling
schedule_meeting_schema = {
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
        "required": ["start_time", "summary"],
    },
}


def schedule_meeting(start_time, summary, user_email="pablo@nider.ai", description=""):
    """Schedule a meeting on the user's calendar.

    Creates a Google Calendar event.

    Args:
        user_email (str): Email of the user scheduling the meeting
        start_time (str): ISO format datetime string for meeting start time
        summary (str): Meeting summary/title
        description (str, optional): Meeting description

    Returns:
        dict: The created Google Calendar event object, or error message if failed
    """
    try:
        creds = Credentials.from_service_account_file(
            "service_account.json", scopes=SCOPES
        )
        service = build("calendar", "v3", credentials=creds)

        start = datetime.fromisoformat(start_time)
        end = start + timedelta(hours=1)

        event = {
            "summary": summary,
            "description": description,
            "start": {"dateTime": start.isoformat(), "timeZone": "Europe/London"},
            "end": {"dateTime": end.isoformat(), "timeZone": "Europe/London"},
            "organizer": {"email": user_email},
            "creator": {"email": user_email},
        }

        event = service.events().insert(calendarId=user_email, body=event).execute()

        logging.info(f"Event created: {event.get('htmlLink')}")
        return event
    except Exception as e:
        error_msg = str(e)
        logging.error(f"Failed to schedule meeting: {error_msg}")
        return error_msg
