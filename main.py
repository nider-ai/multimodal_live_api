from agent import graph


def run_graph(message, session_id="0"):
    config = {
        "configurable": {
            "thread_id": session_id,
            "model_name": "gemini",
        }
    }
    events = graph.invoke(
        {"messages": ("user", message)},
        config=config,
        stream_mode="values",
    )
    messages = events.get("messages", [])
    response = messages[-1].content
    print(response)
    return response
