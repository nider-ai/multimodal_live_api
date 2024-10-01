from main import run_graph
import streamlit as st
import uuid
from dotenv import load_dotenv

load_dotenv()


def main():
    # Initialize Streamlit app
    st.title("LangGraph Chatbot")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    # Add welcome message if chat history is empty
    if not st.session_state.chat_history:
        welcome_message = """
         Welcome!

        I'm here to assist you with:

        - Flight policies for Swiss Airlines.


        **How can I help you today?**
        """
        st.session_state.chat_history.append(("assistant", welcome_message))

    # Display chat messages from history on app rerun
    for speaker, message in st.session_state.chat_history:
        with st.chat_message(speaker):
            st.markdown(message)

    # Accept user input
    if user_input := st.chat_input("You: "):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(user_input)
        # Add user message to chat history
        st.session_state.chat_history.append(("user", user_input))

        response = run_graph(user_input, st.session_state.thread_id)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.chat_history.append(("assistant", response))


if __name__ == "__main__":
    main()
