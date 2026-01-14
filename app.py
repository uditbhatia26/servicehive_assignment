import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import uuid
from chatbot import chatbot

# Page configuration
st.set_page_config(
    page_title="AutoStream AI Assistant",
    page_icon="🤖",
    layout="centered"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Header
st.title("🤖 AutoStream AI Assistant")
st.caption("Your intelligent companion for AutoStream platform inquiries")

# Sidebar with controls and info
with st.sidebar:
    st.header("Chat Controls")
    
    if st.button("New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()
    
    st.divider()
    
    st.header("Session Info")
    st.metric("Total Messages", len(st.session_state.messages))
    st.text(f"Thread ID: {st.session_state.thread_id[:8]}...")
    
    # Debug section
    if st.session_state.messages:
        with st.expander("View Current State"):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            try:
                state_snapshot = chatbot.get_state(config)
                st.json(state_snapshot.values if hasattr(state_snapshot, 'values') else {})
            except Exception as e:
                st.error(f"Could not retrieve state: {e}")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Create config with thread ID for memory
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                
                # Invoke the chatbot
                response = chatbot.invoke(
                    {"messages": [HumanMessage(content=prompt)]},
                    config=config
                )
                
                # Extract the assistant's response
                assistant_messages = [
                    msg for msg in response["messages"] 
                    if isinstance(msg, (AIMessage, SystemMessage))
                ]
                
                if assistant_messages:
                    bot_response = assistant_messages[-1].content
                else:
                    bot_response = "I'm sorry, I didn't generate a response. Please try again."
                
                # Display bot response
                st.write(bot_response)
                
                # Add to session state
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": bot_response
                })
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# Handle sample prompt clicks
if "temp_input" in st.session_state and st.session_state.temp_input:
    st.session_state.messages.append({
        "role": "user",
        "content": st.session_state.temp_input
    })
    
    with st.spinner("Thinking..."):
        try:
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            response = chatbot.invoke(
                {"messages": [HumanMessage(content=st.session_state.temp_input)]},
                config=config
            )
            
            assistant_messages = [
                msg for msg in response["messages"] 
                if isinstance(msg, (AIMessage, SystemMessage))
            ]
            
            if assistant_messages:
                bot_response = assistant_messages[-1].content
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": bot_response
                })
        except Exception as e:
            st.error(f"Error: {e}")
    
    del st.session_state.temp_input
    st.rerun()