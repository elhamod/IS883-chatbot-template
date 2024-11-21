import streamlit as st
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent, create_react_agent
from langchain import hub

# Show title and description.
st.title("💬 Chatbot")

### Important part.
# Create a session state variable to flag whether the app has been initialized.
# This code will only be run first time the app is loaded.
if "memory" not in st.session_state: ### IMPORTANT.
    model_type="gpt-4o-mini"

    # initialize the momory
    max_number_of_exchanges = 10
    st.session_state.memory = ConversationBufferWindowMemory(memory_key="chat_history", k=max_number_of_exchanges, return_messages=True) ### IMPORTANT to use st.session_state.memory.

    # LLM
    chat = ChatOpenAI(openai_api_key=st.secrets["OpenAI_API_KEY"], model=model_type)

    # tools
    tools = []
    
    # Now we add the memory object to the agent executor
    prompt = hub.pull("hwchase17/react-chat")
    agent = create_react_agent(chat, tools, prompt)
    st.session_state.agent_executor = AgentExecutor(agent=agent, tools=tools,  memory=st.session_state.memory, stream_runnable=False)  # , verbose= True ### IMPORTANT to use st.session_state.memory and st.session_state.agent_executor.

# Display the existing chat messages via `st.chat_message`.
for message in st.session_state.memory.buffer:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Create a chat input field to allow the user to enter a message. This will display
# automatically at the bottom of the page.
if prompt := st.chat_input("What is up?"):

    # Store and display the current prompt.
    # st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate a response using the OpenAI API.
    response = st.session_state.agent_executor.invoke({"input":prompt})['output']
    
    # Stream the response to the chat using `st.write_stream`, then store it in 
    # session state.
    with st.chat_message("assistant"):
        st.markdown(response)
        # response = st.write(response)
    # st.session_state.messages.append({"role": "assistant", "content": response})
