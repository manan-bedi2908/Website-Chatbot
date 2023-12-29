import streamlit as st

def main():
    st.set_page_config(page_title = 'Chatbot for our Own Website', 
                       page_icon = ':chatbot:')
    st.header('Chat with your Own Website')
    with st.sidebar:
        st.title('LLM Chatapp using LangChain')
        st.markdown('''
        This App is an LLM powered Chatbot built using:
                    - [Streamlit](https://streamlit.io)
                    - [OpenAI](https://platform.openai.com/docs/models) LLM
                    - [LangChain](https://python.langchain.com/)
        ''')
        

if __name__ == '__main__':
    main()