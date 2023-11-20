import streamlit as st
import os
import openai
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.llms import Cohere
from langchain.document_loaders import TextLoader
from langchain.embeddings import CohereEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_KEY")


def process_long_text(long_text):
    # Save the long text to a file
    with open("input.txt", "w", encoding="utf-8") as file:
        file.write(long_text)

    loader = TextLoader("input.txt")
    documents = loader.load()

    # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=0,separators=[" ", ",", "\n"])

    texts = text_splitter.split_documents(documents)

    embeddings = CohereEmbeddings(
        cohere_api_key='U25sqdQV6D0w5OGJ7eS2VD0MSVyfAlKDC9KIWhe4')
    doc_search = Chroma.from_documents(texts, embeddings)
    llm = Cohere(cohere_api_key='U25sqdQV6D0w5OGJ7eS2VD0MSVyfAlKDC9KIWhe4')
    retriever = doc_search.as_retriever(search_kwargs={"k": 1})
    ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(llm, retriever)
    return qa


def suggest_tip_for_answer(answer):
    prompt = f"Give me tip on how to remember the following answer:{answer}"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=1200,
        temperature=0.7, )
    return response.choices[0].text.strip()


def question_answering_app():
    # Set up the Streamlit app
    st.title("Question Answering App")

    # Create input box for long text
    long_text = st.text_area("Enter the text", height=300)

    # Create a placeholder for the qa variable
    session_state = st.session_state
    if 'qa' not in session_state:
        session_state.qa = None

    # Button to convert long text to text file
    if st.button("Learn"):
        if session_state.qa is not None:
            session_state.qa = None
        session_state.qa = process_long_text(long_text)

    # Process user input when question is provided
    if session_state.qa is not None:
        if 'answer' not in session_state:
            session_state.answer = None
        if 'answer_generated' not in session_state:
            session_state.answer_generated = False

        # Create input box for asking questions
        question = st.text_input("Ask a question")
        button = st.button("Answer")

        if session_state.answer_generated is False and button:
            chat_history = []
            result = session_state.qa({"question": question, "chat_history": chat_history})

            answer = result['answer']
            chat_history = [(question, result["answer"])]
            # Display the answer
            st.write("Answer:", answer)
            session_state.answer = answer
            session_state.answer_generated = True
        elif button and question:
            chat_history = []
            result = session_state.qa({"question": question, "chat_history": chat_history})

            answer = result['answer']
            chat_history = [(question, result["answer"])]
            # Display the answer
            st.write("Answer:", answer)
            session_state.answer = answer
            session_state.answer_generated = True
        elif session_state.answer is not None:
            st.write("Answer:", session_state.answer)
        else:
            st.write("")
        # Add a button for tip
        if session_state.answer_generated is True:
            if st.button("Tip 💡"):
                tip = suggest_tip_for_answer(session_state.answer)
                st.text_area("Tip:-", tip, height=200)
                session_state.answer_generated = False

if __name__ == "__main__":
    question_answering_app()

