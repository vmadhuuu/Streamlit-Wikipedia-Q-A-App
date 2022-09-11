import torch
import wikipedia
import transformers
import streamlit as st
from transformers import pipeline, Pipeline

def load_qa_pipeline() -> Pipeline:
    qa_pipeline = pipeline("question-answering", model = "distilbert-base-uncased-distilled-squad")
    return qa_pipeline


def load_wiki_summary(query: str) -> str:
    results = wikipedia.search(query)
    summary = wikipedia.summary(results[0], sentences = 10)
    return summary

def answer_question(pipeline: Pipeline, question: str, paragraph: str) -> dict:
    input = {
        "question": question,
        "context": paragraph
    }
    output = pipeline(input)
    return output

# Main app engine
if __name__ == "__main__":
    # display title and description
    st.title("Wikipedia Question Anwering")
    st.write("Search your topic, Ask your questions, Get your answers. ")

    
    
    # Text box for topic 
    topic = st.text_input("SEARCH TOPIC","")

    # Display article paragraph
    article_para = st.empty()

    # Text box for answer 
    question = st.text_input("QUESTION","")

    # Display question


    if topic:

        # load wikipedia summary
        summary = load_wiki_summary(topic)

        # Display article summary in paragraph
        article_para.markdown(summary)


    # perform question answering
        if question != "":
            qa_pipeline = load_qa_pipeline()

            result = answer_question(qa_pipeline, question, summary)
            answer = result["answer"]

            st.write(answer)