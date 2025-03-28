import streamlit as st

import pandas as pd
import numpy as np

import joblib

pipe_lr = joblib.load(open("model/emotion_classifier_pipeline.pkl","rb"))

def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

emotions_emoji_dict = {"anger":"😠", "disgust":"🤮", "fear":"😨", "happy":"😁", "joy":"😂","neutral":"😐","sadness":"😭","shame":"😣","surprise":"🙀"}

def main():
    st.title("Sentiment Analyzer")
    menu=["Sentiment Analysis", "Text Generation", "Text Summarization"]
    choice = st.sidebar.selectbox("Menu",menu)

    if choice == "Sentiment Analysis":
        st.subheader("Emotion Detection")

        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label = 'Submit')

        if submit_text:
            col1,col2 =st.columns(2)

            # apply functions
            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)

            with col1:
                st.success("Original Text")
                st.write(raw_text)
                st.success("Prediction Probability")
                st.write(probability)

            with col2:
                st.success("Prediction")
                emoji_icon = emotions_emoji_dict[prediction]
                st.write("{}:{}".format(prediction,emoji_icon))

    elif choice == "Monitor":
        st.subheader("Monitor")
    
    else:
        st.subheader("About")


if __name__ == '__main__':
    main()