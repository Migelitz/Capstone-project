import streamlit as st
from streamlit_chat import message
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

@st.cache(allow_output_mutation=True)
def cached_model():
    model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    return model

@st.cache(allow_output_mutation=True)
def get_dataset():
    df = pd.read_csv('AI_embedding_with_vectors.csv')
    df['embedding'] = df['user_embedding'].apply(json.loads)
    return df

model = cached_model()
df = get_dataset()

st.header('Multilingual Counseling Chatbot')
st.markdown("[❤️ Inspired by 빵형의 개발도상국](https://www.youtube.com/c/빵형의개발도상국)")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

with st.form('form', clear_on_submit=True):
    user_input = st.text_input('You: ', '')  # Multilingual input
    submitted = st.form_submit_button('Send')

if submitted and user_input:
    # Generate embedding for user input
    embedding = model.encode(user_input)

    # Compute cosine similarity with dataset
    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]

    # Save conversation history
    st.session_state.past.append(user_input)
    st.session_state.generated.append(answer['Chatbot'])  # Ensure dataset column name matches

for i in range(len(st.session_state['past'])):
    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
    if len(st.session_state['generated']) > i:
        message(st.session_state['generated'][i], key=str(i) + '_bot')
