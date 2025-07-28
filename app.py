import streamlit as st
import json
import torch
import pickle
import random
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F

# Load model, tokenizer, label encoder, and intents
@st.cache_resource
def load_components():
    model = BertForSequenceClassification.from_pretrained("model")
    tokenizer = BertTokenizer.from_pretrained("model")

    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    with open("intents.json") as f:
        intents = json.load(f)["intents"]

    intent_responses = {intent["tag"]: intent["responses"] for intent in intents}
    return model, tokenizer, label_encoder, intent_responses

model, tokenizer, label_encoder, intent_responses = load_components()

# Prediction function
def predict_intent_with_response(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_class_id = torch.argmax(outputs.logits, dim=1).item()
    intent_tag = label_encoder.inverse_transform([predicted_class_id])[0]
    response = random.choice(intent_responses[intent_tag])
    return {
        "input_text": text,
        "predicted_intent": intent_tag,
        "response": response
    }

# Streamlit UI
st.set_page_config(page_title="Intent Chatbot", layout="centered")
st.title("🤖 BERT Intent Chatbot")
st.markdown("Ask a question and get a response based on your intent!")

# Store chat history in session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input form
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", "")
    submit_button = st.form_submit_button(label="Send")

if submit_button and user_input:
    result = predict_intent_with_response(user_input)
    st.session_state.chat_history.append(("🧍 You", user_input))
    st.session_state.chat_history.append(("🤖 Bot", result["response"]))

# Clear chat
if st.button("🧹 Clear Chat"):
    st.session_state.chat_history = []

# Show chat history
st.markdown("---")
for sender, message in st.session_state.chat_history[::-1]:
    st.markdown(f"**{sender}:** {message}")
