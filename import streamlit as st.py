import streamlit as st
import joblib

# Load model
model = joblib.load("fraud_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
c
# Bank official data
official_emails = [
    "alerts@sbi.co.in",
    "noreply@hdfcbank.com",
    "alerts@icicibank.com"
]

official_numbers = [
    "SBIINB",
    "HDFCBK", 
    "ICICIB"
]

st.title("🏦 AI-Based Bank Fraud Message Detection")

sender = st.text_input("Enter Sender Email / Number")
message = st.text_area("Enter Message")

if st.button("Check Fraud"):
    sender_fraud = False

    if "@" in sender:
        if sender.lower() not in official_emails:
            sender_fraud = True
    else:
        if sender.upper() not in official_numbers:
            sender_fraud = True

    msg_vector = vectorizer.transform([message])
    prediction = model.predict(msg_vector)[0]

    if sender_fraud or prediction == "fraud":
        st.error("❌ FRAUD MESSAGE DETECTED")
    else:
        st.success("✅ LEGITIMATE BANK MESSAGE")
