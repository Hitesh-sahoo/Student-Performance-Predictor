import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import io

# Load trained model
model = joblib.load("student_model.pkl")

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓")
st.title("🎓 Student Performance Predictor")
st.write("Enter student details to predict the final score and get personalized tips!")

# Sidebar for inputs
st.sidebar.header("Student Details")
hours = st.sidebar.number_input("Hours Studied", 0, 10, 5)
attendance = st.sidebar.number_input("Attendance (%)", 0, 100, 75)
assignments = st.sidebar.number_input("Assignments Completed (%)", 0, 100, 80)

# Predict button
if st.button("Predict Final Score"):
    # Prepare input for model
    input_data = pd.DataFrame([[hours, attendance, assignments]],
                              columns=["hours_studied","attendance","assignments"])
    score = model.predict(input_data)[0]

    # Show score visually
    st.subheader("Predicted Score")
    st.success(f"{score:.2f} / 100")

    # Gauge chart using Plotly
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': "Score"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "green" if score>=70 else "orange" if score>=50 else "red"}}))
    st.plotly_chart(fig)

    # Feedback based on score
    if score < 50:
        st.error("Score Category: Poor 😞")
        st.warning("You will pass hardly… practice and study more! 📚")
    elif 50 <= score < 70:
        st.warning("Score Category: Average 😐")
        st.info("You will pass, keep practicing! ✨")
    elif 70 <= score < 90:
        st.success("Score Category: Good 🙂")
        st.success("You are a good student! 👍")
    else:
        st.balloons()
        st.success("Score Category: Excellent 🌟")
        st.success("Excellent! You are a very good student! 🌟")

    # Personalized tips
    st.subheader("📌 Study Tips")
    if hours < 4:
        st.info("Try studying at least 4-5 hours for better results.")
    if attendance < 70:
        st.info("Improve your attendance to understand topics better.")
    if assignments < 80:
        st.info("Complete more assignments to practice and reinforce learning.")

    # Option to download prediction
    output = io.StringIO()
    pd.DataFrame([[hours, attendance, assignments, score]],
                 columns=["Hours Studied", "Attendance (%)", "Assignments (%)", "Predicted Score"]).to_csv(output, index=False)
    st.download_button("Download Prediction as CSV", output.getvalue(), "prediction.csv")
