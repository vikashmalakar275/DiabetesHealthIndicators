import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load dataset
df = pd.read_csv("data/diabetes_012_health_indicators_BRFSS2015.csv")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Prediction", "Data Visualization"])

# ------------------ Data Visualization Page ------------------
if page == "Data Visualization":
    st.title("Diabetes Data Visualization")

    # 1. Distribution of target classes
    st.subheader("Distribution of Diabetes Classes")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.countplot(x="Diabetes_012", data=df, palette="Set2", ax=ax1)
    ax1.set_title("Distribution of Diabetes Classes")
    ax1.set_xlabel("Diabetes Class (0 = No, 1 = Pre, 2 = Yes)")
    ax1.set_ylabel("Count")
    st.pyplot(fig1)

    # 2. Correlation heatmap
    st.subheader("Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(16, 12))
    sns.heatmap(df.corr(), cmap="coolwarm", annot=False, linewidths=0.5, ax=ax2)
    ax2.set_title("Correlation Heatmap")
    st.pyplot(fig2)

# ------------------ Prediction Page ------------------
elif page == "Prediction":
    st.title("Diabetes Prediction App")

    # Section 1: Basic Information
    st.subheader("Basic Information")
    sex = st.radio("Sex", ["Male", "Female"])
    age = st.number_input("Enter your age:", min_value=0, max_value=120, step=1, value=25)
    # Height and Weight Inputs for BMI Calculation
    st.subheader("BMI Calculator")
    height_cm = st.number_input("Enter your height (in centimeters):", min_value=50, max_value=250, step=1, value=170)
    weight = st.number_input("Enter your weight (in kilograms):", min_value=10, max_value=300, step=1, value=70)
    
    # Convert height to meters
    height = height_cm / 100

    # BMI Calculation
    if height > 0:
        bmi = weight / (height ** 2)
        st.write(f"Your BMI is: {bmi:.2f}")
    else:
        st.write("Please enter a valid height.")

    # Map "Male" and "Female" to 1 and 0
    sex = 1 if sex == "Male" else 0

    # Section 2: Health Conditions
    st.subheader("Health Conditions")
    high_bp = st.radio("High Blood Pressure", ["Yes", "No"])
    high_chol = st.radio("High Cholesterol", ["Yes", "No"])
    chol_check = st.radio("Cholesterol Check in past 5 years", ["Yes", "No"])
    smoker = st.radio("Smoker", ["Yes", "No"])
    stroke = st.radio("Ever had a stroke?", ["Yes", "No"])
    heart_disease = st.radio("Heart Disease or Attack", ["Yes", "No"])
    phys_activity = st.radio("Physical Activity in last 30 days", ["Yes", "No"])
    fruits = st.radio("Consumes Fruit 1+ times/day", ["Yes", "No"])
    veggies = st.radio("Consumes Vegetables 1+ times/day", ["Yes", "No"])
    alcohol = st.radio("Heavy Alcohol Consumption", ["Yes", "No"])
    healthcare = st.radio("Has any kind of healthcare coverage", ["Yes", "No"])
    nodoc_cost = st.radio("Couldn’t see doctor due to cost", ["Yes", "No"])
    diffwalk = st.radio("Difficulty Walking", ["Yes", "No"])

    # Map "Yes" and "No" to 1 and 0
    high_bp = 1 if high_bp == "Yes" else 0
    high_chol = 1 if high_chol == "Yes" else 0
    chol_check = 1 if chol_check == "Yes" else 0
    smoker = 1 if smoker == "Yes" else 0
    stroke = 1 if stroke == "Yes" else 0
    heart_disease = 1 if heart_disease == "Yes" else 0
    phys_activity = 1 if phys_activity == "Yes" else 0
    fruits = 1 if fruits == "Yes" else 0
    veggies = 1 if veggies == "Yes" else 0
    alcohol = 1 if alcohol == "Yes" else 0
    healthcare = 1 if healthcare == "Yes" else 0
    nodoc_cost = 1 if nodoc_cost == "Yes" else 0
    diffwalk = 1 if diffwalk == "Yes" else 0

    # Section 3: General Health
    st.subheader("General Health")
    genhlth = st.selectbox(
        "General Health",
        ["Excellent", "Very Good", "Good", "Fair", "Poor"]
    )
    genhlth_map = {"Excellent": 1, "Very Good": 2, "Good": 3, "Fair": 4, "Poor": 5}
    genhlth = genhlth_map[genhlth]

    menthlth = st.selectbox(
        "Mental Health (days not good in last 30 days)",
        [i for i in range(31)]  # Dropdown with 0 to 30 days
    )
    physhlth = st.selectbox(
        "Physical Health (days not good in last 30 days)",
        [i for i in range(31)]  # Dropdown with 0 to 30 days
    )
    
    # Section 4: Socioeconomic Information
    st.subheader("Socioeconomic Information")
    education = st.selectbox(
        "Education Level",
        [
            "Never attended school",
            "Elementary school",
            "Some high school",
            "High school graduate",
            "Some college",
            "College graduate"
        ]
    )
    education_map = {
        "Never attended school": 1,
        "Elementary school": 2,
        "Some high school": 3,
        "High school graduate": 4,
        "Some college": 5,
        "College graduate": 6
    }
    education = education_map[education]

    income = st.selectbox(
        "Income Level",
        [
            "Less than $10,000",
            "$10,000 to $15,000",
            "$15,000 to $20,000",
            "$20,000 to $25,000",
            "$25,000 to $35,000",
            "$35,000 to $50,000",
            "$50,000 to $75,000",
            "$75,000 or more"
        ]
    )
    income_map = {
        "Less than $10,000": 1,
        "$10,000 to $15,000": 2,
        "$15,000 to $20,000": 3,
        "$20,000 to $25,000": 4,
        "$25,000 to $35,000": 5,
        "$35,000 to $50,000": 6,
        "$50,000 to $75,000": 7,
        "$75,000 or more": 8
    }
    income = income_map[income]

    # Collect all inputs in a single-row DataFrame
    input_data = pd.DataFrame([[
        high_bp, high_chol, chol_check, bmi, smoker, stroke, heart_disease,
        phys_activity, fruits, veggies, alcohol, healthcare, nodoc_cost,
        genhlth, menthlth, physhlth, diffwalk, sex, age, education, income
    ]], columns=[
        'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
        'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
        'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
    ])

    # Load the pre-trained model
    model = joblib.load("model/model.pkl")

    # Generate the report when the button is clicked
    if st.button("Predict Diabetes"):
        prediction = model.predict(input_data)
        predicted_class = {0: 'No Diabetes', 1: 'Prediabetes', 2: 'Diabetes'}[prediction[0]]

        # Determine health status based on BMI
        if bmi < 18.5:
            health_status = "Underweight"
            health_color = "orange"
        elif 18.5 <= bmi <= 24.9:
            health_status = "Healthy"
            health_color = "green"
        elif 25 <= bmi <= 29.9:
            health_status = "Overweight"
            health_color = "yellow"
        else:
            health_status = "Obese"
            health_color = "red"

        # Determine diabetes prediction color
        diabetes_color = "red" if predicted_class == "Diabetes" else "orange" if predicted_class == "Prediabetes" else "green"

        # Display the report as a "report card"
        st.markdown("## Diabetes Report")
        st.markdown("---")

        # Create a styled table for the report
        st.markdown(
            f"""
            <style>
                .report-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                    font-size: 18px;
                    text-align: left;
                }}
                .report-table th, .report-table td {{
                    border: 1px solid #dddddd;
                    padding: 8px 12px;
                }}
                .report-table th {{
                    background-color: #f2f2f2;
                }}
            </style>
            <table class="report-table">
                <tr>
                    <th>Field</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Predicted Diabetes Class</td>
                    <td style="color: {diabetes_color};"><strong>{predicted_class}</strong></td>
                </tr>
                <tr>
                    <td>Health Status</td>
                    <td style="color: {health_color};"><strong>{health_status}</strong></td>
                </tr>
                <tr>
                    <td>BMI</td>
                    <td>{bmi}</td>
                </tr>
                <tr>
                    <td>High Blood Pressure</td>
                    <td>{'Yes' if high_bp else 'No'}</td>
                </tr>
                <tr>
                    <td>High Cholesterol</td>
                    <td>{'Yes' if high_chol else 'No'}</td>
                </tr>
                <tr>
                    <td>Cholesterol Check</td>
                    <td>{'Yes' if chol_check else 'No'}</td>
                </tr>
                <tr>
                    <td>Smoker</td>
                    <td>{'Yes' if smoker else 'No'}</td>
                </tr>
                <tr>
                    <td>Stroke</td>
                    <td>{'Yes' if stroke else 'No'}</td>
                </tr>
                <tr>
                    <td>Heart Disease</td>
                    <td>{'Yes' if heart_disease else 'No'}</td>
                </tr>
                <tr>
                    <td>Physical Activity</td>
                    <td>{'Yes' if phys_activity else 'No'}</td>
                </tr>
                <tr>
                    <td>Fruits Consumption</td>
                    <td>{'Yes' if fruits else 'No'}</td>
                </tr>
                <tr>
                    <td>Vegetables Consumption</td>
                    <td>{'Yes' if veggies else 'No'}</td>
                </tr>
                <tr>
                    <td>Heavy Alcohol Consumption</td>
                    <td>{'Yes' if alcohol else 'No'}</td>
                </tr>
                <tr>
                    <td>Healthcare Coverage</td>
                    <td>{'Yes' if healthcare else 'No'}</td>
                </tr>
                <tr>
                    <td>Difficulty Walking</td>
                    <td>{'Yes' if diffwalk else 'No'}</td>
                </tr>
                <tr>
                    <td>General Health</td>
                    <td>{genhlth}</td>
                </tr>
                <tr>
                    <td>Mental Health (days not good)</td>
                    <td>{menthlth}</td>
                </tr>
                <tr>
                    <td>Physical Health (days not good)</td>
                    <td>{physhlth}</td>
                </tr>
                <tr>
                    <td>Education Level</td>
                    <td>{education}</td>
                </tr>
                <tr>
                    <td>Income Level</td>
                    <td>{income}</td>
                </tr>
            </table>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")
        st.markdown(
            """
            **Disclaimer:** This report is generated by an AI model based on the data you provided. 
            For accurate diagnosis and treatment, please consult a healthcare professional.
            """
        )

# Footer Section
st.markdown(
    """
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #f9f9f9;
            text-align: center;
            height: 40px; /* Adjusted height for a thinner footer */
            line-height: 40px; /* Align text vertically */
            border-top: 1px solid #eaeaea;
            font-size: 14px; /* Smaller font size */
        }
        .footer a {
            color: #0073b1; /* LinkedIn blue color */
            text-decoration: none;
            margin-left: 5px;
        }
        .footer a:hover {
            text-decoration: underline;
        }
    </style>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css" rel="stylesheet">
    <div class="footer">
        <p>
            Connect with us on 
            <a href="https://www.linkedin.com/in/vikash-malakar-25aa90116/" target="_blank">
                <i class="fab fa-linkedin"></i> Vikash Malakar
            </a> 
            and 
            <a href="https://www.linkedin.com/in/netajik/" target="_blank">
                <i class="fab fa-linkedin"></i> Netaji K
            </a> 
            | © 2025 Diabetes Prediction App
        </p>
    </div>
    """,
    unsafe_allow_html=True
)