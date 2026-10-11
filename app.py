
from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "student_model (1).pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "student_model (1).pkl not found. Train the model first."
    )

model = joblib.load(MODEL_PATH)

FEATURES = [
    "studytime", "absences", "G1", "G2",
    "failures", "schoolsup", "famsup", "internet"
]


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None
    values = {}

    if request.method == "POST":
        try:
            studytime = int(request.form["studytime"])
            absences = int(request.form["absences"])
            g1 = int(request.form["g1"])
            g2 = int(request.form["g2"])
            failures = int(request.form["failures"])

            schoolsup = request.form["schoolsup"]
            famsup = request.form["famsup"]
            internet = request.form["internet"]

            values = {
                "studytime": str(studytime),
                "absences": str(absences),
                "g1": str(g1),
                "g2": str(g2),
                "failures": str(failures),
                "schoolsup": schoolsup,
                "famsup": famsup,
                "internet": internet
            }

            if studytime not in [1, 2, 3, 4]:
                raise ValueError("Select valid study time.")

            if not 0 <= absences <= 93:
                raise ValueError("Absences must be between 0 and 93.")

            if not 0 <= g1 <= 20 or not 0 <= g2 <= 20:
                raise ValueError("G1 and G2 must be between 0 and 20.")

            if not 0 <= failures <= 4:
                raise ValueError("Failures must be between 0 and 4.")

            if schoolsup not in ["Yes", "No"]:
                raise ValueError("Select school support.")

            if famsup not in ["Yes", "No"]:
                raise ValueError("Select family support.")

            if internet not in ["Yes", "No"]:
                raise ValueError("Select internet access.")

            student = pd.DataFrame([{
                "studytime": studytime,
                "absences": absences,
                "G1": g1,
                "G2": g2,
                "failures": failures,
                "schoolsup": 1 if schoolsup == "Yes" else 0,
                "famsup": 1 if famsup == "Yes" else 0,
                "internet": 1 if internet == "Yes" else 0
            }], columns=FEATURES)

            prediction = float(model.predict(student)[0])
            prediction = max(0.0, min(20.0, prediction))

            percentage = prediction * 5

            if prediction < 10:
                category = "Needs Support"
                level = "support"
                message = "Extra academic guidance may be helpful."
            elif prediction < 14:
                category = "Average"
                level = "average"
                message = "Continue developing consistent study habits."
            else:
                category = "Good"
                level = "good"
                message = "Continue practising and maintaining good habits."

            result = {
                "marks": round(prediction, 2),
                "percentage": round(percentage, 1),
                "category": category,
                "level": level,
                "message": message
            }

        except (ValueError, TypeError, KeyError):
            error = "Please complete every field using valid values."

    return render_template(
        "index.html",
        result=result,
        error=error,
        values=values
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)