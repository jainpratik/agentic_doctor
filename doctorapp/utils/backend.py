import pandas as pd


def agent_response(query):
    return f"Simulated answer for: {query}", "Simulated reasoning trace..."


def parse_lab_pdf(uploaded_file):
    data = {
        "Test Name": ["WBC", "RBC", "Hemoglobin", "Glucose"],
        "Value": [4.5, 5.2, 13.5, 180],
        "Reference Range": ["4.0-11.0", "4.5-5.9", "13.0-17.0", "70-140"],
    }
    return pd.DataFrame(data)


def color_code_cells(row):
    value = row["Value"]
    if row["Test Name"] == "Glucose" and value > 140:
        return ["background-color: yellow"] * len(row)
    return [""] * len(row)


def generate_care_plan(patient):
    return {
        "patient": patient,
        "diagnosis": "Possible prediabetes",
        "recommendations": [
            "Re-check glucose in 3 months",
            "Start light exercise 3x/week",
            "Reduce sugar intake",
        ],
    }


def get_patient_events(patient):
    return [
        {"date": "2024-04-01", "note": "Routine check-up"},
        {"date": "2024-04-15", "note": "Complained of chest pain"},
        {"date": "2024-04-20", "note": "Lab tests done"},
    ]


def get_patient_vitals(patient):
    return pd.DataFrame(
        {
            "Date": ["2024-04-01", "2024-04-15", "2024-04-20"],
            "Heart Rate": [72, 78, 85],
            "Blood Pressure": [120, 130, 135],
            "Temperature": [98.6, 99.1, 99.3],
        }
    )
