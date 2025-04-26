import pandas as pd
import re
import os
import pytesseract
from pdf2image import convert_from_bytes
from PyPDF2 import PdfReader


def color_lab_value(value):
    try:
        val = float(value)
        if val < 0:
            return "❌"
        elif val > 1000:
            return "❌"
        elif val > 100:
            return "⚠️"
        else:
            return "✅"
    except:
        return ""


def extract_entities(text):
    pattern = r"(?P<test>[A-Za-z /]+)[:\\-\\s]+(?P<value>\\d+(\\.\\d+)?)\\s*(?P<unit>[a-zA-Z/%µ]*)"
    matches = re.findall(pattern, text)
    data = []
    scores = {}
    for m in matches:
        test = m[0].strip().upper()
        value = float(m[1])
        unit = m[3]
        flag = "✅"
        if "WBC" in test and value > 12000:
            flag = "❌"
        if "A1C" in test and value >= 6.5:
            flag = "❌"
            scores["A1C"] = value
        if "GLUCOSE" in test and value > 126:
            flag = "⚠️"
            scores["Glucose"] = value
        if "SBP" in test:
            scores["SBP"] = value
        if "RR" in test:
            scores["RR"] = value
        if "SPO2" in test:
            scores["SpO2"] = value
        if "GCS" in test:
            scores["GCS"] = value
        if "UREA" in test:
            scores["Urea"] = value
        data.append({"Test": test, "Value": value, "Unit": unit, "Flag": flag})
    return pd.DataFrame(data), scores


def get_lab_report_path(patient_id):
    base_dir = "data/lab_reports"
    filename = f"{patient_id}.pdf"
    full_path = os.path.join(base_dir, filename)
    return full_path if os.path.exists(full_path) else None
