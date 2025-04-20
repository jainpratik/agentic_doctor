def check_sepsis_criteria(lab_values):
    flags = []
    if lab_values.get("WBC", 0) > 12000:
        flags.append("⬆️ Elevated WBC (>12,000)")
    if lab_values.get("Lactate", 0) > 2.0:
        flags.append("⬆️ Elevated Lactate (>2.0 mmol/L)")
    if lab_values.get("MAP", 100) < 65:
        flags.append("⬇️ Low Mean Arterial Pressure (<65 mmHg)")
    if lab_values.get("RR", 0) >= 22:
        flags.append("⬆️ High Respiratory Rate (≥22)")

    if len(flags) >= 2:
        return "⚠️ Sepsis criteria met:\n" + "\\n".join(flags)
    return "✅ Sepsis criteria not met."


def check_pneumonia_criteria(symptoms):
    matches = []
    for s in symptoms:
        if s.lower() in ["cough", "fever", "chest pain", "shortness of breath"]:
            matches.append(s)
    if len(matches) >= 2:
        return f"⚠️ Possible pneumonia. Matching symptoms: {', '.join(matches)}"
    return "✅ No strong pneumonia indicators found."


def check_diabetes_criteria(labs):
    a1c = labs.get("A1C", 0)
    glucose = labs.get("Glucose", 0)
    flags = []

    if a1c >= 6.5:
        flags.append("⬆️ A1C ≥ 6.5% → diagnostic for diabetes")
    if glucose >= 126:
        flags.append("⬆️ Fasting Glucose ≥ 126 mg/dL → diagnostic")
    if glucose >= 200:
        flags.append("⬆️ Random Glucose ≥ 200 mg/dL → possible diabetes")

    return flags or ["✅ No diabetes criteria met."]


def check_hypertension(labs):
    sbp = labs.get("SBP", 0)
    dbp = labs.get("DBP", 0)
    if sbp >= 140 or dbp >= 90:
        return ["❌ Stage 2 Hypertension"]
    elif 130 <= sbp < 140 or 80 <= dbp < 90:
        return ["⚠️ Stage 1 Hypertension"]
    elif sbp < 120 and dbp < 80:
        return ["✅ Normal blood pressure"]
    else:
        return ["🟡 Elevated blood pressure"]


def check_covid_severity(labs):
    rr = labs.get("RR", 0)
    spo2 = labs.get("SpO2", 100)
    if rr >= 30 or spo2 < 90:
        return ["❌ Severe COVID-19 suspicion"]
    elif rr > 20 or 90 <= spo2 <= 94:
        return ["⚠️ Moderate COVID-19 suspicion"]
    return ["✅ No severe COVID indicators"]


def calc_qsofa(labs):
    score = 0
    if labs.get("RR", 0) >= 22:
        score += 1
    if labs.get("GCS", 15) < 15:
        score += 1
    if labs.get("SBP", 200) <= 100:
        score += 1
    return f"🧮 qSOFA Score: {score} (≥2 suggests high mortality risk)"


def calc_curb65(labs):
    score = 0
    if labs.get("Confusion", False):
        score += 1
    if labs.get("Urea", 0) > 7:
        score += 1
    if labs.get("RR", 0) >= 30:
        score += 1
    if labs.get("SBP", 120) < 90 or labs.get("DBP", 80) <= 60:
        score += 1
    if labs.get("Age", 0) >= 65:
        score += 1
    return f"🧮 CURB-65 Score: {score} (higher = higher pneumonia mortality risk)"


def guideline_checker_expanded(input_text):
    symptoms = [s.strip().lower() for s in input_text.split(",") if s.strip()]

    # Dummy simulated parsed lab values
    labs = {
        "WBC": 16000,
        "Lactate": 3.1,
        "MAP": 60,
        "RR": 24,
        "A1C": 7.1,
        "Glucose": 135,
        "SBP": 145,
        "DBP": 95,
        "RR": 32,
        "SpO2": 89,
        "GCS": 14,
        "Urea": 8,
        "Confusion": True,
        "Age": 67,
    }

    output = []
    output.extend(check_sepsis_criteria(labs))
    output.extend(check_pneumonia_criteria(symptoms))
    output.extend(check_diabetes_criteria(labs))
    output.extend(check_hypertension(labs))
    output.extend(check_covid_severity(labs))
    output.append(calc_qsofa(labs))
    output.append(calc_curb65(labs))

    return "\n".join(output)
