from flask import Flask, render_template, request
import numpy as np
import pickle
import joblib

app = Flask(__name__)

# -------------------------
# 1️⃣ Load Models & Encoders
# -------------------------
voting_model = pickle.load(open("voting_model.pkl", "rb"))
dt_model = pickle.load(open("decision_tree_model.pkl","rb"))
rf_model = pickle.load(open("random_forest_model.pkl","rb"))
xgb_model = pickle.load(open("xgboost_model.pkl","rb"))

le_caste = joblib.load("le_caste.pkl")
le_target = joblib.load("le_target.pkl")

models = {"Decision Tree": dt_model, "Random Forest": rf_model, "XGBoost": xgb_model}

# -------------------------
# 2️⃣ Home Route
# -------------------------
@app.route('/')
def home():
    return render_template('index.html')

# -------------------------
# 3️⃣ Prediction Route
# -------------------------
@app.route('/predict', methods=['POST'])
def predict():
    # Get user input
    cet_score = float(request.form['cet_score'])
    jee_score = float(request.form['jee_score'])
    caste_input = request.form['caste']
    annual_income = float(request.form['annual_income'])
    ews = int(request.form['ews'])
    tfws = int(request.form['tfws'])

    caste_enc = le_caste.transform([caste_input])[0]
    user_data = np.array([[cet_score, jee_score, caste_enc, annual_income, ews, tfws]])

    # Voting Classifier Prediction
    pred_encoded = voting_model.predict(user_data)[0]
    pred_label = le_target.inverse_transform([pred_encoded])[0]
    branch, admission_status, fee_type = pred_label.split('_')

    # Individual Model Confidences
    model_confidences = {}
    proba_for_final = []
    agree_models = 0
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(user_data)[0]
            top_idx = np.argmax(proba)
            confidence = proba[top_idx]*100
            top_label = le_target.inverse_transform([top_idx])[0]
            model_confidences[name] = f"{confidence:.2f}% for {top_label}"
            # Track agreement and probability for the final predicted class
            proba_for_final.append(proba[pred_encoded]*100)
            if top_idx == pred_encoded:
                agree_models += 1
        else:
            model_confidences[name] = "No probability info"

    aggregate_confidence = float(np.mean(proba_for_final)) if proba_for_final else None

    # Fee Calculation
    fee_dict = {"Normal":138787,"EWS":13878,"TFWS":17787,"OBC":78787,"SC":2787,"ST":2787}
    fee_amount = fee_dict.get(fee_type,0)

    # Partial Admission Reason
    reason = ""
    if admission_status == "PartiallyAdmitted":
        if tfws == 1:
            reason = "Did not meet TFWS higher cutoff (2% above normal)"
        elif caste_input in ["SC","ST","OBC"]:
            reason = f"Admitted under reserved category — seat subject to quota"
        else:
            reason = "Partial admission due to cutoff rules"

    # Decision criteria met (human-friendly summary)
    criteria_met = []
    reserved_categories = {"SC", "ST", "OBC"}
    if ews == 1:
        criteria_met.append("EWS eligibility selected by applicant")
    if tfws == 1:
        criteria_met.append("TFWS eligibility selected by applicant")
    if caste_input in reserved_categories:
        criteria_met.append(f"Reserved category provided: {caste_input}")
    # High-level academic sufficiency message (model-based, generic)
    if admission_status in ["Admitted", "PartiallyAdmitted"]:
        criteria_met.append("Academic profile considered sufficient by the model")
    else:
        criteria_met.append("Academic profile considered below required level by the model")
    # Fee category alignment
    criteria_met.append(f"Fee category determined as: {fee_type}")

    # User-friendly highlights and recommendations
    highlights = []
    recommendations = []

    # CET score assessment
    if cet_score >= 90:
        highlights.append("Strong CET score (90+)")
    elif cet_score >= 75:
        highlights.append("Competitive CET score (75+)")
    else:
        recommendations.append("Improve CET score to strengthen admission chances")

    # JEE score assessment
    if jee_score >= 90:
        highlights.append("Strong JEE score (90+)")
    elif jee_score >= 75:
        highlights.append("Competitive JEE score (75+)")
    else:
        recommendations.append("Improve JEE score for better outcomes")

    # Income and concession categories (informational, not determinative)
    if ews == 1:
        highlights.append("EWS selected; fee concession considered where applicable")
    if tfws == 1:
        highlights.append("TFWS selected; tuition fee waiver considered where applicable")
    if annual_income <= 800000:
        highlights.append("Lower annual income may support concession categories")
    else:
        recommendations.append("Higher annual income may limit certain concessions")

    if caste_input in reserved_categories:
        highlights.append(f"Reserved category provided: {caste_input}")

    # Outcome-aware suggestions
    if admission_status == "Rejected":
        recommendations.append("Consider alternative branches or institutions")
        recommendations.append("Focus on raising entrance scores and revisiting category options if valid")
    elif admission_status == "PartiallyAdmitted":
        recommendations.append("Complete required steps to confirm partial admission or explore alternatives")

    return render_template(
        'result.html',
        branch=branch,
        status=admission_status,
        fee_amount=fee_amount,
        reason=reason,
        model_confidences=model_confidences,
        criteria_met=criteria_met,
        highlights=highlights,
        recommendations=recommendations,
        aggregate_confidence=aggregate_confidence,
        agree_models=agree_models,
        cet_score=cet_score,
        jee_score=jee_score,
        annual_income=annual_income,
        ews=ews,
        tfws=tfws,
    )

# -------------------------
# 4️⃣ Run Flask
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
