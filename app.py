from flask import Flask, render_template, request
from knowledge_base import SYMPTOMS, SYMPTOM_GROUPS, SYMPTOM_KEYWORDS, FOLLOWUP_RULES
from inference_engine import infer_disease

app = Flask(__name__)


def extract_symptoms_from_text(symptom_text):
    symptom_text_normalized = f" {(symptom_text or '').strip().lower()} "
    matched = []
    for code, keywords in SYMPTOM_KEYWORDS.items():
        for keyword in keywords:
            keyword_normalized = f" {keyword.strip().lower()} "
            if keyword_normalized in symptom_text_normalized:
                matched.append(code)
                break
    return list(dict.fromkeys(matched))


def build_followup_questions(selected_symptoms):
    questions = []
    selected_set = set(selected_symptoms)
    for rule in FOLLOWUP_RULES:
        if any(code in selected_set for code in rule["trigger_symptoms"]):
            questions.append(rule)
    return questions


def parse_followup_answers(form_data, questions):
    extra_symptoms = []
    adjustments = {}
    answered = []

    for question in questions:
        answer_value = form_data.get(f"followup_{question['id']}")
        if not answer_value:
            continue
        for option in question["options"]:
            if option["value"] == answer_value:
                extra_symptoms.extend(option.get("add_symptoms", []))
                for disease_id, delta in option.get("weight_delta", {}).items():
                    adjustments[disease_id] = adjustments.get(disease_id, 0) + delta
                answered.append({
                    "question": question["question"],
                    "answer": option["label"]
                })
                break

    return list(dict.fromkeys(extra_symptoms)), adjustments, answered


@app.route("/")
def index():
    grouped_symptoms = {
        group_name: [(code, SYMPTOMS[code]) for code in codes]
        for group_name, codes in SYMPTOM_GROUPS.items()
    }
    return render_template(
        "index.html",
        grouped_symptoms=grouped_symptoms,
        stage="intake"
    )


@app.route("/predict", methods=["POST"])
def predict():
    stage = request.form.get("stage", "initial")
    name = (request.form.get("name") or "Người bệnh").strip()
    age_raw = (request.form.get("age") or "").strip()
    gender = (request.form.get("gender") or "all").strip()
    symptom_text = (request.form.get("symptom_text") or "").strip()
    severity = (request.form.get("severity") or "moderate").strip()
    duration = (request.form.get("duration") or "fewdays").strip()

    age = None
    try:
        age = int(age_raw) if age_raw else None
    except ValueError:
        age = None

    selected_from_checkboxes = request.form.getlist("symptoms")
    selected_from_text = extract_symptoms_from_text(symptom_text)
    base_symptoms = list(dict.fromkeys(selected_from_checkboxes + selected_from_text))

    if stage == "initial":
        followup_questions = build_followup_questions(base_symptoms)
        if followup_questions:
            preliminary_results = infer_disease(
                base_symptoms,
                age=age,
                gender=gender,
                severity=severity,
                duration=duration,
                followup_adjustments={}
            )
            return render_template(
                "result.html",
                stage="followup",
                name=name,
                age=age,
                gender=gender,
                severity=severity,
                duration=duration,
                symptom_text=symptom_text,
                selected_symptoms=[SYMPTOMS[code] for code in base_symptoms if code in SYMPTOMS],
                selected_codes=base_symptoms,
                symptom_count=len(base_symptoms),
                followup_questions=followup_questions,
                preliminary_results=preliminary_results[:3]
            )

    followup_questions = build_followup_questions(base_symptoms)
    extra_symptoms, adjustments, answered_followups = parse_followup_answers(request.form, followup_questions)
    final_symptoms = list(dict.fromkeys(base_symptoms + extra_symptoms))
    results = infer_disease(
        final_symptoms,
        age=age,
        gender=gender,
        severity=severity,
        duration=duration,
        followup_adjustments=adjustments
    )

    return render_template(
        "result.html",
        stage="final",
        name=name,
        age=age,
        gender=gender,
        severity=severity,
        duration=duration,
        symptom_text=symptom_text,
        results=results,
        selected_symptoms=[SYMPTOMS[code] for code in final_symptoms if code in SYMPTOMS],
        symptom_count=len(final_symptoms),
        answered_followups=answered_followups
    )


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5050))
    app.run(debug=True, host="0.0.0.0", port=port)