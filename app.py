import os
import uuid

from flask import Flask, abort, g, jsonify, redirect, render_template, request, url_for
from sqlalchemy import text

from inference_engine import infer_disease
from knowledge_base import FOLLOWUP_RULES, SYMPTOMS, SYMPTOM_GROUPS, SYMPTOM_KEYWORDS
from models import Patient, Screening, ScreeningResult, db

try:
    from flask_migrate import Migrate
except ImportError:
    Migrate = None


def create_app():
    app = Flask(__name__)
    os.makedirs(app.instance_path, exist_ok=True)

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        database_url = f"sqlite:///{os.path.join(app.instance_path, 'medai_screening.db')}"
    elif database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)

    app.config["SQLALCHEMY_DATABASE_URI"] = database_url
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["JSON_AS_ASCII"] = False

    db.init_app(app)
    if Migrate is not None:
        Migrate(app, db)

    return app


app = create_app()

with app.app_context():
    db.create_all()
    inspector = db.inspect(db.engine)
    screening_columns = {column["name"] for column in inspector.get_columns("screenings")}
    if "client_id" not in screening_columns:
        with db.engine.begin() as connection:
            connection.execute(text("ALTER TABLE screenings ADD COLUMN client_id VARCHAR(64)"))
            connection.execute(text("CREATE INDEX IF NOT EXISTS ix_screenings_client_id ON screenings (client_id)"))


def get_client_id():
    client_id = request.cookies.get("medai_client_id")
    if not client_id:
        client_id = uuid.uuid4().hex
        g.set_client_cookie = client_id
    return client_id


@app.after_request
def add_no_cache_headers(response):
    client_id = getattr(g, "set_client_cookie", None)
    if client_id:
        response.set_cookie(
            "medai_client_id",
            client_id,
            max_age=60 * 60 * 24 * 365,
            httponly=True,
            samesite="Lax",
        )
    if not request.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


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


def normalize_symptom_codes(symptom_codes):
    return [code for code in dict.fromkeys(symptom_codes or []) if code in SYMPTOMS]


def symptom_names(symptom_codes):
    return [SYMPTOMS[code] for code in symptom_codes if code in SYMPTOMS]


def parse_age(age_raw):
    if age_raw in (None, ""):
        return None
    try:
        age = int(age_raw)
    except (TypeError, ValueError):
        return None
    return age if 0 <= age <= 120 else None


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
                answered.append({"question": question["question"], "answer": option["label"]})
                break
    return normalize_symptom_codes(extra_symptoms), adjustments, answered


def serialize_result_payload(result):
    formula = result.get("formula") or {}
    return {
        "disease_id": result.get("id") or result.get("disease_id", ""),
        "name": result.get("name") or result.get("disease_name", ""),
        "icon": result.get("icon", "stethoscope"),
        "percent": result.get("percent", 0),
        "differential_percent": result.get("differential_percent", 0),
        "core_percent": result.get("core_percent", 0),
        "coverage_percent": result.get("coverage_percent", 0),
        "precision_percent": result.get("precision_percent", 0),
        "competitor_level": result.get("competitor_level", ""),
        "description": result.get("description", ""),
        "advice": result.get("advice", ""),
        "flags": result.get("flags", []),
        "matched": result.get("matched", []),
        "missing_core": result.get("missing_core", []),
        "reasoning_steps": result.get("reasoning_steps", []),
        "bayes_percent": result.get("bayes_percent", 0),
        "rule_activation": result.get("rule_activation", formula.get("rule_activation", 0)),
        "raw_score": result.get("raw_score", 0),
    }


def save_screening(
    client_id,
    patient,
    stage,
    severity,
    duration,
    symptom_text,
    selected_codes,
    selected_symptoms,
    results_data,
    followup_answers=None,
):
    screening = Screening(
        patient_id=patient.id,
        client_id=client_id,
        stage=stage,
        severity=severity,
        duration=duration,
        symptom_text=symptom_text,
        selected_codes=selected_codes,
        selected_symptoms=selected_symptoms,
        symptom_count=len(selected_symptoms),
        followup_answers=followup_answers or {},
    )
    db.session.add(screening)
    db.session.flush()

    for rank, item in enumerate(results_data, 1):
        payload = serialize_result_payload(item)
        result = ScreeningResult(
            screening_id=screening.id,
            rank=rank,
            disease_id=payload["disease_id"],
            disease_name=payload["name"],
            icon=payload["icon"],
            percent=payload["percent"],
            differential_percent=payload["differential_percent"],
            core_percent=payload["core_percent"],
            coverage_percent=payload["coverage_percent"],
            precision_percent=payload["precision_percent"],
            competitor_level=payload["competitor_level"],
            description=payload["description"],
            advice=payload["advice"],
            flags=payload["flags"],
            matched=payload["matched"],
            missing_core=payload["missing_core"],
            reasoning_steps=payload["reasoning_steps"],
            bayes_percent=payload["bayes_percent"],
            rule_activation=payload["rule_activation"],
            raw_score=payload["raw_score"],
        )
        db.session.add(result)

    db.session.commit()
    return screening


def get_or_create_patient(name, age, gender):
    patient = Patient.query.filter_by(name=name, age=age, gender=gender).first()
    if not patient:
        patient = Patient(name=name, age=age, gender=gender)
        db.session.add(patient)
        db.session.flush()
    return patient


@app.route("/", methods=["GET"])
@app.route("/gioi-thieu", methods=["GET"])
def gioi_thieu():
    return render_template("gioithieu.html")


@app.route("/chat", methods=["GET"])
def chat():
    return render_template("chat.html")


@app.route("/sang-loc", methods=["GET"])
def index():
    grouped_symptoms = {
        group_name: [(code, SYMPTOMS[code]) for code in codes]
        for group_name, codes in SYMPTOM_GROUPS.items()
    }
    return render_template("index.html", grouped_symptoms=grouped_symptoms, stage="intake")


@app.route("/predict", methods=["POST"])
def predict():
    client_id = get_client_id()
    stage = request.form.get("stage", "initial")
    name = (request.form.get("name") or "Người bệnh").strip() or "Người bệnh"
    age = parse_age((request.form.get("age") or "").strip())
    gender = (request.form.get("gender") or "all").strip()
    symptom_text = (request.form.get("symptom_text") or "").strip()
    severity = (request.form.get("severity") or "moderate").strip()
    duration = (request.form.get("duration") or "fewdays").strip()

    selected_from_checkboxes = normalize_symptom_codes(request.form.getlist("symptoms"))
    selected_from_text = extract_symptoms_from_text(symptom_text)
    base_symptoms = normalize_symptom_codes(selected_from_checkboxes + selected_from_text)

    if stage == "initial":
        followup_questions = build_followup_questions(base_symptoms)
        if followup_questions:
            preliminary_results = infer_disease(
                base_symptoms,
                age=age,
                gender=gender,
                severity=severity,
                duration=duration,
                followup_adjustments={},
            )
            patient = get_or_create_patient(name, age, gender)
            save_screening(
                client_id,
                patient,
                "followup",
                severity,
                duration,
                symptom_text,
                base_symptoms,
                symptom_names(base_symptoms),
                preliminary_results[:3],
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
                selected_symptoms=symptom_names(base_symptoms),
                selected_codes=base_symptoms,
                symptom_count=len(base_symptoms),
                followup_questions=followup_questions,
                preliminary_results=preliminary_results[:3],
            )

    followup_questions = build_followup_questions(base_symptoms)
    extra_symptoms, adjustments, answered_followups = parse_followup_answers(request.form, followup_questions)
    final_symptoms = normalize_symptom_codes(base_symptoms + extra_symptoms)
    results = infer_disease(
        final_symptoms,
        age=age,
        gender=gender,
        severity=severity,
        duration=duration,
        followup_adjustments=adjustments,
    )

    patient = get_or_create_patient(name, age, gender)
    screening = save_screening(
        client_id,
        patient,
        "final",
        severity,
        duration,
        symptom_text,
        final_symptoms,
        symptom_names(final_symptoms),
        results,
        followup_answers={item["question"]: item["answer"] for item in answered_followups},
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
        selected_symptoms=symptom_names(final_symptoms),
        symptom_count=len(final_symptoms),
        answered_followups=answered_followups,
        screening_id=screening.id,
    )


@app.route("/lich-su", methods=["GET"])
def history():
    client_id = get_client_id()
    page = request.args.get("page", 1, type=int)
    screenings = Screening.query.filter_by(stage="final", client_id=client_id).order_by(
        Screening.created_at.desc()
    ).paginate(page=page, per_page=20, error_out=False)
    return render_template("history.html", screenings=screenings)


@app.route("/lich-su/<int:screening_id>", methods=["GET"])
def history_detail(screening_id):
    client_id = get_client_id()
    screening = Screening.query.filter_by(id=screening_id, client_id=client_id).first()
    if screening is None:
        abort(404)
    results = screening.results.all()
    results_json = [item.to_dict() for item in results]
    return render_template(
        "history_detail.html",
        screening=screening,
        results=results,
        results_json=results_json,
    )


@app.route("/lich-su/xoa", methods=["POST"])
def clear_history():
    client_id = get_client_id()
    screenings = Screening.query.filter_by(client_id=client_id).all()
    screening_ids = [item.id for item in screenings]
    if screening_ids:
        ScreeningResult.query.filter(ScreeningResult.screening_id.in_(screening_ids)).delete(
            synchronize_session=False
        )
        Screening.query.filter(Screening.id.in_(screening_ids)).delete(synchronize_session=False)
        db.session.commit()
    return redirect(url_for("history"))


@app.route("/api/v1/health", methods=["GET"])
def api_health():
    return jsonify(
        {
            "status": "ok",
            "database": app.config["SQLALCHEMY_DATABASE_URI"].split(":", 1)[0],
            "symptom_count": len(SYMPTOMS),
            "group_count": len(SYMPTOM_GROUPS),
        }
    )


@app.route("/api/v1/screen", methods=["POST"])
def api_screen():
    client_id = get_client_id()
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    name = (data.get("name") or "API Patient").strip() or "API Patient"
    age = parse_age(data.get("age"))
    gender = (data.get("gender") or "all").strip()
    symptom_text = (data.get("symptom_text") or "").strip()
    severity = (data.get("severity") or "moderate").strip()
    duration = (data.get("duration") or "fewdays").strip()
    symptom_codes = normalize_symptom_codes(data.get("symptoms", []))

    selected_from_text = extract_symptoms_from_text(symptom_text)
    all_symptoms = normalize_symptom_codes(symptom_codes + selected_from_text)
    if not all_symptoms:
        return jsonify({"error": "At least one valid symptom is required"}), 400

    results = infer_disease(
        all_symptoms,
        age=age,
        gender=gender,
        severity=severity,
        duration=duration,
        followup_adjustments={},
    )

    patient = get_or_create_patient(name, age, gender)
    screening = save_screening(
        client_id,
        patient,
        "final",
        severity,
        duration,
        symptom_text,
        all_symptoms,
        symptom_names(all_symptoms),
        results,
    )

    return jsonify(
        {
            "screening_id": screening.id,
            "patient": patient.to_dict(),
            "symptom_count": len(all_symptoms),
            "selected_symptoms": symptom_names(all_symptoms),
            "results": [item.to_dict() for item in screening.results.all()],
        }
    )


@app.route("/api/v1/diseases", methods=["GET"])
def api_diseases():
    from knowledge_base import DISEASES

    disease_list = []
    for disease_id, disease in DISEASES.items():
        disease_list.append(
            {
                "id": disease_id,
                "name": disease.get("name", ""),
                "group": disease.get("group", ""),
                "icon": disease.get("icon", "stethoscope"),
            }
        )
    return jsonify({"diseases": disease_list, "count": len(disease_list)})


@app.route("/api/v1/symptoms", methods=["GET"])
def api_symptoms():
    query = (request.args.get("q") or "").strip().lower()
    symptoms = [{"code": code, "name": name} for code, name in SYMPTOMS.items()]
    if query:
        symptoms = [
            item
            for item in symptoms
            if query in item["name"].lower() or query in item["code"].lower()
        ]
    return jsonify({"symptoms": symptoms, "count": len(symptoms), "query": query})


@app.route("/api/v1/history", methods=["GET"])
def api_history():
    client_id = get_client_id()
    page = request.args.get("page", 1, type=int)
    per_page = min(max(request.args.get("per_page", 20, type=int), 1), 100)
    screenings = Screening.query.filter_by(stage="final", client_id=client_id).order_by(
        Screening.created_at.desc()
    ).paginate(page=page, per_page=per_page, error_out=False)
    return jsonify(
        {
            "screenings": [item.to_dict() for item in screenings.items],
            "total": screenings.total,
            "page": screenings.page,
            "pages": screenings.pages,
        }
    )


@app.route("/api/v1/screening/<int:screening_id>", methods=["GET"])
def api_screening_detail(screening_id):
    client_id = get_client_id()
    screening = Screening.query.filter_by(id=screening_id, client_id=client_id).first()
    if screening is None:
        abort(404)
    payload = screening.to_dict()
    payload["results"] = [item.to_dict() for item in screening.results.all()]
    payload["patient"] = screening.patient.to_dict()
    return jsonify(payload)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(debug=True, host="0.0.0.0", port=port)
