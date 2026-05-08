from datetime import datetime

from flask_sqlalchemy import SQLAlchemy


db = SQLAlchemy()


class Patient(db.Model):
    __tablename__ = "patients"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    age = db.Column(db.Integer, nullable=True)
    gender = db.Column(db.String(20), nullable=False, default="all")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    screenings = db.relationship(
        "Screening",
        backref="patient",
        lazy="dynamic",
        order_by="Screening.created_at.desc()",
    )

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "age": self.age,
            "gender": self.gender,
            "created_at": self.created_at.isoformat(),
            "screening_count": self.screenings.count(),
        }


class Screening(db.Model):
    __tablename__ = "screenings"

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("patients.id"), nullable=False)
    client_id = db.Column(db.String(64), nullable=True, index=True)
    stage = db.Column(db.String(20), nullable=False)
    severity = db.Column(db.String(20), default="moderate")
    duration = db.Column(db.String(20), default="fewdays")
    symptom_text = db.Column(db.Text, default="")
    selected_codes = db.Column(db.JSON, default=list)
    selected_symptoms = db.Column(db.JSON, default=list)
    symptom_count = db.Column(db.Integer, default=0)
    followup_answers = db.Column(db.JSON, default=dict)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    results = db.relationship(
        "ScreeningResult",
        backref="screening",
        lazy="dynamic",
        order_by="ScreeningResult.rank",
    )

    def to_dict(self):
        top_result = self.results.first()
        return {
            "id": self.id,
            "patient_id": self.patient_id,
            "patient_name": self.patient.name,
            "client_id": self.client_id,
            "stage": self.stage,
            "severity": self.severity,
            "duration": self.duration,
            "symptom_text": self.symptom_text,
            "selected_symptoms": self.selected_symptoms,
            "symptom_count": self.symptom_count,
            "created_at": self.created_at.isoformat(),
            "result_count": self.results.count(),
            "top_result": top_result.disease_name if top_result else None,
        }


class ScreeningResult(db.Model):
    __tablename__ = "screening_results"

    id = db.Column(db.Integer, primary_key=True)
    screening_id = db.Column(db.Integer, db.ForeignKey("screenings.id"), nullable=False)
    rank = db.Column(db.Integer, nullable=False)
    disease_id = db.Column(db.String(100), nullable=False)
    disease_name = db.Column(db.String(200), nullable=False)
    icon = db.Column(db.String(50), default="stethoscope")
    percent = db.Column(db.Float, default=0)
    differential_percent = db.Column(db.Float, default=0)
    core_percent = db.Column(db.Float, default=0)
    coverage_percent = db.Column(db.Float, default=0)
    precision_percent = db.Column(db.Float, default=0)
    competitor_level = db.Column(db.String(50), default="")
    description = db.Column(db.Text, default="")
    advice = db.Column(db.Text, default="")
    flags = db.Column(db.JSON, default=list)
    matched = db.Column(db.JSON, default=list)
    missing_core = db.Column(db.JSON, default=list)
    reasoning_steps = db.Column(db.JSON, default=list)
    bayes_percent = db.Column(db.Float, default=0)
    rule_activation = db.Column(db.Float, default=0)
    raw_score = db.Column(db.Float, default=0)

    def to_dict(self):
        return {
            "id": self.id,
            "rank": self.rank,
            "disease_id": self.disease_id,
            "disease_name": self.disease_name,
            "icon": self.icon,
            "percent": self.percent,
            "differential_percent": self.differential_percent,
            "core_percent": self.core_percent,
            "coverage_percent": self.coverage_percent,
            "precision_percent": self.precision_percent,
            "competitor_level": self.competitor_level,
            "description": self.description,
            "advice": self.advice,
            "flags": self.flags,
            "matched": self.matched,
            "missing_core": self.missing_core,
            "reasoning_steps": self.reasoning_steps,
            "bayes_percent": self.bayes_percent,
            "rule_activation": self.rule_activation,
            "raw_score": self.raw_score,
        }
