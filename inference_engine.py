import math

from knowledge_base import DISEASES, SYMPTOMS


EMERGENCY_DISEASES = {"stroke", "heart_attack"}
HIGH_ALERT_DISEASES = {
    "lung_cancer",
    "liver_cancer",
    "stomach_cancer",
    "breast_cancer",
    "colorectal_cancer",
    "tuberculosis",
}

SEVERITY_FACTORS = {"mild": 0.96, "moderate": 1.0, "severe": 1.08}
DURATION_FACTORS = {"1day": 0.96, "fewdays": 1.0, "weeks": 1.10}
PRIORITY_PRIORS = {"low": 0.36, "medium": 0.48, "high": 0.60, "critical": 0.70}

MAX_RESULTS = 8
MIN_DISPLAY_SCORE_THRESHOLD = 0.08

# giam bot hien tuong 1 benh trong cung nhom nuot het benh con lai
GROUP_COMPETITION_PENALTY = 0.08

# phat khi thieu core symptom
CORE_MISSING_PENALTY = 0.20

# thuong them khi co cum trieu chung rat dac hieu
PAIR_SYNERGY = {
    "covid": [
        ({"S03", "S05"}, 0.16),
        ({"S02", "S03", "S04"}, 0.12),
        ({"S06", "S07"}, 0.05),
    ],
    "influenza": [
        ({"S02", "S06", "S07"}, 0.18),
        ({"S06", "S07"}, 0.08),
    ],
    "common_cold": [
        ({"S08", "S09"}, 0.16),
        ({"S03", "S09"}, 0.08),
    ],
    "acute_pharyngitis": [
        ({"S08", "S01"}, 0.14),
        ({"S08", "S03"}, 0.06),
    ],
    "pneumonia": [
        ({"S02", "S04"}, 0.14),
        ({"S02", "S04", "S23"}, 0.18),
        ({"S21", "S23"}, 0.06),
    ],
    "bronchial_asthma": [
        ({"S04", "S20"}, 0.18),
        ({"S21", "S22"}, 0.10),
    ],
    "tuberculosis": [
        ({"S16", "S17"}, 0.18),
        ({"S16", "S18"}, 0.16),
        ({"S16", "S19"}, 0.10),
        ({"S16", "S18", "S84"}, 0.14),
    ],
    "lung_cancer": [
        ({"S17", "S18"}, 0.14),
        ({"S99", "S100"}, 0.12),
        ({"S17", "S99"}, 0.12),
    ],
    "measles": [
        ({"S02", "S25"}, 0.16),
        ({"S25", "S26"}, 0.14),
    ],
    "hand_foot_mouth": [
        ({"S27", "S28"}, 0.18),
        ({"S28", "S92"}, 0.10),
    ],
    "otitis_media": [
        ({"S90", "S91"}, 0.16),
    ],
    "gastritis": [
        ({"S14", "S12"}, 0.10),
    ],
    "gerd": [
        ({"S38", "S39"}, 0.18),
        ({"S38", "S40"}, 0.12),
    ],
    "acute_hepatitis": [
        ({"S29", "S30"}, 0.18),
        ({"S29", "S31"}, 0.12),
        ({"S30", "S31"}, 0.12),
    ],
    "gallstones": [
        ({"S37", "S29"}, 0.10),
        ({"S37", "S30"}, 0.10),
    ],
    "liver_cancer": [
        ({"S18", "S37"}, 0.12),
        ({"S34", "S36"}, 0.14),
    ],
    "stomach_cancer": [
        ({"S102", "S103"}, 0.18),
        ({"S14", "S18"}, 0.10),
    ],
    "colorectal_cancer": [
        ({"S108", "S109"}, 0.20),
    ],
    "heart_failure": [
        ({"S41", "S42"}, 0.18),
        ({"S35", "S41"}, 0.12),
    ],
    "heart_attack": [
        ({"S45", "S46"}, 0.22),
        ({"S45", "S04"}, 0.10),
    ],
    "stroke": [
        ({"S47", "S48"}, 0.22),
        ({"S47", "S49"}, 0.18),
        ({"S48", "S49"}, 0.18),
    ],
    "anemia": [
        ({"S44", "S43"}, 0.14),
        ({"S44", "S42"}, 0.10),
    ],
    "hypertension": [
        ({"S70", "S07"}, 0.08),
        ({"S70", "S15"}, 0.08),
    ],
    "sciatica": [
        ({"S50", "S51"}, 0.16),
    ],
    "gout": [
        ({"S52", "S53"}, 0.08),
    ],
    "osteoarthritis": [
        ({"S53", "S54"}, 0.06),
    ],
    "rheumatoid_arthritis": [
        ({"S54", "S55"}, 0.16),
        ({"S55", "S56"}, 0.14),
    ],
    "diabetes": [
        ({"S57", "S58", "S59"}, 0.20),
    ],
    "hyperthyroidism": [
        ({"S60", "S61"}, 0.14),
        ({"S61", "S62"}, 0.12),
        ({"S61", "S63"}, 0.10),
    ],
    "kidney_stones": [
        ({"S64", "S65"}, 0.14),
        ({"S64", "S66"}, 0.16),
    ],
    "chronic_kidney_disease": [
        ({"S67", "S68"}, 0.18),
        ({"S67", "S70"}, 0.10),
    ],
    "bph": [
        ({"S71", "S72", "S73"}, 0.18),
    ],
    "erectile_dysfunction": [
        ({"S74", "S75"}, 0.12),
    ],
    "vaginitis": [
        ({"S76", "S77"}, 0.18),
        ({"S77", "S78"}, 0.10),
    ],
    "pcos": [
        ({"S79", "S81"}, 0.16),
        ({"S79", "S80"}, 0.10),
        ({"S81", "S82"}, 0.10),
    ],
    "menopause": [
        ({"S83", "S85"}, 0.12),
        ({"S83", "S86"}, 0.12),
    ],
    "breast_cancer": [
        ({"S105", "S106"}, 0.18),
        ({"S105", "S107"}, 0.18),
    ],
    "rickets": [
        ({"S88", "S89"}, 0.18),
    ],
    "alzheimer": [
        ({"S93", "S94"}, 0.16),
        ({"S93", "S95"}, 0.10),
    ],
    "parkinson": [
        ({"S96", "S97"}, 0.16),
        ({"S96", "S98"}, 0.14),
        ({"S97", "S98"}, 0.12),
    ],
}

DISEASE_NEGATIVE_HINTS = {
    "common_cold": {"S17", "S18", "S99", "S102", "S103", "S04"},
    "acute_pharyngitis": {"S17", "S18", "S99", "S102", "S103", "S04"},
    "covid": {"S17", "S102", "S103", "S108", "S109"},
    "influenza": {"S17", "S102", "S103", "S108", "S109"},
    "pneumonia": {"S05", "S108", "S109"},
    "bronchial_asthma": {"S05", "S17", "S18"},
    "tuberculosis": {"S05", "S09"},
    "lung_cancer": {"S05", "S09"},
    "gerd": {"S17", "S18", "S109"},
    "bph": {"S77", "S78", "S79"},
    "vaginitis": {"S71", "S72", "S73"},
}

HIGH_SPECIFICITY_SYMPTOMS = {
    "S05",
    "S17",
    "S25",
    "S27",
    "S28",
    "S29",
    "S30",
    "S31",
    "S45",
    "S46",
    "S47",
    "S48",
    "S49",
    "S52",
    "S57",
    "S58",
    "S59",
    "S64",
    "S66",
    "S71",
    "S72",
    "S73",
    "S76",
    "S77",
    "S83",
    "S88",
    "S89",
    "S93",
    "S94",
    "S96",
    "S97",
    "S98",
    "S102",
    "S103",
    "S105",
    "S106",
    "S107",
    "S108",
    "S109",
}


def _round_percent(value):
    return round(max(0, min(value, 100)), 1)


def _clamp01(value):
    return max(0.0, min(value, 1.0))


def _sigmoid(value):
    return 1.0 / (1.0 + math.exp(-value))


def _gender_matches(rule, gender):
    if rule == "all":
        return True
    return rule == gender


def _build_age_note(age, age_min, age_max):
    if age is None:
        return "Không có dữ liệu tuổi để hiệu chỉnh."
    if age < age_min:
        return f"Độ tuổi {age} nhỏ hơn nhóm tuổi thường gặp ({age_min}-{age_max})."
    if age > age_max:
        return f"Độ tuổi {age} lớn hơn nhóm tuổi thường gặp ({age_min}-{age_max})."
    return f"Độ tuổi {age} nằm trong nhóm tuổi thường gặp ({age_min}-{age_max})."


def _build_patient_facts(selected_symptoms, age, gender, severity, duration):
    facts = {f"has({code})" for code in selected_symptoms}
    if age is not None:
        facts.add(f"age({age})")
    facts.add(f"gender({gender})")
    facts.add(f"severity({severity})")
    facts.add(f"duration({duration})")
    return facts


def _compute_support_scores(selected_symptoms, disease):
    weights = disease["weights"]
    core = disease.get("core", [])
    total_weight = sum(weights.values())

    matched_codes = [code for code in selected_symptoms if code in weights]
    matched_weight = sum(weights[code] for code in matched_codes)
    core_matched = [code for code in core if code in selected_symptoms]

    coverage_score = matched_weight / total_weight if total_weight > 0 else 0.0

    selected_weight_total = sum(weights.get(code, 0.22) for code in selected_symptoms)
    precision_score = matched_weight / selected_weight_total if selected_weight_total > 0 else 0.0

    core_score = len(core_matched) / len(core) if core else 0.0

    missing_core = [code for code in core if code not in selected_symptoms]
    missing_core_ratio = len(missing_core) / len(core) if core else 0.0

    supportive_score = 0.0
    if matched_codes:
        non_core_codes = [code for code in matched_codes if code not in core]
        if non_core_codes:
            supportive_score = min(
                sum(weights.get(code, 0) for code in non_core_codes) / max(total_weight, 1.0),
                1.0,
            )

    return {
        "matched_codes": matched_codes,
        "matched_weight": matched_weight,
        "core_matched": core_matched,
        "coverage_score": coverage_score,
        "precision_score": precision_score,
        "core_score": core_score,
        "missing_core_ratio": missing_core_ratio,
        "supportive_score": supportive_score,
    }


def _compute_pair_synergy(disease_id, selected_symptoms):
    selected_set = set(selected_symptoms)
    bonus = 0.0
    matched_pairs = []
    for symptom_set, delta in PAIR_SYNERGY.get(disease_id, []):
        if symptom_set.issubset(selected_set):
            bonus += delta
            matched_pairs.append(sorted(symptom_set))
    return min(bonus, 0.24), matched_pairs


def _compute_negative_hint_penalty(disease_id, selected_symptoms):
    selected_set = set(selected_symptoms)
    bad_set = DISEASE_NEGATIVE_HINTS.get(disease_id, set())
    hit = len(selected_set.intersection(bad_set))
    return min(hit * 0.05, 0.16)


def _compute_specificity_penalty(selected_symptoms, matched_codes):
    selected_set = set(selected_symptoms)
    matched_set = set(matched_codes)
    missed_specific = [
        symptom
        for symptom in selected_set
        if symptom in HIGH_SPECIFICITY_SYMPTOMS and symptom not in matched_set
    ]
    return min(len(missed_specific) * 0.04, 0.20)


def _compute_logic_bonus(scores, pair_synergy_bonus):
    bonus = 0.0

    if scores["core_score"] == 1:
        bonus += 0.08
    elif scores["core_score"] >= 0.67:
        bonus += 0.04

    if scores["coverage_score"] >= 0.60:
        bonus += 0.03

    if scores["supportive_score"] >= 0.15:
        bonus += 0.02

    if scores["core_score"] >= 0.50 and scores["precision_score"] >= 0.50:
        bonus += 0.05

    bonus += pair_synergy_bonus
    return bonus


def _compute_logic_penalty(disease_id, scores, selected_symptoms):
    penalty = 0.0

    if scores["core_score"] == 0:
        penalty += 0.20
    elif scores["core_score"] < 0.34:
        penalty += 0.10

    penalty += scores["missing_core_ratio"] * CORE_MISSING_PENALTY
    penalty += _compute_negative_hint_penalty(disease_id, selected_symptoms)
    penalty += _compute_specificity_penalty(selected_symptoms, scores["matched_codes"])

    return penalty


def _compute_rule_activation(disease, scores, pair_synergy_bonus, age, gender):
    activation = 0.0
    traces = []

    if scores["core_score"] == 1:
        activation += 0.16
        traces.append("Luật sản xuất: tất cả triệu chứng cốt lõi đều đúng => kích hoạt mạnh bệnh.")
    elif scores["core_score"] >= 0.5:
        activation += 0.08
        traces.append("Luật sản xuất: đã khớp một phần lớn triệu chứng cốt lõi => giữ bệnh trong tập giả thuyết.")

    if scores["coverage_score"] >= 0.5 and scores["precision_score"] >= 0.45:
        activation += 0.08
        traces.append("Luật mệnh đề: (độ phủ cao AND độ tập trung cao) => hồ sơ phù hợp với bệnh.")

    if pair_synergy_bonus > 0:
        activation += min(pair_synergy_bonus * 0.5, 0.10)
        traces.append("Luật cụm triệu chứng đặc hiệu được kích hoạt => tăng độ tin cậy.")

    age_min, age_max = disease["age_range"]
    if age is not None and age_min <= age <= age_max:
        activation += 0.03
        traces.append("Ràng buộc ngữ cảnh: tuổi nằm trong nhóm thường gặp.")

    if gender != "all" and _gender_matches(disease["gender"], gender):
        if disease["gender"] != "all":
            activation += 0.02
            traces.append("Ràng buộc ngữ cảnh: giới tính phù hợp với bệnh cảnh đặc thù.")

    return min(activation, 0.24), traces


def _compute_bayesian_support(disease, selected_symptoms, scores):
    weights = disease["weights"]
    max_weight = max(weights.values()) if weights else 1.0
    prior = PRIORITY_PRIORS.get(disease.get("priority", "medium"), 0.48)

    log_odds = math.log(prior / max(1e-6, 1.0 - prior))

    for code in selected_symptoms:
        if code in weights:
            likelihood = 0.58 + (0.34 * (weights[code] / max_weight))
        else:
            likelihood = 0.20
        likelihood = min(max(likelihood, 0.05), 0.95)
        log_odds += math.log(likelihood / max(1e-6, 1.0 - likelihood))

    for code in disease.get("core", []):
        if code not in selected_symptoms:
            miss_factor = 0.34 if code in HIGH_SPECIFICITY_SYMPTOMS else 0.42
            log_odds += math.log(miss_factor / max(1e-6, 1.0 - miss_factor))

    unexplained_specific = len(
        [
            code
            for code in selected_symptoms
            if code in HIGH_SPECIFICITY_SYMPTOMS and code not in weights
        ]
    )
    if unexplained_specific:
        log_odds -= unexplained_specific * 0.45

    posterior = _sigmoid(log_odds)
    blended = (posterior * 0.7) + (scores["coverage_score"] * 0.2) + (scores["core_score"] * 0.1)
    return _clamp01(blended)


def _normalize_group_probabilities(results):
    if not results:
        return results

    grouped = {}
    for item in results:
        grouped.setdefault(item["group"], []).append(item)

    adjusted_results = []
    for items in grouped.values():
        items.sort(key=lambda x: x["raw_score"], reverse=True)
        leader_score = items[0]["raw_score"]

        for index, item in enumerate(items):
            competition_penalty = 0.0
            if index > 0:
                gap = max(0.0, leader_score - item["raw_score"])
                competition_penalty = min(gap * GROUP_COMPETITION_PENALTY, 0.12)

            item["raw_score"] = _clamp01(item["raw_score"] - competition_penalty)
            item["raw_percent"] = _round_percent(item["raw_score"] * 100)
            item["group_competition_penalty"] = _round_percent(competition_penalty * 100)
            adjusted_results.append(item)

    adjusted_results.sort(
        key=lambda x: (
            x["raw_percent"],
            x["core_percent"],
            x["coverage_percent"],
            x["precision_percent"],
        ),
        reverse=True,
    )
    return adjusted_results


def _normalize_display_percent(results):
    if not results:
        return results

    total_raw = sum(item["raw_score"] for item in results if item["raw_score"] > 0)
    if total_raw <= 0:
        for item in results:
            item["display_percent"] = 0.0
        return results

    for item in results:
        item["display_percent"] = round((item["raw_score"] / total_raw) * 100, 1)

    current_sum = round(sum(item["display_percent"] for item in results), 1)
    diff = round(100.0 - current_sum, 1)
    if results and diff != 0:
        results[0]["display_percent"] = round(results[0]["display_percent"] + diff, 1)

    return results


def infer_disease(
    user_symptoms,
    age=None,
    gender="all",
    severity="moderate",
    duration="fewdays",
    followup_adjustments=None,
):
    selected_symptoms = list(dict.fromkeys(user_symptoms))
    followup_adjustments = followup_adjustments or {}

    if not selected_symptoms:
        return []

    patient_facts = _build_patient_facts(selected_symptoms, age, gender, severity, duration)
    results = []
    severity_factor = SEVERITY_FACTORS.get(severity, 1.0)
    duration_factor = DURATION_FACTORS.get(duration, 1.0)

    for disease_id, disease in DISEASES.items():
        scores = _compute_support_scores(selected_symptoms, disease)
        matched_codes = scores["matched_codes"]
        matched_weight = scores["matched_weight"]

        if matched_weight <= 0:
            continue

        pair_synergy_bonus, matched_pairs = _compute_pair_synergy(disease_id, selected_symptoms)
        rule_activation, rule_traces = _compute_rule_activation(
            disease,
            scores,
            pair_synergy_bonus,
            age,
            gender,
        )
        bayes_score = _compute_bayesian_support(disease, selected_symptoms, scores)
        bonus = _compute_logic_bonus(scores, pair_synergy_bonus)
        logic_penalty = _compute_logic_penalty(disease_id, scores, selected_symptoms)

        deterministic_score = (
            (scores["coverage_score"] * 0.28)
            + (scores["precision_score"] * 0.16)
            + (scores["core_score"] * 0.32)
            + (scores["supportive_score"] * 0.10)
            + rule_activation
            + bonus
            - logic_penalty
        )

        confidence = (deterministic_score * 0.72) + (bayes_score * 0.28)
        confidence *= severity_factor
        confidence *= duration_factor

        age_adjustment = 0.0
        age_note = "Không có dữ liệu tuổi để hiệu chỉnh."
        if age is not None:
            age_min, age_max = disease["age_range"]
            if age < age_min or age > age_max:
                gap = min(abs(age - age_min), abs(age - age_max))
                if gap <= 5:
                    age_adjustment = -0.08
                elif gap <= 15:
                    age_adjustment = -0.18
                else:
                    age_adjustment = -0.32

                if disease_id == "breast_cancer" and age < 12:
                    age_adjustment = -0.65
                if disease_id in {"alzheimer", "parkinson"} and age < 35:
                    age_adjustment = -0.70
                if disease_id == "bph" and age < 35:
                    age_adjustment = -0.65
            else:
                age_adjustment = 0.04

            age_note = _build_age_note(age, age_min, age_max)

        gender_adjustment = 0.0
        gender_note = "Không có dữ liệu giới tính để hiệu chỉnh."
        if gender != "all":
            if _gender_matches(disease["gender"], gender):
                gender_adjustment = 0.03 if disease["gender"] != "all" else 0.0
                gender_note = "Giới tính phù hợp với bệnh cảnh thường gặp."
            else:
                gender_adjustment = -0.85
                gender_note = "Giới tính không phù hợp với bệnh cảnh này nên hệ thống giảm mạnh độ tin cậy."

        doctor_adjustment = followup_adjustments.get(disease_id, 0.0)

        confidence = confidence + age_adjustment + gender_adjustment + doctor_adjustment
        confidence = _clamp01(confidence)
        raw_percent = _round_percent(confidence * 100)

        matched = [SYMPTOMS[code] for code in matched_codes if code in SYMPTOMS]
        missing_core = [
            SYMPTOMS[code]
            for code in disease.get("core", [])
            if code not in selected_symptoms and code in SYMPTOMS
        ]

        flags = []
        if disease_id in EMERGENCY_DISEASES and raw_percent >= 30:
            flags.append("Cần cấp cứu ngay")
        if disease_id in HIGH_ALERT_DISEASES and raw_percent >= 35:
            flags.append("Cần khám chuyên khoa sớm")
        if age is not None and age_adjustment < -0.25:
            flags.append("Độ tuổi làm giảm độ tin cậy")
        if gender_adjustment < -0.5:
            flags.append("Giới tính không phù hợp")
        if doctor_adjustment > 0.12:
            flags.append("Câu trả lời follow-up làm tăng nghi ngờ")
        if scores["core_score"] < 0.34:
            flags.append("Thiếu triệu chứng cốt lõi")
        if len(selected_symptoms) <= 2:
            flags.append("Dữ kiện đầu vào còn ít")

        reasoning_steps = [
            f"Tập sự kiện đầu vào: {len(patient_facts)} fact, trong đó có {len(selected_symptoms)} fact triệu chứng.",
            f"Suy diễn tiến kích hoạt {len(rule_traces)} luật sản xuất cho {disease['name']}.",
        ]
        reasoning_steps.extend(rule_traces)
        if matched_pairs:
            pretty_pairs = []
            for pair in matched_pairs:
                pretty_pairs.append(" + ".join(SYMPTOMS.get(code, code) for code in pair))
            reasoning_steps.append(
                "Cụm triệu chứng đặc hiệu: " + "; ".join(pretty_pairs) + "."
            )
        reasoning_steps.append(
            f"Thành phần xác suất: điểm Bayes giả lập đạt {_round_percent(bayes_score * 100)}%."
        )

        results.append(
            {
                "id": disease_id,
                "name": disease["name"],
                "icon": disease["icon"],
                "theme": disease["theme"],
                "description": disease["description"],
                "group": disease.get("group", "Chưa phân nhóm"),
                "raw_score": confidence,
                "raw_percent": raw_percent,
                "display_percent": raw_percent,
                "matched": matched,
                "missing_core": missing_core,
                "matched_count": len(matched_codes),
                "selected_count": len(selected_symptoms),
                "coverage_percent": _round_percent(scores["coverage_score"] * 100),
                "precision_percent": _round_percent(scores["precision_score"] * 100),
                "core_percent": _round_percent(scores["core_score"] * 100),
                "supportive_percent": _round_percent(scores["supportive_score"] * 100),
                "missing_core_percent": _round_percent(scores["missing_core_ratio"] * 100),
                "bayes_percent": _round_percent(bayes_score * 100),
                "advice": disease["advice"],
                "age_note": age_note,
                "gender_note": gender_note,
                "flags": list(dict.fromkeys(flags)),
                "reasoning_steps": reasoning_steps,
                "doctor_summary": (
                    f"Hệ thống ghi nhận {len(matched)} triệu chứng phù hợp với {disease['name']}. "
                    f"Điểm được sinh ra từ 3 lớp suy luận: logic mệnh đề trên tập fact, "
                    f"luật sản xuất IF-THEN và một lớp xác suất giả lập theo Bayes để xếp hạng chẩn đoán phân biệt. "
                    f"Điểm trước chuẩn hóa của bệnh này là {raw_percent}%."
                ),
                "explanation": (
                    f"Độ phủ triệu chứng đạt {_round_percent(scores['coverage_score'] * 100)}%, "
                    f"độ tập trung đạt {_round_percent(scores['precision_score'] * 100)}%, "
                    f"khớp triệu chứng cốt lõi đạt {_round_percent(scores['core_score'] * 100)}%, "
                    f"triệu chứng hỗ trợ đạt {_round_percent(scores['supportive_score'] * 100)}%. "
                    f"Hệ thống cộng điểm khi luật cốt lõi, luật cụm triệu chứng đặc hiệu và ràng buộc ngữ cảnh được kích hoạt; "
                    f"đồng thời trừ điểm khi bệnh không giải thích được triệu chứng đặc hiệu hoặc thiếu core symptom."
                ),
                "formula": {
                    "coverage": _round_percent(scores["coverage_score"] * 100),
                    "precision": _round_percent(scores["precision_score"] * 100),
                    "core": _round_percent(scores["core_score"] * 100),
                    "supportive": _round_percent(scores["supportive_score"] * 100),
                    "rule_activation": _round_percent(rule_activation * 100),
                    "bayes": _round_percent(bayes_score * 100),
                    "bonus": _round_percent(bonus * 100),
                    "logic_penalty": _round_percent(logic_penalty * 100),
                    "missing_core_percent": _round_percent(scores["missing_core_ratio"] * 100),
                    "age_adjustment": _round_percent(age_adjustment * 100),
                    "gender_adjustment": _round_percent(gender_adjustment * 100),
                    "doctor_adjustment": _round_percent(doctor_adjustment * 100),
                    "severity_factor": _round_percent((severity_factor - 1) * 100),
                    "duration_factor": _round_percent((duration_factor - 1) * 100),
                },
            }
        )

    results = _normalize_group_probabilities(results)
    ranked_results = list(results)
    results = [item for item in ranked_results if item["raw_score"] >= MIN_DISPLAY_SCORE_THRESHOLD]

    if not results and ranked_results:
        fallback_results = [item for item in ranked_results if item["matched_count"] > 0]
        results = fallback_results[:MAX_RESULTS] if fallback_results else ranked_results[:MAX_RESULTS]

        for item in results:
            item["raw_score"] = max(item["raw_score"], 0.01)
            item["raw_percent"] = _round_percent(item["raw_score"] * 100)
            if "Tín hiệu triệu chứng còn yếu" not in item["flags"]:
                item["flags"].append("Tín hiệu triệu chứng còn yếu")
            item["doctor_summary"] = (
                f"Hệ thống đã ghi nhận một số dấu hiệu bước đầu liên quan tới {item['name']}, "
                f"nhưng dữ kiện hiện tại còn ít hoặc chưa đủ đặc hiệu để kết luận chắc hơn. "
                f"Bệnh này vẫn được giữ lại trong danh sách chẩn đoán phân biệt để tham khảo."
            )
            item["explanation"] = (
                "Bệnh này vẫn được giữ lại vì có ít nhất một phần triệu chứng trùng khớp với hồ sơ đầu vào. "
                "Tuy nhiên mức khớp hiện còn thấp, nên tỷ lệ chỉ mang ý nghĩa định hướng ban đầu và cần thêm triệu chứng "
                "hoặc câu hỏi làm rõ."
            )

    if not results:
        return []

    results = results[:MAX_RESULTS]
    results = _normalize_display_percent(results)

    top_score = results[0]["display_percent"]
    total_display = sum(item["display_percent"] for item in results)

    for item in results:
        item["percent"] = _round_percent(item["display_percent"])
        item["relative_level"] = (
            _round_percent((item["display_percent"] / top_score) * 100) if top_score else 0
        )
        item["differential_percent"] = (
            _round_percent((item["display_percent"] / total_display) * 100) if total_display else 0
        )

        if item["percent"] <= 0 and item["matched_count"] > 0:
            item["percent"] = 0.1

        distance = top_score - item["display_percent"]
        if distance <= 5:
            item["competitor_level"] = "Rất gần bệnh đứng đầu"
        elif distance <= 12:
            item["competitor_level"] = "Có thể nhầm lẫn"
        elif distance <= 22:
            item["competitor_level"] = "Khả năng phụ"
        else:
            item["competitor_level"] = "Ít nghĩ tới hơn"

    return results
