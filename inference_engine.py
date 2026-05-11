import math

from knowledge_base import DISEASES, SYMPTOMS


# File này là "bộ máy suy diễn" của hệ chuyên gia.
# Đầu vào: danh sách mã triệu chứng người dùng chọn/nhập, ví dụ ["S02", "S03"].
# Đầu ra: danh sách bệnh nghi ngờ đã được xếp hạng theo điểm tin cậy.
# Thuật toán không kết luận y khoa tuyệt đối; nó chỉ tính điểm định hướng bằng cách kết hợp:
# 1. Điểm khớp triệu chứng có trọng số.
# 2. Luật IF-THEN của hệ chuyên gia.
# 3. Điểm xác suất Bayes giả lập.
# 4. Hiệu chỉnh theo tuổi, giới tính, mức độ nặng, thời gian bệnh và câu hỏi bổ sung.
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

# Giảm bớt hiện tượng một bệnh trong cùng nhóm lấn át hết các bệnh còn lại.
GROUP_COMPETITION_PENALTY = 0.08

# Phạt khi thiếu triệu chứng cốt lõi.
CORE_MISSING_PENALTY = 0.20

# Thưởng thêm khi có cụm triệu chứng rất đặc hiệu.
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
    # Đưa mọi giá trị phần trăm về khoảng hợp lệ [0, 100].
    # Công thức: phần_trăm = min(max(value, 0), 100), sau đó làm tròn 1 chữ số thập phân.
    return round(max(0, min(value, 100)), 1)


def _clamp01(value):
    # Chuẩn hóa điểm xác suất/độ tin cậy về khoảng [0, 1].
    # 0 nghĩa là không có tín hiệu hỗ trợ, 1 nghĩa là tín hiệu rất mạnh.
    return max(0.0, min(value, 1.0))


def _sigmoid(value):
    # Hàm sigmoid chuyển log-odds về xác suất.
    # Công thức: sigmoid(x) = 1 / (1 + e^(-x)).
    # Trong file này dùng để đổi điểm Bayes ở dạng log-odds thành xác suất hậu nghiệm.
    return 1.0 / (1.0 + math.exp(-value))


def _gender_matches(rule, gender):
    # Nếu luật ghi "all" thì bệnh áp dụng cho mọi giới tính.
    if rule == "all":
        # Trả về True vì không cần kiểm tra nam/nữ.
        return True
    # Nếu không phải "all", giới tính người dùng phải đúng bằng giới tính trong luật.
    return rule == gender


def _is_gender_incompatible(disease, gender):
    # Kiểm tra bệnh có bị loại tuyệt đối theo giới tính hay không.
    # Quy tắc:
    # - Nếu người dùng chưa chọn giới tính cụ thể ("all") thì không loại theo giới.
    # - Nếu bệnh áp dụng cho mọi giới ("all") thì không loại.
    # - Nếu bệnh chỉ dành cho một giới cụ thể nhưng người dùng chọn giới khác thì loại.
    # Ví dụ:
    # - gender = "male", disease["gender"] = "female"  => loại bệnh phụ khoa.
    # - gender = "female", disease["gender"] = "male" => loại bệnh nam khoa.
    # Đây là ràng buộc cứng, không chỉ là trừ điểm, để tránh kết luận phi lý.
    disease_gender = disease.get("gender", "all")
    return gender != "all" and disease_gender != "all" and disease_gender != gender


def _is_age_clinically_implausible(disease_id, age):
    # Kiểm tra các trường hợp tuổi gần như không hợp lý với bệnh trong mô hình.
    # Khác với giới tính, tuổi thường không phải điều kiện loại trừ tuyệt đối cho mọi bệnh.
    # Tuy nhiên một số bệnh có ngưỡng tuổi rất đặc thù; nếu để lọt vào kết quả sẽ gây kết luận phi lý.
    # Vì vậy chỉ loại cứng những ca lệch tuổi rất rõ:
    # - Mãn kinh ở người quá trẻ.
    # - Phì đại/ung thư tuyến tiền liệt ở nam quá trẻ.
    # - Alzheimer/Parkinson ở người rất trẻ trong mô hình sàng lọc này.
    # - Ung thư vú ở trẻ nhỏ.
    if age is None:
        return False
    if disease_id == "menopause" and age < 35:
        return True
    if disease_id in {"bph", "prostate_cancer"} and age < 35:
        return True
    if disease_id in {"alzheimer", "parkinson"} and age < 35:
        return True
    if disease_id == "breast_cancer" and age < 12:
        return True
    return False


def _build_age_note(age, age_min, age_max):
    # Nếu người dùng không nhập tuổi, hệ thống không thể hiệu chỉnh theo tuổi.
    if age is None:
        # Trả về câu giải thích để hiển thị trong kết quả.
        return "Không có dữ liệu tuổi để hiệu chỉnh."
    # Nếu tuổi nhỏ hơn tuổi tối thiểu thường gặp của bệnh.
    if age < age_min:
        # Trả về ghi chú tuổi thấp hơn khoảng thường gặp.
        return f"Độ tuổi {age} nhỏ hơn nhóm tuổi thường gặp ({age_min}-{age_max})."
    # Nếu tuổi lớn hơn tuổi tối đa thường gặp của bệnh.
    if age > age_max:
        # Trả về ghi chú tuổi cao hơn khoảng thường gặp.
        return f"Độ tuổi {age} lớn hơn nhóm tuổi thường gặp ({age_min}-{age_max})."
    # Nếu không rơi vào hai trường hợp trên, tuổi nằm trong khoảng thường gặp.
    return f"Độ tuổi {age} nằm trong nhóm tuổi thường gặp ({age_min}-{age_max})."


def _build_patient_facts(selected_symptoms, age, gender, severity, duration):
    # Biến dữ liệu đầu vào thành tập "fact" để mô phỏng logic vị từ cấp 1.
    # Ví dụ:
    # - Người bệnh có triệu chứng S03  -> has(S03)
    # - Tuổi 25                       -> age(25)
    # - Giới tính nam                 -> gender(male)
    # Những fact này được dùng trong phần giải thích suy diễn.
    # Tạo fact has(Sxx) cho từng triệu chứng người dùng có.
    facts = {f"has({code})" for code in selected_symptoms}
    # Chỉ thêm fact tuổi nếu người dùng nhập tuổi hợp lệ.
    if age is not None:
        # Thêm fact age(n), ví dụ age(25).
        facts.add(f"age({age})")
    # Thêm fact giới tính để luật ngữ cảnh có thể sử dụng.
    facts.add(f"gender({gender})")
    # Thêm fact mức độ nặng để giải thích trạng thái đầu vào.
    facts.add(f"severity({severity})")
    # Thêm fact thời gian mắc bệnh để giải thích trạng thái đầu vào.
    facts.add(f"duration({duration})")
    # Trả về toàn bộ tập fact phục vụ phần reasoning_steps.
    return facts


def _compute_support_scores(selected_symptoms, disease):
    # Tính các điểm nền tảng cho một bệnh dựa trên mức độ khớp triệu chứng.
    # Ký hiệu:
    # - U: tập triệu chứng người dùng cung cấp.
    # - W_d(s): trọng số của triệu chứng s đối với bệnh d.
    # - C_d: tập triệu chứng cốt lõi của bệnh d.
    # Lấy bảng trọng số triệu chứng của bệnh đang xét.
    weights = disease["weights"]
    # Lấy danh sách triệu chứng cốt lõi; nếu không có thì dùng danh sách rỗng.
    core = disease.get("core", [])
    # Tính tổng trọng số toàn bộ triệu chứng của bệnh.
    total_weight = sum(weights.values())

    # matched_codes = U giao với tập triệu chứng có trong bệnh d.
    matched_codes = [code for code in selected_symptoms if code in weights]

    # matched_weight = tổng trọng số các triệu chứng đã khớp.
    # Công thức: matched_weight = sum(W_d(s)) với mọi s thuộc matched_codes.
    matched_weight = sum(weights[code] for code in matched_codes)

    # core_matched là các triệu chứng cốt lõi của bệnh d xuất hiện trong đầu vào.
    core_matched = [code for code in core if code in selected_symptoms]

    # coverage_score đo bệnh d đã được "phủ" bởi triệu chứng đầu vào bao nhiêu.
    # Công thức:
    # coverage = matched_weight / total_weight
    # Nếu độ phủ cao, người dùng có nhiều dấu hiệu quan trọng của bệnh đó.
    coverage_score = matched_weight / total_weight if total_weight > 0 else 0.0

    # precision_score đo các triệu chứng người dùng đưa vào có tập trung vào bệnh d không.
    # Với triệu chứng không thuộc bệnh d, hệ thống gán trọng số nền 0.22 để tránh chia 0
    # và để phạt trường hợp người dùng chọn nhiều triệu chứng nhưng bệnh chỉ giải thích được ít.
    # Công thức:
    # precision = matched_weight / sum(W_d(s) nếu s thuộc bệnh d, ngược lại 0.22)
    # Tính tổng trọng số của toàn bộ triệu chứng người dùng nhập dưới góc nhìn bệnh d.
    selected_weight_total = sum(weights.get(code, 0.22) for code in selected_symptoms)
    # Nếu mẫu số > 0 thì chia bình thường, ngược lại trả 0 để tránh lỗi chia cho 0.
    precision_score = matched_weight / selected_weight_total if selected_weight_total > 0 else 0.0

    # core_score đo tỷ lệ triệu chứng cốt lõi đã xuất hiện.
    # Công thức: core_score = |C_d giao U| / |C_d|.
    # Điểm này quan trọng vì thiếu triệu chứng cốt lõi thì bệnh phải bị giảm độ tin cậy.
    core_score = len(core_matched) / len(core) if core else 0.0

    # missing_core_ratio đo tỷ lệ triệu chứng cốt lõi còn thiếu.
    # Công thức: missing_core_ratio = |C_d - U| / |C_d|.
    missing_core = [code for code in core if code not in selected_symptoms]
    missing_core_ratio = len(missing_core) / len(core) if core else 0.0

    # supportive_score đo phần triệu chứng hỗ trợ, tức khớp với bệnh nhưng không nằm trong nhóm cốt lõi.
    # Công thức gần đúng:
    # supportive = sum(W_d(s) với s thuộc matched_codes và s không thuộc C_d) / total_weight.
    # Khởi tạo điểm triệu chứng hỗ trợ bằng 0.
    supportive_score = 0.0
    # Chỉ tính điểm hỗ trợ nếu bệnh có ít nhất một triệu chứng khớp.
    if matched_codes:
        # Lọc ra các triệu chứng khớp nhưng không thuộc nhóm cốt lõi.
        non_core_codes = [code for code in matched_codes if code not in core]
        # Nếu có triệu chứng hỗ trợ thì mới tính điểm.
        if non_core_codes:
            # Điểm hỗ trợ = tổng trọng số triệu chứng hỗ trợ / tổng trọng số bệnh.
            supportive_score = min(
                # Cộng trọng số của từng triệu chứng hỗ trợ.
                sum(weights.get(code, 0) for code in non_core_codes) / max(total_weight, 1.0),
                # Chặn trên là 1.0 để điểm luôn nằm trong khoảng [0, 1].
                1.0,
            )

    # Trả về toàn bộ chỉ số trung gian để các hàm sau dùng tiếp.
    return {
        # Danh sách mã triệu chứng đã khớp với bệnh.
        "matched_codes": matched_codes,
        # Tổng trọng số của các triệu chứng đã khớp.
        "matched_weight": matched_weight,
        # Danh sách triệu chứng cốt lõi đã khớp.
        "core_matched": core_matched,
        # Điểm độ phủ triệu chứng.
        "coverage_score": coverage_score,
        # Điểm độ tập trung triệu chứng.
        "precision_score": precision_score,
        # Điểm khớp triệu chứng cốt lõi.
        "core_score": core_score,
        # Tỷ lệ triệu chứng cốt lõi còn thiếu.
        "missing_core_ratio": missing_core_ratio,
        # Điểm triệu chứng hỗ trợ.
        "supportive_score": supportive_score,
    }


def _compute_pair_synergy(disease_id, selected_symptoms):
    # Tính điểm thưởng khi xuất hiện một cụm triệu chứng đặc hiệu.
    # Ví dụ: đau ngực trái + đau lan tay trái là cụm rất đặc hiệu cho nhồi máu cơ tim.
    # Công thức: pair_bonus = min(sum(delta của các cụm khớp), 0.24).
    # Giới hạn 0.24 để một vài cụm đặc hiệu không làm điểm tăng quá mức.
    # Chuyển danh sách triệu chứng thành tập hợp để kiểm tra tập con nhanh hơn.
    selected_set = set(selected_symptoms)
    # Khởi tạo tổng điểm thưởng cụm triệu chứng.
    # Khởi tạo tổng điểm thưởng logic.
    bonus = 0.0
    # Lưu lại các cụm đã khớp để hiển thị trong phần giải thích.
    matched_pairs = []
    # Duyệt toàn bộ cụm triệu chứng đặc hiệu của bệnh đang xét.
    for symptom_set, delta in PAIR_SYNERGY.get(disease_id, []):
        # Nếu toàn bộ triệu chứng trong cụm đều nằm trong đầu vào người dùng.
        if symptom_set.issubset(selected_set):
            # Cộng điểm thưởng delta của cụm đó.
            bonus += delta
            # Lưu cụm đã khớp, sắp xếp để hiển thị ổn định.
            matched_pairs.append(sorted(symptom_set))
    # Trả về bonus đã chặn trên 0.24 và danh sách cụm đã khớp.
    return min(bonus, 0.24), matched_pairs


def _compute_negative_hint_penalty(disease_id, selected_symptoms):
    # Trừ điểm nếu người dùng có các triệu chứng "gợi ý ngược" với bệnh đang xét.
    # Ví dụ cảm lạnh thông thường mà có ho ra máu/sụt cân thì phải giảm điểm.
    # Công thức: penalty = min(số triệu chứng gợi ý ngược * 0.05, 0.16).
    # Chuyển triệu chứng đầu vào thành tập hợp để lấy giao nhanh.
    selected_set = set(selected_symptoms)
    # Lấy tập triệu chứng gợi ý ngược của bệnh; nếu bệnh không có thì dùng tập rỗng.
    bad_set = DISEASE_NEGATIVE_HINTS.get(disease_id, set())
    # Đếm số triệu chứng gợi ý ngược xuất hiện trong đầu vào.
    hit = len(selected_set.intersection(bad_set))
    # Mỗi triệu chứng gợi ý ngược phạt 0.05, tối đa 0.16.
    return min(hit * 0.05, 0.16)


def _compute_specificity_penalty(selected_symptoms, matched_codes):
    # Trừ điểm khi người dùng có triệu chứng rất đặc hiệu nhưng bệnh hiện tại không giải thích được.
    # Ví dụ người dùng có liệt mặt/méo miệng nhưng bệnh đang xét không liên quan thần kinh.
    # Công thức: penalty = min(số triệu chứng đặc hiệu bị bỏ sót * 0.04, 0.20).
    # Tập triệu chứng người dùng đưa vào.
    selected_set = set(selected_symptoms)
    # Tập triệu chứng mà bệnh hiện tại giải thích được.
    matched_set = set(matched_codes)
    # Tìm các triệu chứng đặc hiệu có trong đầu vào nhưng không được bệnh hiện tại giải thích.
    missed_specific = [
        # Giữ lại mã triệu chứng đặc hiệu bị bỏ sót.
        symptom
        # Duyệt từng triệu chứng người dùng đã nhập.
        for symptom in selected_set
        # Điều kiện: triệu chứng đặc hiệu và không nằm trong tập triệu chứng đã khớp.
        if symptom in HIGH_SPECIFICITY_SYMPTOMS and symptom not in matched_set
    ]
    # Mỗi triệu chứng đặc hiệu bị bỏ sót phạt 0.04, tối đa 0.20.
    return min(len(missed_specific) * 0.04, 0.20)


def _compute_logic_bonus(scores, pair_synergy_bonus):
    # Tính điểm cộng theo logic luật.
    # Mục tiêu: thưởng cho bệnh có điểm cốt lõi tốt, độ phủ cao, triệu chứng hỗ trợ tốt,
    # và có cụm triệu chứng đặc hiệu.
    bonus = 0.0

    # Nếu khớp đủ 100% triệu chứng cốt lõi, tăng 0.08.
    # Nếu khớp từ 67% trở lên, tăng 0.04.
    if scores["core_score"] == 1:
        # Cộng 0.08 vì bệnh khớp đủ toàn bộ triệu chứng cốt lõi.
        bonus += 0.08
    elif scores["core_score"] >= 0.67:
        # Cộng 0.04 vì bệnh khớp phần lớn triệu chứng cốt lõi.
        bonus += 0.04

    # Nếu độ phủ >= 60%, nghĩa là phần lớn cấu trúc triệu chứng của bệnh đã được phủ.
    if scores["coverage_score"] >= 0.60:
        # Cộng 0.03 vì độ phủ triệu chứng đã đủ cao.
        bonus += 0.03

    # Nếu có triệu chứng hỗ trợ đáng kể, tăng nhẹ để phân biệt với bệnh chỉ khớp nhóm cốt lõi.
    if scores["supportive_score"] >= 0.15:
        # Cộng 0.02 vì có thêm triệu chứng hỗ trợ ngoài nhóm cốt lõi.
        bonus += 0.02

    # Điều kiện AND: vừa có ít nhất nửa nhóm cốt lõi, vừa có độ tập trung tốt.
    # Đây là luật "hồ sơ đầu vào tập trung vào bệnh đang xét".
    if scores["core_score"] >= 0.50 and scores["precision_score"] >= 0.50:
        # Cộng 0.05 vì vừa khớp nhóm cốt lõi vừa có độ tập trung tốt.
        bonus += 0.05

    # Cộng thêm điểm cụm triệu chứng đặc hiệu đã tính ở _compute_pair_synergy.
    bonus += pair_synergy_bonus
    # Trả về tổng điểm thưởng cho bệnh đang xét.
    return bonus


def _compute_logic_penalty(disease_id, scores, selected_symptoms):
    # Tính điểm phạt theo logic loại trừ.
    # Bệnh bị giảm điểm khi thiếu triệu chứng cốt lõi, có triệu chứng gợi ý ngược,
    # hoặc không giải thích được triệu chứng đặc hiệu trong đầu vào.
    # Khởi tạo tổng điểm phạt logic.
    penalty = 0.0

    # Không khớp triệu chứng cốt lõi nào thì phạt mạnh.
    if scores["core_score"] == 0:
        # Phạt 0.20 vì không có triệu chứng cốt lõi nào khớp.
        penalty += 0.20
    elif scores["core_score"] < 0.34:
        # Phạt 0.10 vì tỷ lệ triệu chứng cốt lõi khớp quá thấp.
        penalty += 0.10

    # Phạt theo tỷ lệ triệu chứng cốt lõi còn thiếu.
    penalty += scores["missing_core_ratio"] * CORE_MISSING_PENALTY
    # Phạt nếu có triệu chứng gợi ý ngược với bệnh.
    penalty += _compute_negative_hint_penalty(disease_id, selected_symptoms)
    # Phạt nếu đầu vào có triệu chứng đặc hiệu nhưng bệnh không giải thích được.
    penalty += _compute_specificity_penalty(selected_symptoms, scores["matched_codes"])

    # Trả về tổng điểm phạt để trừ khỏi điểm logic tất định.
    return penalty


def _compute_rule_activation(disease, scores, pair_synergy_bonus, age, gender):
    # Mô phỏng luật sản xuất IF-THEN trong hệ chuyên gia.
    # Mỗi luật đúng sẽ cộng một phần điểm kích hoạt và ghi lại vết giải thích.
    # Tổng rule_activation bị chặn tối đa 0.24 để không lấn át toàn bộ mô hình.
    # Khởi tạo điểm kích hoạt luật.
    activation = 0.0
    # Khởi tạo danh sách câu giải thích luật nào đã được kích hoạt.
    traces = []

    # Luật 1:
    # IF khớp toàn bộ triệu chứng cốt lõi THEN kích hoạt mạnh giả thuyết bệnh.
    if scores["core_score"] == 1:
        # Cộng 0.16 khi toàn bộ triệu chứng cốt lõi đều đúng.
        activation += 0.16
        # Ghi lại vết suy diễn để người dùng hiểu vì sao bệnh tăng điểm.
        traces.append("Luật sản xuất: tất cả triệu chứng cốt lõi đều đúng => kích hoạt mạnh bệnh.")
    elif scores["core_score"] >= 0.5:
        # Cộng 0.08 khi ít nhất một nửa triệu chứng cốt lõi đúng.
        activation += 0.08
        # Ghi lại vết suy diễn mức trung bình.
        traces.append("Luật sản xuất: đã khớp một phần lớn triệu chứng cốt lõi => giữ bệnh trong tập giả thuyết.")

    # Luật 2:
    # IF độ phủ >= 0.5 AND độ tập trung >= 0.45 THEN hồ sơ phù hợp với bệnh.
    if scores["coverage_score"] >= 0.5 and scores["precision_score"] >= 0.45:
        # Cộng 0.08 khi cả độ phủ và độ tập trung đều đạt ngưỡng.
        activation += 0.08
        # Ghi lại luật AND giữa độ phủ và độ tập trung.
        traces.append("Luật mệnh đề: (độ phủ cao AND độ tập trung cao) => hồ sơ phù hợp với bệnh.")

    # Luật 3:
    # IF có cụm triệu chứng đặc hiệu THEN tăng độ tin cậy.
    if pair_synergy_bonus > 0:
        # Cộng một nửa điểm thưởng cụm, tối đa 0.10, vào điểm kích hoạt luật.
        activation += min(pair_synergy_bonus * 0.5, 0.10)
        # Ghi lại rằng cụm triệu chứng đặc hiệu đã kích hoạt.
        traces.append("Luật cụm triệu chứng đặc hiệu được kích hoạt => tăng độ tin cậy.")

    # Luật 4:
    # IF tuổi thuộc khoảng tuổi thường gặp của bệnh THEN tăng nhẹ.
    # Lấy khoảng tuổi thường gặp của bệnh.
    age_min, age_max = disease["age_range"]
    # Chỉ xét tuổi nếu người dùng có nhập tuổi và tuổi nằm trong khoảng.
    if age is not None and age_min <= age <= age_max:
        # Cộng 0.03 vì ngữ cảnh tuổi phù hợp.
        activation += 0.03
        # Ghi lại vết suy diễn theo tuổi.
        traces.append("Ràng buộc ngữ cảnh: tuổi nằm trong nhóm thường gặp.")

    # Luật 5:
    # IF giới tính phù hợp với bệnh đặc thù THEN tăng nhẹ.
    # Chỉ xét giới tính nếu người dùng chọn giới tính cụ thể.
    if gender != "all" and _gender_matches(disease["gender"], gender):
        # Chỉ cộng điểm nếu bản thân bệnh có giới tính đặc thù.
        if disease["gender"] != "all":
            # Cộng 0.02 vì giới tính phù hợp với bệnh đặc thù.
            activation += 0.02
            # Ghi lại vết suy diễn theo giới tính.
            traces.append("Ràng buộc ngữ cảnh: giới tính phù hợp với bệnh cảnh đặc thù.")

    # Chặn điểm kích hoạt tối đa 0.24 và trả kèm danh sách vết suy diễn.
    return min(activation, 0.24), traces


def _compute_bayesian_support(disease, selected_symptoms, scores):
    # Tính điểm xác suất kiểu Bayes giả lập.
    # Vì dự án không có dữ liệu thống kê y khoa thật, độ hợp lý được xấp xỉ từ trọng số triệu chứng.
    #
    # Bước 1: xác suất tiên nghiệm lấy theo mức ưu tiên bệnh.
    # odds = P(d) / (1 - P(d))
    # log_odds = ln(odds)
    #
    # Bước 2: với từng triệu chứng s:
    # - Nếu s thuộc bệnh d, độ hợp lý tăng theo trọng số W_d(s).
    # - Nếu s không thuộc bệnh d, độ hợp lý thấp để giảm xác suất.
    # log_odds += ln(P(s|d) / (1 - P(s|d)))
    #
    # Bước 3: thiếu triệu chứng cốt lõi thì cộng log của hệ số thiếu < 0.5, làm log_odds giảm.
    #
    # Bước 4: xác suất hậu nghiệm = sigmoid(log_odds).
    # Cuối cùng trộn xác suất hậu nghiệm với độ phủ/điểm cốt lõi để kết quả ổn định hơn.
    # Lấy trọng số triệu chứng của bệnh để ước lượng độ hợp lý.
    weights = disease["weights"]
    # Lấy trọng số lớn nhất để chuẩn hóa W_d(s) / max_weight.
    max_weight = max(weights.values()) if weights else 1.0
    # Lấy xác suất tiên nghiệm theo mức ưu tiên; nếu thiếu mức ưu tiên thì dùng medium = 0.48.
    prior = PRIORITY_PRIORS.get(disease.get("priority", "medium"), 0.48)

    # Chuyển xác suất tiên nghiệm sang log-odds: ln(P / (1 - P)).
    log_odds = math.log(prior / max(1e-6, 1.0 - prior))

    # Duyệt từng triệu chứng người dùng nhập để cập nhật bằng độ hợp lý.
    for code in selected_symptoms:
        # Nếu triệu chứng có trong bảng trọng số của bệnh.
        if code in weights:
            # Trọng số càng cao thì độ hợp lý càng gần 0.92.
            # Công thức: độ_hợp_lý = 0.58 + 0.34 * W_d(s) / trọng_số_lớn_nhất.
            likelihood = 0.58 + (0.34 * (weights[code] / max_weight))
        else:
            # Triệu chứng không thuộc bệnh đang xét làm khả năng giải thích thấp.
            likelihood = 0.20
        # Chặn độ hợp lý trong [0.05, 0.95] để tránh log(0) hoặc xác suất tuyệt đối.
        likelihood = min(max(likelihood, 0.05), 0.95)
        # Cộng log của tỷ số hợp lý vào log_odds.
        log_odds += math.log(likelihood / max(1e-6, 1.0 - likelihood))

    # Duyệt các triệu chứng cốt lõi của bệnh để phạt nếu thiếu.
    for code in disease.get("core", []):
        # Nếu triệu chứng cốt lõi không xuất hiện trong đầu vào.
        if code not in selected_symptoms:
            # Triệu chứng đặc hiệu bị thiếu thì phạt mạnh hơn bằng miss_factor 0.34.
            miss_factor = 0.34 if code in HIGH_SPECIFICITY_SYMPTOMS else 0.42
            # Cộng log của miss_factor vào log_odds; vì < 0.5 nên làm xác suất giảm.
            log_odds += math.log(miss_factor / max(1e-6, 1.0 - miss_factor))

    # Đếm triệu chứng đặc hiệu có trong đầu vào nhưng bệnh không giải thích được.
    unexplained_specific = len(
        [
            # Mã triệu chứng đặc hiệu không nằm trong weights của bệnh.
            code
            # Duyệt từng triệu chứng đầu vào.
            for code in selected_symptoms
            # Điều kiện đặc hiệu nhưng bệnh không có triệu chứng này.
            if code in HIGH_SPECIFICITY_SYMPTOMS and code not in weights
        ]
    )
    # Nếu có triệu chứng đặc hiệu không giải thích được.
    if unexplained_specific:
        # Mỗi triệu chứng trừ 0.45 trực tiếp trên log_odds.
        log_odds -= unexplained_specific * 0.45

    # Đưa log_odds về xác suất hậu nghiệm bằng sigmoid.
    posterior = _sigmoid(log_odds)

    # Trộn điểm:
    # điểm_Bayes = xác_suất_hậu_nghiệm * 0.7 + độ_phủ * 0.2 + điểm_cốt_lõi * 0.1.
    # Xác suất hậu nghiệm là thành phần chính; độ phủ/điểm cốt lõi giúp tránh điểm Bayes dao động quá mạnh.
    # Tính điểm Bayes cuối cùng sau khi trộn.
    blended = (posterior * 0.7) + (scores["coverage_score"] * 0.2) + (scores["core_score"] * 0.1)
    # Chặn kết quả trong [0, 1] rồi trả về.
    return _clamp01(blended)


def _normalize_group_probabilities(results):
    # Chuẩn hóa cạnh tranh trong cùng nhóm bệnh.
    # Mục tiêu: nếu nhiều bệnh cùng nhóm đều khớp giống nhau, bệnh đứng sau bị trừ nhẹ
    # để tránh danh sách kết quả bị một nhóm bệnh chiếm hết.
    # Công thức phạt:
    # điểm_phạt_cạnh_tranh = min((điểm_dẫn_đầu - điểm_hiện_tại) * 0.08, 0.12).
    # Nếu danh sách rỗng thì không có gì để chuẩn hóa.
    if not results:
        # Trả nguyên danh sách rỗng.
        return results

    # Tạo từ điển gom bệnh theo nhóm.
    grouped = {}
    # Duyệt từng kết quả bệnh.
    for item in results:
        # Nếu nhóm chưa tồn tại thì tạo danh sách mới, sau đó thêm bệnh hiện tại vào nhóm.
        grouped.setdefault(item["group"], []).append(item)

    # Danh sách kết quả sau khi đã phạt cạnh tranh trong nhóm.
    adjusted_results = []
    # Duyệt từng nhóm bệnh.
    for items in grouped.values():
        # Sắp xếp bệnh trong nhóm theo điểm thô giảm dần.
        items.sort(key=lambda x: x["raw_score"], reverse=True)
        # Bệnh đầu nhóm là bệnh mạnh nhất của nhóm.
        leader_score = items[0]["raw_score"]

        # Duyệt từng bệnh trong nhóm sau khi đã sắp xếp.
        for index, item in enumerate(items):
            # Mặc định bệnh đứng đầu không bị phạt.
            competition_penalty = 0.0
            # Chỉ phạt các bệnh đứng sau bệnh dẫn đầu.
            if index > 0:
                # Tính khoảng cách điểm giữa bệnh dẫn đầu và bệnh hiện tại.
                gap = max(0.0, leader_score - item["raw_score"])
                # Điểm phạt = gap * hệ số cạnh tranh, tối đa 0.12.
                competition_penalty = min(gap * GROUP_COMPETITION_PENALTY, 0.12)

            # Trừ điểm phạt khỏi điểm thô và chặn trong [0, 1].
            item["raw_score"] = _clamp01(item["raw_score"] - competition_penalty)
            # Cập nhật lại phần trăm thô sau khi phạt.
            item["raw_percent"] = _round_percent(item["raw_score"] * 100)
            # Lưu lại mức phạt để có thể giải thích hoặc kiểm tra lỗi.
            item["group_competition_penalty"] = _round_percent(competition_penalty * 100)
            # Đưa bệnh đã hiệu chỉnh vào danh sách chung.
            adjusted_results.append(item)

    # Sắp xếp lại toàn bộ kết quả sau phạt.
    adjusted_results.sort(
        # Ưu tiên phần trăm thô, sau đó điểm cốt lõi, độ phủ, độ tập trung.
        key=lambda x: (
            x["raw_percent"],
            x["core_percent"],
            x["coverage_percent"],
            x["precision_percent"],
        ),
        # Sắp xếp giảm dần để bệnh mạnh nhất đứng trước.
        reverse=True,
    )
    # Trả về danh sách đã chuẩn hóa cạnh tranh nhóm.
    return adjusted_results


def _normalize_display_percent(results):
    # Chuẩn hóa điểm thô thành phần trăm hiển thị sao cho tổng các bệnh hiển thị = 100%.
    # Công thức:
    # phần_trăm_hiển_thị_i = điểm_thô_i / tổng_điểm_thô * 100.
    # Nếu không có kết quả thì trả nguyên danh sách.
    if not results:
        return results

    # Tổng điểm thô của các bệnh có điểm > 0.
    total_raw = sum(item["raw_score"] for item in results if item["raw_score"] > 0)
    # Nếu tổng điểm <= 0 thì không thể chia tỷ lệ phần trăm.
    if total_raw <= 0:
        # Gán 0% cho từng bệnh.
        for item in results:
            item["display_percent"] = 0.0
        # Trả về kết quả đã gán 0%.
        return results

    # Duyệt từng bệnh để tính phần trăm hiển thị.
    for item in results:
        # Công thức: phần_trăm_hiển_thị = điểm_thô / tổng_điểm_thô * 100.
        item["display_percent"] = round((item["raw_score"] / total_raw) * 100, 1)

    # Tính tổng sau khi làm tròn 1 chữ số thập phân.
    current_sum = round(sum(item["display_percent"] for item in results), 1)
    # Tính sai lệch so với 100%.
    diff = round(100.0 - current_sum, 1)
    # Nếu có sai lệch do làm tròn, cộng/trừ vào bệnh đứng đầu.
    if results and diff != 0:
        # Điều chỉnh bệnh đầu tiên để tổng đúng bằng 100%.
        results[0]["display_percent"] = round(results[0]["display_percent"] + diff, 1)

    # Trả về kết quả đã có phần trăm hiển thị.
    return results


def infer_disease(
    user_symptoms,
    age=None,
    gender="all",
    severity="moderate",
    duration="fewdays",
    followup_adjustments=None,
):
    # ================= THUẬT TOÁN TÌM KIẾM BỆNH PHÙ HỢP =================
    # Đây là hàm chính của hệ thống chẩn đoán định hướng.
    #
    # Chiến lược tìm kiếm cụ thể:
    # - Không dùng BFS/DFS trên đồ thị, vì kho bệnh không được biểu diễn thành các cạnh.
    # - Hệ thống dùng "generate-and-test" / tìm kiếm vét cạn có xếp hạng:
    #   1. Generate: lấy từng bệnh trong DISEASES làm một giả thuyết chẩn đoán.
    #   2. Test: so khớp triệu chứng người dùng với weights/core của bệnh đó.
    #   3. Score: tính điểm phù hợp bằng logic luật + Bayes giả lập + ngữ cảnh.
    #   4. Rank: sắp xếp các giả thuyết theo điểm giảm dần.
    #
    # Vì số bệnh trong DISEASES hữu hạn, duyệt hết từng bệnh giúp không bỏ sót giả thuyết.
    # Độ phức tạp xấp xỉ O(D * S), trong đó:
    # - D là số bệnh trong kho tri thức.
    # - S là số triệu chứng người dùng nhập.
    #
    # Quy trình chi tiết:
    # 1. Chuẩn hóa danh sách triệu chứng đầu vào.
    # 2. Duyệt từng bệnh trong kho tri thức DISEASES.
    # 3. Tính độ phủ, độ tập trung, điểm cốt lõi, điểm hỗ trợ.
    # 4. Tính điểm thưởng/điểm phạt theo luật IF-THEN.
    # 5. Tính điểm Bayes xấp xỉ để mô phỏng lập luận xác suất.
    # 6. Ghép điểm, hiệu chỉnh tuổi/giới/mức độ/thời gian/follow-up.
    # 7. Lọc ngưỡng, chuẩn hóa phần trăm và trả về danh sách bệnh được xếp hạng.
    # Loại bỏ triệu chứng trùng nhưng vẫn giữ thứ tự nhập ban đầu.
    selected_symptoms = list(dict.fromkeys(user_symptoms))
    # Nếu không truyền điều chỉnh từ câu hỏi bổ sung thì dùng từ điển rỗng.
    followup_adjustments = followup_adjustments or {}

    # Nếu người dùng không có triệu chứng nào.
    if not selected_symptoms:
        # Không đủ dữ liệu để suy diễn nên trả danh sách rỗng.
        return []

    # Tạo tập fact phục vụ suy diễn và giải thích.
    patient_facts = _build_patient_facts(selected_symptoms, age, gender, severity, duration)
    # Khởi tạo danh sách kết quả bệnh.
    results = []
    # Lấy hệ số hiệu chỉnh theo mức độ nặng; nếu không hợp lệ thì dùng 1.0.
    severity_factor = SEVERITY_FACTORS.get(severity, 1.0)
    # Lấy hệ số hiệu chỉnh theo thời gian mắc; nếu không hợp lệ thì dùng 1.0.
    duration_factor = DURATION_FACTORS.get(duration, 1.0)

    # VÒNG LẶP TÌM KIẾM CHÍNH:
    # Mỗi vòng lặp xét đúng một bệnh trong DISEASES.
    # Nếu bệnh có ít nhất một triệu chứng khớp, nó được giữ làm ứng viên.
    # Nếu không khớp triệu chứng nào, bệnh bị loại ngay bằng câu lệnh continue bên dưới.
    for disease_id, disease in DISEASES.items():
        # Với mỗi bệnh d, hệ thống xem d là một giả thuyết cần kiểm tra.
        # Ràng buộc giới tính là điều kiện loại trừ cứng.
        # Nếu người dùng là nam thì bệnh chỉ gặp ở nữ không được đưa vào kết quả.
        # Nếu người dùng là nữ thì bệnh chỉ gặp ở nam không được đưa vào kết quả.
        # Việc loại ở đầu vòng lặp giúp bệnh khác giới không còn xuất hiện dù triệu chứng khớp 100%.
        if _is_gender_incompatible(disease, gender):
            continue
        # Ràng buộc tuổi đặc biệt cũng được loại sớm để tránh bệnh rất lệch tuổi lọt vào top kết quả.
        # Các bệnh chỉ hơi lệch tuổi vẫn được giữ lại nhưng sẽ bị trừ điểm và có ghi chú age_note.
        if _is_age_clinically_implausible(disease_id, age):
            continue

        # Tính các điểm độ phủ, độ tập trung, cốt lõi, hỗ trợ cho bệnh này.
        scores = _compute_support_scores(selected_symptoms, disease)
        # Lấy danh sách triệu chứng đã khớp từ bộ điểm trung gian.
        matched_codes = scores["matched_codes"]
        # Lấy tổng trọng số triệu chứng đã khớp.
        matched_weight = scores["matched_weight"]

        # Nếu không khớp triệu chứng nào thì bỏ qua bệnh này.
        if matched_weight <= 0:
            continue

        # Tính điểm thưởng nếu đầu vào có cụm triệu chứng đặc hiệu của bệnh.
        pair_synergy_bonus, matched_pairs = _compute_pair_synergy(disease_id, selected_symptoms)
        # Tính điểm kích hoạt luật IF-THEN và lấy vết giải thích.
        rule_activation, rule_traces = _compute_rule_activation(
            # Truyền dữ liệu bệnh đang xét.
            disease,
            # Truyền các điểm nền tảng đã tính.
            scores,
            # Truyền điểm thưởng cụm triệu chứng.
            pair_synergy_bonus,
            # Truyền tuổi để xét luật ngữ cảnh tuổi.
            age,
            # Truyền giới tính để xét luật ngữ cảnh giới.
            gender,
        )
        # Tính điểm xác suất Bayes giả lập.
        bayes_score = _compute_bayesian_support(disease, selected_symptoms, scores)
        # Tính điểm cộng logic.
        bonus = _compute_logic_bonus(scores, pair_synergy_bonus)
        # Tính điểm phạt logic.
        logic_penalty = _compute_logic_penalty(disease_id, scores, selected_symptoms)

        # Điểm logic tất định.
        # Công thức:
        # deterministic =
        #   độ_phủ * 0.28
        # + độ_tập_trung * 0.16
        # + điểm_cốt_lõi * 0.32
        # + điểm_hỗ_trợ * 0.10
        # + rule_activation
        # + điểm_thưởng
        # - điểm_phạt_logic
        # Trong đó điểm cốt lõi được đặt trọng số cao nhất vì triệu chứng cốt lõi quyết định mạnh.
        deterministic_score = (
            (scores["coverage_score"] * 0.28)
            + (scores["precision_score"] * 0.16)
            + (scores["core_score"] * 0.32)
            + (scores["supportive_score"] * 0.10)
            + rule_activation
            + bonus
            - logic_penalty
        )

        # Kết hợp điểm logic với điểm Bayes.
        # Công thức:
        # confidence = deterministic_score * 0.72 + bayes_score * 0.28.
        # Logic chiếm 72% vì đây là hệ chuyên gia dựa luật; Bayes chiếm 28% để hỗ trợ xếp hạng.
        confidence = (deterministic_score * 0.72) + (bayes_score * 0.28)

        # Hiệu chỉnh theo mức độ nặng và thời gian mắc.
        # Mức severe hoặc thời gian weeks làm điểm tăng nhẹ; mild hoặc 1day làm điểm giảm nhẹ.
        confidence *= severity_factor
        confidence *= duration_factor

        # Hiệu chỉnh theo tuổi:
        # - Tuổi nằm trong age_range: cộng 0.04.
        # - Lệch nhẹ <= 5 tuổi: trừ 0.08.
        # - Lệch vừa <= 15 tuổi: trừ 0.18.
        # - Lệch xa hơn: trừ 0.32.
        # Một số bệnh đặc thù có phạt mạnh hơn để tránh kết quả phi lý.
        # Khởi tạo điểm hiệu chỉnh tuổi.
        age_adjustment = 0.0
        # Khởi tạo câu giải thích tuổi mặc định.
        age_note = "Không có dữ liệu tuổi để hiệu chỉnh."
        # Chỉ hiệu chỉnh tuổi nếu có tuổi hợp lệ.
        if age is not None:
            # Lấy khoảng tuổi thường gặp của bệnh.
            age_min, age_max = disease["age_range"]
            # Nếu tuổi nằm ngoài khoảng thường gặp.
            if age < age_min or age > age_max:
                # Tính khoảng cách gần nhất từ tuổi người dùng đến biên khoảng tuổi.
                gap = min(abs(age - age_min), abs(age - age_max))
                # Nếu lệch không quá 5 tuổi.
                if gap <= 5:
                    # Phạt nhẹ 0.08.
                    age_adjustment = -0.08
                # Nếu lệch không quá 15 tuổi.
                elif gap <= 15:
                    # Phạt vừa 0.18.
                    age_adjustment = -0.18
                # Nếu lệch xa hơn 15 tuổi.
                else:
                    # Phạt mạnh 0.32.
                    age_adjustment = -0.32

                # Nếu nghi ung thư vú ở trẻ dưới 12 tuổi thì rất không phù hợp.
                if disease_id == "breast_cancer" and age < 12:
                    # Phạt đặc biệt 0.65.
                    age_adjustment = -0.65
                # Alzheimer/Parkinson thường không phù hợp người dưới 35 tuổi trong mô hình này.
                if disease_id in {"alzheimer", "parkinson"} and age < 35:
                    # Phạt đặc biệt 0.70.
                    age_adjustment = -0.70
                # Phì đại tuyến tiền liệt là bệnh thường gặp ở nam lớn tuổi.
                if disease_id == "bph" and age < 35:
                    # Phạt đặc biệt 0.65 nếu tuổi quá trẻ.
                    age_adjustment = -0.65
            else:
                # Nếu tuổi nằm trong khoảng thường gặp, cộng nhẹ 0.04.
                age_adjustment = 0.04

            # Tạo câu giải thích tuổi để đưa ra giao diện.
            age_note = _build_age_note(age, age_min, age_max)

        # Hiệu chỉnh theo giới tính:
        # - Bệnh sai giới đã bị loại ở đầu vòng lặp.
        # - Nếu bệnh áp dụng cho mọi giới thì không cộng.
        # - Nếu bệnh đặc thù và giới tính khớp thì cộng 0.03.
        # Khởi tạo điểm hiệu chỉnh giới tính.
        gender_adjustment = 0.0
        # Khởi tạo câu giải thích giới tính mặc định.
        gender_note = "Không có dữ liệu giới tính để hiệu chỉnh."
        # Chỉ xét giới tính nếu người dùng đã chọn nam/nữ cụ thể.
        if gender != "all":
            # Nếu giới tính phù hợp với luật giới tính của bệnh.
            if _gender_matches(disease["gender"], gender):
                # Nếu bệnh đặc thù giới tính thì cộng 0.03, còn bệnh áp dụng mọi giới thì cộng 0.
                gender_adjustment = 0.03 if disease["gender"] != "all" else 0.0
                # Ghi chú rằng giới tính phù hợp.
                gender_note = "Giới tính phù hợp với bệnh cảnh thường gặp."

        # Follow-up là phần suy diễn lùi: hệ thống hỏi thêm rồi dùng câu trả lời
        # để tăng/giảm điểm một bệnh cụ thể.
        # Lấy điểm điều chỉnh từ câu trả lời bổ sung cho bệnh hiện tại.
        doctor_adjustment = followup_adjustments.get(disease_id, 0.0)

        # Điểm cuối cùng trước chuẩn hóa hiển thị.
        # Công thức:
        # confidence = clamp01(confidence + age_adjustment + gender_adjustment + followup_adjustment)
        # Cộng các hiệu chỉnh ngữ cảnh vào điểm tin cậy.
        confidence = confidence + age_adjustment + gender_adjustment + doctor_adjustment
        # Chặn điểm tin cậy trong khoảng [0, 1].
        confidence = _clamp01(confidence)
        # Đổi điểm [0, 1] sang phần trăm [0, 100].
        raw_percent = _round_percent(confidence * 100)

        # Chuyển mã triệu chứng khớp sang tên tiếng Việt để hiển thị.
        matched = [SYMPTOMS[code] for code in matched_codes if code in SYMPTOMS]
        # Tạo danh sách tên triệu chứng cốt lõi còn thiếu.
        missing_core = [
            # Lấy tên triệu chứng từ bảng SYMPTOMS.
            SYMPTOMS[code]
            # Duyệt từng triệu chứng cốt lõi của bệnh.
            for code in disease.get("core", [])
            # Chỉ lấy triệu chứng cốt lõi chưa có trong đầu vào và tồn tại trong SYMPTOMS.
            if code not in selected_symptoms and code in SYMPTOMS
        ]

        # Khởi tạo danh sách cảnh báo/nhãn giải thích.
        flags = []
        # Nếu là bệnh cấp cứu và điểm đủ cao thì gắn cảnh báo cấp cứu.
        if disease_id in EMERGENCY_DISEASES and raw_percent >= 30:
            flags.append("Cần cấp cứu ngay")
        # Nếu là bệnh nguy cơ cao và điểm đủ cao thì khuyên khám chuyên khoa sớm.
        if disease_id in HIGH_ALERT_DISEASES and raw_percent >= 35:
            flags.append("Cần khám chuyên khoa sớm")
        # Nếu tuổi làm giảm điểm mạnh thì thông báo cho người dùng.
        if age is not None and age_adjustment < -0.25:
            flags.append("Độ tuổi làm giảm độ tin cậy")
        # Nếu giới tính không phù hợp thì thêm cờ giải thích.
        if gender_adjustment < -0.5:
            flags.append("Giới tính không phù hợp")
        # Nếu câu hỏi bổ sung làm tăng nghi ngờ rõ rệt thì thêm cờ.
        if doctor_adjustment > 0.12:
            flags.append("Câu trả lời follow-up làm tăng nghi ngờ")
        # Nếu điểm cốt lõi thấp thì báo thiếu triệu chứng cốt lõi.
        if scores["core_score"] < 0.34:
            flags.append("Thiếu triệu chứng cốt lõi")
        # Nếu người dùng nhập quá ít triệu chứng thì cảnh báo dữ kiện yếu.
        if len(selected_symptoms) <= 2:
            flags.append("Dữ kiện đầu vào còn ít")

        # ================= CÂY SUY DIỄN CHẨN ĐOÁN =================
        # Hệ thống không dựng một class Tree riêng, nhưng mỗi bệnh ứng viên tạo ra một nhánh suy diễn:
        #
        # Gốc cây:
        #   Tập fact đầu vào của người bệnh, ví dụ has(S03), has(S04), age(30).
        #
        # Nhánh bệnh:
        #   Một bệnh trong DISEASES, ví dụ covid, influenza, pneumonia.
        #
        # Nút suy diễn trên nhánh:
        #   - so khớp triệu chứng cốt lõi;
        #   - luật IF-THEN được kích hoạt;
        #   - cụm triệu chứng đặc hiệu;
        #   - điểm Bayes;
        #   - hiệu chỉnh tuổi/giới/follow-up.
        #
        # Lá cây:
        #   Điểm phần trăm cuối cùng và nhãn cạnh tranh của bệnh.
        #
        # reasoning_steps chính là vết của nhánh suy diễn này để giao diện giải thích được
        # vì sao một bệnh được xếp hạng cao hoặc thấp.
        reasoning_steps = [
            # Bước 1: tóm tắt số fact đầu vào.
            f"Tập sự kiện đầu vào: {len(patient_facts)} fact, trong đó có {len(selected_symptoms)} fact triệu chứng.",
            # Bước 2: liệt kê rõ các triệu chứng đã khớp với bệnh đang xét.
            "Triệu chứng đã khớp: " + (", ".join(matched) if matched else "chưa có triệu chứng khớp rõ."),
            # Bước 3: ghi rõ trạng thái core symptom để người dùng hiểu vì sao điểm core cao/thấp.
            (
                "Đã khớp toàn bộ core symptom."
                if not missing_core
                else "Core symptom còn thiếu: " + ", ".join(missing_core) + "."
            ),
            # Bước 4: giải thích tuổi có phù hợp với bệnh hay không.
            f"Đánh giá độ tuổi: {age_note}",
            # Bước 5: giải thích giới tính có phù hợp với bệnh hay không.
            f"Đánh giá giới tính: {gender_note}",
            # Bước 2: tóm tắt số luật IF-THEN được kích hoạt.
            f"Suy diễn tiến kích hoạt {len(rule_traces)} luật sản xuất cho {disease['name']}.",
        ]
        # Thêm các vết suy diễn chi tiết từ _compute_rule_activation.
        reasoning_steps.extend(rule_traces)
        # Nếu có cụm triệu chứng đặc hiệu đã khớp.
        if matched_pairs:
            # Khởi tạo danh sách cụm ở dạng tên triệu chứng.
            pretty_pairs = []
            # Duyệt từng cụm mã triệu chứng.
            for pair in matched_pairs:
                # Chuyển mã Sxx thành tên triệu chứng và nối bằng dấu +.
                pretty_pairs.append(" + ".join(SYMPTOMS.get(code, code) for code in pair))
            # Thêm câu giải thích cụm triệu chứng đặc hiệu vào reasoning_steps.
            reasoning_steps.append(
                "Cụm triệu chứng đặc hiệu: " + "; ".join(pretty_pairs) + "."
            )
        # Thêm bước giải thích điểm Bayes.
        reasoning_steps.append(
            f"Thành phần xác suất: điểm Bayes giả lập đạt {_round_percent(bayes_score * 100)}%."
        )

        # Thêm bệnh hiện tại vào danh sách kết quả.
        results.append(
            # Mỗi phần tử là một từ điển chứa điểm, giải thích và dữ liệu hiển thị.
            {
                # Mã định danh bệnh.
                "id": disease_id,
                # Tên bệnh tiếng Việt.
                "name": disease["name"],
                # Biểu tượng dùng trên giao diện.
                "icon": disease["icon"],
                # Theme màu giao diện.
                "theme": disease["theme"],
                # Mô tả bệnh.
                "description": disease["description"],
                # Nhóm bệnh, nếu thiếu thì ghi "Chưa phân nhóm".
                "group": disease.get("group", "Chưa phân nhóm"),
                # Điểm thô sau toàn bộ hiệu chỉnh, nằm trong [0, 1].
                "raw_score": confidence,
                # Điểm thô đổi sang phần trăm.
                "raw_percent": raw_percent,
                # Tạm gán phần trăm hiển thị bằng phần trăm thô trước khi chuẩn hóa tổng 100%.
                "display_percent": raw_percent,
                # Danh sách tên triệu chứng đã khớp.
                "matched": matched,
                # Danh sách triệu chứng cốt lõi còn thiếu.
                "missing_core": missing_core,
                # Số triệu chứng đã khớp.
                "matched_count": len(matched_codes),
                # Tổng số triệu chứng người dùng cung cấp.
                "selected_count": len(selected_symptoms),
                # Coverage đổi sang phần trăm.
                "coverage_percent": _round_percent(scores["coverage_score"] * 100),
                # Precision đổi sang phần trăm.
                "precision_percent": _round_percent(scores["precision_score"] * 100),
                # Điểm cốt lõi đổi sang phần trăm.
                "core_percent": _round_percent(scores["core_score"] * 100),
                # Điểm hỗ trợ đổi sang phần trăm.
                "supportive_percent": _round_percent(scores["supportive_score"] * 100),
                # Tỷ lệ thiếu triệu chứng cốt lõi đổi sang phần trăm.
                "missing_core_percent": _round_percent(scores["missing_core_ratio"] * 100),
                # Điểm Bayes đổi sang phần trăm.
                "bayes_percent": _round_percent(bayes_score * 100),
                # Lời khuyên lấy từ kho tri thức.
                "advice": disease["advice"],
                # Ghi chú hiệu chỉnh tuổi.
                "age_note": age_note,
                # Ghi chú hiệu chỉnh giới tính.
                "gender_note": gender_note,
                # Loại bỏ cờ trùng lặp nhưng giữ thứ tự.
                "flags": list(dict.fromkeys(flags)),
                # Các bước suy diễn để hiển thị.
                "reasoning_steps": reasoning_steps,
                # Tóm tắt ngắn gọn cho người dùng/bác sĩ đọc kết quả.
                "doctor_summary": (
                    f"Hệ thống ghi nhận {len(matched)} triệu chứng phù hợp với {disease['name']}. "
                    f"Điểm được sinh ra từ 3 lớp suy luận: logic mệnh đề trên tập fact, "
                    f"luật sản xuất IF-THEN và một lớp xác suất giả lập theo Bayes để xếp hạng chẩn đoán phân biệt. "
                    f"Điểm trước chuẩn hóa của bệnh này là {raw_percent}%."
                ),
                # Giải thích chi tiết hơn về các thành phần điểm chính.
                "explanation": (
                    f"Độ phủ triệu chứng đạt {_round_percent(scores['coverage_score'] * 100)}%, "
                    f"độ tập trung đạt {_round_percent(scores['precision_score'] * 100)}%, "
                    f"khớp triệu chứng cốt lõi đạt {_round_percent(scores['core_score'] * 100)}%, "
                    f"triệu chứng hỗ trợ đạt {_round_percent(scores['supportive_score'] * 100)}%. "
                    f"Hệ thống cộng điểm khi luật cốt lõi, luật cụm triệu chứng đặc hiệu và ràng buộc ngữ cảnh được kích hoạt; "
                    f"đồng thời trừ điểm khi bệnh không giải thích được triệu chứng đặc hiệu hoặc thiếu core symptom."
                ),
                # Lưu lại các thành phần công thức để giao diện có thể hiển thị hoặc kiểm tra lỗi.
                "formula": {
                    # coverage = matched_weight / total_weight.
                    "coverage": _round_percent(scores["coverage_score"] * 100),
                    # precision = matched_weight / selected_weight_total.
                    "precision": _round_percent(scores["precision_score"] * 100),
                    # core = số triệu chứng cốt lõi khớp / tổng số triệu chứng cốt lõi.
                    "core": _round_percent(scores["core_score"] * 100),
                    # supportive = điểm triệu chứng hỗ trợ ngoài nhóm cốt lõi.
                    "supportive": _round_percent(scores["supportive_score"] * 100),
                    # rule_activation = tổng điểm luật IF-THEN đã kích hoạt.
                    "rule_activation": _round_percent(rule_activation * 100),
                    # bayes = điểm xác suất Bayes giả lập.
                    "bayes": _round_percent(bayes_score * 100),
                    # bonus = tổng điểm thưởng logic.
                    "bonus": _round_percent(bonus * 100),
                    # logic_penalty = tổng điểm phạt logic.
                    "logic_penalty": _round_percent(logic_penalty * 100),
                    # missing_core_percent = tỷ lệ triệu chứng cốt lõi còn thiếu.
                    "missing_core_percent": _round_percent(scores["missing_core_ratio"] * 100),
                    # age_adjustment = hiệu chỉnh theo tuổi.
                    "age_adjustment": _round_percent(age_adjustment * 100),
                    # gender_adjustment = hiệu chỉnh theo giới tính.
                    "gender_adjustment": _round_percent(gender_adjustment * 100),
                    # doctor_adjustment = hiệu chỉnh từ câu trả lời bổ sung.
                    "doctor_adjustment": _round_percent(doctor_adjustment * 100),
                    # severity_factor lưu mức tăng/giảm do mức độ nặng.
                    "severity_factor": _round_percent((severity_factor - 1) * 100),
                    # duration_factor lưu mức tăng/giảm do thời gian mắc.
                    "duration_factor": _round_percent((duration_factor - 1) * 100),
                },
            }
        )

    # Sau khi duyệt xong toàn bộ DISEASES, danh sách results chính là tập ứng viên tìm được.
    # Từ đây hệ thống không sinh thêm bệnh mới, mà chỉ chuẩn hóa và xếp hạng các ứng viên.
    # Chuẩn hóa cạnh tranh giữa các bệnh trong cùng nhóm.
    results = _normalize_group_probabilities(results)
    # Sao lưu danh sách đã xếp hạng để dùng phương án dự phòng nếu lọc ngưỡng làm rỗng kết quả.
    ranked_results = list(results)

    # Chỉ giữ bệnh có điểm thô đủ ngưỡng hiển thị.
    # MIN_DISPLAY_SCORE_THRESHOLD = 0.08 nghĩa là tối thiểu 8%.
    # Lọc các bệnh có điểm thô đạt ngưỡng tối thiểu.
    results = [item for item in ranked_results if item["raw_score"] >= MIN_DISPLAY_SCORE_THRESHOLD]

    # Nếu sau khi lọc không còn kết quả nhưng trước đó vẫn có bệnh khớp.
    if not results and ranked_results:
        # Chỉ lấy bệnh có ít nhất một triệu chứng khớp.
        fallback_results = [item for item in ranked_results if item["matched_count"] > 0]
        # Ưu tiên kết quả dự phòng, nếu không có thì lấy danh sách đã xếp hạng.
        results = fallback_results[:MAX_RESULTS] if fallback_results else ranked_results[:MAX_RESULTS]

        # Duyệt từng bệnh dự phòng để gắn nhãn dữ kiện yếu.
        for item in results:
            # Đảm bảo điểm thô tối thiểu 0.01 để vẫn có thể hiển thị.
            item["raw_score"] = max(item["raw_score"], 0.01)
            # Cập nhật lại phần trăm thô.
            item["raw_percent"] = _round_percent(item["raw_score"] * 100)
            # Tránh thêm trùng flag.
            if "Tín hiệu triệu chứng còn yếu" not in item["flags"]:
                # Gắn cờ giải thích vì sao kết quả vẫn được giữ.
                item["flags"].append("Tín hiệu triệu chứng còn yếu")
            # Ghi lại tóm tắt dự phòng.
            item["doctor_summary"] = (
                f"Hệ thống đã ghi nhận một số dấu hiệu bước đầu liên quan tới {item['name']}, "
                f"nhưng dữ kiện hiện tại còn ít hoặc chưa đủ đặc hiệu để kết luận chắc hơn. "
                f"Bệnh này vẫn được giữ lại trong danh sách chẩn đoán phân biệt để tham khảo."
            )
            # Ghi lại giải thích dự phòng.
            item["explanation"] = (
                "Bệnh này vẫn được giữ lại vì có ít nhất một phần triệu chứng trùng khớp với hồ sơ đầu vào. "
                "Tuy nhiên mức khớp hiện còn thấp, nên tỷ lệ chỉ mang ý nghĩa định hướng ban đầu và cần thêm triệu chứng "
                "hoặc câu hỏi làm rõ."
            )

    # Nếu vẫn không có bệnh nào thì trả danh sách rỗng.
    if not results:
        return []

    # Giới hạn số kết quả tối đa hiển thị.
    results = results[:MAX_RESULTS]
    # Chuẩn hóa phần trăm hiển thị để tổng các bệnh bằng 100%.
    results = _normalize_display_percent(results)

    # Lấy điểm hiển thị của bệnh đứng đầu.
    top_score = results[0]["display_percent"]
    # Tính tổng phần trăm hiển thị của tất cả bệnh đang hiển thị.
    total_display = sum(item["display_percent"] for item in results)

    # Duyệt từng bệnh để bổ sung chỉ số so sánh và nhãn cạnh tranh.
    for item in results:
        # percent: phần trăm sau chuẩn hóa để hiển thị cho người dùng.
        item["percent"] = _round_percent(item["display_percent"])

        # relative_level: so với bệnh đứng đầu.
        # Công thức: mức_tương_đối = phần_trăm_hiển_thị_i / phần_trăm_hiển_thị_cao_nhất * 100.
        item["relative_level"] = (
            _round_percent((item["display_percent"] / top_score) * 100) if top_score else 0
        )

        # differential_percent: tỷ lệ trong tập chẩn đoán phân biệt đang hiển thị.
        # Vì phần trăm hiển thị đã chuẩn hóa tổng bằng 100, giá trị này thường gần phần trăm chính.
        item["differential_percent"] = (
            _round_percent((item["display_percent"] / total_display) * 100) if total_display else 0
        )

        if item["percent"] <= 0 and item["matched_count"] > 0:
            # Nếu bị làm tròn về 0 nhưng vẫn có triệu chứng khớp, giữ tối thiểu 0.1%.
            item["percent"] = 0.1

        # Khoảng cách điểm giữa bệnh đứng đầu và bệnh hiện tại.
        distance = top_score - item["display_percent"]
        # Nếu chênh lệch <= 5%, bệnh này rất gần bệnh đứng đầu.
        if distance <= 5:
            item["competitor_level"] = "Rất gần bệnh đứng đầu"
        # Nếu chênh lệch <= 12%, vẫn có khả năng nhầm lẫn.
        elif distance <= 12:
            item["competitor_level"] = "Có thể nhầm lẫn"
        # Nếu chênh lệch <= 22%, coi là khả năng phụ.
        elif distance <= 22:
            item["competitor_level"] = "Khả năng phụ"
        # Nếu chênh lệch lớn hơn, bệnh ít được nghĩ tới hơn.
        else:
            item["competitor_level"] = "Ít nghĩ tới hơn"

    # Trả về danh sách bệnh đã xếp hạng và giải thích đầy đủ.
    return results
