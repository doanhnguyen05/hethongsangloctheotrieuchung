import os
import uuid

from flask import Flask, abort, g, jsonify, redirect, render_template, request, url_for
from sqlalchemy import text

from inference_engine import infer_disease
from knowledge_base import DISEASES, FOLLOWUP_RULES, SYMPTOMS, SYMPTOM_GROUPS, SYMPTOM_KEYWORDS
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
    # Thuật toán nhận diện triệu chứng từ đoạn văn người dùng nhập.
    # Ý tưởng: chuẩn hóa chuỗi về chữ thường, sau đó dò từng từ khóa trong SYMPTOM_KEYWORDS.
    # Nếu từ khóa của triệu chứng Sxx xuất hiện trong câu nhập, hệ thống thêm Sxx vào tập triệu chứng.
    # Đây là cách so khớp theo từ khóa, không phải xử lý ngôn ngữ tự nhiên phức tạp.
    # Chuẩn hóa văn bản: nếu là None thì đổi thành "", bỏ khoảng trắng hai đầu, chuyển chữ thường.
    symptom_text_normalized = f" {(symptom_text or '').strip().lower()} "
    # Khởi tạo danh sách mã triệu chứng tìm được.
    matched = []
    # Duyệt từng mã triệu chứng và danh sách từ khóa tương ứng.
    for code, keywords in SYMPTOM_KEYWORDS.items():
        # Duyệt từng từ khóa của triệu chứng hiện tại.
        for keyword in keywords:
            # Chuẩn hóa từ khóa giống văn bản để so khớp thống nhất.
            keyword_normalized = f" {keyword.strip().lower()} "
            # Nếu từ khóa xuất hiện trong văn bản người dùng nhập.
            if keyword_normalized in symptom_text_normalized:
                # Thêm mã triệu chứng vào danh sách đã khớp.
                matched.append(code)
                # Dừng duyệt từ khóa của triệu chứng này để tránh thêm trùng.
                break
    # Loại mã trùng và trả về danh sách mã triệu chứng đã phát hiện.
    return list(dict.fromkeys(matched))


def normalize_symptom_codes(symptom_codes):
    # Chuẩn hóa danh sách mã triệu chứng:
    # - dict.fromkeys(...) giữ đúng thứ tự nhưng loại bỏ mã trùng.
    # - chỉ giữ mã tồn tại trong SYMPTOMS để tránh dữ liệu rác từ biểu mẫu/API.
    # Duyệt danh sách đã loại trùng, chỉ giữ code tồn tại trong SYMPTOMS.
    return [code for code in dict.fromkeys(symptom_codes or []) if code in SYMPTOMS]


def symptom_names(symptom_codes):
    # Chuyển danh sách mã Sxx thành tên triệu chứng tiếng Việt.
    return [SYMPTOMS[code] for code in symptom_codes if code in SYMPTOMS]


def gender_label(gender):
    # Chuẩn hóa giới tính nội bộ sang nhãn tiếng Việt để giao diện không hiện raw value male/female.
    labels = {"male": "Nam", "female": "Nữ", "all": "Khác / Chưa xác định"}
    return labels.get(gender, "Khác / Chưa xác định")


def disease_matches_patient_context(disease_id, gender, age):
    # Lọc kết quả theo ràng buộc giới tính và tuổi khi đọc lại lịch sử.
    # Các bản ghi cũ có thể được lưu trước khi sửa thuật toán, nên vẫn có bệnh sai giới trong DB.
    # Hàm này giúp trang chi tiết không hiển thị bệnh phụ khoa cho nam, bệnh nam khoa cho nữ,
    # hoặc các bệnh rất lệch tuổi như mãn kinh ở tuổi 24.
    disease = DISEASES.get(disease_id)
    if not disease:
        return True
    disease_gender = disease.get("gender", "all")
    if gender != "all" and disease_gender != "all" and disease_gender != gender:
        return False
    if age is not None:
        if disease_id == "menopause" and age < 35:
            return False
        if disease_id in {"bph", "prostate_cancer"} and age < 35:
            return False
        if disease_id in {"alzheimer", "parkinson"} and age < 35:
            return False
        if disease_id == "breast_cancer" and age < 12:
            return False
    return True


def normalize_stored_result_percent(results):
    # Chuẩn hóa lại phần trăm khi trang chi tiết lịch sử lọc bỏ các bệnh sai giới.
    # Nếu không làm bước này, các phần trăm cũ có thể không còn tổng 100% sau khi lọc.
    if not results:
        return results

    total_raw = sum((item.raw_score or 0) for item in results if (item.raw_score or 0) > 0)
    if total_raw <= 0:
        total_raw = sum((item.differential_percent or 0) for item in results if (item.differential_percent or 0) > 0)

    if total_raw <= 0:
        return results

    for item in results:
        base_score = item.raw_score if (item.raw_score or 0) > 0 else item.differential_percent
        item.differential_percent = round((base_score / total_raw) * 100, 1)

    diff = round(100.0 - sum(item.differential_percent for item in results), 1)
    if diff:
        results[0].differential_percent = round(results[0].differential_percent + diff, 1)
    return results


def parse_age(age_raw):
    # Nếu tuổi rỗng hoặc là None thì coi như người dùng không nhập tuổi.
    if age_raw in (None, ""):
        # Trả về None để thuật toán bỏ qua hiệu chỉnh tuổi.
        return None
    # Thử chuyển tuổi sang số nguyên.
    try:
        # Ép kiểu tuổi từ chuỗi hoặc giá trị biểu mẫu sang số nguyên.
        age = int(age_raw)
    # Nếu dữ liệu tuổi không hợp lệ, ví dụ chữ cái.
    except (TypeError, ValueError):
        # Trả về None để không làm hỏng luồng xử lý.
        return None
    # Chỉ chấp nhận tuổi từ 0 đến 120, ngoài khoảng này coi là không hợp lệ.
    return age if 0 <= age <= 120 else None


def build_followup_questions(selected_symptoms):
    # Suy diễn lùi ở mức giao diện.
    # Khi người dùng đã có một số triệu chứng ban đầu, hệ thống kiểm tra FOLLOWUP_RULES.
    # Nếu triệu chứng hiện tại kích hoạt trigger_symptoms của luật, hệ thống hỏi thêm
    # để xác minh các triệu chứng còn thiếu hoặc tăng/giảm trọng số bệnh liên quan.
    # Khởi tạo danh sách câu hỏi cần hỏi thêm.
    questions = []
    # Chuyển triệu chứng đã chọn thành tập hợp để kiểm tra nhanh.
    selected_set = set(selected_symptoms)
    # Duyệt từng luật hỏi bổ sung trong kho tri thức.
    for rule in FOLLOWUP_RULES:
        # Nếu ít nhất một trigger symptom của luật xuất hiện trong triệu chứng đã chọn.
        if any(code in selected_set for code in rule["trigger_symptoms"]):
            # Thêm câu hỏi này vào danh sách cần hiển thị.
            questions.append(rule)
    # Trả về toàn bộ câu hỏi bổ sung phù hợp.
    return questions


def parse_followup_answers(form_data, questions):
    # Xử lý câu trả lời cho câu hỏi bổ sung.
    # Mỗi lựa chọn có thể:
    # - add_symptoms: bổ sung triệu chứng mới vào tập sự kiện.
    # - weight_delta: cộng/trừ điểm trực tiếp cho bệnh cụ thể.
    # Đây là bước cập nhật trạng thái trước khi gọi infer_disease(...) lần cuối.
    # Danh sách triệu chứng bổ sung suy ra từ câu trả lời.
    extra_symptoms = []
    # Dictionary lưu điểm cộng/trừ cho từng bệnh.
    adjustments = {}
    # Danh sách câu hỏi đã trả lời để lưu lịch sử/hiển thị.
    answered = []
    # Duyệt từng câu hỏi bổ sung đã được hiển thị.
    for question in questions:
        # Lấy giá trị người dùng chọn theo thuộc tính name="followup_<id>".
        answer_value = form_data.get(f"followup_{question['id']}")
        # Nếu câu này chưa có câu trả lời thì bỏ qua.
        if not answer_value:
            continue
        # Duyệt từng lựa chọn của câu hỏi.
        for option in question["options"]:
            # Tìm lựa chọn khớp với giá trị người dùng gửi lên.
            if option["value"] == answer_value:
                # Bổ sung các triệu chứng mà lựa chọn này xác nhận.
                extra_symptoms.extend(option.get("add_symptoms", []))
                # Duyệt các điểm điều chỉnh bệnh của lựa chọn.
                for disease_id, delta in option.get("weight_delta", {}).items():
                    # Cộng dồn độ lệch điểm nếu một bệnh được nhiều lựa chọn tác động.
                    adjustments[disease_id] = adjustments.get(disease_id, 0) + delta
                # Lưu lại nội dung câu hỏi và câu trả lời dạng dễ đọc.
                answered.append({"question": question["question"], "answer": option["label"]})
                # Đã tìm thấy lựa chọn đúng nên thoát vòng lặp lựa chọn.
                break
    # Chuẩn hóa triệu chứng bổ sung, rồi trả về triệu chứng bổ sung, điểm điều chỉnh và câu đã trả lời.
    return normalize_symptom_codes(extra_symptoms), adjustments, answered


def serialize_result_payload(result):
    # Chuẩn hóa đầu ra từ inference_engine trước khi lưu vào cơ sở dữ liệu.
    # Các trường như coverage_percent, precision_percent, core_percent, bayes_percent
    # được giữ lại để giao diện và lịch sử có thể giải thích kết quả bằng công thức.
    # Lấy từ điển công thức từ kết quả; nếu không có thì dùng từ điển rỗng.
    formula = result.get("formula") or {}
    # Trả về gói dữ liệu đã chuẩn hóa tên trường.
    return {
        # Mã bệnh, hỗ trợ cả khóa "id" và "disease_id".
        "disease_id": result.get("id") or result.get("disease_id", ""),
        # Tên bệnh, hỗ trợ cả khóa "name" và "disease_name".
        "name": result.get("name") or result.get("disease_name", ""),
        # Biểu tượng bệnh, mặc định là stethoscope nếu thiếu.
        "icon": result.get("icon", "stethoscope"),
        # Phần trăm hiển thị sau chuẩn hóa.
        "percent": result.get("percent", 0),
        # Tỷ lệ trong tập chẩn đoán phân biệt.
        "differential_percent": result.get("differential_percent", 0),
        # Tỷ lệ khớp triệu chứng cốt lõi.
        "core_percent": result.get("core_percent", 0),
        # Tỷ lệ độ phủ triệu chứng.
        "coverage_percent": result.get("coverage_percent", 0),
        # Tỷ lệ độ tập trung triệu chứng.
        "precision_percent": result.get("precision_percent", 0),
        # Nhãn so sánh với bệnh đứng đầu.
        "competitor_level": result.get("competitor_level", ""),
        # Mô tả bệnh.
        "description": result.get("description", ""),
        # Lời khuyên.
        "advice": result.get("advice", ""),
        # Danh sách cảnh báo/cờ trạng thái.
        "flags": result.get("flags", []),
        # Danh sách triệu chứng đã khớp.
        "matched": result.get("matched", []),
        # Danh sách triệu chứng cốt lõi còn thiếu.
        "missing_core": result.get("missing_core", []),
        # Các bước suy diễn.
        "reasoning_steps": result.get("reasoning_steps", []),
        # Điểm Bayes dạng phần trăm.
        "bayes_percent": result.get("bayes_percent", 0),
        # Điểm kích hoạt luật, ưu tiên kết quả, nếu thiếu thì lấy từ công thức.
        "rule_activation": result.get("rule_activation", formula.get("rule_activation", 0)),
        # Điểm thô trước chuẩn hóa hiển thị.
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
    # Lưu một lần sàng lọc vào cơ sở dữ liệu.
    # Bảng screenings lưu đầu vào: bệnh nhân, triệu chứng, mức độ, thời gian.
    # Bảng screening_results lưu đầu ra: các bệnh có điểm cao nhất, phần trăm, triệu chứng khớp,
    # triệu chứng cốt lõi còn thiếu và các bước suy diễn.
    # Tạo bản ghi Screening lưu thông tin lần sàng lọc.
    screening = Screening(
        # Liên kết với bệnh nhân.
        patient_id=patient.id,
        # Mã máy khách lưu trong cookie để tách lịch sử từng máy/người dùng.
        client_id=client_id,
        # Giai đoạn: followup hoặc final.
        stage=stage,
        # Mức độ nặng.
        severity=severity,
        # Thời gian mắc.
        duration=duration,
        # Đoạn mô tả triệu chứng tự do.
        symptom_text=symptom_text,
        # Danh sách mã triệu chứng.
        selected_codes=selected_codes,
        # Danh sách tên triệu chứng.
        selected_symptoms=selected_symptoms,
        # Số lượng triệu chứng.
        symptom_count=len(selected_symptoms),
        # Các câu trả lời cho câu hỏi bổ sung.
        followup_answers=followup_answers or {},
    )
    # Thêm bản ghi sàng lọc vào phiên làm việc SQLAlchemy.
    db.session.add(screening)
    # Đẩy tạm dữ liệu để bản ghi sàng lọc có id trước khi tạo ScreeningResult.
    db.session.flush()

    # Duyệt từng kết quả bệnh để lưu vào bảng kết quả sàng lọc.
    for rank, item in enumerate(results_data, 1):
        # Chuẩn hóa kết quả từ inference_engine thành gói dữ liệu ổn định.
        payload = serialize_result_payload(item)
        # Tạo bản ghi kết quả cho một bệnh ở thứ hạng hiện tại.
        result = ScreeningResult(
            # Khóa ngoại trỏ về bản ghi sàng lọc.
            screening_id=screening.id,
            # Thứ hạng bệnh trong danh sách kết quả.
            rank=rank,
            # Mã bệnh.
            disease_id=payload["disease_id"],
            # Tên bệnh.
            disease_name=payload["name"],
            # Biểu tượng bệnh.
            icon=payload["icon"],
            # Phần trăm hiển thị.
            percent=payload["percent"],
            # Tỷ lệ chẩn đoán phân biệt.
            differential_percent=payload["differential_percent"],
            # Điểm triệu chứng cốt lõi.
            core_percent=payload["core_percent"],
            # Điểm độ phủ.
            coverage_percent=payload["coverage_percent"],
            # Điểm độ tập trung.
            precision_percent=payload["precision_percent"],
            # Nhãn cạnh tranh với bệnh đứng đầu.
            competitor_level=payload["competitor_level"],
            # Mô tả bệnh.
            description=payload["description"],
            # Lời khuyên.
            advice=payload["advice"],
            # Các cờ cảnh báo.
            flags=payload["flags"],
            # Triệu chứng đã khớp.
            matched=payload["matched"],
            # Triệu chứng cốt lõi còn thiếu.
            missing_core=payload["missing_core"],
            # Các bước suy diễn.
            reasoning_steps=payload["reasoning_steps"],
            # Điểm Bayes.
            bayes_percent=payload["bayes_percent"],
            # Điểm kích hoạt luật.
            rule_activation=payload["rule_activation"],
            # Điểm thô.
            raw_score=payload["raw_score"],
        )
        # Thêm bản ghi kết quả vào phiên làm việc.
        db.session.add(result)

    # Ghi toàn bộ bản ghi sàng lọc và kết quả vào cơ sở dữ liệu.
    db.session.commit()
    # Trả về bản ghi sàng lọc vừa lưu.
    return screening


def get_or_create_patient(name, age, gender):
    # Tìm bệnh nhân đã tồn tại theo bộ name-age-gender.
    patient = Patient.query.filter_by(name=name, age=age, gender=gender).first()
    # Nếu chưa có bệnh nhân này.
    if not patient:
        # Tạo bản ghi Patient mới.
        patient = Patient(name=name, age=age, gender=gender)
        # Thêm vào phiên làm việc.
        db.session.add(patient)
        # Đẩy tạm dữ liệu để bệnh nhân có id trước khi lưu bản ghi sàng lọc.
        db.session.flush()
    # Trả về patient cũ hoặc mới tạo.
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
    # Luồng xử lý chính của biểu mẫu web.
    # Bước 1: nhận thông tin người bệnh và triệu chứng từ biểu mẫu.
    # Bước 2: gom triệu chứng từ ô chọn + triệu chứng tách từ văn bản.
    # Bước 3: nếu là lần nhập đầu và có luật hỏi bổ sung, hỏi thêm trước khi chẩn đoán cuối.
    # Bước 4: sau câu hỏi bổ sung, gọi infer_disease(...) để tính điểm và xếp hạng bệnh.
    # Bước 5: lưu lịch sử và trả kết quả ra result.html.
    # Lấy hoặc tạo mã máy khách từ cookie để lưu lịch sử riêng cho từng người dùng.
    client_id = get_client_id()
    # Lấy giai đoạn từ biểu mẫu; mặc định là initial nếu chưa có.
    stage = request.form.get("stage", "initial")
    # Lấy tên người bệnh; nếu rỗng thì dùng "Người bệnh".
    name = (request.form.get("name") or "Người bệnh").strip() or "Người bệnh"
    # Lấy tuổi từ biểu mẫu và chuyển sang số nguyên hợp lệ bằng parse_age.
    age = parse_age((request.form.get("age") or "").strip())
    # Lấy giới tính; mặc định là "all" nếu người dùng không chọn.
    gender = (request.form.get("gender") or "all").strip()
    # Lấy đoạn mô tả triệu chứng tự do.
    symptom_text = (request.form.get("symptom_text") or "").strip()
    # Lấy mức độ nặng; mặc định moderate.
    severity = (request.form.get("severity") or "moderate").strip()
    # Lấy thời gian mắc; mặc định fewdays.
    duration = (request.form.get("duration") or "fewdays").strip()

    # Lấy danh sách triệu chứng người dùng tick checkbox và chuẩn hóa mã Sxx.
    selected_from_checkboxes = normalize_symptom_codes(request.form.getlist("symptoms"))
    # Tách thêm triệu chứng từ đoạn mô tả tự do bằng thuật toán so khớp từ khóa.
    selected_from_text = extract_symptoms_from_text(symptom_text)

    # base_symptoms = triệu chứng người dùng chọn trực tiếp + triệu chứng phát hiện từ mô tả tự do.
    base_symptoms = normalize_symptom_codes(selected_from_checkboxes + selected_from_text)

    # Nếu đây là lần gửi đầu tiên từ biểu mẫu.
    if stage == "initial":
        # Tìm các câu hỏi bổ sung phù hợp với triệu chứng ban đầu.
        followup_questions = build_followup_questions(base_symptoms)
        # Nếu có câu hỏi cần hỏi thêm.
        if followup_questions:
            # Chạy suy diễn sơ bộ để hiển thị vài bệnh nghi ngờ trước khi hỏi thêm.
            # Kết quả này chưa phải kết luận cuối vì chưa có câu trả lời bổ sung.
            # Gọi thuật toán suy diễn với triệu chứng ban đầu.
            preliminary_results = infer_disease(
                # Triệu chứng đầu vào chưa có câu hỏi bổ sung.
                base_symptoms,
                # Tuổi người bệnh.
                age=age,
                # Giới tính người bệnh.
                gender=gender,
                # Mức độ nặng.
                severity=severity,
                # Thời gian mắc.
                duration=duration,
                # Chưa có điều chỉnh từ câu hỏi bổ sung.
                followup_adjustments={},
            )
            # Lấy hoặc tạo bệnh nhân để lưu lịch sử.
            patient = get_or_create_patient(name, age, gender)
            # Lưu kết quả sơ bộ vào cơ sở dữ liệu.
            save_screening(
                # Client hiện tại.
                client_id,
                # Bệnh nhân hiện tại.
                patient,
                # Giai đoạn followup cho biết đây chưa phải kết quả cuối.
                "followup",
                # Mức độ nặng.
                severity,
                # Thời gian mắc.
                duration,
                # Text triệu chứng.
                symptom_text,
                # Mã triệu chứng ban đầu.
                base_symptoms,
                # Tên triệu chứng ban đầu.
                symptom_names(base_symptoms),
                # Chỉ lưu 3 kết quả sơ bộ có điểm cao nhất.
                preliminary_results[:3],
            )
            # Trả trang kết quả ở chế độ hỏi bổ sung để người dùng trả lời thêm.
            return render_template(
                # Mẫu giao diện hiển thị.
                "result.html",
                # Báo giao diện biết đang ở bước hỏi bổ sung.
                stage="followup",
                # Tên người bệnh.
                name=name,
                # Tuổi.
                age=age,
                # Giới tính.
                gender=gender,
                # Mức độ nặng.
                severity=severity,
                # Thời gian mắc.
                duration=duration,
                # Text triệu chứng.
                symptom_text=symptom_text,
                # Tên triệu chứng đã chọn.
                selected_symptoms=symptom_names(base_symptoms),
                # Mã triệu chứng đã chọn.
                selected_codes=base_symptoms,
                # Số lượng triệu chứng.
                symptom_count=len(base_symptoms),
                # Danh sách câu hỏi bổ sung.
                followup_questions=followup_questions,
                # Top 3 kết quả sơ bộ.
                preliminary_results=preliminary_results[:3],
            )

    # Nếu không phải initial hoặc đã qua bước hỏi bổ sung, tiếp tục xử lý kết quả cuối.
    followup_questions = build_followup_questions(base_symptoms)
    # Phân tích câu trả lời bổ sung thành triệu chứng bổ sung và điểm điều chỉnh.
    extra_symptoms, adjustments, answered_followups = parse_followup_answers(request.form, followup_questions)

    # final_symptoms = triệu chứng ban đầu + triệu chứng suy ra từ câu trả lời bổ sung.
    final_symptoms = normalize_symptom_codes(base_symptoms + extra_symptoms)

    # Gọi bộ máy suy diễn chính.
    # followup_adjustments truyền vào các độ lệch điểm theo từng bệnh.
    # Chạy thuật toán suy diễn cuối cùng.
    results = infer_disease(
        # Triệu chứng cuối cùng sau khi cộng câu trả lời bổ sung.
        final_symptoms,
        # Tuổi.
        age=age,
        # Giới tính.
        gender=gender,
        # Mức độ nặng.
        severity=severity,
        # Thời gian mắc.
        duration=duration,
        # Điểm điều chỉnh theo câu hỏi bổ sung.
        followup_adjustments=adjustments,
    )

    # Lấy hoặc tạo bệnh nhân.
    patient = get_or_create_patient(name, age, gender)
    # Lưu kết quả cuối vào cơ sở dữ liệu.
    screening = save_screening(
        # Client hiện tại.
        client_id,
        # Bệnh nhân hiện tại.
        patient,
        # Giai đoạn final cho biết đây là kết quả cuối.
        "final",
        # Mức độ nặng.
        severity,
        # Thời gian mắc.
        duration,
        # Text triệu chứng.
        symptom_text,
        # Mã triệu chứng cuối.
        final_symptoms,
        # Tên triệu chứng cuối.
        symptom_names(final_symptoms),
        # Danh sách kết quả bệnh.
        results,
        # Lưu câu hỏi và câu trả lời bổ sung dạng từ điển.
        followup_answers={item["question"]: item["answer"] for item in answered_followups},
    )

    # Trả trang kết quả cuối cho người dùng.
    return render_template(
        # Mẫu giao diện hiển thị kết quả.
        "result.html",
        # Giai đoạn final.
        stage="final",
        # Tên người bệnh.
        name=name,
        # Tuổi.
        age=age,
        # Giới tính.
        gender=gender,
        # Mức độ nặng.
        severity=severity,
        # Thời gian mắc.
        duration=duration,
        # Text triệu chứng.
        symptom_text=symptom_text,
        # Danh sách kết quả bệnh.
        results=results,
        # Tên triệu chứng cuối.
        selected_symptoms=symptom_names(final_symptoms),
        # Số triệu chứng cuối.
        symptom_count=len(final_symptoms),
        # Danh sách câu hỏi bổ sung đã trả lời.
        answered_followups=answered_followups,
        # Id lần sàng lọc để xem lại lịch sử.
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
    results = [
        item
        for item in screening.results.all()
        if disease_matches_patient_context(item.disease_id, screening.patient.gender, screening.patient.age)
    ]
    results = normalize_stored_result_percent(results)
    results_json = [item.to_dict() for item in results]
    return render_template(
        "history_detail.html",
        screening=screening,
        results=results,
        results_json=results_json,
        gender_label=gender_label,
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
    # API tương đương biểu mẫu /predict nhưng nhận JSON.
    # Thuật toán phía sau vẫn là:
    # chuẩn hóa triệu chứng -> infer_disease(...) -> lưu bản ghi sàng lọc -> trả JSON.
    # Lấy mã máy khách từ cookie để API cũng lưu lịch sử theo máy khách.
    client_id = get_client_id()
    # Đọc phần thân JSON; silent=True để không ném lỗi khi phần thân không hợp lệ.
    data = request.get_json(silent=True)
    # Nếu yêu cầu không có phần thân JSON.
    if not data:
        # Trả lỗi 400 vì API yêu cầu JSON.
        return jsonify({"error": "JSON body required"}), 400

    # Lấy tên bệnh nhân từ JSON, mặc định là API Patient.
    name = (data.get("name") or "API Patient").strip() or "API Patient"
    # Phân tích tuổi từ JSON.
    age = parse_age(data.get("age"))
    # Lấy giới tính, mặc định là all.
    gender = (data.get("gender") or "all").strip()
    # Lấy mô tả triệu chứng tự do.
    symptom_text = (data.get("symptom_text") or "").strip()
    # Lấy mức độ nặng, mặc định moderate.
    severity = (data.get("severity") or "moderate").strip()
    # Lấy thời gian mắc, mặc định fewdays.
    duration = (data.get("duration") or "fewdays").strip()
    # Lấy danh sách mã triệu chứng từ JSON và chuẩn hóa.
    symptom_codes = normalize_symptom_codes(data.get("symptoms", []))

    # Tách thêm triệu chứng từ văn bản mô tả triệu chứng.
    selected_from_text = extract_symptoms_from_text(symptom_text)
    # Gộp triệu chứng trong JSON với triệu chứng tách từ văn bản.
    all_symptoms = normalize_symptom_codes(symptom_codes + selected_from_text)
    # Nếu không có triệu chứng hợp lệ nào.
    if not all_symptoms:
        # Trả lỗi 400 vì thuật toán cần ít nhất một triệu chứng.
        return jsonify({"error": "At least one valid symptom is required"}), 400

    # Gọi thuật toán suy diễn bệnh.
    results = infer_disease(
        # Toàn bộ triệu chứng hợp lệ.
        all_symptoms,
        # Tuổi.
        age=age,
        # Giới tính.
        gender=gender,
        # Mức độ nặng.
        severity=severity,
        # Thời gian mắc.
        duration=duration,
        # API không có câu hỏi bổ sung nên truyền từ điển rỗng.
        followup_adjustments={},
    )

    # Lấy hoặc tạo bệnh nhân.
    patient = get_or_create_patient(name, age, gender)
    # Lưu lần sàng lọc vào cơ sở dữ liệu.
    screening = save_screening(
        # Client hiện tại.
        client_id,
        # Bệnh nhân.
        patient,
        # API trả luôn kết quả cuối.
        "final",
        # Mức độ nặng.
        severity,
        # Thời gian mắc.
        duration,
        # Text triệu chứng.
        symptom_text,
        # Mã triệu chứng hợp lệ.
        all_symptoms,
        # Tên triệu chứng.
        symptom_names(all_symptoms),
        # Kết quả suy diễn.
        results,
    )

    # Trả phản hồi JSON cho máy khách gọi API.
    return jsonify(
        {
            # Id lần sàng lọc vừa lưu.
            "screening_id": screening.id,
            # Thông tin bệnh nhân.
            "patient": patient.to_dict(),
            # Số triệu chứng đầu vào.
            "symptom_count": len(all_symptoms),
            # Tên triệu chứng đã chọn.
            "selected_symptoms": symptom_names(all_symptoms),
            # Danh sách kết quả bệnh lấy từ cơ sở dữ liệu.
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
    results = [
        item
        for item in screening.results.all()
        if disease_matches_patient_context(item.disease_id, screening.patient.gender, screening.patient.age)
    ]
    results = normalize_stored_result_percent(results)
    payload["results"] = [item.to_dict() for item in results]
    payload["patient"] = screening.patient.to_dict()
    return jsonify(payload)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(debug=True, host="0.0.0.0", port=port)
