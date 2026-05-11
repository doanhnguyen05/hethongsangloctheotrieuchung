# Phân tích AI chuyên sâu trong hệ chuyên gia chẩn đoán bệnh

Tài liệu này giải thích chi tiết các thành phần trí tuệ nhân tạo trong hệ thống:

- Thuật toán tìm kiếm bệnh phù hợp.
- Biểu diễn tri thức.
- Suy diễn tiến.
- Suy diễn lùi.
- Cây suy diễn chẩn đoán.
- Lập luận xác suất.
- Hệ thống gợi ý và cảnh báo.

Mục tiêu của hệ thống không phải thay thế bác sĩ, mà là mô phỏng một hệ chuyên gia sàng lọc bệnh dựa trên triệu chứng đầu vào.

## 0. Tổng quan luồng xử lý của hệ thống

Luồng xử lý tổng thể:

1. Người dùng nhập thông tin bệnh nhân, triệu chứng đã chọn và mô tả triệu chứng tự do.
2. `app.py` chuẩn hóa dữ liệu đầu vào.
3. `extract_symptoms_from_text(...)` nhận diện thêm triệu chứng từ văn bản tự do.
4. `build_followup_questions(...)` kiểm tra có cần hỏi thêm hay không.
5. Nếu cần hỏi thêm, hệ thống hiển thị câu hỏi follow-up.
6. `parse_followup_answers(...)` chuyển câu trả lời follow-up thành triệu chứng bổ sung hoặc điểm điều chỉnh.
7. `infer_disease(...)` trong `inference_engine.py` duyệt toàn bộ kho bệnh `DISEASES`.
8. Với mỗi bệnh, hệ thống tính điểm logic, điểm Bayes, điểm hiệu chỉnh ngữ cảnh.
9. Hệ thống lọc, chuẩn hóa, xếp hạng và trả về danh sách bệnh nghi ngờ.
10. Giao diện hiển thị phần trăm, lời khuyên, cảnh báo và các bước suy diễn.

Bảng ánh xạ nhanh:

| Khái niệm AI | File | Hàm/biến chính | Vai trò |
|---|---|---|---|
| Biểu diễn tri thức | `knowledge_base.py` | `DISEASES`, `SYMPTOMS`, `FOLLOWUP_RULES` | Lưu tri thức bệnh, triệu chứng, luật hỏi thêm |
| Tìm kiếm bệnh | `inference_engine.py` | `infer_disease(...)` | Duyệt từng bệnh, chấm điểm, xếp hạng |
| Suy diễn tiến | `inference_engine.py` | `_compute_rule_activation(...)`, `_compute_logic_bonus(...)` | Từ fact đầu vào kích hoạt luật để tăng/giảm điểm |
| Suy diễn lùi | `app.py`, `knowledge_base.py` | `build_followup_questions(...)`, `parse_followup_answers(...)`, `FOLLOWUP_RULES` | Hỏi thêm để xác minh giả thuyết |
| Cây suy diễn | `inference_engine.py` | `reasoning_steps` | Lưu vết giải thích cho từng nhánh bệnh |
| Lập luận xác suất | `inference_engine.py` | `_compute_bayesian_support(...)` | Tính điểm Bayes giả lập |
| Hệ thống gợi ý | `inference_engine.py`, `app.py` | `advice`, `flags`, `doctor_summary`, `explanation` | Tạo lời khuyên, cảnh báo và giải thích |

## 1. Thuật toán tìm kiếm bệnh phù hợp

Vị trí chính:

- File: `inference_engine.py`
- Hàm: `infer_disease(...)`
- Đoạn quan trọng: vòng lặp `for disease_id, disease in DISEASES.items():`

Chiến lược tìm kiếm:

Hệ thống dùng chiến lược `generate-and-test` có xếp hạng. Nghĩa là:

1. Sinh giả thuyết: mỗi bệnh trong `DISEASES` được xem là một giả thuyết chẩn đoán.
2. Kiểm tra giả thuyết: so sánh triệu chứng người dùng với `weights` và `core` của bệnh.
3. Chấm điểm: tính điểm bằng độ phủ, độ tập trung, điểm cốt lõi, luật IF-THEN, Bayes giả lập và điều chỉnh ngữ cảnh.
4. Xếp hạng: sắp xếp các bệnh theo điểm giảm dần và trả về top kết quả.

Đây không phải BFS/DFS vì kho tri thức không phải đồ thị có cạnh. Đây là tìm kiếm vét cạn trên tập bệnh hữu hạn, sau đó dùng hàm đánh giá để xếp hạng. Độ phức tạp xấp xỉ `O(D * S)`, trong đó `D` là số bệnh và `S` là số triệu chứng đầu vào.

### 1.1. Vì sao gọi là tìm kiếm?

Trong bài toán này, “không gian tìm kiếm” là tập tất cả bệnh có trong `DISEASES`. Mỗi bệnh là một trạng thái ứng viên. Hệ thống cần tìm bệnh nào phù hợp nhất với tập triệu chứng đầu vào.

Ví dụ người dùng nhập:

- `S02`: sốt cao.
- `S03`: ho khan.
- `S04`: khó thở.

Hệ thống sẽ lần lượt xét:

- COVID-19 có khớp không?
- Cúm mùa có khớp không?
- Viêm phổi có khớp không?
- Lao phổi có khớp không?
- Các bệnh khác có khớp không?

Sau khi xét hết, hệ thống không trả một bệnh duy nhất ngay lập tức, mà tạo danh sách chẩn đoán phân biệt có xếp hạng.

### 1.2. Các bước chấm điểm trong một vòng lặp bệnh

Trong `infer_disease(...)`, mỗi bệnh được xử lý theo chuỗi:

1. `_compute_support_scores(...)`: tính độ khớp triệu chứng.
2. `_compute_pair_synergy(...)`: kiểm tra cụm triệu chứng đặc hiệu.
3. `_compute_rule_activation(...)`: kích hoạt luật IF-THEN.
4. `_compute_bayesian_support(...)`: tính điểm Bayes giả lập.
5. `_compute_logic_bonus(...)`: tính điểm cộng logic.
6. `_compute_logic_penalty(...)`: tính điểm phạt logic.
7. Hiệu chỉnh theo tuổi, giới tính, mức độ nặng, thời gian mắc và câu hỏi follow-up.
8. Ghi kết quả vào `results`.

### 1.3. Công thức điểm tổng quát

Điểm logic tất định:

```text
deterministic_score =
    coverage_score   * 0.28
  + precision_score  * 0.16
  + core_score       * 0.32
  + supportive_score * 0.10
  + rule_activation
  + bonus
  - logic_penalty
```

Sau đó kết hợp với điểm Bayes:

```text
confidence = deterministic_score * 0.72 + bayes_score * 0.28
```

Tiếp tục hiệu chỉnh:

```text
confidence =
    confidence
  + age_adjustment
  + gender_adjustment
  + followup_adjustment
```

Cuối cùng hệ thống chặn điểm trong khoảng `[0, 1]`, đổi sang phần trăm và xếp hạng.

### 1.4. Ý nghĩa các thành phần điểm

`coverage_score`:

- Đo mức độ bệnh được “phủ” bởi các triệu chứng người dùng.
- Công thức: tổng trọng số triệu chứng khớp / tổng trọng số toàn bộ triệu chứng của bệnh.
- Nếu cao, nghĩa là người dùng có nhiều dấu hiệu quan trọng của bệnh đó.

`precision_score`:

- Đo mức độ triệu chứng người dùng có tập trung vào bệnh đó không.
- Nếu người dùng nhập quá nhiều triệu chứng không liên quan, điểm này giảm.

`core_score`:

- Đo tỷ lệ triệu chứng cốt lõi đã khớp.
- Đây là thành phần quan trọng nhất trong điểm logic vì triệu chứng cốt lõi quyết định mạnh đến bệnh.

`supportive_score`:

- Đo mức độ khớp triệu chứng hỗ trợ ngoài nhóm cốt lõi.

`rule_activation`:

- Điểm cộng khi luật IF-THEN được kích hoạt.
- Ví dụ: nếu khớp toàn bộ triệu chứng cốt lõi thì tăng mạnh.

`bonus`:

- Điểm thưởng cho cụm triệu chứng đặc hiệu.
- Ví dụ đau ngực trái + đau lan tay trái là cụm đặc hiệu cho nhồi máu cơ tim.

`logic_penalty`:

- Điểm phạt nếu thiếu triệu chứng cốt lõi, có triệu chứng gợi ý ngược, hoặc bệnh không giải thích được triệu chứng đặc hiệu.

## 2. Biểu diễn tri thức

Vị trí chính:

- File: `knowledge_base.py`
- Các biến: `SYMPTOMS`, `SYMPTOM_GROUPS`, `SYMPTOM_KEYWORDS`, `FOLLOWUP_RULES`, `DISEASES`

Ý nghĩa từng phần:

- `SYMPTOMS`: bảng mã hóa triệu chứng. Ví dụ `S03` là `Ho khan`.
- `SYMPTOM_GROUPS`: gom triệu chứng theo nhóm cơ quan/hệ bệnh để hiển thị giao diện.
- `SYMPTOM_KEYWORDS`: tập từ khóa để nhận diện triệu chứng từ văn bản tự do.
- `FOLLOWUP_RULES`: tập luật hỏi bổ sung, dùng cho suy diễn lùi.
- `DISEASES`: kho bệnh chính, mỗi bệnh là một frame tri thức.

Trong `DISEASES`, mỗi bệnh có các trường:

- `name`: tên bệnh hiển thị.
- `group`: nhóm bệnh.
- `description`: mô tả bệnh.
- `advice`: gợi ý/lời khuyên.
- `age_range`: khoảng tuổi thường gặp.
- `gender`: giới tính phù hợp.
- `priority`: mức ưu tiên, dùng như xác suất tiên nghiệm tương đối.
- `core`: các triệu chứng cốt lõi.
- `weights`: trọng số liên quan của từng triệu chứng với bệnh.

### 2.1. Biểu diễn triệu chứng

`SYMPTOMS` là bảng mã hóa triệu chứng:

```python
SYMPTOMS = {
    "S03": "Ho khan",
    "S04": "Khó thở",
    ...
}
```

Ý nghĩa:

- Giúp hệ thống xử lý bằng mã ngắn thay vì chuỗi tiếng Việt dài.
- Tránh lỗi khi so sánh chuỗi.
- Dễ dùng trong `DISEASES`, `SYMPTOM_GROUPS`, `SYMPTOM_KEYWORDS`.

### 2.2. Biểu diễn bệnh bằng frame

Mỗi bệnh trong `DISEASES` giống một frame tri thức:

```python
"covid": {
    "name": "COVID-19",
    "group": "Hô hấp & truyền nhiễm",
    "age_range": (5, 90),
    "gender": "all",
    "priority": "high",
    "core": ["S03", "S04", "S05"],
    "weights": {
        "S02": 2.6,
        "S03": 2.8,
        "S04": 3.0,
        "S05": 3.5
    }
}
```

Ý nghĩa:

- `core` là tri thức dạng “bệnh này thường cần có những dấu hiệu chính nào”.
- `weights` là tri thức định lượng, cho biết triệu chứng nào quan trọng hơn.
- `age_range`, `gender` là tri thức ngữ cảnh.
- `priority` là tri thức nền để hỗ trợ lập luận xác suất.

### 2.3. Biểu diễn luật hỏi bổ sung

`FOLLOWUP_RULES` biểu diễn luật theo dạng:

```text
IF người dùng có trigger_symptoms
THEN hỏi question
IF chọn option
THEN thêm add_symptoms và chỉnh weight_delta
```

Ví dụ:

```text
IF có đau ngực trái
THEN hỏi cơn đau có lan lên hàm hoặc tay trái không
IF có
THEN thêm triệu chứng S46 và tăng điểm heart_attack
```

Đây là phần tri thức động vì nó không chỉ lưu bệnh, mà còn quyết định hệ thống cần hỏi gì tiếp theo.

### 2.4. Biểu diễn từ khóa nhận diện triệu chứng

`SYMPTOM_KEYWORDS` giúp ánh xạ văn bản tự do sang mã triệu chứng.

Ví dụ:

```python
"S03": ["ho khan", "ho không đờm"]
```

Nếu người dùng nhập “tôi bị ho khan nhiều ngày”, hàm `extract_symptoms_from_text(...)` trong `app.py` sẽ phát hiện `S03`.

## 3. Suy diễn tiến

Vị trí chính:

- File: `inference_engine.py`
- Hàm: `infer_disease(...)`
- Hàm phụ: `_compute_rule_activation(...)`, `_compute_logic_bonus(...)`, `_compute_logic_penalty(...)`

Cách hoạt động:

Suy diễn tiến bắt đầu từ các fact đầu vào của người bệnh, ví dụ:

- `has(S03)` = có ho khan.
- `has(S04)` = có khó thở.
- `age(30)` = tuổi 30.
- `gender(male)` = giới tính nam.

Từ các fact này, hệ thống duyệt từng bệnh và kích hoạt các luật:

- Nếu khớp toàn bộ triệu chứng cốt lõi thì tăng điểm mạnh.
- Nếu độ phủ và độ tập trung cao thì tăng điểm.
- Nếu có cụm triệu chứng đặc hiệu thì tăng điểm.
- Nếu tuổi/giới tính phù hợp thì tăng điểm nhẹ.

Kết quả của suy diễn tiến được lưu trong `reasoning_steps` để hiển thị giải thích.

### 3.1. Fact trong hệ thống

Fact là sự kiện đã biết về người bệnh. Hệ thống tạo fact trong hàm `_build_patient_facts(...)`.

Ví dụ đầu vào:

```python
selected_symptoms = ["S03", "S04"]
age = 30
gender = "male"
severity = "moderate"
duration = "fewdays"
```

Fact được tạo:

```text
has(S03)
has(S04)
age(30)
gender(male)
severity(moderate)
duration(fewdays)
```

Các fact này là điểm xuất phát của suy diễn tiến.

### 3.2. Luật sản xuất IF-THEN

Trong `_compute_rule_activation(...)`, hệ thống mô phỏng luật sản xuất:

Luật 1:

```text
IF tất cả triệu chứng cốt lõi của bệnh đều xuất hiện
THEN kích hoạt mạnh giả thuyết bệnh
```

Luật 2:

```text
IF độ phủ cao AND độ tập trung cao
THEN hồ sơ người bệnh phù hợp với bệnh đang xét
```

Luật 3:

```text
IF có cụm triệu chứng đặc hiệu
THEN tăng độ tin cậy
```

Luật 4:

```text
IF tuổi nằm trong khoảng thường gặp
THEN tăng nhẹ độ tin cậy
```

Luật 5:

```text
IF giới tính phù hợp với bệnh đặc thù
THEN tăng nhẹ độ tin cậy
```

### 3.3. Ví dụ suy diễn tiến

Giả sử người dùng có:

- `S45`: đau thắt ngực trái.
- `S46`: đau lan lên hàm hoặc tay trái.
- `S04`: khó thở.

Khi xét bệnh `heart_attack`, hệ thống nhận thấy:

- `S45` và `S46` là triệu chứng cốt lõi.
- Cụm `S45 + S46` có trong `PAIR_SYNERGY`.
- Độ phủ và điểm cốt lõi cao.

Kết quả:

- `_compute_pair_synergy(...)` cộng điểm cụm đặc hiệu.
- `_compute_rule_activation(...)` kích hoạt luật triệu chứng cốt lõi.
- `reasoning_steps` ghi lại các bước suy diễn.

## 4. Suy diễn lùi

Vị trí chính:

- File: `knowledge_base.py`
- Biến: `FOLLOWUP_RULES`
- File: `app.py`
- Hàm: `build_followup_questions(...)`, `parse_followup_answers(...)`

Cách hoạt động:

Suy diễn lùi xuất hiện khi hệ thống chưa đủ thông tin và cần hỏi ngược lại người dùng.

Quy trình:

1. Người dùng nhập triệu chứng ban đầu.
2. `build_followup_questions(...)` kiểm tra triệu chứng nào kích hoạt `trigger_symptoms`.
3. Hệ thống hỏi thêm câu hỏi trong `FOLLOWUP_RULES`.
4. `parse_followup_answers(...)` biến câu trả lời thành:
   - triệu chứng bổ sung trong `add_symptoms`;
   - điểm cộng/trừ cho bệnh trong `weight_delta`.
5. `infer_disease(...)` chạy lại với tập triệu chứng và điều chỉnh mới.

Ví dụ: nếu người dùng có đau ngực trái, hệ thống hỏi cơn đau có lan lên hàm/tay trái không. Nếu có, hệ thống bổ sung `S46` và tăng điểm `heart_attack`.

### 4.1. Vì sao cần suy diễn lùi?

Suy diễn tiến chỉ dùng dữ liệu đã có. Nhưng trong y khoa, nhiều trường hợp cần hỏi thêm để phân biệt bệnh.

Ví dụ:

- Đau ngực có thể do trào ngược, viêm phổi, nhồi máu cơ tim.
- Ho ra máu có thể liên quan lao phổi, ung thư phổi hoặc tổn thương cấp.
- Vàng da có thể do viêm gan, sỏi mật, bệnh gan nặng.

Vì vậy hệ thống dùng suy diễn lùi để hỏi lại người dùng nhằm xác minh giả thuyết.

### 4.2. Cơ chế trong code

Trong `app.py`:

```python
followup_questions = build_followup_questions(base_symptoms)
```

Hàm này duyệt `FOLLOWUP_RULES`. Nếu triệu chứng ban đầu chứa `trigger_symptoms`, câu hỏi tương ứng được đưa vào giao diện.

Sau khi người dùng trả lời:

```python
extra_symptoms, adjustments, answered_followups = parse_followup_answers(...)
```

Kết quả có thể gồm:

- `extra_symptoms`: triệu chứng mới được suy ra từ câu trả lời.
- `adjustments`: điểm cộng/trừ trực tiếp cho bệnh.
- `answered_followups`: lịch sử câu hỏi và câu trả lời.

### 4.3. Ví dụ suy diễn lùi

Luật F02 trong `FOLLOWUP_RULES`:

```text
IF có S45 hoặc S99
THEN hỏi: "Cơn đau ngực có lan lên hàm hoặc tay trái không?"
```

Nếu người dùng trả lời “Có lan rõ lên hàm hoặc tay trái”:

```text
add_symptoms = ["S46"]
weight_delta = {"heart_attack": 0.18}
```

Nghĩa là hệ thống:

- Bổ sung fact `has(S46)`.
- Tăng điểm bệnh `heart_attack`.
- Chạy lại suy diễn để kết quả chính xác hơn.

## 5. Cây suy diễn chẩn đoán

Vị trí chính:

- File: `inference_engine.py`
- Đoạn tạo `reasoning_steps` trong hàm `infer_disease(...)`

Hệ thống không tạo class `Tree` riêng. Cây suy diễn được biểu diễn ngầm qua các nhánh ứng viên:

- Gốc cây: tập fact đầu vào của người bệnh.
- Mỗi nhánh: một bệnh trong `DISEASES`.
- Nút trên nhánh: các phép so khớp, luật IF-THEN, cụm triệu chứng đặc hiệu, Bayes, tuổi, giới tính, follow-up.
- Lá cây: điểm cuối cùng và nhãn cạnh tranh của bệnh.

Danh sách `reasoning_steps` là vết suy diễn của từng nhánh bệnh, dùng để giải thích tại sao bệnh được chọn.

### 5.1. Cấu trúc cây suy diễn dạng khái niệm

Có thể mô tả cây suy diễn như sau:

```text
Tập triệu chứng đầu vào
|
+-- Bệnh ứng viên 1: COVID-19
|   +-- Khớp triệu chứng nào?
|   +-- Thiếu triệu chứng cốt lõi nào?
|   +-- Luật nào được kích hoạt?
|   +-- Điểm Bayes bao nhiêu?
|   +-- Điểm cuối cùng bao nhiêu?
|
+-- Bệnh ứng viên 2: Cúm mùa
|   +-- Khớp triệu chứng nào?
|   +-- Thiếu triệu chứng cốt lõi nào?
|   +-- Luật nào được kích hoạt?
|   +-- Điểm Bayes bao nhiêu?
|   +-- Điểm cuối cùng bao nhiêu?
|
+-- Bệnh ứng viên 3: Viêm phổi
    +-- Khớp triệu chứng nào?
    +-- Thiếu triệu chứng cốt lõi nào?
    +-- Luật nào được kích hoạt?
    +-- Điểm Bayes bao nhiêu?
    +-- Điểm cuối cùng bao nhiêu?
```

### 5.2. Vì sao hệ thống không cần class Tree?

Hệ thống không cần tạo cấu trúc cây vật lý vì:

- Mỗi bệnh được xét độc lập như một nhánh.
- Mỗi nhánh đã có `reasoning_steps` để lưu vết suy diễn.
- Kết quả cuối cùng là danh sách các nhánh được xếp hạng.

Do đó cây suy diễn tồn tại ở dạng logic/giải thích, không phải ở dạng object riêng.

### 5.3. Vết suy diễn trong kết quả

Mỗi bệnh trong `results` có:

- `matched`: triệu chứng đã khớp.
- `missing_core`: triệu chứng cốt lõi còn thiếu.
- `reasoning_steps`: các bước suy diễn.
- `bayes_percent`: điểm xác suất.
- `formula`: các thành phần điểm.
- `doctor_summary`: tóm tắt lý do.
- `explanation`: giải thích chi tiết.

## 6. Lập luận xác suất

Vị trí chính:

- File: `inference_engine.py`
- Hàm: `_compute_bayesian_support(...)`

Cách hoạt động:

Hệ thống dùng Bayes giả lập vì không có dữ liệu thống kê y khoa thật.

Quy trình:

1. Lấy xác suất tiên nghiệm từ `priority` của bệnh thông qua `PRIORITY_PRIORS`.
2. Chuyển tiên nghiệm thành `log_odds`.
3. Với mỗi triệu chứng:
   - nếu triệu chứng thuộc bệnh, tăng `log_odds` theo trọng số;
   - nếu triệu chứng không thuộc bệnh, giảm khả năng giải thích.
4. Nếu thiếu triệu chứng cốt lõi, giảm `log_odds`.
5. Dùng sigmoid để đổi `log_odds` thành xác suất hậu nghiệm.
6. Trộn xác suất này với độ phủ và điểm cốt lõi để có `bayes_score`.

Sau đó `infer_disease(...)` kết hợp:

`confidence = deterministic_score * 0.72 + bayes_score * 0.28`

Trong đó điểm logic chiếm 72%, Bayes chiếm 28%.

### 6.1. Vì sao gọi là Bayes giả lập?

Bayes chuẩn cần các xác suất y khoa thật:

```text
P(bệnh)
P(triệu chứng | bệnh)
P(triệu chứng | không bệnh)
```

Dự án không có bộ dữ liệu thống kê y khoa thật, nên hệ thống ước lượng xác suất từ:

- `priority`: mức ưu tiên/nguy cơ của bệnh.
- `weights`: trọng số triệu chứng.
- `core`: triệu chứng cốt lõi.
- triệu chứng đặc hiệu không giải thích được.

Vì vậy đây là Bayes giả lập, dùng để hỗ trợ xếp hạng chứ không phải xác suất y khoa thật.

### 6.2. Các bước tính trong `_compute_bayesian_support(...)`

Bước 1: lấy xác suất tiên nghiệm.

```python
prior = PRIORITY_PRIORS.get(disease.get("priority", "medium"), 0.48)
```

Ví dụ:

- `low`: 0.36.
- `medium`: 0.48.
- `high`: 0.60.
- `critical`: 0.70.

Bước 2: đổi sang log-odds.

```text
log_odds = ln(prior / (1 - prior))
```

Bước 3: cập nhật theo từng triệu chứng.

Nếu triệu chứng thuộc bệnh:

```text
likelihood = 0.58 + 0.34 * weight / max_weight
```

Nếu triệu chứng không thuộc bệnh:

```text
likelihood = 0.20
```

Bước 4: phạt thiếu triệu chứng cốt lõi.

Nếu bệnh cần một triệu chứng cốt lõi nhưng người dùng không có triệu chứng đó, `log_odds` bị giảm.

Bước 5: đổi về xác suất hậu nghiệm.

```text
posterior = sigmoid(log_odds)
```

Bước 6: trộn với độ phủ và điểm cốt lõi.

```text
bayes_score = posterior * 0.7 + coverage_score * 0.2 + core_score * 0.1
```

### 6.3. Vai trò của xác suất trong hệ thống

Điểm Bayes giúp:

- Phân biệt các bệnh có điểm logic gần nhau.
- Tăng tính mềm dẻo khi triệu chứng không khớp tuyệt đối.
- Giảm việc hệ thống chỉ dựa cứng vào luật IF-THEN.

Tuy nhiên điểm Bayes chỉ chiếm 28% trong công thức cuối để giữ bản chất hệ chuyên gia dựa luật.

## 7. Hệ thống gợi ý

Vị trí chính:

- File: `inference_engine.py`
- Trường kết quả: `advice`, `flags`, `doctor_summary`, `explanation`, `reasoning_steps`
- File: `app.py`
- Hàm: `serialize_result_payload(...)`, `save_screening(...)`, route `/predict`, route `/api/v1/screen`

Cách hoạt động:

Sau khi xếp hạng bệnh, hệ thống tạo gợi ý theo từng bệnh:

- `advice`: lời khuyên lấy từ kho tri thức `DISEASES`.
- `flags`: cảnh báo như `Cần cấp cứu ngay`, `Cần khám chuyên khoa sớm`, `Thiếu triệu chứng cốt lõi`.
- `doctor_summary`: tóm tắt vì sao hệ thống nghĩ đến bệnh này.
- `explanation`: giải thích các thành phần điểm.
- `reasoning_steps`: các bước suy diễn đã kích hoạt.

Gợi ý không thay thế chẩn đoán bác sĩ. Nó chỉ là kết quả sàng lọc/tham khảo dựa trên tập triệu chứng người dùng cung cấp.

### 7.1. Các loại gợi ý

Hệ thống tạo nhiều lớp gợi ý:

1. Gợi ý bệnh:
   - danh sách bệnh nghi ngờ;
   - phần trăm tương đối;
   - mức cạnh tranh với bệnh đứng đầu.

2. Gợi ý hành động:
   - `advice` lấy từ kho tri thức.
   - Ví dụ nghỉ ngơi, theo dõi, đi khám sớm.

3. Cảnh báo nguy cơ:
   - `Cần cấp cứu ngay`.
   - `Cần khám chuyên khoa sớm`.
   - `Thiếu triệu chứng cốt lõi`.
   - `Dữ kiện đầu vào còn ít`.

4. Gợi ý giải thích:
   - `doctor_summary`.
   - `explanation`.
   - `reasoning_steps`.

### 7.2. Cách tạo cảnh báo

Trong `infer_disease(...)`, hệ thống gắn `flags` dựa trên điều kiện:

```text
IF bệnh thuộc nhóm cấp cứu AND điểm đủ cao
THEN thêm "Cần cấp cứu ngay"
```

```text
IF bệnh thuộc nhóm nguy cơ cao AND điểm đủ cao
THEN thêm "Cần khám chuyên khoa sớm"
```

```text
IF điểm cốt lõi thấp
THEN thêm "Thiếu triệu chứng cốt lõi"
```

```text
IF số triệu chứng đầu vào quá ít
THEN thêm "Dữ kiện đầu vào còn ít"
```

### 7.3. Chuẩn hóa phần trăm gợi ý

Sau khi có điểm thô, hệ thống gọi `_normalize_display_percent(...)`.

Mục tiêu:

- Chuyển điểm thô thành phần trăm hiển thị.
- Tổng phần trăm các bệnh hiển thị bằng 100%.
- Người dùng dễ so sánh bệnh nào nghi ngờ cao hơn.

Công thức:

```text
display_percent_i = raw_score_i / tổng raw_score * 100
```

Lưu ý: đây là phần trăm trong danh sách chẩn đoán phân biệt, không phải xác suất y khoa tuyệt đối.

## 8. Kết luận chuyên sâu

Hệ thống này là một hệ chuyên gia lai:

- Dựa luật: dùng `core`, `weights`, `FOLLOWUP_RULES`, `PAIR_SYNERGY`, luật IF-THEN.
- Dựa tìm kiếm: duyệt toàn bộ `DISEASES` để tìm bệnh phù hợp.
- Dựa xác suất: dùng Bayes giả lập để hỗ trợ xếp hạng.
- Có giải thích: tạo `reasoning_steps`, `doctor_summary`, `explanation`.
- Có gợi ý: dùng `advice` và `flags`.

Điểm mạnh:

- Dễ mở rộng kho bệnh.
- Có giải thích rõ vì sao bệnh được chọn.
- Kết hợp luật cứng và điểm mềm.
- Có hỏi bổ sung để cải thiện đầu vào.

Giới hạn:

- Không có dữ liệu y khoa thật để học xác suất chính xác.
- Nhận diện văn bản dựa trên từ khóa, chưa phải NLP nâng cao.
- Cây suy diễn là biểu diễn logic qua `reasoning_steps`, không phải cấu trúc cây vật lý.
- Kết quả chỉ mang tính sàng lọc/tham khảo.
