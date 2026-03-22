# Giai thich chi tiet cach ap dung AI vao bai

## 1. Muc tieu sua `inference_engine.py`

File `knowledge_base.py` dang mo ta tri thuc theo dung tinh than cua he chuyen gia:
- `SYMPTOMS`: tap ky hieu trieu chung.
- `DISEASES`: moi benh co `core`, `weights`, `age_range`, `gender`, `priority`, `group`.
- `FOLLOWUP_RULES`: tap luat hoi them de bo sung du kien.

Vi vay `inference_engine.py` duoc sua lai theo huong:
- xem trieu chung nguoi dung nhap la **fact**;
- xem moi benh trong `DISEASES` la **tap luat + rang buoc ngu canh**;
- suy dien tu fact sang gia thuyet benh theo **luat san xuat**;
- sau do ket hop them lop **danh gia xac suat** de xep hang chan doan phan biet.

Noi ngan gon: file nay khong chi "cong diem theo trong so" nua, ma da duoc to chuc lai thanh mo hinh gan hon voi:
- Chuong 3: Bieu dien tri thuc
- Chuong 4: Phuong phap suy dien
- Chuong 5: Lap luan xac suat

---

## 2. Ap dung Chuong 3: Bieu dien tri thuc

## 2.1. Logic menh de

### Khai niem
Logic menh de xem moi menh de la mot phat bieu dung/sai.

Trong bai nay, moi fact co the duoc xem la mot menh de:
- `has(S03)`: nguoi benh co ho khan
- `has(S05)`: nguoi benh mat vi giac
- `has(S04)`: nguoi benh kho tho

Khi do co the viet luat:
- `has(S03) AND has(S05) => nghi_ngo(COVID)`
- `has(S45) AND has(S46) => nghi_ngo(nhoi_mau_co_tim)`
- `has(S47) AND has(S48) => nghi_ngo(dot_quy)`

### Da ap dung trong code nhu the nao

Trong `inference_engine.py`, phan nay duoc the hien qua:
- `_build_patient_facts(...)`: tao tap fact dau vao.
- `_compute_support_scores(...)`: xet benh nao khop voi cac menh de symptom.
- `_compute_rule_activation(...)`: kich hoat luat khi cac dieu kien menh de dung.

Vi du:
- Neu tat ca `core symptoms` cua benh deu xuat hien, luat "benh duoc kich hoat manh" se dung.
- Neu `coverage` cao va `precision` cao, luat "ho so phu hop voi benh" se dung.

### Loi ich
- De cai dat.
- Giai thich duoc vi sao benh duoc tang diem hay giam diem.
- Rat phu hop voi he chuyen gia dua tren tap luat.

---

## 2.2. Logic vi tu cap 1

### Khai niem
Logic vi tu cap 1 mo rong logic menh de bang cach them:
- doi tuong
- thuoc tinh
- quan he

Dang tong quat:
- `HasSymptom(Patient, S03)`
- `Age(Patient, 25)`
- `Gender(Patient, male)`
- `SuitableAge(Patient, covid)`

### Da ap dung trong code nhu the nao

Trong code moi, cac fact khong chi dung o muc symptom, ma con them ngu canh:
- `age(25)`
- `gender(male)`
- `severity(severe)`
- `duration(weeks)`

Ham `_build_patient_facts(...)` tao tap fact kieu nay.

Ham `_compute_rule_activation(...)` dung cac fact ngu canh do de kich hoat them luat:
- tuoi nam trong `age_range` cua benh => tang diem
- gioi tinh phu hop voi benh dac thu => tang diem

Ham `infer_disease(...)` con dung logic vi tu de:
- giam diem neu tuoi qua xa `age_range`
- giam rat manh neu gioi tinh khong phu hop voi benh dac thu

### Vi du

Voi benh `bph`:
- Neu `gender = male` va `age` nam trong nhom thuong gap => giu gia thuyet manh hon.
- Neu `gender = female` => he thong tru diem rat manh.

Dieu nay la tinh than cua logic vi tu:
- cung mot trieu chung, nhung doi tuong khac nhau thi ket luan khac nhau.

---

## 2.3. Luat san xuat

### Khai niem
Luat san xuat co dang:

`IF dieu_kien THEN ket_luan`

Vi du:
- IF `ho keo dai` AND `sut can` THEN `nghi lao phoi`
- IF `dau nguc trai` AND `lan tay trai` THEN `nghi nhoi mau co tim`

### Da ap dung trong code nhu the nao

Phan trung tam la ham:
- `_compute_rule_activation(...)`

Ham nay hien thuc luat san xuat theo 3 lop:

1. Luat cot loi
- Neu `core_score == 1`:
  - nghia la tat ca trieu chung cot loi cua benh deu xuat hien
  - he thong tang diem kich hoat manh

2. Luat phu hop tong the
- Neu `coverage_score >= 0.5` va `precision_score >= 0.45`
  - coi nhu bo trieu chung dau vao giai thich benh kha tot

3. Luat cum trieu chung dac hieu
- Duoc model hoa trong `PAIR_SYNERGY`
- Vi du:
  - `S03 + S05` la cum dac hieu cho `covid`
  - `S45 + S46` la cum dac hieu cho `heart_attack`
  - `S47 + S48` la cum dac hieu cho `stroke`

Khi cum nay dung, he thong tang diem vi luat dac hieu da duoc kich hoat.

### Vi sao cach nay phu hop voi `knowledge_base.py`

Vi `knowledge_base.py` da luu:
- `core`
- `weights`
- `age_range`
- `gender`
- `priority`

Nen `inference_engine.py` moi khong can hard-code lai tri thuc theo kieu roi rac.
No doc truc tiep cau truc tri thuc da co san trong kho tri thuc va bien no thanh:
- fact
- dieu kien
- luat
- ket luan

Do do logic da "phu hop voi knowledge_base.py" hon phien ban cu.

---

## 3. Ap dung Chuong 4: Cac phuong phap suy dien

## 3.1. Cay, luoi va do thi

### Cach hieu trong bai

Co the xem qua trinh suy dien nhu mot do thi:
- Nut dau vao: trieu chung, tuoi, gioi tinh, muc do, thoi gian
- Nut trung gian: `coverage`, `precision`, `core_match`, `pair_synergy`, `rule_activation`
- Nut dich: tung gia thuyet benh

Quan he:
- symptom -> disease
- context -> disease
- follow-up answer -> disease

Moi benh la mot nut dich trong do thi suy dien.

### Ap dung trong code

Trong `infer_disease(...)`, moi benh trong `DISEASES` duoc duyet nhu:
- lay tap symptom khop
- tinh do phu
- kich hoat luat
- tinh diem xac suat
- tao nhan xet giai thich

Day la cach di tren do thi tu fact -> gia thuyet.

---

## 3.2. Trang thai va khong gian

### Khai niem
Trang thai la mo ta hien tai cua bai toan.
Khong gian trang thai la tap tat ca kha nang.

### Ap dung trong bai

Trang thai hien tai = tap du lieu nguoi dung vua cung cap:
- cac trieu chung
- tuoi
- gioi tinh
- muc do
- thoi gian
- cau tra loi follow-up

Moi khi co them follow-up:
- trang thai thay doi
- tap fact thay doi
- diem benh thay doi

Tuc la he thong di chuyen trong khong gian trang thai chan doan.

---

## 3.3. Cay AND-OR

### Cach hieu

Co nhieu benh can:
- AND: can cung luc nhieu trieu chung cot loi
- OR: chi can mot so trieu chung ho tro van co the giu gia thuyet

### Ap dung trong code

AND:
- `core_score == 1` nghia la toan bo core match
- cac cap dac hieu trong `PAIR_SYNERGY` yeu cau tat ca symptom trong cap cung xuat hien

OR:
- benh van duoc giu lai neu co `matched_weight > 0`
- follow-up co the bo sung them symptom de day manh mot benh cu the

Vi du:
- Lao phoi:
  - AND manh: `ho keo dai` AND `sut can`
  - OR ho tro: them `sot ve chieu` OR `do mo hoi dem`

---

## 3.4. Luat suy dien

### Luat tien
Tu symptom suy ra benh.

### Luat lui
Tu benh nghi ngo quay lai hoi symptom con thieu de xac minh.

### Ap dung trong bai

#### Suy dien tien
Phan chinh nam o:
- `infer_disease(...)`
- `_compute_rule_activation(...)`
- `_compute_support_scores(...)`

He thong lay symptom dau vao roi suy ra benh.

#### Suy dien lui
Phan nay duoc the hien giap tiep qua:
- `FOLLOWUP_RULES` trong `knowledge_base.py`
- `build_followup_questions(...)`
- `parse_followup_answers(...)`

Co nghia la:
1. He thong tam nghi mot nhom benh
2. Sau do quay nguoc lai hoi them symptom quan trong
3. Neu cau tra loi xac nhan, diem benh tang
4. Neu phu dinh, diem benh giam hoac khong tang

Day chinh la tinh than cua lap luan lui.

---

## 4. Ap dung Chuong 5: Lap luan xac suat

## 4.1. Van de thong tin khong chac chan

Trong y khoa, mot symptom co the thuoc nhieu benh:
- sot + ho co the la cum, covid, viem phoi
- dau bung co the la viem da day, soi mat, ung thu da day

Neu chi dung luat logic cung:
- he thong rat de ket luan cung nhac

Vi vay can mot lop xac suat de:
- xep hang
- mo ta muc nghi ngo tuong doi
- xu ly du lieu thieu

---

## 4.2. Nguyen tac lap luan xac suat

File moi bo sung:
- `PRIORITY_PRIORS`
- `_compute_bayesian_support(...)`

### Y tuong

Moi benh co:
- prior xap xi theo `priority`
- likelihood xap xi theo muc do quan trong cua symptom (`weights`)

Neu symptom xuat hien va co weight cao:
- xac suat ho tro benh tang

Neu core symptom vang:
- xac suat giam

Neu co symptom dac hieu nhung benh khong giai thich duoc:
- xac suat giam tiep

### Cach lam trong code

Ham `_compute_bayesian_support(...)`:
1. Lay prior tu `priority`
2. Di qua tung symptom nguoi dung nhap
3. Neu symptom nam trong `weights` cua benh:
   - cho likelihood cao hon
4. Neu symptom khong nam trong `weights`:
   - cho likelihood thap
5. Neu thieu `core symptom`:
   - cong them he so am vao log-odds
6. Chuyen ve posterior bang sigmoid
7. Tron nhe voi `coverage_score` va `core_score`

Ket qua la mot `bayes_score` trong [0, 1].

---

## 4.3. Vi sao khong dung Bayes thuan tuy?

Vi bai nay la demo hoc tap, kho tri thuc hien tai:
- chua co tan suat xac suat that
- chua co P(symptom|disease) duoc thong ke tu du lieu y khoa that

Nen cach dang dung la:
- **Bayes gia lap / Bayes proxy**
- du de minh hoa chuong 5
- van ton trong tri thuc da co trong `knowledge_base.py`

No phu hop voi muc tieu mon hoc hon la co gang gia vo co xac suat chinh xac y khoa.

---

## 5. Cong thuc tong hop sau khi sua

Cho moi benh, he thong tinh:

1. Diem logic:
- coverage
- precision
- core match
- supportive
- rule activation
- bonus do cum symptom dac hieu
- penalty do thieu core, symptom mau thuan, symptom dac hieu khong giai thich duoc

2. Diem xac suat:
- `bayes_score`

3. Tron hai thanh phan:
- `confidence = deterministic_score * 0.72 + bayes_score * 0.28`

4. Hieu chinh ngu canh:
- muc do
- thoi gian
- tuoi
- gioi tinh
- follow-up

5. Chuan hoa:
- so sanh trong tung nhom benh
- roi chuan hoa thanh `% chan doan phan biet`

---

## 6. Cac diem sua quan trong trong `inference_engine.py`

## 6.1. Tao tap fact ro rang

Ham moi:
- `_build_patient_facts(...)`

Tac dung:
- bien input thanh dang fact co cau truc
- phuc vu cho logic menh de va logic vi tu

## 6.2. Kich hoat luat san xuat

Ham moi:
- `_compute_rule_activation(...)`

Tac dung:
- gom cac luat IF-THEN thanh mot cho
- de giai thich va de mo rong

## 6.3. Bo sung lop xac suat

Ham moi:
- `_compute_bayesian_support(...)`

Tac dung:
- ho tro xep hang khi nhieu benh co symptom giong nhau
- tao lien he voi chuong 5

## 6.4. Sua nguong hien thi

Phien ban cu dung:
- `MIN_DISPLAY_THRESHOLD = 0.8`
- nhung lai so sanh voi `raw_percent`

Dieu nay de gay nham ve don vi.

Phien ban moi doi thanh:
- `MIN_DISPLAY_SCORE_THRESHOLD = 0.08`
- so sanh truc tiep voi `raw_score`

Nghia ro hon va nhat quan hon.

## 6.5. Them vet suy dien de giai thich

Ket qua moi co:
- `reasoning_steps`
- `bayes_percent`
- `rule_activation`

Nen khi hien thi, nguoi hoc de theo doi:
- fact nao da vao
- luat nao da no
- vi sao diem tang/giam

---

## 7. Cach ap dung tung phan ly thuyet vao bai thuyet trinh

Neu can thuyet trinh, co the noi theo thu tu sau:

### Phan 1. Bieu dien tri thuc
- Trieu chung duoc ma hoa thanh ky hieu `Sxx`
- Benh duoc mo ta bang tap core symptom, weight, tuoi, gioi tinh
- Quan he giua symptom va benh duoc bieu dien bang luat san xuat

### Phan 2. Suy dien
- Dau vao tao thanh fact
- Lap luan tien kich hoat cac benh tu tap fact
- Lap luan lui duoc mo phong bang follow-up questions

### Phan 3. Xac suat
- Khi nhieu benh cung thoa mot phan dieu kien, can xep hang
- Bayes proxy duoc dung de ho tro danh gia muc nghi ngo

### Phan 4. Giai thich ket qua
- Moi benh deu co do phu, precision, core match, Bayes score, bonus, penalty
- Nguoi dung khong chi thay ket qua, ma con thay logic ket qua

---

## 8. `gioithieu.html` dung de lam gi

File `gioithieu.html` duoc tao de:
- gioi thieu tong quan bai
- trinh bay muc tieu
- tom tat cac ky thuat AI da dung
- giup giang vien hoac nguoi xem vao la hieu ngay bai nay dang minh hoa noi dung nao cua mon hoc

No rat hop khi demo dau gio thay vi vao thang trang nhap lieu.

---

## 9. Tai sao web lag tren Chrome/Coc Coc nhung Safari muot

Nguyen nhan chinh:
- qua nhieu `blur`
- `backdrop-filter` lon
- animation lien tuc tren pseudo-element kich thuoc rat lon
- hieu ung 3D `translateZ`, `rotateX`, `rotateY`
- selector `:has(input:checked)` ap dung tren rat nhieu `.symptom-card`

Safari thuong toi uu mot so hieu ung do tot hon Chrome.
Chrome/Coc Coc thuong bi ton:
- recalculation style
- paint
- compositing

### Cach da sua

1. Tao `static/app.js`
- tu dong gan class `chromium-optimized` cho Chrome/Coc Coc/Chromium

2. Bo `:has(...)`
- doi sang class `.is-checked`
- update bang JavaScript khi checkbox thay doi

3. Giam hieu ung nang tren Chromium
- tat bieu dien blur/animation qua nang
- giam do sau transform
- giam shadow va hover phuc tap

Nho vay giao dien van dep, nhung re-render nhe hon tren Chrome/Coc Coc.

---

## 10. Ket luan

Sau khi sua, bai nay da ro hon ve mat hoc thuat:
- dung `knowledge_base.py` nhu mot kho tri thuc that su
- suy dien bang luat san xuat thay vi cong diem thu cong don thuan
- co lien he ro voi logic menh de, logic vi tu cap 1
- co lap luan tien, lap luan lui
- co bo sung lop lap luan xac suat kieu Bayes de xep hang
- co trang gioi thieu tong quan de demo bai
- co toi uu render de Chrome/Coc Coc bot lag

Neu can, co the mo rong tiep theo huong:
- them so do AND-OR vao giao dien
- ve graph symptom -> disease
- hien thi "luat nao da kich hoat" thanh bang chi tiet tren trang ket qua
