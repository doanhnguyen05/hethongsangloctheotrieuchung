# Hệ thống sàng lọc bệnh theo triệu chứng

## 1. Mở terminal và chuyển vào thư mục dự án

```bash
cd "/Users/doanhnguyen/Documents/tài liệu/phầm mềm/Trí tuệ nhân tạo"
```

## 2. Tạo hoặc kích hoạt môi trường ảo

```bash
python3 -m venv venv
source venv/bin/activate
```

Nếu lệnh `source venv/bin/activate` không được, dùng:

```bash
. ./venv/bin/activate
```

## 3. Cài đặt phụ thuộc

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu sau khi kích hoạt venv mà `python` hoặc `pip` vẫn không tìm thấy, dùng:

```bash
python3 -m pip install -r requirements.txt
```

## 4. Chạy ứng dụng

```bash
python app.py
```

## 5. Mở ứng dụng trên trình duyệt

Truy cập:

```
http://127.0.0.1:5050
```

## 6. Khắc phục lỗi thường gặp

- Nếu `pip` hoặc `python` không được tìm thấy sau khi kích hoạt venv, môi trường ảo hiện tại có thể bị hỏng.
- Xóa và tạo lại venv:

```bash
rm -rf venv
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

> Lưu ý: vì đường dẫn chứa khoảng trắng và ký tự tiếng Việt, hãy dùng dấu ngoặc kép khi nhập đường dẫn đầy đủ.
