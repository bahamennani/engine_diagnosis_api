FROM python:3.10-slim

# إعداد مجلد العمل
WORKDIR /app

# نسخ وتثبيت التبعيات
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# نسخ بقية ملفات المشروع
COPY . .

# فتح البورت الذي يستخدمه render (5000)
EXPOSE 5000

# استخدام gunicorn لتشغيل التطبيق
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
