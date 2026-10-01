FROM python:3.14

WORKDIR /app

COPY requirements.txt . 

RUN pip install --no-cache-dir -r requirements.txt

COPY apiii.py .
COPY Credit_Default.pkl .

EXPOSE 10000

CMD ["sh", "-c", "uvicorn apiii:app --host 0.0.0.0 --port ${PORT:-10000}"]