FROM python:3.12-slim

WORKDIR /app 

COPY requiremnts.txt .

RUN pip install --no-cache-dir -r requiremnts.txt

COPY . . 

EXPOSE 5000

CMD ["python", "app.py"]