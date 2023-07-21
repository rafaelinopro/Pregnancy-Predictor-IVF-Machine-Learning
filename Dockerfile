FROM python:3.7.4

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY ./src/Flask/ .

EXPOSE 5000

CMD ["flask", "run","--host","0.0.0.0","--port","5000"]