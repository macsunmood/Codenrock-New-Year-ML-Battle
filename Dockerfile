FROM python:3.9-buster
COPY ./ /app
WORKDIR /app
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

# Download pretrained model with weights
RUN gdown 1Id9r8YSq_XCKeF0hz020hvdhQd6jETOE --output ./data/weight

CMD ["python3","run.py"]