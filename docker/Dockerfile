FROM python
RUN pip3 install fastapi
RUN pip3 install "uvicorn[standard]"
RUN pip3 install joblib

EXPOSE 6969

CMD [ "uvicorn", "main:app", "--port", "80" ]