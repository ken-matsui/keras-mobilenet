FROM gcr.io/google-appengine/python@sha256:ea5aa4f84336233ae6a1ca39ca3fd1608cc01ed865e777c58cbd99c45bbe2a1f
LABEL python_version=python3.6
RUN virtualenv --no-download /env -p python3.6
ENV VIRTUAL_ENV /env
ENV PATH /env/bin:$PATH
ADD requirements.txt /app/
RUN pip install -r requirements.txt
RUN apt-get update -qq \
 && apt-get install -y -qq libsm6 libxext6 libxrender-dev
ADD . /app/
CMD exec gunicorn -b :$PORT main:app
