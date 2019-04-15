FROM ubuntu:18.04
MAINTAINER suraj <suraj7.y@gmail.com>


RUN apt-get update && \
    apt-get -y install \
        python3-pip \
	python3-dev \
        python3-pip \
        python-lxml \
        nginx


ENV PYTHONUNBUFFERED 1

RUN mkdir /face_rec

WORKDIR /face_rec

COPY . .

RUN pip3 install -r requirements.txt

RUN pip3 install gunicorn

#RUN python3 manage.py collectstatic  --noinput

ENV LANG C.UTF-8

ENV LC_ALL C.UTF-8

EXPOSE 9092

COPY deploy/django_nginx.conf /etc/nginx/sites-available/

RUN ln -s /etc/nginx/sites-available/django_nginx.conf /etc/nginx/sites-enabled

RUN echo "daemon off;" >> /etc/nginx/nginx.conf

RUN rm -rf deploy && rm -rf Dockerfile && sed -i -e 's/www-data/root/g' /etc/nginx/nginx.conf

ENTRYPOINT ["sh", "entrypoint.sh"]
