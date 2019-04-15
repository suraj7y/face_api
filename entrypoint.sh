#!/bin/bash

# Prepare log files and start outputting logs to stdout
mkdir logs
touch ./logs/gunicorn.log
touch ./logs/gunicorn-access.log
tail -n 0 -f ./logs/gunicorn*.log &

export DJANGO_SETTINGS_MODULE=face_rec.settings

exec /usr/local/bin/gunicorn face_rec.wsgi:application \
    --name face_rec \
    --bind unix:face_rec.sock \
    --workers 3 \
    --timeout 600 \
    --log-level=info \
    --log-file=./logs/gunicorn.log \
    --access-logfile=./logs/access.log &

exec service nginx start



