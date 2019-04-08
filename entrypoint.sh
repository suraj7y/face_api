#!/bin/bash

# Prepare log files and start outputting logs to stdout
mkdir logs
touch ./logs/gunicorn.log
touch ./logs/gunicorn-access.log
tail -n 0 -f ./logs/gunicorn*.log &

export DJANGO_SETTINGS_MODULE=erp.settings

exec /usr/local/bin/gunicorn erp.wsgi:application \
    --name erp \
    --bind unix:erp.sock \
    --workers 3 \
    --log-level=info \
    --log-file=./logs/gunicorn.log \
    --access-logfile=./logs/access.log &

exec service nginx start



