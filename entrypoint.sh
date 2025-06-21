python demonstration/manage.py migrate --noinput
python demonstration/manage.py collectstatic --noinput
cd demonstration
python -m gunicorn --bind 0.0.0.0:8000 --workers 3 config.wsgi:application
