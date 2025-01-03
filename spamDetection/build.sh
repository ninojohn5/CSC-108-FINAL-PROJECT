#!/usr/bin/env bash

set -o errexit

pip install requirements.txt


python manage.py collecstatic --no-input

python manage.py migrate