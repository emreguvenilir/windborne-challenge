#!/bin/bash
set -e  # stop on error

echo "Starting initial data and model setup..."
python3 workingV1.py
echo "Training model..."
python3 model1.py
echo "Setup complete. Launching app..."
exec gunicorn app:app