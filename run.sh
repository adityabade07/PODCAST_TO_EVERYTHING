#!/bin/bash
cd "$(dirname "$0")"
python -m pip install -q -r requirements.txt
python -m streamlit run app.py
