import streamlit as st
import sys
import platform
import os

st.title('Debug: NeuralRetail Streamlit Environment')
st.write('Python version:', sys.version)
st.write('Platform:', platform.platform())
st.write('Current working directory:', os.getcwd())
st.write('Environment variables (sample):')
for k in ['API_URL', 'PORT']:
    st.write(k, '->', os.getenv(k))

st.success('Debug app started successfully. If this loads, Streamlit hosting and Python runtime are OK.')
