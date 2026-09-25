import numpy as np
import joblib
import streamlit as st


object=joblib.load("lr_california.joblib")
model=object["model"]
columns=object["columns"]
st.title("califronia app")

res=[]
for i in columns:
    res.append(st.number_input(f"enter the feature {i}"))


output=model.predict([res])
if st.button("click"):
    st.success(output)