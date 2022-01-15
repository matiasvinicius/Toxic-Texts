import streamlit as st
import pandas as pd
import pickle
from lime.lime_text import LimeTextExplainer

st.title("Toxicity classificator")
txt = st.text_input("Type some text and our AI will retrieve the classification about toxicity", 'I love everybody')
pipe = pickle.load(open('pipe.pickle', 'rb'))

prediction = pipe.predict([txt])[0]

st.write('Is toxic:', prediction)

explainer = LimeTextExplainer(class_names=["Safe", "Toxic"])
st.write("Running explainer")
exp = explainer.explain_instance(txt, pipe.predict_proba, num_features=12)
fig = exp.as_html()
st.components.v1.html(fig, height = 300, scrolling=True)
