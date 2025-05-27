import streamlit_shadcn_ui as ui
import streamlit as st
import pickle
import pandas as pd
from skimage.transform import resize
from skimage.io import imread
import plotly.graph_objects as go
import numpy as np
cm=np.array([[1494,1053],[950,1493]])
with open(r"model_rfc_best.pkl", "rb") as file:
    model = pickle.load(file)
tab=ui.tabs(options=["Predictor","Behind The Scenes"], key="tab1")
if tab == 'Predictor':
    st.markdown('# Hello')
    st.write('This model will predict the differene between a cat or a dog')
    uploaded_file = st.file_uploader("Upload an image(Please upload a downloaded image)", type=["png", "jpg", "jpeg"])
    submit=st.button('Predict')
    if submit == True:
        img=imread(uploaded_file)
        st.image(uploaded_file, caption="Your Image", use_container_width=True)
        img_resize=resize(img,(200,200,3))
        l=[img_resize.flatten()]
        probability=model.predict_proba(l)
        Categories=['Cat','Dog']
        for ind, val in enumerate(Categories):
            st.write(f"{val} = {probability[0][ind]*100:.2f}%")
        st.markdown("## The predicted image is : "+Categories[model.predict(l)[0]])
elif tab =='Behind The Scenes':
    st.markdown('# This is a Random Forest Model')
    st.markdown('### This Model uses RFC(Random Forest Classfier)to identify if the picture given is a cat or dog')
    cols = st.columns(1)
    with cols[0]:
        ui.metric_card(title="Total traing data size", content="3 Billion different values", description="Big Data", key="card12")
    cols=st.columns(2)
    with cols[0]:
        ui.metric_card(title="Accuracy", content="60%", description="Decent Accuracy", key="card124")
    with cols[1]:
        ui.metric_card(title="Algrothim(Ai)", content="Random Forest Classfier", description="n_estimators=1000", key="card232")
    labels = ["Cat", "Dog"]
    fig = go.Figure(data=go.Heatmap(
    z=cm,
    x= labels ,
    y= labels , 
    colorscale='Blues',
    showscale=True
    ))
    annotations = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            annotations.append(
            dict(
                text=str(cm[i][j]),
                x=labels[j],
                y=labels[i],
                font=dict(color="black", size=14),
                showarrow=False
            )
        )
    fig.update_layout(
    title="Confusion Matrix",
    xaxis_title="Predicted Label",
    yaxis_title="True Label",
    annotations=annotations
)
    st.plotly_chart(fig)
    data = [
    {"Factors": "Cats", "Precision": "0.61", "Recall": "0.59", "F1-Score": "0.60","Support":"2547"},
    {"Factors": "Dogs", "Precision": "0.59", "Recall": "0.61", "F1-Score": "0.60","Support":"2443"},
    {"Factors": "Accuracy", "Precision": "0", "Recall": "0", "F1-Score": "0.60","Support":"4990"},
    {"Factors": "Macro Average", "Precision": "0.60", "Recall": "0.60", "F1-Score": "0.60","Support":"4990"},
    {"Factors": "Wieghted Average", "Precision": "0.60", "Recall":"0.60", "F1-Score": "0.60","Support":"4990"},
    ]
