"""Model comparison with only the charts needed to explain class imbalance."""
import pandas as pd
import plotly.express as px
import streamlit as st
from components.charts import confusion_chart
from components.ui import card, goto, page_header
from components.home_lottie import lottie_data_analytics
from components.data import load_results, load_cv, model_result, load_clarity


def render_under_the_lens():
    page_header('DIAGNOSTIC ANALYTICS','Under the <em>Lens</em>',
                'Where the baseline and imbalance strategies differ.',lottie_data_analytics,'analytics_lottie')
    st.info('Predicting **Normal for every image** gives 75% accuracy on this dataset, '
            'but detects **zero cataracts**. Recall and specificity must be considered alongside accuracy.')

    baseline=model_result('EfficientNetB0','baseline')
    weighted=model_result('EfficientNetB0','class weights')
    def cap(r): return f'{int(r.TP)}/20 cataracts detected · {int(r.FN)} missed · {int(r.FP)} false alarms'
    a,b=st.columns(2)
    with a:
        with card('Highest MCC · baseline',f'EfficientNetB0 · MCC {baseline.mcc:.3f}'):
            st.plotly_chart(confusion_chart(baseline),use_container_width=True)
            st.caption(cap(baseline))
    with b:
        with card('Higher sensitivity · class weights',f'EfficientNetB0 · MCC {weighted.mcc:.3f}'):
            st.plotly_chart(confusion_chart(weighted),use_container_width=True)
            st.caption(cap(weighted))

    with card('All models','Sensitivity versus specificity'):
        df=load_results().copy()
        choices=['All','Classical','Neural']
        family=st.segmented_control('Model family',choices,default='All')
        if family and family!='All':df=df.loc[df.family==family]
        fig=px.scatter(df,x='specificity',y='recall',color='model',symbol='strategy',
             hover_data=['strategy','accuracy','mcc','FN','FP'],range_x=[0.7,1.02],range_y=[0.35,1.02],
             labels={'specificity':'Normal specificity','recall':'Cataract sensitivity'})
        fig.update_xaxes(tickformat='.0%');fig.update_yaxes(tickformat='.0%')
        fig.update_layout(height=400,legend_title_text='Model / handling',margin=dict(l=10,r=10,t=10,b=10),paper_bgcolor='rgba(0,0,0,0)',font=dict(family='Inter, sans-serif'))
        st.plotly_chart(fig,use_container_width=True)

    with st.expander('Research process · methodology details'):
        st.markdown('**1. Data and split:** 400 fundus images (300 Normal, 100 Cataract); stratified 320 train / 80 test.')
        st.markdown('**2. Preprocessing:** black-border crop, 224×224 resize and LAB-channel CLAHE.')
        st.markdown('**3. Two model branches:** 401 handcrafted features for SVM / RF / LR; pixels or ImageNet features for three CNN architectures.')
        st.markdown('**4. Handle imbalance:** classical baseline, weights, ROS and SMOTE (12 runs); neural baseline, weights and augmentation (9 runs).')
        st.markdown('**5. Evaluate:** accuracy, cataract sensitivity, normal specificity, F1 and MCC on the unchanged test set.')
        st.caption('Classical tuning used stratified training cross-validation. Neural early stopping used an internal validation partition. Selecting the best among 21 on the test set introduces selection bias.')
    with st.expander('Handling techniques and results'):
        st.write('**Class weights:** change error penalties. **ROS:** copy minority training vectors. '
                 '**SMOTE:** interpolate training features. **Augmentation:** rotate/flip minority training images. '
                 'All techniques were confined to training data.')
        p=load_results().pivot_table(index='model',columns='strategy',values='recall')
        st.dataframe(p.style.format('{:.0%}'),use_container_width=True)
    with st.expander('Repeated cross-validation · classical models'):
        cv=load_cv()[['model','strategy','recall_mean','recall_sd','mcc_mean','mcc_sd']]
        st.dataframe(cv,hide_index=True,use_container_width=True)
        st.caption('5-fold CV × 3 repeats, with previously tuned hyperparameters; not fully nested.')
    with st.expander('Clarity descriptors · full dataset'):
        c=load_clarity()
        desc=c.groupby('class')[['edge_density','corner_density','laplacian_variance']].mean().round(4)
        st.dataframe(desc)
        st.caption('Descriptive measurements only; focus and acquisition conditions can also change image clarity.')
    st.caption('All numbers above are from the latest documented 21-experiment run. No cutoff was optimised on the test set.')

    if st.button('TRY A PHOTO ➜',key='to_prediction'):
        goto('go_to_screen')
