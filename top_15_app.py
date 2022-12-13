#!/usr/bin/env python
# coding: utf-8

# In[1]:

import xgboost as xgb
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image,ImageFile
from numpy import loadtxt
from xgboost import XGBClassifier
import urllib.request
import streamlit.components.v1 as components
#import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
import eli5


model_def=xgb.XGBClassifier()

model_def.load_model("C:/Users/Maty/Documents/App_football_preediction/model_def.json")

model_mid=xgb.XGBClassifier()

model_mid.load_model("C:/Users/Maty/Documents/App_football_preediction/model_mid.json")

model_att=xgb.XGBClassifier()

model_att.load_model("C:/Users/Maty/Documents/App_football_preediction/model_att.json")


@st.cache
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)
    
#def explainer(model):
    #explainer0 = shap.TreeExplainer(model)

    #shap_values0= ""
    #return(explainer0,shap_values0)
    
def st_shap(plot, height=None):
    print(type(shap))
    print(dir(shap))
    js=shap.getjs()
    shap_html = f"<head>{js}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)
    

    
def welcome(): 
    return 'welcome all'
  
def prediction(x1, x2, x3, x4, x5,model):   
    mw=np.array([x1, x2, x3, x4, x5]).reshape(1,-1)
    prediction = model.predict(mw)
    print(prediction) 
    return prediction 
  
def main(): 
    
    st.title("Objetive")

    st.text("The objetive of this App is to predict if the player will or not be part of the Top\n15 in his position.\nEvery input stat was selected regarding their relevance in the prediction model.\nIn order to use the model in each moment of the whole season,\n all input stats must be divided by the total match played by the team.")

    st.title("Top 15 Prediction") 
    html_temp = ""
    
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    urllib.request.urlretrieve('https://img.freepik.com/vector-premium/silueta-jugador-futbol-ilustracion-bola_62860-180.jpg',"JUGADOR.jpg")
    image = Image.open("JUGADOR.jpg").resize((300,400))
    st.image(image)
    st.markdown(html_temp, unsafe_allow_html = True) 

    position=st.selectbox(
    'Which position is the player?',
    ('Defender', 'Midlefielder', 'Forward'))
    if position=="Defender":
        min_per_conceded_overall = st.number_input("Amount of minutes in which the goal recieve a goal") 
        clean_sheets_away = st.number_input("Clean sheets playing away") 
        red_cards_overall = st.number_input("Red Cards received") 
        goals_overall = st.number_input("Goals") 
        minutes_played_overall= st.number_input("Minutes played per match") 
        result =""
        #explainer_1,shap_values0=explainer(model)
        #shap_value = explainer_1.shap_values(np.array([goals_per_90_overall_input,assists_per_90_overall_input,goals_involved_per_90_overall_input,goals_involved_per_90_overall_input,min_per_conceded_overall_input]).reshape(1,-1))
        #exp = explainer.explain_instance(np.array([min_per_conceded_overall,conceded_per_90_overall,minutes_played_overall,goals_overall,clean_sheets_away]), model.predict_proba, num_features=6)

        
                                        
        
    


        
            

    

        if st.button("Predict"): 
            result =prediction(min_per_conceded_overall,red_cards_overall,goals_overall,minutes_played_overall,clean_sheets_away,model=model_def)
            
            
        
        
            if result==1:
                result='top 15 in Defenders'
            else:
                result= 'not top 15 in Defenders'
                
            st.success('The player is {}'.format(result)) 
        
        
            st.subheader('Analizando la prediccion:')
            test= pd.DataFrame(np.array([min_per_conceded_overall,red_cards_overall,minutes_played_overall,goals_overall,clean_sheets_away]).reshape(1,-1), 
             columns=['min_per_conceded_overall', 
                      'red_cards_overall','minutes_played_overall','goals_overall','clean_sheets_away'])
            

            html_object= eli5.show_prediction(model_def,test,show_feature_values=True,feature_names=['Minutes team Concede a goal', 
                      'Red Cards','Minutes played','Goals','Clean Sheets Away'])

            raw_html = html_object._repr_html_()
            components.html(raw_html,height=200)
            #show_prediction(model_def, np.array([min_per_conceded_overall,conceded_per_90_overall,goals_overall,minutes_played_overall,clean_sheets_away]), show_feature_values=True).format_as_html
            #st_shap(shap.force_plot(explainer_1.expected_value, shap_value, np.array([goals_per_90_overall_input,assists_per_90_overall_input,goals_involved_per_90_overall_input,goals_involved_per_90_overall_input,min_per_conceded_overall_input]).reshape(1,-1)))
            #components.html(exp.as_html(show_table=True), height=800)
                                   
    if position=="Midlefielder":
        goals_involved_per_90_overall = st.number_input("Goals involved per match") 
        assists_overall = st.number_input("Assits") 
        goals_per_90_overall = st.number_input("Goals per 90 minutes") 
        min_per_conceded_overall = st.number_input("Minutes in which team received a goal") 
        yellow_cards_overall= st.number_input("Yellow Cards received") 
        result =""
        #explainer_1,shap_values0=explainer(model)
        #shap_value = explainer_1.shap_values(np.array([goals_per_90_overall_input,assists_per_90_overall_input,goals_involved_per_90_overall_input,goals_involved_per_90_overall_input,min_per_conceded_overall_input]).reshape(1,-1))
        #exp = explainer.explain_instance(np.array([min_per_conceded_overall,conceded_per_90_overall,minutes_played_overall,goals_overall,clean_sheets_away]), model.predict_proba, num_features=6)

      
        if st.button("Predict"): 
            result =prediction(goals_involved_per_90_overall,goals_per_90_overall,assists_overall,min_per_conceded_overall,yellow_cards_overall,model=model_mid)
            
        
        
            if result==1:
                result='top 15 in midfielders'
            else:
                result= 'not top 15 in midfielders'
                
            st.success('The player is {}'.format(result)) 
        
        
            st.subheader('Analizando la prediccion:')
            test= pd.DataFrame(np.array([goals_involved_per_90_overall,goals_per_90_overall,assists_overall,min_per_conceded_overall,yellow_cards_overall]).reshape(1,-1), 
             columns=['goals_involved_per_90_overall', 'goals_per_90_overall',
       'assists_overall', 'min_per_conceded_overall', 'yellow_cards_overall' ])
            

            html_object= eli5.show_prediction(model_mid,test,show_feature_values=True,feature_names=['Goals involved per match', 'Goals per Match',
       'Assists', 'Minutes in which team conced a goal', 'Yellow Cards'])

            raw_html = html_object._repr_html_()
            components.html(raw_html,height=200)
            #st_shap(shap.force_plot(explainer_1.expected_value, shap_value, np.array([goals_per_90_overall_input,assists_per_90_overall_input,goals_involved_per_90_overall_input,goals_involved_per_90_overall_input,min_per_conceded_overall_input]).reshape(1,-1)))
            #components.html(exp.as_html(show_table=True), height=800)
    if position=="Forward":
        goals_per_90_overall = st.number_input("Goals per match") 
        goals_involved_per_90_overall = st.number_input("Goals involved per match") 
        clean_sheets_away = st.number_input("Clean Sheets Away") 
        min_per_conceded_overall = st.number_input("Minutes in which team received a goal") 
        penalty_goals= st.number_input("Penalty Goals") 
        result =""
        #explainer_1,shap_values0=explainer(model)
        #shap_value = explainer_1.shap_values(np.array([goals_per_90_overall_input,assists_per_90_overall_input,goals_involved_per_90_overall_input,goals_involved_per_90_overall_input,min_per_conceded_overall_input]).reshape(1,-1))
        #exp = explainer.explain_instance(np.array([min_per_conceded_overall,conceded_per_90_overall,minutes_played_overall,goals_overall,clean_sheets_away]), model.predict_proba, num_features=6)

        
    


        
            

    

        if st.button("Predict"): 
            result =prediction(goals_per_90_overall, goals_involved_per_90_overall,min_per_conceded_overall,clean_sheets_away,penalty_goals,model=model_att)
            
        
        
            if result==1:
                result='top 15 in Forwards'
            else:
                result= 'not top 15 in Forwards'
                
            st.success('The player is {}'.format(result)) 
        
        
            st.subheader('Analizando la prediccion:')
            test= pd.DataFrame(np.array([goals_per_90_overall, goals_involved_per_90_overall,min_per_conceded_overall,clean_sheets_away,penalty_goals]).reshape(1,-1), 
             columns=['goals_per_90_overall', 'goals_involved_per_90_overall','min_per_conceded_overall', 'clean_sheets_away', 'penalty_goals'])
            
            html_object= eli5.show_prediction(model_att,test,show_feature_values=True,feature_names=['Goals per match', 
                      'Goals involved','Minutes teams conced a goal','Clean Sheets playing away','Penalty goals'])
            

            raw_html = html_object._repr_html_()
            components.html(raw_html,height=200)
            #st_shap(shap.force_plot(explainer_1.expected_value, shap_value, np.array([goals_per_90_overall_input,assists_per_90_overall_input,goals_involved_per_90_overall_input,goals_involved_per_90_overall_input,min_per_conceded_overall_input]).reshape(1,-1)))
            #components.html(exp.as_html(show_table=True), height=800)                                    


if __name__=='__main__': 
        main() 
        


    # In[2]:





    # In[ ]:





    # In[ ]:





    # In[ ]:





    # In[ ]:





    # In[ ]:





    # In[ ]:




