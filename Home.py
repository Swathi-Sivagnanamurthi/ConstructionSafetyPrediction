import streamlit as st
import pandas as pd
import pickle 

#Load CSS
with open('./css/Stylesheet.css') as f:
    css = f.read()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

#Load the dataset for input fields
df = pd.read_pickle('data.pkl')
#df
constr_enduse_data=df[['Construction End Use','con_end']].sort_values(by='con_end',ascending=True).dropna()
constr_enduse_data=constr_enduse_data[constr_enduse_data.con_end!=0].drop_duplicates()

env_fact_data=df[['Environmental Factor','evn_factor']].sort_values(by='evn_factor',ascending=True).dropna()
env_fact_data=env_fact_data.drop_duplicates()

hum_fact_data=df[['Human Factor','hum_factor']].sort_values(by='hum_factor',ascending=True).dropna()
hum_fact_data=hum_fact_data.drop_duplicates()

proj_type_data=df[['Project Type','proj_type']].sort_values(by='proj_type',ascending=True).dropna()
proj_type_data=proj_type_data[proj_type_data.proj_type!=0].drop_duplicates()

nature_injury_data=df[['Nature of Injury','nature_of_inj']].sort_values(by='nature_of_inj',ascending=True).dropna()
nature_injury_data=nature_injury_data.drop_duplicates()

parts_inj_data=df[['Part of Body','part_of_body']].sort_values(by='part_of_body',ascending=True).dropna()
parts_inj_data=parts_inj_data[parts_inj_data.part_of_body!=0].drop_duplicates()


Event_type_data=df[['Event type','event_type']].sort_values(by='event_type',ascending=True)
Event_type_data=Event_type_data[Event_type_data.event_type!=0].drop_duplicates()

task_assigned_data=df[['Task Assigned','task_assigned']].sort_values(by='task_assigned',ascending=True)
task_assigned_data=task_assigned_data.drop_duplicates()

#proj_cost_data1=df[['Project Cost','proj_cost']].sort_values(by='Project Cost',ascending=True).drop_duplicates()
#proj_cost_data1=proj_cost_data1[proj_cost_data1.proj_cost!='0']
#proj_cost_data1['Project Cost'] = proj_cost_data1['Project Cost'].str.replace(",","")
#proj_cost_data1=proj_cost_data1[['Project Cost','proj_cost']]
#proj_cost_data1['Project Cost']=proj_cost_data1['Project Cost'].str.replace("to"," to ")

#Load th trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
print(model.feature_names_in_)

# Initialize an empty list to store the data
data_list=[]

# Define the get_data function
def get_data(con_enduse, floors, env_fact, hum_fact, proj_type, task, nature_injury, parts_inj,Event_type):
    input_data={"con_end": con_enduse_value,
                "build_stor": floors,
                "proj_type": proj_type_value,
                "nature_of_inj": nature_injury_value ,
                "part_of_body": parts_inj_value,
                "event_type":Event_type_value,
                "evn_factor": env_fact_value,
                "hum_factor": hum_fact_value,
                "task_assigned": task_value
                }
    #st.write(pd.DataFrame(input_data))
    input_df=pd.DataFrame([input_data])
    #st.write(input_df)

    # Make prediction using the loaded model
    prediction = model.predict(input_df)
    if prediction=='1':
        pred="NonFatal"
        st.warning("Prediction: Nonfatal So safety measure for activity will be HIGH")
    else:
        pred="Fatal"
    st.success("Prediction: Fatal So Safety measure for this acitivity is NORMAL")
    

with st.form("my_form",border=False,clear_on_submit=True):
    st.header("CONSTRUCTION SAFETY LEVEL PREDICTION")
    grid = st.columns(2)
    with grid[0]:
        #Construction End Use
        con_enduse_text=st.selectbox('Construction End Use',constr_enduse_data)
        con_enduse_value=constr_enduse_data.loc[constr_enduse_data['Construction End Use']==con_enduse_text,'con_end'].values[0]
        #Environmental Factor
        env_fact_text=st.selectbox('Environmental Factor',env_fact_data)
        env_fact_value=env_fact_data.loc[env_fact_data['Environmental Factor']==env_fact_text,'evn_factor'].values[0]
        #Project Type
        proj_type_text=st.selectbox('Project Type',proj_type_data)
        proj_type_value=proj_type_data.loc[proj_type_data['Project Type']==proj_type_text,'proj_type'].values[0]
    with grid[1]:
        #Number of floors
        floors=st.number_input('Number of Floors',value=0,step=1,placeholder="Type a number")
        #Human factor
        hum_fact_text=st.selectbox('Human Factors',hum_fact_data)
        hum_fact_value=hum_fact_data.loc[hum_fact_data['Human Factor']==hum_fact_text,'hum_factor'].values[0]
        #Task Assigned
        task_text=st.radio("Task",task_assigned_data['Task Assigned'])
        task_value=task_assigned_data.loc[task_assigned_data['Task Assigned']==task_text,'task_assigned'].values[0]
    #Event type
    Event_type_text=st.selectbox("Event Type",Event_type_data)
    Event_type_value=Event_type_data.loc[Event_type_data['Event type']==Event_type_text,'event_type'].values[0]
    #Project Cost
    #project_cost_text=st.radio("Choose Project cost range",proj_cost_data1['Project Cost'],horizontal=True)
    #proj_cost_value=proj_cost_data1.loc[proj_cost_data1['Project Cost']==project_cost_text,'proj_cost'].values[0]
    #Nature of injury
    nature_injury_text=st.selectbox('Nature of Injury',nature_injury_data)
    nature_injury_value=nature_injury_data.loc[nature_injury_data['Nature of Injury']==nature_injury_text,'nature_of_inj'].values[0]
    #Parts of the body
    parts_inj_text=st.selectbox('Part of Body',parts_inj_data)
    parts_inj_value=parts_inj_data.loc[parts_inj_data['Part of Body']==parts_inj_text,'part_of_body'].values[0]

    Predict=st.form_submit_button('Predict')
    if Predict:
        with st.spinner('Wait for it...'):
            get_data(con_enduse_value,floors,env_fact_value,hum_fact_value,proj_type_value,task_value,nature_injury_value,parts_inj_value,Event_type_value)