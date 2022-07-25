import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import modelModule as mm
import database


st.title("Analytic FORK")
st.write("---------------------------------------------")


st.subheader("Upload CSV")
csv_file = st.file_uploader("",type = "csv")


bt_1 = st.button("Upload")
bt_2 = st.button("Start") 

st.write("---------------------------------------------")


st.sidebar.subheader("FORK Settings 🛠")
seq_length = st.sidebar.slider("Sequence Length",1,10,4)
fut_preds    = st.sidebar.slider("Number of Output",1,10,1) 



if bt_1:
    inputCsv = pd.read_csv(csv_file)
    database.csv_var = inputCsv



if bt_2:
     
    input_csv  = database.csv_var
    keys = input_csv.keys()

    st.subheader("Uploaded CSV")
    st.dataframe(input_csv)

    container = {}

    check_point = 0
    column_a = 1
    column_b = 2

    while check_point < len(keys)-1:

        out_putBox = []
        log = f"Processing feature: {keys[column_a]}"
        st.write(log)    
            
        data   = mm.readCSV(input_csv,column_a,column_b)
        dX,dY  = mm.sequencer(data,seq_length)
        model  = mm.modelTrainer(dX,dY,seq_length)
        output = mm.futurepred(fut_preds,model,data)

        index_out = 0

        while index_out<len(output):
            out_putBox.append(float(output[index_out]))
            index_out = index_out+1


        container[keys[column_a]]  = out_putBox
        check_point = check_point + 1
        column_a    = column_a + 1
        column_b    = column_b + 1

    database.keys = keys
    database.container = container
    database.stamp = fut_preds
    database.csv_var = input_csv     


st.sidebar.write("---------------------------------------------")

st.sidebar.subheader("Output Visualization 👓")
feature = st.sidebar.selectbox("",database.keys[1:])
bt_3 = st.sidebar.button("Plot it")

st.sidebar.write("---------------------------------------------")

st.sidebar.subheader("Support 💻")
bt_4 = st.sidebar.button("About FORK")
bt_5 = st.sidebar.button("HELP")



if bt_3:
    st.subheader("Output Visualization")
           
    present_values = database.csv_var[feature].values
    fut_values = database.container[feature]
    total_values = np.append(present_values,fut_values)

    fig = plt.figure(figsize=(10, 4))

    plt.axvline(x=len(database.csv_var[database.keys[0]])-1, c='r', linestyle='--') 
    plt.plot(total_values) 
           
    st.pyplot(fig)

    if present_values[-1] < fut_values[-1]:
        st.write(f"After {database.stamp} Timestamp value of {feature} should have increased spot.")

    elif present_values[-1] > fut_values[-1]:
        st.write(f"After {database.stamp} Timestamp value of {feature} should have decreased spot.")
   
    elif present_values[-1] == fut_values[-1]:
        st.write(f"After {database.stamp} Timestamp value of {feature} should have no change.")


if bt_4:
    st.subheader("ABOUT FORK 📕")
    st.write("Analytic FORK is a analytical too initialy designed to analyize energy and water consumption figure of a manufacturing industries and similar compaies at each stage of manufacturing and forcast the values for the same.")
    st.write("But FORCK have alot more potential so after some modification FORCK is ready to analyize and do forcasting on any enterd data (in the required format).")                

if bt_5:
    st.subheader("Getting Started With FORK 🔌")
    st.write("Analytic FORK is a analytical too initialy designed to analyize energy and water consumption figure of a manufacturing industries and similar compaies at each stage of manufacturing and forcast the values for the same.")
    st.write("But FORCK have alot more potential so after some modification FORCK is ready to analyize and do forcasting on any enterd data (in the required format).")                
