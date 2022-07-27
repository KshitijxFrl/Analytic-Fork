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
st.info('"If you are new to FORK try visiting support section."')

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
        st.write(f"After {database.stamp} Timestamp value of {feature} should have increased spot (around {fut_values[-1]}).")

    elif present_values[-1] > fut_values[-1]:
        st.write(f"After {database.stamp} Timestamp value of {feature} should have decreased spot around ({fut_values[-1]}).")
   
    elif present_values[-1] == fut_values[-1]:
        st.write(f"After {database.stamp} Timestamp value of {feature} should have no change.")


if bt_4:
    st.subheader("ABOUT FORK 📕")
    st.write("Analytic FORK is a analytical tool initially designed to analyse energy and water consumption figure of manufacturing industries and similar companies at each stage of manufacturing and forcast the values for the same.")
    st.write("But FORCK have alot more potential so after some modification FORCK is ready to analyize and do forcasting on any entered data (in the required format).")                
    st.write("At the present time FORK is using LSTM as its forcasting model.")
    st.write("To get the required format please click on format in support section.")


if bt_5:
    
    st.subheader("Getting Started With FORK 🔌")
    st.write("Step 1:- Upload CSV.")
    st.write("a) Select a csv file using Browes File.")
    st.write("b) Click on upload.") 
    st.image("./assets/step1.png")

    st.write("Step 2:- Adjest The FORK.")
    st.write("a) Look in FORK Settings and adjest the sequence which LSTM model will use to get train. Please use this feature if you are familear with LSTM.")                
    st.write("b) Look in FORK Settings and adjest the number future output you want. Recommended between 1 to 5")
    st.write("c) Click on start which initiate FORK (this process will take some time).")
    st.image("./assets/step2(a)(b).png")
    st.image("./assets/step2(c).png")

    st.write("Step 3:- Visualize output.")
    st.write("Go in output visualization section select the feature you want to visualize and click on plot it.")
    st.image("./assets/step3.png")

    st.write("Note:- Check This Sample CSV to understand format (All the entries in this sample are fictional and completely random)")
    st.image("./assets/sample_csv.png")
    
