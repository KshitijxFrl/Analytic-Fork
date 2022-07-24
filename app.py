from venv import create
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import modelModule as mm

def main():
        
    st.title("Analytic FORK")

    tab1, tab2, tab3 = st.tabs(["Home", "Help", "About AF"])
    
    st.sidebar.subheader("Select A feature")
    plot_bt = st.sidebar.button("Plot It")

    # HOME TAB
    with tab1:

        st.subheader("Upload Csv")
        csv_file = st.file_uploader("",type = "csv")


        if csv_file != None:
            input_csv  = pd.read_csv(csv_file)
            keys = input_csv.keys()

            st.dataframe(input_csv)

            container = {}
            feature =  st.sidebar.selectbox("",keys[1:])

            check_point = 0
            column_a = 1
            column_b = 2

            while check_point < len(keys)-1:

                out_putBox = []    
            
                data   = mm.readCSV(input_csv,column_a,column_b)
                dX,dY  = mm.sequencer(data,4)
                model  = mm.modelTrainer(dX,dY,4)
                output = mm.futurepred(5,model,data)

                index_out = 0

                while index_out<len(output):
                    out_putBox.append(float(output[index_out]))
                    index_out = index_out+1

                container[keys[column_a]]  = out_putBox
                check_point = check_point + 1
                column_a    = column_a + 1
                column_b    = column_b + 1

            print(container)
        
        if plot_bt:
          
           st.subheader("Graphical Visualization")
           
           present_values = input_csv[feature].values
           fut_values = container[feature]
           total_values = np.append(present_values,fut_values)

           fig = plt.figure(figsize=(10, 4))

           plt.axvline(x=len(input_csv[keys[0]]), c='r', linestyle='--') 
           plt.plot(total_values) 
           
           st.pyplot(fig)
  
            
   
   #HELP TAB
    with tab2:
        st.write("Help")
   
   
   #ABOUT AF TAB
    with tab3:         
        st.write("About")


        
        
if __name__ == '__main__':
    main()

    

