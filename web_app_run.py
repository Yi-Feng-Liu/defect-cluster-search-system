import streamlit as st
import time
import matplotlib.pyplot as plt
import numpy as np
from modules.main_TFTxCOC2 import main
import mpld3
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Cluster Defect Search System",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# st.title('Cluster Defect Search System')

### Initialize values in Session State
if 'submit_form1' not in st.session_state:
    st.session_state.submit_form1 = False

with st.container():
    with st.form(key='form1'):
        min_value=3
        
        search_mode = st.radio(label="Mode", options=["COC2", "TFT", "TFT+COC2"], key="Search_Mode")

        sheet_id = st.text_area(label='SHEET_ID', height=120, key='SHEET_ID', placeholder='換行輸入, ex:\nSHEET_ID_1\nSHEET_ID_2\n...')
        
        list_sheet_id_tmp = sheet_id.split('\n')
        list_sheet_id = [value for value in list_sheet_id_tmp if value != '']

        threshold = st.number_input(label='Enter Cluster Threshold', min_value=min_value, step=1)

        submit_form1 = st.form_submit_button(label='確認搜尋條件')

    if submit_form1 or st.session_state.submit_form1:
            
        with st.spinner('Wait for it...'):
            time.sleep(1)
            
            for id in list_sheet_id:
                for fig in (main(sheet_ID=id, threshold=abs(threshold), option=search_mode)):
                    st.pyplot(fig=fig, dpi=500)
                    
                    # fig_html = mpld3.fig_to_html(fig)
                    # components.html(fig_html, height=700, scrolling=True)
