import streamlit as st
import time
import matplotlib.pyplot as plt
import numpy as np
from modules.main_TFTxCOC2 import main
import mpld3
import streamlit.components.v1 as components
from modules.utils import plot_defect_info


st.set_page_config(
    page_title="Cluster Defect Search System",
    page_icon="?",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title('Cluster Defect Search System')
light_on_ins_list = ["MT-ACL", "MT+ACL", "MT+ACL2", "MD-ACL", "MT-ACL2", "MT-ACL3"]


### Initialize values in Session State
if 'tft_form' not in st.session_state:
    st.session_state.tft_form = False
    
if 'tft_coc2_form' not in st.session_state:
    st.session_state.tft_coc2_form = False
    
if 'coc2_form' not in st.session_state:
    st.session_state.coc2_form = False

if 'm_tft_form' not in st.session_state:
    st.session_state.m_tft_form = False

if 'm_tft_coc2_form' not in st.session_state:
    st.session_state.m_tft_coc2_form = False

if 'm_coc2_form' not in st.session_state:
    st.session_state.m_coc2_form = False

    
# if 'submit_form2' not in st.session_state:
#     st.session_state.submit_form2 = False



st.sidebar.title(':blue[Configuration Settings] :sunglasses:')
st.sidebar.markdown('**:red[_173的資料量龐大, 建議以單片搜尋_]**')
min_value = 4
mode = st.sidebar.radio(label="Mode", options=["單片", "多片"], key="Choose_type")
search_mode = st.sidebar.radio(label="Categories", options=["COC2", "TFT", "TFT+COC2"], key="Search_Mode")
threshold = st.sidebar.number_input(label='Enter Cluster Threshold', min_value=min_value, step=1)
single_sheet_option_ls=[]


if mode == "單片":
    if search_mode=='TFT':
        st.session_state.coc2_form = False
        st.session_state.tft_coc2_form = False
        col1, col2 = st.columns(2)
        with col1:
            with st.form(key='tft_form1'):
                s_tft_sheet_id = st.text_input(label='Sheet_ID')
                tft_form = st.form_submit_button(label='Search')
                    
                if tft_form or st.session_state.tft_form:
                    st.session_state.tft_form = True
                    if s_tft_sheet_id.strip() == "":
                        st.error('Enter Sheet ID')
                        
                    else:    
                        with st.spinner('Executing...'):
                            pdi = plot_defect_info(sheet_ID=s_tft_sheet_id.strip())
                            df = pdi.get_TFT_CreateTime_df()
                            single_sheet_option_ls = pdi.get_option_list(df)
                            st.success('Success! Please Choose Date!')
                            
        with col2:
            with st.form(key='tft_form2'):
                select_date = st.selectbox(label='Date', options=single_sheet_option_ls)
                submit_form2 = st.form_submit_button(label='Submit')
                if submit_form2:
                    st.success('Success')
    
                        
        if submit_form2:
            with st.spinner('Executing...'):
                df = df[df['CreateTime']==select_date.split(" ")[0]]
                for fig in (main(sheet_ID=s_tft_sheet_id.strip(), threshold=abs(threshold), option=search_mode, TFT_df=df)):
                    st.pyplot(fig=fig, dpi=500)
            
    
    if search_mode=='TFT+COC2':
        st.session_state.coc2_form = False
        st.session_state.tft_form = False
        col1, col2 = st.columns(2)
        with col1:
            with st.form(key='tft_coc2_form1'):
                s_tft_coc2_sheet_id = st.text_input(label='Sheet_ID')
                tft_coc2_form = st.form_submit_button(label='Search')
                
                if tft_coc2_form or st.session_state.tft_coc2_form:
                    if s_tft_coc2_sheet_id.strip() == "" :
                        st.error('Enter Sheet ID')
                    else:
                        st.session_state.tft_coc2_form = True
                        
                        with st.spinner('Executing...'):
                            pdi = plot_defect_info(sheet_ID=s_tft_coc2_sheet_id.strip())
                            df = pdi.get_TFT_CreateTime_df()
                            single_sheet_option_ls = pdi.get_option_list(df)
                            st.success('Success! Please Choose Date!')
                            
        with col2:
            with st.form(key='tft_coc2_form2'):
                select_date = st.selectbox(label='Date', options=single_sheet_option_ls)
                submit_form2 = st.form_submit_button(label='Submit')
                if submit_form2:
                    st.success('Success')
    
                        
        if submit_form2:
            with st.spinner('Executing...'):
                df = df[df['CreateTime']==select_date.split(" ")[0]]
                for fig in (main(sheet_ID=s_tft_coc2_sheet_id.strip(), threshold=abs(threshold), option=search_mode, TFT_df=df)):
                    st.pyplot(fig=fig, dpi=500)
                    
                    # fig_html = mpld3.fig_to_html(fig)
                    # components.html(fig_html, height=700, scrolling=True)

            
            
    elif search_mode == 'COC2':
        st.session_state.tft_form = False
        st.session_state.tft_coc2_form = False
        with st.form(key='coc2_form1'):
            s_coc2_sheet_id = st.text_input(label='Sheet_ID')
            coc2_form = st.form_submit_button(label='Submit')
            
            if coc2_form or st.session_state.coc2_form:
                if s_coc2_sheet_id.strip() == "":
                    st.error('Enter Sheet ID')
                    
                else:
                    with st.spinner('Executing...'):
                        for fig in (main(sheet_ID=s_coc2_sheet_id, threshold=abs(threshold), option=search_mode)):
                            st.pyplot(fig=fig, dpi=500)
                            
                            fig_html = mpld3.fig_to_html(fig)
                            components.html(fig_html, height=700, scrolling=True)
                    
    


if mode == "多片":
    if search_mode=='TFT':
        select_OPID = st.sidebar.selectbox(label='OPID', options=light_on_ins_list, help='Only Show the Newest Data from Choose OPID')
        st.session_state.m_coc2_form = False
        st.session_state.m_tft_coc2_form = False
        with st.form(key='tft_form1'):
            m_tft_sheet_id = st.text_area(label='SHEET_ID', height=120, key='SHEET_ID', placeholder='換行輸入, ex:\nSHEET_ID_1\nSHEET_ID_2\n...')
            tft_list_sheet_id_tmp = m_tft_sheet_id.split('\n')
            
            tft_list_sheet_id = [value for value in tft_list_sheet_id_tmp if value != '']
            m_tft_form = st.form_submit_button(label='Submit')
                
            if m_tft_form or st.session_state.m_tft_form:
                st.session_state.m_tft_form = False
                
                if len(tft_list_sheet_id)==0:
                    st.error('Enter Sheet ID')
                    
                else:
                    not_found_list = []
                    with st.spinner('Executing...'):
                        for tft in tft_list_sheet_id:
                            try:
                                for fig in (main(sheet_ID=tft, threshold=abs(threshold), option=search_mode, OPID=select_OPID)):
                                    st.pyplot(fig=fig, dpi=500)
                            except:
                                not_found_list.append(tft)
                                
                    if len(not_found_list)!=0: 
                        st.warning(f'{select_OPID} not found {not_found_list}')
                    
                    
    if search_mode=='TFT+COC2':
        select_OPID = st.sidebar.selectbox(label='OPID', options=light_on_ins_list, help='Only Show the Newest Data from Choose OPID')
        st.session_state.m_coc2_form = False
        st.session_state.m_tft_form = False
        with st.form(key='tft_coc2_form1'):
            m_tft_coc2_sheet_id = st.text_area(label='SHEET_ID', height=120, key='SHEET_ID', placeholder='換行輸入, ex:\nSHEET_ID_1\nSHEET_ID_2\n...')
            
            tft_list_sheet_id_tmp = m_tft_coc2_sheet_id.split('\n')
            
            tft_list_sheet_id = [value for value in tft_list_sheet_id_tmp if value != '']
            m_tft_coc2_form = st.form_submit_button(label='Submit')
            
            if m_tft_coc2_form or st.session_state.m_tft_coc2_form:
                if len(tft_list_sheet_id) == 0:
                    st.error('Enter Sheet ID')
                else:
                    with st.spinner('Executing...'):
                        not_found_list = []
                        for tft in tft_list_sheet_id:
                            try:
                                for fig in (main(sheet_ID=tft, threshold=abs(threshold), option=search_mode, OPID=select_OPID)):
                                    st.pyplot(fig=fig, dpi=500)
                            except:
                                not_found_list.append(tft)
                    
                    if len(not_found_list) != 0:
                        st.warning(f'{select_OPID} not found {not_found_list}')
                    # fig_html = mpld3.fig_to_html(fig)
                    # components.html(fig_html, height=700, scrolling=True)

            
            
    elif search_mode == 'COC2':
        st.session_state.m_tft_form = False
        st.session_state.m_tft_coc2_form = False
        with st.form(key='coc2_form1'):
            coc2_sheet_ids = st.text_area(label='SHEET_ID', height=120, key='SHEET_ID', placeholder='換行輸入, ex:\nSHEET_ID_1\nSHEET_ID_2\n...')
        
            coc2_list_sheet_id_tmp = coc2_sheet_ids.split('\n')
            coc2_list_sheet_id = [value for value in coc2_list_sheet_id_tmp if value != '']
            m_coc2_form = st.form_submit_button(label='Submit')
            
            if m_coc2_form or st.session_state.m_coc2_form:
                if len(coc2_list_sheet_id) == 0:
                    st.error('Enter Sheet ID')
                    
                else:
                    not_found_list = []
                    st.success('Success')
                    with st.spinner('Executing...'):
                        for coc2 in coc2_list_sheet_id:
                            try:
                                for fig in (main(sheet_ID=coc2, threshold=abs(threshold), option=search_mode)):
                                    st.pyplot(fig=fig, dpi=500)
                            except:
                                not_found_list.append(coc2)
                                continue 
                    if len(not_found_list) != 0:        
                        st.warning(f'{not_found_list} not found')



