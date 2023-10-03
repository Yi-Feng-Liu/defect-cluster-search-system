import streamlit as st
import matplotlib.pyplot as plt
from modules.plot_TFTxCOC2 import main, main_forShipping
import streamlit.components.v1 as components
from modules.utils import plot_defect_info, get_df_from_period_of_time
import base64
import streamlit.components.v1 as components
from modules.utils import CreatePPT 


# def download_button(object_to_download):
#     """
#     Generates a link to download the given object_to_download.
#     Params:
#     ------
#     object_to_download:  The object to be downloaded.
#     download_filename (str): filename and extension of file. e.g. mydata.csv,
#     some_txt_output.txt download_link_text (str): Text to display for download
#     link.
#     button_text (str): Text to display on download button (e.g. 'click here to download file')
#     pickle_it (bool): If True, pickle file.
#     Returns:
#     -------
#     (str): the anchor tag to download object_to_download
#     Examples:
#     --------
#     download_link(your_df, 'YOUR_DF.csv', 'Click to download data!')
#     download_link(your_str, 'YOUR_STRING.txt', 'Click to download text!')
#     """
    
#     # some strings <-> bytes conversions necessary here
#     # try:
#     #     b64 = base64.b64encode(object_to_download.encode()).decode()
#     # except:
#     b64 = base64.b64encode(object_to_download).decode()


#     dl_link = f"""
#         <html>
#         <head>
#         <script src="http://code.jquery.com/jquery-3.2.1.min.js"></script>
#         <script>
#         $(function() {{
#         $('a[data-auto-download]').each(function(){{
#         var $this = $(this);
#         setTimeout(function() {{
#         window.location = $this.attr('href');
#         }}, 500);
#         }});
#         }});
#         </script>
#         </head>
#         <body>
#         <div class="wrapper">
#         <a data-auto-download href="data:application/vnd.openxmlformats-officedocument.presentationml.presentation;base64,{b64}"></a>
#         </div>
#         </body>
#         </html>"""

#     return dl_link



# def download_ppt():
#     ppt_type = CreatePPT().load_ppt()
#     components.html(
#         download_button(ppt_type.getvalue()),
#         height=0,
#     )
#     return ppt_type

# def check_and_download():
#     import os
#     if os.path.exists('./modules/temp/temp.pptx'):
#         download_ppt()
#         CreatePPT().delete_temp_file()    
      

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



st.sidebar.title(':blue[Configuration Settings] :sunglasses:')
st.sidebar.markdown('**:red[_173的資料量龐大, 建議以單片搜尋_]**')
st.sidebar.markdown('**:black[Tips:COC2 Defect Info已翻面並與TFT同向, 互動式介面則以原始數據呈現]**')
min_value = 4
mode = st.sidebar.radio(label="Mode", options=["單片", "多片"], key="Choose_type")
search_mode = st.sidebar.radio(label="Categories", options=["COC2", "TFT", "TFT+COC2"], key="Search_Mode")

# cluster defect threshold value
threshold = st.sidebar.number_input(label='Enter Cluster Threshold', min_value=min_value, step=1)
single_sheet_option_ls=[]


if mode == "單片":
    if search_mode=='TFT':
        Instype = st.sidebar.selectbox(label='Inspection Type', options=['L255', 'L10', 'L0'])
        st.session_state.coc2_form = False
        st.session_state.tft_coc2_form = False
        col1, col2 = st.columns(2)
        with col1:
            with st.form(key='tft_form1'):
                s_tft_sheet_id = st.text_input(label='Sheet_ID')
                tft_form = st.form_submit_button(label='Search')
                    
                if tft_form or st.session_state.tft_form:
                    if s_tft_sheet_id.strip() == "":
                        st.error('Enter Sheet ID')
                        
                    else:
                        with st.spinner('Executing...'):
                            st.session_state.tft_form = True
                            pdi = plot_defect_info(sheet_ID=s_tft_sheet_id.strip())
                            df = pdi.get_TFT_CreateTime_df()
                            # single_sheet_option_ls 該產品的各檢測時間序列
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
                full_df, defect_code_col, lumiance_col = pdi.get_TFT_full_RGB_Dataframe(df, Instype)
                
                TFT_interact_fig = pdi.interact_scatter(
                    df=full_df,
                    col_led_index_x='LED_Index_I',
                    col_led_index_y='LED_Index_J',
                    col_of_color='LED_TYPE',
                    symbol = defect_code_col,
                    hover_data=[lumiance_col]
                )
                
                st.plotly_chart(TFT_interact_fig, use_container_width=True)
        
                figs = main(
                    sheet_ID=s_tft_sheet_id.strip(), 
                    threshold=abs(threshold), 
                    option=search_mode, 
                    Ins_type=Instype, 
                    TFT_df=df
                )
                for fig in figs:
                    st.pyplot(fig=fig, dpi=500)
                
    
    if search_mode=='TFT+COC2':
        Instype = st.sidebar.selectbox(label='Inspection Type', options=['L255', 'L10', 'L0'])
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
                figs = main(
                    sheet_ID=s_tft_coc2_sheet_id.strip(), 
                    threshold=abs(threshold), 
                    option=search_mode,
                    Ins_type=Instype, 
                    TFT_df=df
                )
                for fig in figs:
                    st.pyplot(fig=fig, dpi=500)
                    
                    

            
            
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
                        pdi = plot_defect_info(sheet_ID=s_coc2_sheet_id.strip())
                        full_df = pdi.get_COC2_full_RGB_Dataframe()

                        coc2_interact_fig = pdi.interact_scatter(
                            df=full_df,
                            col_led_index_x='LED_Index_X',
                            col_led_index_y='LED_Index_Y',
                            col_of_color='LED_TYPE',
                            symbol = 'Defect_Reciepe',
                            hover_data=['Shift_X', 'Shif_Y' ,'Rotate', 'LINK']
                        )
    
                        st.plotly_chart(coc2_interact_fig, use_container_width=True)
                        for fig in (main(sheet_ID=s_coc2_sheet_id, threshold=abs(threshold), option=search_mode)):
                            st.pyplot(fig=fig, dpi=500)
                    
    
    
if mode == "多片":
    if search_mode=='TFT':
        
        # SheetID的輸入方式 手key或由日期區間判定
        date_toggle = st.sidebar.toggle('Date Mode')
        
        # for shipping 給 M01的特定圖片格式
        forShipping = st.sidebar.toggle(label="FOR SHIPPING")
        
        # 檢查條件
        Instype = st.sidebar.selectbox(label='Inspection Type', options=['L255', 'L10', 'L0'])
        
        # 站點選定
        select_OPID = st.sidebar.selectbox(label='OPID', options=light_on_ins_list, help='Only Show the Newest Data from Choose OPID')
        st.session_state.m_coc2_form = False
        st.session_state.m_tft_coc2_form = False
        
        with st.form(key='tft_form1'):
            from datetime import timedelta, datetime
            
            if date_toggle:
                start_col, end_col = st.columns(2)
                
                with start_col:
                    start_date = st.date_input('Start', value= datetime.now() -timedelta(days=1))
                    
                with end_col:
                    end_date = st.date_input('End')
                    
                m_tft_form = st.form_submit_button(label='Submit')
                start_date = start_date.strftime("%Y%m%d%H%M%S")
                end_date = end_date.strftime("%Y%m%d%H%M%S")
                
                if m_tft_form or st.session_state.m_tft_form:
                    df = get_df_from_period_of_time(start_date=start_date, end_date=end_date, InsType=Instype, OPID=select_OPID)
                    st.info(f'Totally got {len(df.SHEET_ID.unique())-1} products in period of time.')
                    progress_text = "Operation in progress. Please wait."
                    my_bar = st.progress(0, text=progress_text)
                    
                    # For shipping 需要製作出貨報告
                    if forShipping:
                        for i in range(len(df.SHEET_ID.unique())):
                            my_bar.progress((i+1)*(int(100/(len(df.SHEET_ID.unique())))), text=progress_text)
                            SHEET_ID = df.SHEET_ID.unique()[i]
                            fig = main_forShipping(df=df, SHEET_ID=SHEET_ID, Ins_type=Instype)
                            st.pyplot(fig=fig, dpi=500)
                            plt.close(fig)
                        my_bar.empty()
                        
                    else:
                        for i in range(len(df.SHEET_ID.unique())):
                            my_bar.progress((i+1)*(int(100/(len(df.SHEET_ID.unique())))), text=progress_text)
                            fig_ls = main(
                                sheet_ID=df.SHEET_ID.unique()[i], 
                                threshold=abs(threshold), 
                                option=search_mode, 
                                Ins_type=Instype, 
                                OPID=select_OPID
                            )
                            for fig in fig_ls:
                                st.pyplot(fig=fig, dpi=500)
                                plt.close(fig) 
                        my_bar.empty()
                        
                    st.info('Done')
                    
            else:
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
                        
                        # For shipping 需要製作出貨報告
                        if forShipping:
                            with st.spinner('Executing...'):
                                for tft in tft_list_sheet_id:
                                    try:
                                        pdi = plot_defect_info(tft)
                                        df = pdi.get_TFT_CreateTime_df()
                                        df = pdi.get_TFT_newest_df(df=df, InsType=Instype, OPID=select_OPID)
                                        fig = main_forShipping(df=df, SHEET_ID=tft, Ins_type=Instype)
                                        st.pyplot(fig=fig, dpi=500)
                                        plt.close(fig) 
                                    except:
                                        not_found_list.append(tft)
                                    
                        else:
                            with st.spinner('Executing...'):
                                for tft in tft_list_sheet_id:
                                    try:
                                        figs = main(
                                            sheet_ID=tft, 
                                            threshold=abs(threshold), 
                                            option=search_mode, 
                                            Ins_type=Instype, 
                                            OPID=select_OPID
                                        )
                                        for fig in figs:
                                            st.pyplot(fig=fig, dpi=500)
                                            plt.close(fig)
                                    except:
                                        not_found_list.append(tft)
                                    
                        if len(not_found_list)!=0: 
                            st.warning(f'{select_OPID} not found {not_found_list}')
                        st.info('Done')
                        
        
        if forShipping:
            from datetime import datetime
            ppt_type = CreatePPT().load_ppt()
            button = st.download_button(
                label='Download',
                data=ppt_type,
                file_name=f'{datetime.now().strftime("%Y%m%d%H%M%S")} Shipping Report.pptx',
                help='Note:若沒有執行先前的Submit, 將會下載空的PPT檔案'
            )
            CreatePPT().delete_temp_file() 
                 
                    
    if search_mode=='TFT+COC2':
        Instype = st.sidebar.selectbox(label='Inspection Type', options=['L255', 'L10', 'L0'])
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
                                figs = main(
                                    sheet_ID=tft, 
                                    threshold=abs(threshold), 
                                    option=search_mode, 
                                    Ins_type=Instype, 
                                    OPID=select_OPID
                                )
                                for fig in figs:
                                    st.pyplot(fig=fig, dpi=500)
                                    plt.close(fig)
                            except:
                                not_found_list.append(tft)
                    
                    if len(not_found_list) != 0:
                        st.warning(f'{select_OPID} not found {not_found_list}')
                    st.info('Done')
            
            
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
                                figs = main(sheet_ID=coc2, threshold=abs(threshold), option=search_mode)
                                for fig in figs:
                                    st.pyplot(fig=fig, dpi=500)
                                    plt.close(fig)
                            except:
                                not_found_list.append(coc2)
                                continue 
                    if len(not_found_list) != 0:        
                        st.warning(f'{not_found_list} not found')
                    st.info('Done')


