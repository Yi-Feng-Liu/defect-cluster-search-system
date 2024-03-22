import streamlit as st
import matplotlib.pyplot as plt
from modules.plot_TFTxCOC2 import main, main_forShipping, plot_fullChip_coc2, Chargemap, combineTwoInstypeMartix
from modules.utils import plot_defect_info, get_df_from_period_of_time, grid_fs, calculateExecutedTime
from modules.utils import CreatePPT
from page.defectCompare import defectCompare, defect_code_dict
from page.atmb import ATMB
import pandas as pd
import re
from bson import ObjectId
import pickle
import plotly.graph_objects as go
from PIL import Image

st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(
    page_title="Cluster Defect Search System",
    page_icon="?",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title('Cluster Defect Search System')

DEFAULT_LIGHT_ON_OPID_LIST = ["MT-ACL", "MT+ACL", "MT+ACL2", "MD-ACL", "MT-ACL2", "MT-ACL3", "JI-DMU"]

st.markdown(
    """
    <style>
        .frank-text {
            color: blue;
            font-size: 24px;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True
)

def fine_tune_offset_value() -> int:
    value = st.sidebar.number_input(
        label = 'Enter COC2 Offet Value', 
        min_value = 0, 
        step = 1, 
        help = """**用來調整 COC2 Chip1 & chip2 的 layout **\n"""   
    )
    return value


def conv_config():
    conv_cluster = st.sidebar.toggle(label='Conv Cluster')
    ksize, conv_thresh = 4, 4
    
    if conv_cluster:
        ksize = st.sidebar.number_input(
            label = 'Enter Kernel Size', 
            min_value = ksize, 
            step = 1, 
        )
        
        conv_thresh = st.sidebar.number_input(
            label = 'Enter Block Threshold', 
            min_value = conv_thresh, 
            step = 1, 
        )
            
    return ksize, conv_thresh

            
def get_newest_tft_df(sheet_ID: str, InsType: str, OPID: str, ACTUAL_RECIPE: str) -> pd.DataFrame:
    pdi = plot_defect_info(sheet_ID=sheet_ID)
    df = pdi.get_TFT_CreateTime_df()
    df = pdi.get_TFT_newest_df(
        df = df, 
        InsType = InsType, 
        OPID = OPID,
        recipe = ACTUAL_RECIPE
    )
    return df


@calculateExecutedTime
def concat_dataframe_id(series: pd.Series):
    import pickle
    from bson import ObjectId
    fs = grid_fs(client='mongodb://wma:mamcb1@10.88.26.102:27017', db_name='MT', collection='LUM_SummaryTable') 
    
    df_ls = [pickle.loads(fs.get(ObjectId(x)).read()) for x in series]
    df = pd.concat(df_ls)
    leave_columns = ["LED_TYPE", "LED_Index_I", "LED_Index_J", "Lighting_check", "Defect_Code", "CHIP"]
    df = df[leave_columns]
    df = df[df["Lighting_check"] == '0'].reset_index(drop=True)
    df['defect_appear_count'] = 1
    df.drop(columns="Lighting_check", inplace=True)
    grouped_df = df.groupby(by=["LED_TYPE", "LED_Index_I", "LED_Index_J", "Defect_Code", "CHIP"]).aggregate("sum").reset_index()
    grouped_df = grouped_df.sort_values(by="defect_appear_count", ascending=False).reset_index(drop=True)
    return grouped_df


### Initialize values in Session State ###  
if 'tft_form' not in st.session_state:
    st.session_state.tft_form = False
    
if 'tft_coc2_form' not in st.session_state:
    st.session_state.tft_coc2_form = False
    
if 'coc2_form' not in st.session_state:
    st.session_state.coc2_form = False

if 'multiat_form_btn' not in st.session_state:
    st.session_state.multiat_form_btn = False
    
if 'at_form1_btn' not in st.session_state:
    st.session_state.at_form1_btn = False
    
if 'at_form2_btn' not in st.session_state:
    st.session_state.at_form2_btn = False

if 'at_form1_btn2' not in st.session_state:
    st.session_state.at_form1_btn2 = False
    
if 'at_form3_btn' not in st.session_state:
    st.session_state.at_form3_btn = False
    
if 'at_form4_btn' not in st.session_state:
    st.session_state.at_form4_btn = False    
    
if 'at_tft_form' not in st.session_state:
    st.session_state.at_tft_form = False

if 'at_tft_form_btn2' not in st.session_state:
    st.session_state.at_tft_form_btn2 = False    

if 'at_tft_form_btn3' not in st.session_state:
    st.session_state.at_tft_form_btn3 = False      

if 'atmb_form1_btn' not in st.session_state:
    st.session_state.atmb_form1_btn = False    
    
if 'atmb_form2_btn' not in st.session_state:
    st.session_state.atmb_form2_btn = False   

if 'atmb_form3_btn' not in st.session_state:
    st.session_state.atmb_form3_btn = False          

if 'atmb_form4_btn' not in st.session_state:
    st.session_state.atmb_form4_btn = False   

if 'm_tft_form' not in st.session_state:
    st.session_state.m_tft_form = False

if 'm_tft_coc2_form' not in st.session_state:
    st.session_state.m_tft_coc2_form = False

if 'm_coc2_form' not in st.session_state:
    st.session_state.m_coc2_form = False
    
if 'run' not in st.session_state:
    st.session_state.run = False
    
if 'obj' not in st.session_state:
    st.session_state.obj = object 

if 'atmb_obj' not in st.session_state:
    st.session_state.atmb_obj = object  

if 'df_at_defect' not in st.session_state:
    st.session_state.df_at_defect = pd.DataFrame()

if 'df_mb_defect' not in st.session_state:
    st.session_state.df_mb_defect = pd.DataFrame()
    
if 'match_df' not in st.session_state:
    st.session_state.match_df = pd.DataFrame()    
    
if 'chip_info_lst' not in st.session_state:
    st.session_state.chip_info_lst = []
### Initialize values in Session State ###


### Common Variable ###
INSTYPE_LABEL = 'Inspection Type'
INSTYPE_OPTIONS = ['L255', 'L10', 'L0']
recipe_lst, retest_and_time_lst, step_lst, ct_step_lst, rect_step_lst, createtime_lst, default_single_sheet_option_ls, AT_MULTI_CHIP_ID_LST, chip_info_lst = [], [], [], [], [], [], [], [], []
select_recipes = ['default recipe1','default recipe2']
MIN_VALUE = 2
recipe_yn = ""
ct_select_step = ""
ct_select_retest = ""
ct_select_time = ""
select_retest = ""
judgeTypeOne = 'Type 1 (Sub Pixel)'
judgeTypeTwo = 'Type 2 (Pixel)'
### Common Variable ###


### Sidebar Setting ###
st.sidebar.title(':blue[Configuration Settings] :sunglasses:')
st.sidebar.markdown('**:red[_173 & Z300 & Z123的資料量龐大, 建議以單片搜尋_]**')
st.sidebar.markdown('**:black[Tips:COC2 Defect Info已翻面並與TFT同向, 互動式介面則以原始數據呈現]**')

MODE = st.sidebar.radio(label="Mode", options=["單片", "多片", "Defect比對"], key="Choose_type")

if MODE != "Defect比對":
    mode_options = ["AT", "AT-compare(CT/Re-CT)","Find charge value by light-on xy" , "COC2", "TFT", "TFT+AT", "TFT+COC2", "TFT+COC2+AT"]
    THRESHOLD = st.sidebar.number_input(
        label='Enter Cluster Threshold', 
        min_value=MIN_VALUE, 
        step=1, 
        help="""
            **面對大尺寸的產品例如17吋or 18吋的產品**\n
            **Cluster Threshold值不建議設得太小**\n
            **Defect太多建議設定10~100左右的數值能夠加快系統運作**\n
            """   
    )
else:
    mode_options = ["TFT+COC2","AT+各站點"]
    
SEARCH_MODE = st.sidebar.radio(label="Categories", options=mode_options, key="Search_Mode")

########################## Trigger Condiction ####################################
# S 代表單片的變數 ex: S_TFT_SHEET_ID
# M 代表多片的變數 ex: M_TFT_SHEET_ID
def getJudgeAnwser(res, types):        
    if res == 1:
        st.info(f'{types} Judge Result NG')
    else:
        st.info(f'{types} Judge Result OK')  
        
        
if MODE == "單片":
    if SEARCH_MODE == 'AT':
        st.session_state.tft_form = False
        st.session_state.tft_coc2_form = False
        st.session_state.coc2_form = False
        st.session_state.at_tft_form = False

        col1, col2, col3 = st.columns(3)
        
        with col1:
            
            with st.form(key='at_form1'):
                
                AT_CHIP_ID = st.text_input(label='Enter the chip id').strip()
                at_form1_btn = st.form_submit_button(label='Submit')
                
                if at_form1_btn or st.session_state.at_form1_btn:
                    if AT_CHIP_ID == "":
                        st.error('Enter chip ID')
                    else:
                        with st.spinner('Executing...'):
                            st.session_state.at_form1_btn = True
                            st.session_state.obj = Chargemap(chip_id=AT_CHIP_ID)
                            recipe_lst = st.session_state.obj.get_recipes()
                            st.success('Success!')
                            
        with col2:
            
            with st.form(key='at_form2'): 
                
                select_recipe = st.selectbox(
                    label = 'Select recipe', 
                    options = recipe_lst
                )
                at_form2_btn = st.form_submit_button(label='Submit')
                
                if at_form2_btn or st.session_state.at_form2_btn:
                    with st.spinner('Executing...'):
                        st.session_state.at_form2_btn = True
                        retest_and_time_lst = st.session_state.obj.get_retest_and_time(select_recipe)
                        st.success('Success!')
                        
        with col3:
            
            with st.form(key='at_form3'): 
                
                select_retest = st.selectbox(
                    label = 'Select retest', 
                    options = retest_and_time_lst
                )
                at_form3_btn = st.form_submit_button(label='Submit')
                
                if at_form3_btn or st.session_state.at_form3_btn:
                    select_time = re.sub(r'\D', '', select_retest.split("-")[1][1:])
                    select_retest = select_retest.split("-")[0][:-1] 
                    step_lst = st.session_state.obj.get_steps(
                        select_retest, 
                        select_recipe,
                        select_time
                    )
                    st.success('Success!')                
    
        if at_form3_btn:
            
            with st.spinner('Executing...'):

                for step in step_lst:
                    
                    try:

                        chargemap_fig = st.session_state.obj.plot_chargemap_img(
                            step, 
                            select_retest, 
                            select_recipe,
                            select_time
                        )
                        if isinstance(chargemap_fig,plt.Figure):
                            st.pyplot(chargemap_fig, dpi=500)
                        
                        defect_fig = st.session_state.obj.plot_defectmap_img(
                            step, 
                            select_retest, 
                            select_time, 
                            select_recipe
                        )
                        if isinstance(defect_fig,plt.Figure):
                            st.pyplot(defect_fig, dpi=500)
                            
                    except:
                        
                        continue
                
                try:
                    
                    defect_count_fig = st.session_state.obj.get_total_defect_count(
                        select_retest, 
                        select_time, 
                        select_recipe
                    )
                    if isinstance(defect_count_fig,plt.Figure):
                        st.pyplot(defect_count_fig, dpi=500)
                        
                except:
                    st.write('<div class="frank-text"> Zero Defect </div>', unsafe_allow_html=True)

                df_lst = st.session_state.obj.gen_defect_summary(
                    [str(select_retest)],
                    [select_time],
                    [select_recipe],
                    select_recipe[:4]
                )
                
                st.markdown('# 全座標點位，包含 Step, Defect code, LED_TYPE')
                st.markdown(f'## {select_recipe}')
                st.dataframe(df_lst[0], use_container_width=True)
                st.markdown('# Defect count，包含 Step, Defect code, LED_TYPE')   
                st.markdown(f'## {select_recipe}')   
                st.dataframe(df_lst[1], use_container_width=True)                    
                
                st.info("Done")


    if SEARCH_MODE == 'AT-compare(CT/Re-CT)':
        st.session_state.tft_form = False
        st.session_state.tft_coc2_form = False
        st.session_state.coc2_form = False
        st.session_state.at_tft_form = False

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            with st.form(key='at_form1'):
                
                AT_CHIP_ID = st.text_input(label='Enter the chip id').strip()
                at_form1_btn = st.form_submit_button(label='Submit')
                
                if at_form1_btn or st.session_state.at_form1_btn:
                    if AT_CHIP_ID == "":
                        st.error('Enter chip ID')
                    else:
                        with st.spinner('Executing...'):
                            st.session_state.at_form1_btn = True
                            st.session_state.obj = Chargemap(chip_id=AT_CHIP_ID)
                            recipe_lst = st.session_state.obj.get_recipes()
                            st.success('Success!')
                            
                            recipe_yn = st.radio(
                                "Same recipe?",
                                ["same", "different"]
                            )

                            at_form1_btn2 = st.form_submit_button(label='Submit(2)')
                            
                            if at_form1_btn2 or st.session_state.at_form1_btn2:
                                with st.spinner('Executing...'):
                                    st.session_state.at_form2_btn2 = True
                                    st.success('Success!')                               
                            
        if len(recipe_lst) == 1:
            st.info("只有1個CT")
        else:
            with col2:
                with st.form(key='at_form2'): 
                    
                    if recipe_yn == "same":
                        select_recipe = st.selectbox(
                            label = 'Select recipe', 
                            options = recipe_lst
                        )
                        
                        select_recipes = [select_recipe]*2
                        
                        at_form2_btn = st.form_submit_button(label='Submit')
                        if at_form2_btn or st.session_state.at_form2_btn:
                            with st.spinner('Executing...'):
                                st.session_state.at_form2_btn = True
                                st.success('Success!')                        
                    else:
                        select_recipes = st.multiselect(
                            label = 'Select recipe', 
                            options = recipe_lst
                        )
                        
                        at_form2_btn = st.form_submit_button(label='Submit')
                        if at_form2_btn or st.session_state.at_form2_btn:
                            with st.spinner('Executing...'):
                                st.session_state.at_form2_btn = True
                                st.success('Success!')
                
            with col3:
                with st.form(key='at_form3'): 
                    
                    ct_retest_and_time_lst = st.session_state.obj.get_retest_and_time(select_recipes[0])
                    rect_retest_and_time_lst = st.session_state.obj.get_retest_and_time(select_recipes[1])
                                                                
                    ct_select_retest = st.selectbox(
                        label = f'Select 『{select_recipes[0]}』 retest',
                        options = ct_retest_and_time_lst
                    )
                    rect_select_retest = st.selectbox(
                        label = f'Select 『{select_recipes[1]}』 retest(2)',
                        options = rect_retest_and_time_lst
                    )

                    at_form3_btn = st.form_submit_button(label='Submit')
                    if at_form3_btn or st.session_state.at_form3_btn:
                        with st.spinner('Executing...'):
                            
                            st.session_state.at_form3_btn = True
                            ct_select_time = re.sub(r'\D', '', ct_select_retest.split("-")[1][1:])
                            ct_select_retest = ct_select_retest.split("-")[0][:-1]
                            ct_step_lst = st.session_state.obj.get_steps(
                                ct_select_retest, 
                                select_recipes[0],
                                ct_select_time
                            )
                                                                
                            rect_select_time = re.sub(r'\D', '', rect_select_retest.split("-")[1][1:])
                            rect_select_retest = rect_select_retest.split("-")[0][:-1]
                            rect_step_lst = st.session_state.obj.get_steps(
                                rect_select_retest, 
                                select_recipes[1],
                                rect_select_time
                            )
                            
                            st.success('Success!')               

            with col4:
                with st.form(key='at_form4'): 
                                            
                    ct_select_step = st.selectbox(
                        label = f'Select 『{select_recipes[0]}』 step',
                        options = ct_step_lst
                    )
                    
                    rect_select_step = st.selectbox(
                        label = f'Select 『{select_recipes[1]}』 step(2)',
                        options = rect_step_lst
                    )     
                    
                    at_form4_btn = st.form_submit_button(label='Submit')
                    
                    if at_form4_btn or st.session_state.at_form4_btn:
                        with st.spinner('Executing...'):
                            st.session_state.at_form4_btn = True                     
                            st.success('Success!') 
                            
            if at_form4_btn:
                with st.spinner('Executing...'):
                    st.session_state.at_form4_btn = True

                    # CT
                    ct_chargemap_fig = st.session_state.obj.plot_chargemap_img(
                        ct_select_step, 
                        ct_select_retest, 
                        select_recipes[0],
                        ct_select_time
                    )
                    if isinstance(ct_chargemap_fig,plt.Figure):
                        st.pyplot(ct_chargemap_fig, dpi=500)

                    # Re-CT
                    rect_chargemap_fig = st.session_state.obj.plot_chargemap_img(
                        rect_select_step, 
                        rect_select_retest, 
                        select_recipes[1],
                        rect_select_time
                    )
                    if isinstance(rect_chargemap_fig,plt.Figure):
                        st.pyplot(rect_chargemap_fig, dpi=500)   
                    
                    
                    
                    # CT 減 Re-CT
                    both_chargemap_fig = st.session_state.obj.plot_chargemap_img(
                        "","","","",
                        [ct_select_step,rect_select_step],
                        [ct_select_retest,rect_select_retest],
                        [select_recipes[0],select_recipes[1]],
                        [ct_select_time,rect_select_time]
                    )
                    if isinstance(both_chargemap_fig,plt.Figure):
                        st.pyplot(both_chargemap_fig, dpi=500)   
                    
                    df_lst = st.session_state.obj.gen_defect_summary(
                        [ct_select_retest, rect_select_retest],
                        [ct_select_time, rect_select_time],
                        [select_recipes[0], select_recipes[1]],
                        select_recipes[0][:4]
                    )
                    
                    st.markdown('# 全座標點位，包含 Step, Defect code, LED_TYPE')
                    st.markdown(f'## 只有 {select_recipes[0]} 有')
                    st.dataframe(df_lst[0], use_container_width=True)
                    st.markdown(f'## 只有 {select_recipes[1]} 有')
                    st.dataframe(df_lst[1], use_container_width=True)
                    st.markdown(f'## {select_recipes[0]} 和 {select_recipes[1]} 交集')
                    st.dataframe(df_lst[2], use_container_width=True)
                    st.markdown('# Defect count，包含 Step, Defect code, LED_TYPE')   
                    st.markdown(f'## 只有 {select_recipes[0]} 有')   
                    st.dataframe(df_lst[3], use_container_width=True)   
                    st.markdown(f'## 只有 {select_recipes[1]} 有')
                    st.dataframe(df_lst[4], use_container_width=True)        
                    st.markdown(f'## {select_recipes[0]} 和 {select_recipes[1]} 交集')   
                    st.dataframe(df_lst[5], use_container_width=True)  
                    st.info("Done")                       
                    

    if SEARCH_MODE == 'Find charge value by light-on xy':
        st.session_state.tft_form = False
        st.session_state.tft_coc2_form = False
        st.session_state.coc2_form = False
        st.session_state.at_tft_form = False

        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            
            with st.form(key='at_form1'):
                
                AT_CHIP_ID = st.text_input(label='Enter the chip id').strip()
                at_form1_btn = st.form_submit_button(label='Submit')
                
                if at_form1_btn or st.session_state.at_form1_btn:
                    if AT_CHIP_ID == "":
                        st.error('Enter chip ID')
                    else:
                        with st.spinner('Executing...'):
                            st.session_state.at_form1_btn = True
                            st.session_state.obj = Chargemap(chip_id=AT_CHIP_ID)
                            recipe_lst = st.session_state.obj.get_recipes()
                            st.success('Success!')
                            
        with col2:
            
            with st.form(key='at_form2'): 
                
                select_recipe = st.selectbox(
                    label = 'Select recipe', 
                    options = recipe_lst
                )
                at_form2_btn = st.form_submit_button(label='Submit')
                
                if at_form2_btn or st.session_state.at_form2_btn:
                    with st.spinner('Executing...'):
                        st.session_state.at_form2_btn = True
                        retest_and_time_lst = st.session_state.obj.get_retest_and_time(select_recipe)
                        st.success('Success!')
                        
        with col3:
            
            with st.form(key='at_form3'): 
                
                select_retest = st.selectbox(
                    label = 'Select retest', 
                    options = retest_and_time_lst
                )
                at_form3_btn = st.form_submit_button(label='Submit')
                
                if at_form3_btn or st.session_state.at_form3_btn:
                    st.session_state.at_form3_btn = True
                    select_time = re.sub(r'\D', '', select_retest.split("-")[1][1:])
                    select_retest = select_retest.split("-")[0][:-1] 
                    step_lst = st.session_state.obj.get_steps(
                        select_retest, 
                        select_recipe,
                        select_time
                    )
                    st.success('Success!')                

        with col4:
            
            with st.form(key='at_form4'):
                
                light_on_x = st.text_input(label='Enter light-on Y')
                light_on_y = st.text_input(label='Enter light-on X')
                at_form4_btn = st.form_submit_button(label='Submit')            
                
        if at_form4_btn or st.session_state.at_form4_btn:
            
            with st.spinner('Executing...'):
                
                st.markdown('## light-on 點位: (x,y): (' + str(light_on_x) + "," + str(light_on_y) + ")")
                
                df = st.session_state.obj.df_charge2d
                df['lm_time'] = pd.to_datetime(df['lm_time'], format="%Y/%m/%d %H:%M:%S.%f")   
                df = df[(df['recipe_id'] == select_recipe) & (df['ins_cnt'] == select_retest) & (df['lm_time'].dt.date == pd.to_datetime(select_time).date())]
                
                product = df["recipe_id"].values[0][:4]
                
                at_x, at_y = st.session_state.obj.mb2at(product,int(light_on_x),int(light_on_y))
                st.markdown('## AT 點位: (x,y): (' + str(at_x) + "," + str(at_y) + ")")                
                
                df = df[["step","2d_r_object_id","2d_g_object_id","2d_b_object_id"]]
                df[["2d_r_object_id","2d_g_object_id","2d_b_object_id"]] = df[["2d_r_object_id","2d_g_object_id","2d_b_object_id"]].map(lambda x: st.session_state.obj.get_charge_value(x,at_x,at_y))
                
                df.columns = ["step","R charge value","G charge value","B charge value"]
                df = df.reset_index(drop=True)

                st.dataframe(df, use_container_width=True)                  
                
                
    if SEARCH_MODE == 'TFT+COC2+AT':
        INSTYPE = st.sidebar.selectbox(
            label = INSTYPE_LABEL, 
            options = INSTYPE_OPTIONS
        )
        
        st.session_state.tft_form = False
        st.session_state.tft_coc2_form = False
        st.session_state.coc2_form = False
        st.session_state.at_form1_btn = False

            
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            with st.form(key='at_tft_coc2_form1'):
                AT_TFT_COC2_CHIP_ID = st.text_input(label='Enter the chip id').strip()
                at_tft_form = st.form_submit_button(label='Submit')
                
                if at_tft_form or st.session_state.at_tft_form:
                    if AT_TFT_COC2_CHIP_ID == "":
                        st.error('Enter Sheet ID')
                            
                    else:
                        with st.spinner('Executing...'):
                            st.session_state.at_tft_form = True
                            ### AT function 放這邊 ###
                            st.session_state.obj = Chargemap(chip_id=AT_TFT_COC2_CHIP_ID)
                            recipe_lst = st.session_state.obj.get_recipes()
                            ### TFT ###
                            pdi = plot_defect_info(sheet_ID=AT_TFT_COC2_CHIP_ID)
                            df = pdi.get_TFT_CreateTime_df()
                            # default_single_sheet_option_ls 該產品的各檢測時間序列
                            default_single_sheet_option_ls = pdi.get_option_list(df)
                            st.success('Success! Please select Step')
        
        with col2:
            with st.form(key='at_tft_form2'): 
                select_recipe = st.selectbox(
                    label = 'Select recipe', 
                    options = recipe_lst
                ) 
                at_tft_form2 = st.form_submit_button(label='Submit')
                
                if at_tft_form2 or st.session_state.at_tft_form_btn2:
                    st.session_state.at_tft_form_btn2 = True
                    retest_and_time_lst = st.session_state.obj.get_retest_and_time(select_recipe)
                    select_retest = str(max(retest_and_time_lst))
                    select_time = re.sub(r'\D', '', select_retest.split("-")[1][1:])
                    select_retest = select_retest.split("-")[0][:-1]                      
                    step_lst = st.session_state.obj.get_steps(
                        select_retest, 
                        select_recipe,
                        select_time
                    )
                    st.success('Success!' )

        with col3:
            with st.form(key='at_tft_form3'): 
                select_step = st.selectbox(
                    label = 'Select Step', 
                    options = step_lst
                )

                at_tft_form_btn3 = st.form_submit_button(label='Submit')
                if at_tft_form_btn3 or st.session_state.at_tft_form_btn3:               
                    st.success('Success!')   
                            
        with col4:
            with st.form(key='at_tft_form4'):  
                select_date = st.selectbox(
                    label = 'Date', 
                    options = default_single_sheet_option_ls
                ) 
                at_tft_form3 = st.form_submit_button(label='Submit')
                if at_tft_form3:
                    st.success('Success')
                    
        if at_tft_form3:
            with st.spinner('Executing...'):
                chargemap_fig = fig=st.session_state.obj.plot_chargemap_img(
                    select_step, 
                    select_retest, 
                    select_recipe,
                    select_time
                )
                if isinstance(chargemap_fig,plt.Figure):
                    st.pyplot(chargemap_fig, dpi=500)
                
                defect_fig = st.session_state.obj.plot_defectmap_img(
                    select_step, 
                    select_retest, 
                    select_time, 
                    select_recipe
                )
                if isinstance(defect_fig, plt.Figure):
                    st.pyplot(defect_fig, dpi=500)
                
                df = df[df['CreateTime']==select_date.split(" ")[0]]
                
                figs, _, _ = main(
                    sheet_ID = AT_TFT_COC2_CHIP_ID, 
                    threshold = THRESHOLD, 
                    option = SEARCH_MODE, 
                    Ins_type = INSTYPE, 
                    TFT_df = df
                )
                for fig in figs:
                    st.pyplot(fig=fig, dpi=500)
                st.info("Done")
    
    
    if SEARCH_MODE == 'TFT+AT':
        INSTYPE = st.sidebar.selectbox(
            label = INSTYPE_LABEL, 
            options = INSTYPE_OPTIONS
        )
        ksize, conv_thresh = conv_config()
        st.session_state.tft_form = False
        st.session_state.tft_coc2_form = False
        st.session_state.coc2_form = False
        st.session_state.at_form1_btn = False

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            with st.form(key='at_tft_form1'):
                AT_TFT_CHIP_ID = st.text_input(label='Enter the chip id').strip()
                at_tft_form = st.form_submit_button(label='Submit')
                
                if at_tft_form or st.session_state.at_tft_form:
                    if AT_TFT_CHIP_ID == "":
                        st.error('Enter Sheet ID')
                            
                    else:
                        with st.spinner('Executing...'):
                            st.session_state.at_tft_form = True
                            ### AT function 放這邊 ###
                            st.session_state.obj = Chargemap(chip_id=AT_TFT_CHIP_ID)
                            recipe_lst = st.session_state.obj.get_recipes()
                            ### TFT ###
                            pdi = plot_defect_info(sheet_ID=AT_TFT_CHIP_ID)
                            df = pdi.get_TFT_CreateTime_df()
                            # default_single_sheet_option_ls 該產品的各檢測時間序列
                            default_single_sheet_option_ls = pdi.get_option_list(df)
                            st.success('Success! Please select Step')

        with col2:
            with st.form(key='at_tft_form2'): 
                select_recipe = st.selectbox(
                    label = 'Select recipe', 
                    options = recipe_lst
                ) 
                at_tft_form_btn2 = st.form_submit_button(label='Submit')
                
                if at_tft_form_btn2 or st.session_state.at_tft_form_btn2:
                    st.session_state.at_tft_form_btn2 = True
                    retest_and_time_lst = st.session_state.obj.get_retest_and_time(select_recipe)
                    select_retest = str(max(retest_and_time_lst))
                    select_time = re.sub(r'\D', '', select_retest.split("-")[1][1:])
                    select_retest = select_retest.split("-")[0][:-1]                      
                    step_lst = st.session_state.obj.get_steps(
                        select_retest, 
                        select_recipe,
                        select_time
                    )                                  
                    st.success('Success!' )
                                                
        with col3:
            with st.form(key='at_tft_form3'): 
                select_step = st.selectbox(
                    label = 'Select Step', 
                    options = step_lst
                )

                at_tft_form_btn3 = st.form_submit_button(label='Submit')
                if at_tft_form_btn3 or st.session_state.at_tft_form_btn3:                  
                    st.success('Success!')      
        
        with col4:
            with st.form(key='at_tft_form4'):  
                select_date = st.selectbox(
                    label = 'Date', 
                    options = default_single_sheet_option_ls
                ) 
                at_tft_form3 = st.form_submit_button(label='Submit')
                if at_tft_form3:
                    st.success('Success')
                    
        if at_tft_form3:
            with st.spinner('Executing...'):
                chargemap_fig = fig=st.session_state.obj.plot_chargemap_img(
                    select_step, 
                    select_retest, 
                    select_recipe,
                    select_time
                )
                if isinstance(chargemap_fig,plt.Figure):
                    st.pyplot(chargemap_fig, dpi=500)
                
                defect_fig = st.session_state.obj.plot_defectmap_img(
                    select_step, 
                    select_retest, 
                    select_time, 
                    select_recipe
                )
                if isinstance(defect_fig, plt.Figure):
                    st.pyplot(defect_fig, dpi=500)
                
                df = df[df['CreateTime']==select_date.split(" ")[0]]
                figs, _, _ = main(
                    sheet_ID = AT_TFT_CHIP_ID, 
                    threshold = THRESHOLD, 
                    option = SEARCH_MODE, 
                    Ins_type = INSTYPE, 
                    TFT_df = df,
                    kernelSize = ksize, 
                    conv_threshold = conv_thresh
                )
                for fig in figs:
                    st.pyplot(fig=fig, dpi=500)
                st.info("Done")
                        
                        
    if SEARCH_MODE == 'TFT':
        INSTYPE_OPTIONS = ['L255', 'L10', 'L0', 'L255 + L0']
        INSTYPE = st.sidebar.selectbox(
            label = INSTYPE_LABEL, 
            options = INSTYPE_OPTIONS
        )
        
        ksize, conv_thresh = conv_config()
        st.session_state.coc2_form = False
        st.session_state.tft_coc2_form = False
        st.session_state.at_form1_btn = False
        
        col1, col2 = st.columns(2)
        with col1:
            with st.form(key='tft_form1'):
                S_TFT_SHEET_ID = st.text_input(label='Sheet_ID').strip()
                tft_form = st.form_submit_button(label='Search')
                    
                if tft_form or st.session_state.tft_form:
                    if S_TFT_SHEET_ID == "":
                        st.error('Enter Sheet ID')
                        
                    else:
                        with st.spinner('Executing...'):
                            st.session_state.tft_form = True
                            pdi = plot_defect_info(sheet_ID=S_TFT_SHEET_ID)
                            df = pdi.get_TFT_CreateTime_df()
                            # default_single_sheet_option_ls 該產品的各檢測時間序列
                            default_single_sheet_option_ls = pdi.get_option_list(df)
                            st.success('Success! Please Choose Date!')
                            
        with col2:
            with st.form(key='tft_form2'):
                select_date = st.selectbox(
                    label = 'Date', 
                    options = default_single_sheet_option_ls
                )
                
                submit_form2 = st.form_submit_button(label='Submit')
                if submit_form2:
                    st.success('Success')
        
                    
        if INSTYPE != INSTYPE_OPTIONS[3]:
            if submit_form2:
                with st.spinner('Executing...'):
                    df = df[df['CreateTime']==select_date.split(" ")[0]]
                    full_df, defect_code_col_name, lumiance_col_name = pdi.get_TFT_full_RGB_Dataframe(df, INSTYPE)
                    lum_dataframe = full_df[['LED_TYPE', 'LED_Luminance']]
                    st.info('Luminance Info')
                    color_ls = ['R', 'G', 'B']
                    dfls = []
                    
                    for i in color_ls:
                        temp_df = pd.DataFrame()
                        color_df = lum_dataframe[lum_dataframe['LED_TYPE']==i]
                        max_lum = color_df['LED_Luminance'].astype(float).max()
                        min_lum = color_df['LED_Luminance'].astype(float).min()
                        gap = float(max_lum) - float(min_lum)
                        temp_df['LED_TYPE'] = i, 
                        temp_df['Max'] = round(float(max_lum), 2)
                        temp_df['min'] = round(float(min_lum), 2)
                        temp_df['gap'] = round(gap, 2)
                        dfls.append(temp_df)
                    # st.dataframe(pd.concat(dfls), use_container_width=True)
                    
                    TFT_interact_fig = pdi.interact_scatter(
                        df = full_df,
                        col_led_index_x = 'LED_Index_I',
                        col_led_index_y = 'LED_Index_J',
                        col_of_color = 'LED_TYPE',
                        symbol = defect_code_col_name,
                        hover_data = [lumiance_col_name]
                    )
                    
                    st.plotly_chart(TFT_interact_fig, use_container_width=True)
                    
                    # res 是經過Conv判片過後的結果 其值為 0 or 1 
                    # 1 表示為 NG
                    figs, resMain, resSub = main(
                        sheet_ID = S_TFT_SHEET_ID, 
                        threshold = THRESHOLD, 
                        option = SEARCH_MODE, 
                        Ins_type = INSTYPE, 
                        TFT_df = df,
                        kernelSize = ksize,
                        conv_threshold = conv_thresh
                    )
                    
                
                    getJudgeAnwser(resMain, types=judgeTypeOne)
                    getJudgeAnwser(resSub, types=judgeTypeTwo)
                
                    for fig in figs:
                        st.pyplot(fig=fig, dpi=500)    
                
        else:
            if submit_form2:
                figs, resMain, resSub = combineTwoInstypeMartix(
                    sheet_ID = S_TFT_SHEET_ID,
                    df = df[df['CreateTime']==select_date.split(" ")[0]],
                    kernelSize = ksize,
                    conv_threshold = conv_thresh,
                    Ins_type = INSTYPE
                )        
                
                getJudgeAnwser(resMain, types=judgeTypeOne)
                getJudgeAnwser(resSub, types=judgeTypeTwo)
                
                for fig in figs:
                    st.pyplot(fig=fig, dpi=500)
    
    
    if SEARCH_MODE == 'TFT+COC2':
        INSTYPE = st.sidebar.selectbox(
            label = INSTYPE_LABEL, 
            options = INSTYPE_OPTIONS
        )
        
        st.session_state.coc2_form = False
        st.session_state.tft_form = False
        st.session_state.at_form1_btn = False
        
        col1, col2 = st.columns(2)
        with col1:
            with st.form(key='tft_coc2_form1'):
                S_TFT_COC2_SHEET_ID = st.text_input(label='Sheet_ID').strip()
                tft_coc2_form = st.form_submit_button(label='Search')
                
                if tft_coc2_form or st.session_state.tft_coc2_form:
                    if S_TFT_COC2_SHEET_ID == "" :
                        st.error('Enter Sheet ID')
                        
                    else:
                        st.session_state.tft_coc2_form = True
                        with st.spinner('Executing...'):
                            pdi = plot_defect_info(sheet_ID=S_TFT_COC2_SHEET_ID)
                            df = pdi.get_TFT_CreateTime_df()
                            default_single_sheet_option_ls = pdi.get_option_list(df)
                            st.success('Success! Please Choose Date!')
                            
        with col2:
            with st.form(key='tft_coc2_form2'):
                select_date = st.selectbox(
                    label = 'Date', 
                    options = default_single_sheet_option_ls
                )
                submit_form2 = st.form_submit_button(label = 'Submit')
                if submit_form2:
                    st.success('Success')
    
                        
        if submit_form2:
            with st.spinner('Executing...'):
                df = df[df['CreateTime']==select_date.split(" ")[0]]
                figs, _, _  = main(
                    sheet_ID = S_TFT_COC2_SHEET_ID, 
                    threshold = THRESHOLD, 
                    option = SEARCH_MODE,
                    Ins_type = INSTYPE, 
                    TFT_df = df
                )
                for fig in figs:
                    st.pyplot(fig=fig, dpi=500)
                    
                    

    elif SEARCH_MODE == 'COC2':
        st.session_state.tft_form = False
        st.session_state.tft_coc2_form = False
        st.session_state.at_form1_btn = False
        
        with st.form(key='coc2_form1'):
            S_COC2_SHEET_ID = st.text_input(label='Sheet_ID').strip()
            coc2_form = st.form_submit_button(label='Submit')
            LAYOUT_OFFSETVALUE = fine_tune_offset_value()
            
            if coc2_form or st.session_state.coc2_form:
                if S_COC2_SHEET_ID == "":
                    st.error('Enter Sheet ID')
                    
                else:
                    with st.spinner('Executing...'):
                        pdi = plot_defect_info(sheet_ID=S_COC2_SHEET_ID)
                        full_df, _ = pdi.get_COC2_full_RGB_Dataframe()
                        haveChip = pdi.seriesHaveChip(full_df['Chip'])
                        
                        if haveChip:
                            chip_items = full_df['Chip'].sort_values(ascending=True).unique()
                            full_figs = plot_fullChip_coc2(S_COC2_SHEET_ID, THRESHOLD, onlyCOC2=True, layoutOffet=LAYOUT_OFFSETVALUE)
                            
                            for fullfig in full_figs:
                                st.pyplot(fig=fullfig, dpi=500)
                                    
                            for chip in chip_items:
                                st.info(chip)
                                chip_df = full_df[full_df['Chip']==chip]
                                
                                coc2_interact_fig = pdi.interact_scatter(
                                    df = chip_df,
                                    col_led_index_x = 'LED_Index_X',
                                    col_led_index_y = 'LED_Index_Y',
                                    col_of_color = 'LED_TYPE',
                                    symbol = 'Defect_Reciepe',
                                    hover_data = ['Shift_X', 'Shif_Y' ,'Rotate']
                                )
                                st.plotly_chart(coc2_interact_fig, use_container_width=True)

                                figs, _, _  = main(
                                    sheet_ID = S_COC2_SHEET_ID, 
                                    threshold = THRESHOLD, 
                                    option = SEARCH_MODE, 
                                    Chip = chip
                                )
                                for fig in figs:
                                    st.pyplot(fig=fig, dpi=500)

                        else:
                            coc2_interact_fig = pdi.interact_scatter(
                                df = full_df,
                                col_led_index_x = 'LED_Index_X',
                                col_led_index_y = 'LED_Index_Y',
                                col_of_color = 'LED_TYPE',
                                symbol = 'Defect_Reciepe',
                                hover_data = ['Shift_X', 'Shif_Y' ,'Rotate', 'LINK']
                            )
                            st.plotly_chart(coc2_interact_fig, use_container_width=True)
                            
                            figs, _, _ = main(
                                sheet_ID = S_COC2_SHEET_ID, 
                                threshold = THRESHOLD, 
                                option = SEARCH_MODE
                            )
                            for fig in figs:
                                st.pyplot(fig=fig, dpi=500)
                    
    
if MODE == "多片":
    
    if SEARCH_MODE == 'AT':
        with st.form(key='multiat_form1'):

            AT_MULTI_CHIP_ID = st.text_area(
                label = 'Enter multi chipids', 
                height = 120, 
                key = 'SHEET_ID', 
                placeholder = '換行輸入, ex:\nchip_1\nchip_2\n...'
            )
            
            AT_MULTI_CHIP_ID_LST = AT_MULTI_CHIP_ID.split('\n')
            AT_MULTI_CHIP_ID_LST = [value for value in AT_MULTI_CHIP_ID_LST if value != '']

            selected_option = st.radio('選擇一個選項', ['chargemap 圖', 'defect count 表'])

            multiat_form_btn = st.form_submit_button(label='Submit')
            
        chip_info_lst = []
        
        if multiat_form_btn or st.session_state.multiat_form_btn:
            if AT_MULTI_CHIP_ID_LST == []:
                st.error('Enter chip ID')
            else:
                st.session_state.multiat_form_btn = True
                for AT_CHIP_ID in AT_MULTI_CHIP_ID_LST:
                
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        
                        st.markdown(f'## {AT_CHIP_ID}')
                        st.session_state.obj = Chargemap(chip_id=AT_CHIP_ID)
                        recipe_lst = st.session_state.obj.get_recipes()
                                        
                    with col2:
                        
                        with st.form(key=f'at_form2_{AT_CHIP_ID}'): 
                            
                            select_recipe = st.selectbox(
                                label = 'Select recipe', 
                                options = recipe_lst
                            )
                            at_form2_btn = st.form_submit_button(label='Submit')
                            
                            if at_form2_btn or st.session_state.at_form2_btn:
                                with st.spinner('Executing...'):
                                    st.session_state.at_form2_btn = True
                                    retest_and_time_lst = st.session_state.obj.get_retest_and_time(select_recipe)
                                    st.success('Success!')
                                    
                    with col3:
                        
                        with st.form(key=f'at_form3_{AT_CHIP_ID}'): 
                            
                            select_retest = st.selectbox(
                                label = 'Select retest', 
                                options = retest_and_time_lst
                            )
                            at_form3_btn = st.form_submit_button(label='Submit')
                            
                            if at_form3_btn or st.session_state.at_form3_btn:
                                select_time = re.sub(r'\D', '', select_retest.split("-")[1][1:])
                                select_retest = select_retest.split("-")[0][:-1] 
                                step_lst = st.session_state.obj.get_steps(
                                    select_retest, 
                                    select_recipe,
                                    select_time
                                )
                                st.success('Success!')                
                
                    if at_form3_btn:
                        
                        with st.spinner('Executing...'):
                            st.session_state.chip_info_lst.append([AT_CHIP_ID,step_lst,select_retest,select_recipe,select_time])

        with st.form(key='at_form4'):        
            at_form4_btn = st.form_submit_button(label='Submit')                 
            if at_form4_btn or st.session_state.at_form4_btn:
                for chip_info in st.session_state.chip_info_lst:
                    
                    st.session_state.obj = Chargemap(chip_id=chip_info[0])
                    step_lst = chip_info[1]
                    select_retest = chip_info[2]
                    select_recipe = chip_info[3]
                    select_time = chip_info[4]
                    
                    if selected_option == "chargemap 圖":
                        
                        for step in step_lst:
                            
                            try:

                                chargemap_fig = st.session_state.obj.plot_chargemap_img(
                                    step, 
                                    select_retest, 
                                    select_recipe,
                                    select_time
                                )
                                if isinstance(chargemap_fig,plt.Figure):
                                    st.pyplot(chargemap_fig, dpi=500)
                                
                                defect_fig = st.session_state.obj.plot_defectmap_img(
                                    step, 
                                    select_retest, 
                                    select_time, 
                                    select_recipe
                                )
                                if isinstance(defect_fig,plt.Figure):
                                    st.pyplot(defect_fig, dpi=500)
                                    
                            except:
                                
                                continue
                    else:
                        try:
                            
                            defect_count_fig = st.session_state.obj.get_total_defect_count(
                                select_retest, 
                                select_time, 
                                select_recipe
                            )
                            if isinstance(defect_count_fig,plt.Figure):
                                st.pyplot(defect_count_fig, dpi=500)
                                
                        except:
                            st.write('<div class="frank-text"> Zero Defect </div>', unsafe_allow_html=True)

                        df_lst = st.session_state.obj.gen_defect_summary(
                            [str(select_retest)],
                            [select_time],
                            [select_recipe],
                            select_recipe[:4]
                        )
                        
                        st.markdown('# 全座標點位，包含 Step, Defect code, LED_TYPE')
                        st.markdown(f'## {select_recipe} - {chip_info[0]}')
                        st.dataframe(df_lst[0], use_container_width=True)
                        st.markdown('# Defect count，包含 Step, Defect code, LED_TYPE')   
                        st.markdown(f'## {select_recipe} - {chip_info[0]}')   
                        st.dataframe(df_lst[1], use_container_width=True)
                    
                st.success('Success!')               
                    
    elif SEARCH_MODE == 'TFT':

        ksize, conv_thresh = conv_config()
            
        # for shipping 給 M01 & Client 的特定圖片格式
        SHIPPING_M01 = st.sidebar.toggle(label="FOR SHIPPING M01")
        SHIPPING_Client = st.sidebar.toggle(label="FOR SHIPPING Client")
        
        if SHIPPING_M01 == True and SHIPPING_Client == True:
            st.error("FOR SHIPPING 只能選擇其中一個模式，或者兩個模式皆為關閉狀態", icon="🚨")
            raise Exception("There is only the option to either turn on A or B, or to turn both off")
            
        # 顯示重複Defect的資訊
        Repeat_defect = st.sidebar.toggle(label="Show Repeat Defect Info")
        
        # 檢查條件
        INSTYPE = st.sidebar.selectbox(
            label = INSTYPE_LABEL, 
            options = INSTYPE_OPTIONS
        )
        
        # 站點選定
        SELECT_OPID = st.sidebar.selectbox(
            label = 'OPID', 
            options = DEFAULT_LIGHT_ON_OPID_LIST, 
            help = 'Only Show the Newest Data from Choose OPID'
        )
        
        SELECT_RECIPE = st.sidebar.selectbox(
            label = 'Recipe', 
            options = [
                "",
                "5V-DVT3-4-16.1-9-bonding_20231228",
                "2V-DVT3-4-16.1-9-bonding_20231228", 
                "9V-DVT3-4-16.1-9-bonding_20231228",
            ],
            help = '若無選擇，將依最新的檔案進行產出'
        )
        
        st.session_state.m_coc2_form = False
        st.session_state.m_tft_coc2_form = False
        
        with st.form(key='tft_form1'):
            from datetime import timedelta, datetime
            M_TFT_SHEET_ID = st.text_area(
                label = 'SHEET_ID', 
                height = 120, 
                key = 'SHEET_ID', 
                placeholder = '換行輸入, ex:\nSHEET_ID_1\nSHEET_ID_2\n...'
            )
            
            TEMP_TFT_SHEET_ID_LIST = M_TFT_SHEET_ID.split('\n')
            TFT_SHEET_ID_LIST = [value for value in TEMP_TFT_SHEET_ID_LIST if value != '']

            m_tft_form = st.form_submit_button(label='Submit')
            if m_tft_form or st.session_state.m_tft_form:
                st.session_state.m_tft_form = False
                
                if len(TFT_SHEET_ID_LIST) == 0:
                    st.error('Enter Sheet ID')
                    
                else:
                    not_found_list = []
                    tft_df_ls = []
                    # For shipping 需要製作出貨報告
                    if SHIPPING_M01:
                        with st.spinner('Executing...'):
                            for TFT_ID in TFT_SHEET_ID_LIST:
                                try:
                                    TFT_ID = TFT_ID.strip()
                                    df = get_newest_tft_df(
                                        sheet_ID=TFT_ID, InsType=INSTYPE, OPID=SELECT_OPID, ACTUAL_RECIPE=SELECT_RECIPE
                                    )
                                    tft_df_ls.append(df)
                                    figs = main_forShipping(
                                        df = df, 
                                        SHEET_ID = TFT_ID, 
                                        Ins_type = INSTYPE,
                                    )
                                    for fig in figs:
                                        st.pyplot(fig=fig, dpi=500)
                                        plt.close(fig) 
                                except:
                                    not_found_list.append(TFT_ID)
                                    
                                    
                    elif SHIPPING_Client:
                        with st.spinner('Executing...'):
                            mainNgcnt = 0
                            byPixelNgcnt = 0
                            for TFT_ID in TFT_SHEET_ID_LIST:
                                try:
                                    TFT_ID = TFT_ID.strip()
                                    df = get_newest_tft_df(
                                        sheet_ID=TFT_ID, InsType=INSTYPE, OPID=SELECT_OPID, ACTUAL_RECIPE=SELECT_RECIPE
                                    )
                                    tft_df_ls.append(df)
                                    figs, resMain, resSub = main_forShipping(
                                        df = df, 
                                        SHEET_ID = TFT_ID, 
                                        Ins_type = INSTYPE,
                                        shipping2Client=True,
                                        kernelSize = ksize, 
                                        conv_threshold = conv_thresh,
                                        recipe = SELECT_RECIPE
                                    )
                                    
                                    getJudgeAnwser(resMain, types=judgeTypeOne)
                                    getJudgeAnwser(resSub, types=judgeTypeTwo) 
                                        
                                    mainNgcnt += resMain
                                    byPixelNgcnt += resSub
                                    
                                    for fig in figs:
                                        st.pyplot(fig=fig, dpi=500)
                                        plt.close(fig) 
                                except:
                                    not_found_list.append(TFT_ID)
                                    
                            st.info(f'(Type MainPixel) 共{len(TFT_SHEET_ID_LIST)}片 不良率為{(mainNgcnt/len(TFT_SHEET_ID_LIST))*100:.2f}%')
                            st.info(f'(Type byPixel) 共{len(TFT_SHEET_ID_LIST)}片 不良率為{(byPixelNgcnt/len(TFT_SHEET_ID_LIST))*100:.2f}%')
                                                
                    else:
                        with st.spinner('Executing...'):
                            mainNgcnt = 0
                            byPixelNgcnt = 0
                            for TFT_ID in TFT_SHEET_ID_LIST:
                                try:
                                    TFT_ID = TFT_ID.strip()
                                    df = get_newest_tft_df(
                                        sheet_ID=TFT_ID, InsType=INSTYPE, OPID=SELECT_OPID, ACTUAL_RECIPE=SELECT_RECIPE
                                    )
                                    
                                    tft_df_ls.append(df)
                                    figs, resMain, resSub = main(
                                        sheet_ID = TFT_ID, 
                                        threshold = THRESHOLD, 
                                        option = SEARCH_MODE, 
                                        Ins_type = INSTYPE, 
                                        OPID = SELECT_OPID,
                                        kernelSize = ksize,
                                        conv_threshold = conv_thresh,
                                        recipe = SELECT_RECIPE
                                    )
                                    
                                    getJudgeAnwser(resMain, types=judgeTypeOne)
                                    getJudgeAnwser(resSub, types=judgeTypeTwo)
                                        
                                    mainNgcnt += resMain
                                    byPixelNgcnt += resSub
                                    
                                    for fig in figs:
                                        st.pyplot(fig=fig, dpi=500)
                                        plt.close(fig)      
                                except:
                                    not_found_list.append(TFT_ID)

                            st.info(f'(Type MainPixel) 共{len(TFT_SHEET_ID_LIST)}片 不良率為{(mainNgcnt/len(TFT_SHEET_ID_LIST))*100:.2f}%')
                            st.info(f'(Type byPixel) 共{len(TFT_SHEET_ID_LIST)}片 不良率為{(byPixelNgcnt/len(TFT_SHEET_ID_LIST))*100:.2f}%')
                            
                    if len(not_found_list)!=0: 
                        st.warning(f'{SELECT_OPID} not found {not_found_list}')
                        
                    if Repeat_defect:
                        with st.spinner('Executing...'):
                            concated_df = pd.concat(tft_df_ls)
                            concated_df = concat_dataframe_id(concated_df['Dataframe_id'])
                            concated_df = concated_df[concated_df['defect_appear_count'] >= 2]
                            pdi = plot_defect_info()
                            tft_repeat_interact_fig = pdi.interact_scatter(
                                df = concated_df,
                                col_led_index_x = 'LED_Index_I',
                                col_led_index_y = 'LED_Index_J',
                                col_of_color = 'LED_TYPE',
                                symbol = 'Defect_Code',
                                hover_data = ['Defect_Code', 'defect_appear_count']
                            )
                            st.plotly_chart(tft_repeat_interact_fig, use_container_width=True)
                            st.dataframe(concated_df, use_container_width=True)
                    st.info('Done')
                        
        if SHIPPING_M01 or SHIPPING_Client:
            from datetime import datetime
            Bytes_PPT = CreatePPT().load_ppt()
            button = st.download_button(
                label = 'Download',
                data = Bytes_PPT,
                file_name = f'{datetime.now().strftime("%Y%m%d%H%M%S")} Shipping Report.pptx',
                help = 'Note:若沒有執行先前的Submit, 將會下載空的PPT檔案'
            )
            CreatePPT().delete_temp_file() 
                 
                    
    if SEARCH_MODE == 'TFT+COC2':
        INSTYPE = st.sidebar.selectbox(
            label = INSTYPE_LABEL, 
            options = INSTYPE_OPTIONS
        )
        
        SELECT_OPID = st.sidebar.selectbox(
            label = 'OPID', 
            options = DEFAULT_LIGHT_ON_OPID_LIST, 
            help = 'Only Show the Newest Data from Choose OPID'
        )
        
        st.session_state.m_coc2_form = False
        st.session_state.m_tft_form = False
        
        with st.form(key='tft_coc2_form1'):
            M_TFT_COC2_SHEET_ID = st.text_area(
                label = 'SHEET_ID', 
                height = 120, 
                key = 'SHEET_ID', 
                placeholder = '換行輸入, ex:\nSHEET_ID_1\nSHEET_ID_2\n...'
            )
            
            TEMP_TFT_SHEET_ID_LIST = M_TFT_COC2_SHEET_ID.split('\n')
            TFT_SHEET_ID_LIST = [value for value in TEMP_TFT_SHEET_ID_LIST if value != '']
            
            m_tft_coc2_form = st.form_submit_button(label='Submit')
            if m_tft_coc2_form or st.session_state.m_tft_coc2_form:
                if len(TFT_SHEET_ID_LIST) == 0:
                    st.error('Enter Sheet ID')
                    
                else:
                    with st.spinner('Executing...'):
                        not_found_list = []
                        for TFT_ID in TFT_SHEET_ID_LIST:
                            # try:
                                figs, _, _  = main(
                                    sheet_ID = TFT_ID, 
                                    threshold = THRESHOLD, 
                                    option = SEARCH_MODE, 
                                    Ins_type = INSTYPE, 
                                    OPID = SELECT_OPID
                                )
                                for fig in figs:
                                    st.pyplot(fig=fig, dpi=500)
                                    plt.close(fig)
                            # except:
                            #     not_found_list.append(TFT_ID)
                    
                    if len(not_found_list) != 0:
                        st.warning(f'{SELECT_OPID} not found {not_found_list}')
                    st.info('Done')
            
            
    elif SEARCH_MODE == 'COC2':
        st.session_state.m_tft_form = False
        st.session_state.m_tft_coc2_form = False
        
        FILTER_DEFECT_THRESHOLD = st.sidebar.number_input(
            label = 'Enter filter Threshold', 
            min_value = 1, 
            step = 1, 
            help = """
                   **資料只會呈現超過設定值的數據**\n
                   """   
        )
        

        with st.form(key='coc2_form1'):
            COC2_SHEET_IDS = st.text_area(
                label = 'SHEET_ID', 
                height = 120, 
                key = 'SHEET_ID', 
                placeholder = '換行輸入, ex:\nSHEET_ID_1\nSHEET_ID_2\n...'
            )

            LAYOUT_OFFSETVALUE = fine_tune_offset_value()
            
            TEMP_COC2_SHEET_ID_LIST = COC2_SHEET_IDS.split('\n')
            COC2_SHEET_ID_LIST = [value for value in TEMP_COC2_SHEET_ID_LIST if value != '']
            
            m_coc2_form = st.form_submit_button(label='Submit')
            if m_coc2_form or st.session_state.m_coc2_form:
                if len(COC2_SHEET_ID_LIST) == 0:
                    st.error('Enter Sheet ID')
                    
                else:
                    not_found_list = []
                    st.success('Success')
                    with st.spinner('Executing...'):
                        concat_df_ls = []
                        for coc2 in COC2_SHEET_ID_LIST:
                            try:
                                pdi = plot_defect_info(sheet_ID=coc2)
                                # 有分chip的COC2回傳值不同
                                try:
                                    full_df, _ = pdi.get_COC2_full_RGB_Dataframe()
                                except:
                                    full_df = pdi.get_COC2_full_RGB_Dataframe()
                                    
                                concat_df_ls.append(full_df)
                                haveChip = pdi.seriesHaveChip(full_df['Chip'])
                                
                                if haveChip:
                                    chip_items = full_df['Chip'].sort_values(ascending=True).unique()
                                    full_figs = plot_fullChip_coc2(coc2, THRESHOLD, onlyCOC2=True, layoutOffet=LAYOUT_OFFSETVALUE)
                                    
                                    for fullfig in full_figs:
                                        st.pyplot(fig=fullfig, dpi=500)
                                            
                                    for chip in chip_items:
                                        st.info(chip)
                                        figs, _  = main(
                                            sheet_ID = coc2, 
                                            threshold = THRESHOLD, 
                                            option = SEARCH_MODE, 
                                            Chip = chip
                                        )
                                        for fig in figs:
                                            st.pyplot(fig=fig, dpi=500)

                                else:
                                    figs, _  = main(
                                        sheet_ID = coc2, 
                                        threshold = THRESHOLD, 
                                        option = SEARCH_MODE
                                    )
                                    for fig in figs:
                                        st.pyplot(fig=fig, dpi=500)
                            except:
                                not_found_list.append(coc2)
                                continue 
                        
                        # plot defect 疊圖 & 出現次數資訊
                        leave_columns = ['LED_TYPE', 'LED_Index_X', 'LED_Index_Y', 'Defect_Reciepe']
                        concated_df = pd.concat(concat_df_ls)
                        concated_df = concated_df[leave_columns]
                        concated_df["Defect_Reciepe"] = concated_df["Defect_Reciepe"].apply(lambda x: defect_code_dict.get(x))
                        concated_df['defect_appear_count'] = 1
                        if LAYOUT_OFFSETVALUE != 0:
                            concated_df['LED_Index_Y'] = concated_df["LED_Index_Y"].apply(
                                lambda x: x + LAYOUT_OFFSETVALUE if x > 156 else x
                            )
                            
                        concated_df = concated_df.groupby(by=['LED_TYPE', 'LED_Index_X', 'LED_Index_Y', 'Defect_Reciepe']).aggregate('sum')
                        filtered_df = concated_df[concated_df['defect_appear_count'] >= FILTER_DEFECT_THRESHOLD].reset_index()
                        
                        coc2_interact_fig = pdi.interact_scatter(
                            df = filtered_df,
                            col_led_index_x = 'LED_Index_X',
                            col_led_index_y = 'LED_Index_Y',
                            col_of_color = 'LED_TYPE',
                            symbol = 'Defect_Reciepe',
                            hover_data = ['Defect_Reciepe', 'defect_appear_count']
                        )
                        
                        st.plotly_chart(coc2_interact_fig, use_container_width=True)
                        st.dataframe(filtered_df, use_container_width=True)
                        
                    if len(not_found_list) != 0:        
                        st.warning(f'{not_found_list} not found')
                    st.info('Done')
                    
##################################################################################

##########################  Defect Comparison ####################################
if MODE == 'Defect比對':
    st.session_state.coc2_form = False
    st.session_state.tft_coc2_form = False
    st.session_state.at_form = False
    
    if SEARCH_MODE == "TFT+COC2":
        defectCompare()
        
    elif SEARCH_MODE == "AT+各站點":
        
        cols = st.columns([.2,.25,.2,.35]) 
        
        with cols[0]:
            with st.form(key="atmb_form1"):
                
                atmb_chipid = st.text_input(label="Enter the chip id").strip()
                atmb_form1_btn = st.form_submit_button(label="Search")
                
                if atmb_form1_btn or st.session_state.atmb_form1_btn:
                    with st.spinner('Executing...'):
                        st.session_state.atmb_form1_btn = True
                        st.session_state.atmb_obj = ATMB(atmb_chipid)
                        recipe_lst = st.session_state.atmb_obj.get_recipes()                        
                        st.success('Success!')                
                
                    with cols[1]:
                        with st.form(key='atmb_form2'): 
                            
                            select_recipe = st.selectbox(
                                label = 'Select AT recipe', 
                                options = recipe_lst
                            )
                            
                            atmb_form2_btn = st.form_submit_button(label='Submit')
                            if atmb_form2_btn or st.session_state.atmb_form2_btn:
                                with st.spinner('Executing...'):
                                    st.session_state.atmb_form2_btn = True
                                    retest_and_time_lst = st.session_state.atmb_obj.get_retest_and_time(select_recipe)
                                    st.success('Success!')

                        with st.form(key='atmb_form3'): 
                            
                            select_retest = st.selectbox(
                                label = f'Select AT 『{select_recipe}』 retest',
                                options = retest_and_time_lst
                            )
                            
                            atmb_form3_btn = st.form_submit_button(label='Submit')
                            if atmb_form3_btn or st.session_state.atmb_form3_btn:
                                with st.spinner('Executing...'):                    
                                    st.session_state.atmb_form3_btn = True
                                    select_time = re.sub(r'\D', '', select_retest.split("-")[1][1:])
                                    select_retest = select_retest.split("-")[0][:-1]

                                    df_at = st.session_state.atmb_obj.df_at[
                                        (st.session_state.atmb_obj.df_at['recipe_id'] == select_recipe) &\
                                        (st.session_state.atmb_obj.df_at['ins_cnt'] == select_retest) &\
                                        (pd.to_datetime(st.session_state.atmb_obj.df_at['lm_time']).dt.date == pd.to_datetime(select_time).date())
                                    ]

                                    df_at_defect = st.session_state.atmb_obj.fs_at.get(ObjectId(df_at['df_defect'].values[0])).read()
                                    df_at_defect = pickle.loads(df_at_defect)
                                    
                                    product = select_recipe[:4]
                                    
                                    # df_at_defect 處理
                                    df_at_defect = st.session_state.atmb_obj.at2mb(df_at_defect, product)
                                    
                                    st.session_state.df_at_defect = df_at_defect
                                    
                                    createtime_lst = st.session_state.atmb_obj.get_createtime_opid()
                                    st.success('Success!')    

                                with cols[2]:
                                    with st.form(key='atmb_form4'): 
                                        
                                        select_createtime = st.selectbox(
                                            label = 'Select light on CreateTime',
                                            options = createtime_lst
                                        )
                                        
                                        site = select_createtime.split(" ")[1]
                                        select_createtime = select_createtime.split(" ")[0]
                                        atmb_form4_btn = st.form_submit_button(label='Submit')

                                        if atmb_form4_btn or st.session_state.atmb_form4_btn == True:
                                            st.session_state.atmb_form4_btn = True
                                            with st.spinner('Executing...'):
                                                
                                                df_mb = st.session_state.atmb_obj.df_mb
                                                df_mb = df_mb[(df_mb["CreateTime"]==select_createtime)]
                                                
                                                df_mb_defect = pd.DataFrame()
                                                for _,row in df_mb.iterrows():
                                                    
                                                    df_mb_temp = st.session_state.atmb_obj.fs_mb.get(ObjectId(row["Dataframe_id"])).read()
                                                    df_mb_temp = pickle.loads(df_mb_temp)
                                                    df_mb_temp = df_mb_temp[df_mb_temp['Defect_Code'] != '']
                                                    df_mb_defect = pd.concat([df_mb_defect,df_mb_temp], axis=0)
                                                    
                                                df_mb_defect = df_mb_defect.reset_index(drop=True)
                                                df_mb_defect = df_mb_defect[
                                                    [
                                                        'LED_Index_I',
                                                        'LED_Index_J',
                                                        'Defect_Code',
                                                        'LED_Luminance',
                                                        'LED_TYPE'
                                                    ]
                                                ]

                                                # df_mb_defect 處理
                                                df_mb_defect = st.session_state.atmb_obj.mb2at(df_mb_defect, product)
                                                
                                                st.session_state.df_mb_defect = df_mb_defect
                                                st.success('Success!') 

                                            st.session_state.match_df, cnt_R, cnt_G, cnt_B = st.session_state.atmb_obj.get_match_points(
                                                st.session_state.df_at_defect,
                                                st.session_state.df_mb_defect
                                            )
                                            
                                            with cols[3]:
                                                with st.expander(label=f"AT & {site} 資料"): 
                                                    if st.session_state.atmb_form4_btn == True:
                                                        st.markdown(f'##### AT 資料 ({len(st.session_state.df_at_defect)}顆 defects)')
                                                        st.dataframe(st.session_state.df_at_defect, height=150)
                                                        st.markdown(f'##### {site} 資料 ({len(st.session_state.df_mb_defect)}顆 defects)')
                                                        st.dataframe(st.session_state.df_mb_defect, height=150)
                                                        st.markdown(f'##### Match 資料 ({len(st.session_state.match_df)}顆 defects, R:{cnt_R} G:{cnt_G} B:{cnt_B})')
                                                        st.dataframe(st.session_state.match_df, height=150)                                                        
                                                
        if st.session_state.atmb_form4_btn == True:

            image = Image.open('./images/image.png')
            st.image(image, caption='Local Image', use_column_width=True)
            
            # Build figure
            fig = go.Figure()

            for led_type in ["red","green","blue"]:
                st.session_state.atmb_obj.plot_mb(
                    fig,
                    st.session_state.df_mb_defect,
                    led_type,
                    site
                )
                
                st.session_state.atmb_obj.plot_at(
                    fig,
                    st.session_state.df_at_defect,
                    led_type,
                    site
                )
                
            WH = st.session_state.atmb_obj.get_WH(product)
            width, heiht = WH[0], WH[1]
            
            # 設定 x 和 y 軸上下界 & layout
            fig.update_xaxes(range=[0,width+30])
            fig.update_yaxes(range=[heiht+30,0])
            fig.update_layout(
                title={
                    'text': f"<b>{atmb_chipid}  [座標體系以 light-on 為準，(1,1)在左上角]</b>",
                    'font': {
                        'family': 'Arial',
                        'size': 36,
                        'color': '#0072B2'
                    }
                },
                xaxis={'side': 'top'},
                width=1000,
                height=800
            )
            
            st.plotly_chart(fig, use_container_width=True) 
