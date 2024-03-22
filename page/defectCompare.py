import streamlit as st
from modules.utils import plot_defect_info



defect_code_dict = {
    "AB01": "Particle",
    "AB02": "Tilt",
    "AB03": "Crack",
    "AB04": "shift",
    "AB05": "Defect Area",
    "AB06": "缺晶/Not Found",
    "AB07": "LED色度",
    "AB08": "LED缺晶",
    "AB09": "LED亮/暗(輝度)",
    "AB10": "Rotate",
    "AB11": "PAD Loss",
    "AB12": "On Dot",
    "AB13": "Bright Dot",
    "AB14": "Multi",
    "AB15": "Edge",
    "OK": "LED已上件"
}


def defectCompare():
    default_single_sheet_option_ls = []
    INSTYPE_LABEL = "Inspection Type"
    INSTYPE_OPTIONS = ["L255", "L10", "L0"]
    tft_leave_cols = ["LED_TYPE", "LED_Index_I", "LED_Index_J", "Defect_Code"]
    coc2_cols_name = ["LED_TYPE", "LED_Index_I", "LED_Index_J", "Shift X", "Shift Y", "Rotate", "Defect Reciepe", "LINK", "Chip"]
    
    INSTYPE = st.sidebar.selectbox(
        label = INSTYPE_LABEL, 
        options = INSTYPE_OPTIONS
    )

    col1, col2 = st.columns(2)
    with col1:
        with st.form(key="tft_form1"):
            S_TFT_SHEET_ID = st.text_input(label="Sheet_ID").strip()
            tft_form = st.form_submit_button(label="Search")
                
            if tft_form or st.session_state.tft_form:
                if S_TFT_SHEET_ID == "":
                    st.error("Enter Sheet ID")
                    
                else:
                    with st.spinner("Executing..."):
                        st.session_state.tft_form = True
                        pdi = plot_defect_info(sheet_ID=S_TFT_SHEET_ID)
                        tft_df = pdi.get_TFT_CreateTime_df()
                        # default_single_sheet_option_ls 該產品的各檢測時間序列
                        default_single_sheet_option_ls = pdi.get_option_list(tft_df)
                        st.success("Success! Please Choose Date!")
                        
    with col2:
        with st.form(key="tft_form2"):
            select_date = st.selectbox(
                label = "Date", 
                options = default_single_sheet_option_ls
            )
            
            submit_form2 = st.form_submit_button(label="Submit")
            if submit_form2:
                with st.spinner("Executing..."):
                    st.success("Success")

    if submit_form2:
        with st.spinner("Executing..."):
            coc2_df, max_index_x = pdi.get_COC2_full_RGB_Dataframe(onlyCOC2=False)
            coc2_df.columns = coc2_cols_name
            
            coc2_df["LED_Index_I"] = coc2_df["LED_Index_I"].apply(lambda x: abs(x-max_index_x) + 1)    
            coc2_df["Defect Reciepe"] = coc2_df["Defect Reciepe"].apply(lambda x: defect_code_dict.get(x))
            
            tft_df = tft_df[tft_df["CreateTime"]==select_date.split(" ")[0]]
            tft_full_df, _, _ = pdi.get_TFT_full_RGB_Dataframe(tft_df, INSTYPE)
            tft_full_df = tft_full_df[tft_leave_cols]
            tft_full_df = tft_full_df[tft_full_df["Defect_Code"] != ""].reset_index(drop=True)
            
            res = tft_full_df.merge(right=coc2_df, how="inner", on=["LED_TYPE", "LED_Index_I", "LED_Index_J"])
            res.drop(columns=["LINK", "Chip"], inplace=True)
            
            res["ck"] = 1
            summary = res.pivot_table(index="LED_TYPE", columns="Defect Reciepe", values="ck", aggfunc="sum")
            summary = summary.fillna(value=0)
            
            fig = pdi.interact_scatter(
                    df = res,
                    col_led_index_x = 'LED_Index_I',
                    col_led_index_y = 'LED_Index_J',
                    col_of_color = 'LED_TYPE',
                    symbol = 'Defect Reciepe',
                    hover_data = ['Shift X', 'Shift Y' ,'Rotate']
                )
            
            st.info("Defect Compare Mapping")
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("Summary Info")
            st.dataframe(summary, use_container_width=True)
            
            st.info("Detail Data")
            res.drop(columns="ck", inplace=True)
            st.dataframe(res, use_container_width=True)
