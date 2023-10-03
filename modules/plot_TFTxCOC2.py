from modules.utils import Mark_cluster, plot_defect_info, grid_fs, CreatePPT
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


font_size = 16
bottom_gap = -0.2


def flip_arr_bottom_right(arr):
    return np.flip(np.flip(arr, 0), 1)


def interact_scatter(df:pd.DataFrame, col_led_index_x:str, col_led_index_y:str, col_of_color:str, symbol:str, labels:list,
                     max_col_led_index_y=None):
    
    if max_col_led_index_y == None:
        max_y = df['LED_Indexf_J'].max()
    
    color_discrete_map = {'R': 'rgb(255,0,0)', 'G': 'rgb(0,255,0)', 'B': 'rgb(0,0,255)'}
    fig = px.scatter(
        df,
        x=col_led_index_x,
        y=col_led_index_y,
        color=col_of_color,
        color_discrete_map=color_discrete_map,
        symbol=symbol,
        hover_data=labels,
        color_continuous_scale="reds",
        width=1000,
        height=500,
        range_y=[max_y, 0],
    )
    
    fig.update_layout(
        xaxis={'side': 'top'},
    )
    
    return fig



def TFT_and_COC2(sheet_ID:str, threshold:int, Ins_type:str, TFT_df=None|pd.DataFrame, OPID=None|pd.DataFrame):
    pdi = plot_defect_info(sheet_ID)
    
    coc2_fs = grid_fs(client=pdi.MongoDBClient, db_name=pdi.DB, collection='COC2_AOI_ARRAY')
    colors = ['white', 'black']
    cmap = mcolors.LinearSegmentedColormap.from_list('CMAP', colors)
    
    lum_fs = grid_fs(client=pdi.MongoDBClient, db_name=pdi.DB, collection='LUM_SummaryTable')
    
    duplicate_col = ['OPID', 'LED_TYPE', 'Inspection_Type']
    
    # for single sheet id
    if isinstance(TFT_df, pd.DataFrame):
        specific_time_df = TFT_df
        sort_col_ls = ['LED_TYPE', 'Inspection_Type']
        specific_time_df = specific_time_df.drop_duplicates(duplicate_col, keep='first')
        tft_newest_df = specific_time_df.sort_values(by=sort_col_ls, ascending=False).reset_index(drop=True)
        fig_tft = plot_tft(sheet_ID=sheet_ID, threshold=threshold, Ins_type=Ins_type, TFT_df=tft_newest_df)
        
    # for multiple sheet id. only select specific opid dataframe
    else:
        tft_newest_df = pdi.get_TFT_CreateTime_df() 
        sort_col_ls = ['CreateTime','LED_TYPE', 'Inspection_Type']
        tft_newest_df = tft_newest_df.sort_values(by=sort_col_ls, ascending=False)
        tft_newest_df = tft_newest_df.drop_duplicates(duplicate_col, keep='first').reset_index(drop=True)
        fig_tft = plot_tft(sheet_ID=sheet_ID, threshold=threshold, Ins_type=Ins_type, OPID=OPID)

    del duplicate_col, sort_col_ls
    
    # tft params
    R_light_ID = pdi.get_specific_object_id(tft_newest_df, 'LightingCheck_2D', 'R', Ins_type)
    G_light_ID = pdi.get_specific_object_id(tft_newest_df, 'LightingCheck_2D', 'G', Ins_type)
    B_light_ID = pdi.get_specific_object_id(tft_newest_df, 'LightingCheck_2D', 'B', Ins_type)
    
    R_lum_ID = pdi.get_specific_object_id(tft_newest_df, 'Luminance_2D', 'R', Ins_type)
    G_lum_ID = pdi.get_specific_object_id(tft_newest_df, 'Luminance_2D', 'G', Ins_type)
    B_lum_ID = pdi.get_specific_object_id(tft_newest_df, 'Luminance_2D', 'B', Ins_type)
    
    R_tft_x, R_tft_y = pdi.get_defect_coord(object_id=R_light_ID, fs=lum_fs, coc2=False)
    G_tft_x, G_tft_y = pdi.get_defect_coord(object_id=G_light_ID, fs=lum_fs, coc2=False)
    B_tft_x, B_tft_y = pdi.get_defect_coord(object_id=B_light_ID, fs=lum_fs, coc2=False)
    
    R_lum_arr, _, _, _ = pdi.get_heat_map_imshow_params(object_id=R_lum_ID, LED_TYPE='R', fs=lum_fs)
    G_lum_arr, _, _, _ = pdi.get_heat_map_imshow_params(object_id=G_lum_ID, LED_TYPE='G', fs=lum_fs)
    B_lum_arr, _, _, _ = pdi.get_heat_map_imshow_params(object_id=B_lum_ID, LED_TYPE='B', fs=lum_fs)

    R_tft_defect_arr, _ = pdi.get_defect_imshow_params(object_id=R_light_ID, LED_TYPE='R', fs=lum_fs, coc2=False)
    G_tft_defect_arr, _ = pdi.get_defect_imshow_params(object_id=G_light_ID, LED_TYPE='G', fs=lum_fs, coc2=False)
    B_tft_defect_arr, _ = pdi.get_defect_imshow_params(object_id=B_light_ID, LED_TYPE='B', fs=lum_fs, coc2=False)
    
    R_tft_defect_arr = flip_arr_bottom_right(R_tft_defect_arr)
    G_tft_defect_arr = flip_arr_bottom_right(G_tft_defect_arr)
    B_tft_defect_arr = flip_arr_bottom_right(B_tft_defect_arr)
    
    R_lum_arr = flip_arr_bottom_right(R_lum_arr)
    G_lum_arr = flip_arr_bottom_right(G_lum_arr)
    B_lum_arr = flip_arr_bottom_right(B_lum_arr)
    
        
    # COC2 params
    coc2_df = pdi.get_COC2_df()
    coc2_id = coc2_df['SHEET_ID'][0]
    
    R_COC2_ID = pdi.get_specific_object_id(coc2_df, 'arr_id', 'R', Ins_tpye=None)
    G_COC2_ID = pdi.get_specific_object_id(coc2_df, 'arr_id', 'G', Ins_tpye=None)
    B_COC2_ID = pdi.get_specific_object_id(coc2_df, 'arr_id', 'B', Ins_tpye=None)

    R_coc2_x, R_coc2_y = pdi.get_defect_coord(object_id=R_COC2_ID, fs=coc2_fs, coc2=True)
    G_coc2_x, G_coc2_y = pdi.get_defect_coord(object_id=G_COC2_ID, fs=coc2_fs, coc2=True)
    B_coc2_x, B_coc2_y = pdi.get_defect_coord(object_id=B_COC2_ID, fs=coc2_fs, coc2=True)

    R_coc2_defect_arr, _ = pdi.get_defect_imshow_params(object_id=R_COC2_ID, LED_TYPE='R', fs=coc2_fs, coc2=True)
    G_coc2_defect_arr, _ = pdi.get_defect_imshow_params(object_id=G_COC2_ID, LED_TYPE='G', fs=coc2_fs, coc2=True)
    B_coc2_defect_arr, _ = pdi.get_defect_imshow_params(object_id=B_COC2_ID, LED_TYPE='B', fs=coc2_fs, coc2=True)
    
    # white COC2
    coc2_white = R_coc2_defect_arr + G_coc2_defect_arr + B_coc2_defect_arr
    coc2_white_cluster = np.where(coc2_white > 0, 1, 0)

    y, x = R_lum_arr.shape
    
    # tft + coc2
    fig_tft_coc2, axs2 = plt.subplots(2,5)
    fig_tft_coc2.set_figheight(5)
    fig_tft_coc2.set_figwidth(17.5)

    axs2[0,0].set_xlim([0, x])
    axs2[0,0].set_ylim([0, y])
    axs2[0,0].scatter(R_coc2_x, R_coc2_y, s=10, marker='.', edgecolors='lightcoral', facecolors='none')
    axs2[0,0].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    axs2[0,0].invert_yaxis()
    
    axs2[0,1].scatter(G_coc2_x, G_coc2_y, s=10, marker='.', edgecolors='mediumseagreen', facecolors='none')
    axs2[0,1].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    axs2[0,1].invert_yaxis()
    
    axs2[0,2].scatter(B_coc2_x, B_coc2_y, s=10, marker='.', edgecolors='blue', facecolors='none')
    axs2[0,2].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    axs2[0,2].invert_yaxis()
    
    axs2[0,3].scatter(R_coc2_x, R_coc2_y, s=10, marker='.', edgecolors='lightcoral', facecolors='none')
    axs2[0,3].scatter(G_coc2_x, G_coc2_y, s=10, marker='.', edgecolors='mediumseagreen', facecolors='none')
    axs2[0,3].scatter(B_coc2_x, B_coc2_y, s=10, marker='.', edgecolors='blue', facecolors='none')
    axs2[0,3].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    axs2[0,3].invert_yaxis()

    axs2[0,0].set_title('COC2 R Defect Map', y=bottom_gap)
    axs2[0,1].set_title('COC2 G Defect Map', y=bottom_gap)
    axs2[0,2].set_title('COC2 B Defect Map', y=bottom_gap)
    axs2[0,3].set_title('COC2 Full Defect Map', y=bottom_gap)


    axs2[1,0].scatter(R_tft_x, R_tft_y, s=10, marker='.', edgecolors='lightcoral', facecolors='none')
    axs2[1,1].scatter(G_tft_x, G_tft_y, s=10, marker='.', edgecolors='mediumseagreen', facecolors='none')
    axs2[1,2].scatter(B_tft_x, B_tft_y, s=10, marker='.', edgecolors='blue', facecolors='none')
    axs2[1,3].scatter(R_tft_x, R_tft_y, s=10, marker='.', edgecolors='lightcoral', facecolors='none')
    axs2[1,3].scatter(G_tft_x, G_tft_y, s=10, marker='.', edgecolors='mediumseagreen', facecolors='none')
    axs2[1,3].scatter(B_tft_x, B_tft_y, s=10, marker='.', edgecolors='blue', facecolors='none')

    axs2[1,0].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    axs2[1,1].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    axs2[1,2].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    axs2[1,3].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)


    axs2[1,0].set_title('TFT R Defect Map', y=bottom_gap)
    axs2[1,1].set_title('TFT G Defect Map', y=bottom_gap)
    axs2[1,2].set_title('TFT B Defect Map', y=bottom_gap)
    axs2[1,3].set_title('TFT Full Defect Map', y=bottom_gap)
    
    axs2[1,0].set_xlim([x, 0])
    axs2[1,1].set_xlim([x, 0])
    axs2[1,2].set_xlim([x, 0])
    axs2[1,3].set_xlim([x, 0])

    axs2[1,0].invert_xaxis()
    axs2[1,0].invert_yaxis()
    axs2[1,1].invert_xaxis()
    axs2[1,1].invert_yaxis()
    axs2[1,2].invert_xaxis()
    axs2[1,2].invert_yaxis()
    axs2[1,3].invert_xaxis()
    axs2[1,3].invert_yaxis()

    mark_cluster = Mark_cluster(coc2_white_cluster, threshold, axs2[0,4])
    
    axs2[0,4] = mark_cluster(coc2_white_cluster, threshold, axs2[0,4])
    axs2[0,4].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    axs2[0,4].set_title('Panel Defect Cluster Detectation', y=bottom_gap)
    axs2[0,4].set_title('COC2 White Defect Map', y=bottom_gap)
    axs2[0,4].imshow(coc2_white_cluster, cmap=cmap)
    

    # white TFT
    tft_white = R_tft_defect_arr + G_tft_defect_arr + B_tft_defect_arr
    tft_white_cluster = np.where(tft_white > 0, 1, 0)
    
    axs2[1,4] = mark_cluster(tft_white_cluster, threshold, axs2[1,4])
    axs2[1,4].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    axs2[1,4].set_title('Panel Defect Cluster Detectation', y=bottom_gap)
    axs2[1,4].imshow(tft_white_cluster, cmap=cmap)
    axs2[1,4].set_title('TFT White Defect Map', y=bottom_gap)
    fig_tft_coc2.suptitle(f'{sheet_ID}_{coc2_id} Comparison', fontsize=font_size, y=1)
    fig_tft_coc2.tight_layout()
    
    
    fig_coc2 = plot_coc2(sheet_ID, threshold, onlyCOC2=False)
    
    figlist = [fig_tft_coc2] + fig_tft + fig_coc2

    return figlist


def plot_coc2(sheet_ID:str, threshold:int, onlyCOC2:bool):
    fig_coc2, axs3 = plt.subplots(4,2)

    fig_coc2.set_figheight(8)
    fig_coc2.set_figwidth(8)
    
    pdi = plot_defect_info(sheet_ID)
    
    colors = ['white', 'black']
    cmap = mcolors.LinearSegmentedColormap.from_list('CMAP', colors)
    
    coc2_fs = grid_fs(client=pdi.MongoDBClient, db_name=pdi.DB, collection='COC2_AOI_ARRAY')
    
    if onlyCOC2:
        coc2_df = pdi.get_COC2_df_without_TFT()
    else:
        coc2_df = pdi.get_COC2_df()
    
    coc2_id = coc2_df['SHEET_ID'][0]
    coc2_opid = coc2_df['OPID'][0]
    
    R_COC2_ID = pdi.get_specific_object_id(coc2_df, 'arr_id', 'R', Ins_tpye=None)
    G_COC2_ID = pdi.get_specific_object_id(coc2_df, 'arr_id', 'G', Ins_tpye=None)
    B_COC2_ID = pdi.get_specific_object_id(coc2_df, 'arr_id', 'B', Ins_tpye=None)

    R_coc2_x, R_coc2_y = pdi.get_defect_coord(object_id=R_COC2_ID, fs=coc2_fs, coc2=True)
    G_coc2_x, G_coc2_y = pdi.get_defect_coord(object_id=G_COC2_ID, fs=coc2_fs, coc2=True)
    B_coc2_x, B_coc2_y = pdi.get_defect_coord(object_id=B_COC2_ID, fs=coc2_fs, coc2=True)
    
    R_coc2_defect_arr, R_coc2_defect_cmp = pdi.get_defect_imshow_params(object_id=R_COC2_ID, LED_TYPE='R', fs=coc2_fs, coc2=True)
    G_coc2_defect_arr, G_coc2_defect_cmp = pdi.get_defect_imshow_params(object_id=G_COC2_ID, LED_TYPE='G', fs=coc2_fs, coc2=True)
    B_coc2_defect_arr, B_coc2_defect_cmp = pdi.get_defect_imshow_params(object_id=B_COC2_ID, LED_TYPE='B', fs=coc2_fs, coc2=True)
    
    y, x = R_coc2_defect_arr.shape
    
    # white COC2
    coc2_white = R_coc2_defect_arr + G_coc2_defect_arr + B_coc2_defect_arr
    coc2_white_cluster = np.where(coc2_white > 0, 1, 0)

    mark_cluster = Mark_cluster(coc2_white_cluster, threshold, axs3[0,1])
    axs3[0, 0].set_xlim([0, x])
    axs3[0, 0].set_ylim([y, 0])
    axs3[0, 0].scatter(R_coc2_x, R_coc2_y, s=10, marker='.', edgecolors='lightcoral', facecolors='none')
    axs3[0, 0].scatter(G_coc2_x, G_coc2_y, s=10, marker='.', edgecolors='mediumseagreen', facecolors='none')
    axs3[0, 0].scatter(B_coc2_x, B_coc2_y, s=10, marker='.', edgecolors='blue', facecolors='none')
    axs3[0, 0].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    axs3[0, 0].set_title('COC2 White Defect Map', y=bottom_gap)
    # axs3[0, 0].invert_yaxis()

    axs3[0, 1].set_xlim([0, x])
    axs3[0, 1].set_ylim([0, y])
    axs3[0, 1] = mark_cluster(coc2_white_cluster, threshold, axs3[0,1])
    axs3[0, 1].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    axs3[0, 1].set_title('Panel Defect Cluster Detectation', y=bottom_gap)
    axs3[0, 1].imshow(coc2_white_cluster, cmap=cmap)
    axs3[0, 1].invert_yaxis()


    axs3[1, 0].set_xlim([0, x])
    axs3[1, 0].set_ylim([0, y])
    axs3[1, 0].scatter(R_coc2_x, R_coc2_y, s=10, marker='.', edgecolors='lightcoral', facecolors='none')
    axs3[1, 0].set_title('R Defect Scatter', y=bottom_gap)
    axs3[1, 0].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    axs3[1, 0].invert_yaxis()

    axs3[1, 1] = mark_cluster(R_coc2_defect_arr, threshold, axs3[1, 1])
    axs3[1, 1].imshow(R_coc2_defect_arr, cmap=R_coc2_defect_cmp)
    axs3[1, 1].set_title('R Defect Cluster Detectation', y=bottom_gap)
    axs3[1, 1].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)


    axs3[2, 0].set_xlim([0, x])
    axs3[2, 0].set_ylim([0, y])
    axs3[2, 0].scatter(G_coc2_x, G_coc2_y, s=10, marker='.', edgecolors='mediumseagreen', facecolors='none')
    axs3[2, 0].set_title('G Defect Scatter', y=bottom_gap)
    axs3[2, 0].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    axs3[2, 0].invert_yaxis()

    axs3[2, 1] = mark_cluster(G_coc2_defect_arr, threshold, axs3[2, 1])
    axs3[2, 1].imshow(G_coc2_defect_arr, cmap=G_coc2_defect_cmp)
    axs3[2, 1].set_title('G Defect Cluster Detectation', y=bottom_gap)
    axs3[2, 1].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

    axs3[3, 0].set_xlim([0, x])
    axs3[3, 0].set_ylim([0, y])
    axs3[3, 0].scatter(B_coc2_x, B_coc2_y, s=10, marker='.', edgecolors='blue', facecolors='none')
    axs3[3, 0].set_title('B Defect Scatter', y=bottom_gap)
    axs3[3, 0].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    axs3[3, 0].invert_yaxis()

    axs3[3, 1] = mark_cluster(B_coc2_defect_arr, threshold, axs3[3, 1])
    axs3[3, 1].imshow(B_coc2_defect_arr, cmap=B_coc2_defect_cmp)
    axs3[3, 1].set_title('B Defect Cluster Detectation', y=bottom_gap)
    axs3[3, 1].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    fig_coc2.suptitle(f'COC2 {coc2_id} {coc2_opid} Defect Info', fontsize=font_size)
    fig_coc2.tight_layout()
    
    return [fig_coc2]


def plot_tft(sheet_ID:str, threshold:int, Ins_type:str, TFT_df=None|pd.DataFrame, OPID=None|str):
    pdi = plot_defect_info(sheet_ID)
    
    colors = ['white', 'black']
    cmap = mcolors.LinearSegmentedColormap.from_list('CMAP', colors)

    lum_fs = grid_fs(client=pdi.MongoDBClient, db_name=pdi.DB, collection='LUM_SummaryTable')
    
    duplicate_col = ['OPID', 'LED_TYPE', 'Inspection_Type']
    
    # for single sheet id
    if isinstance(TFT_df, pd.DataFrame):
        specific_time_df = TFT_df
        sort_col_ls = ['LED_TYPE', 'Inspection_Type']
        specific_time_df = specific_time_df.drop_duplicates(duplicate_col, keep='first')
        tft_newest_df = specific_time_df.sort_values(by=sort_col_ls, ascending=False).reset_index(drop=True)
    
    # for multiple sheet id    
    else:
        tft_newest_df = pdi.get_TFT_CreateTime_df() 
        sort_col_ls = ['CreateTime','LED_TYPE', 'Inspection_Type']
        tft_newest_df = tft_newest_df.sort_values(by=sort_col_ls, ascending=False)
        
        newest_ct = tft_newest_df['CreateTime'].unique()[0]
        duplicate_col = ['CreateTime', 'OPID', 'LED_TYPE', 'Inspection_Type']
        tft_newest_df = tft_newest_df[(tft_newest_df['CreateTime'] == newest_ct) & (tft_newest_df['OPID'] == OPID)]
        tft_newest_df = tft_newest_df.drop_duplicates(duplicate_col, keep='first').reset_index(drop=True)
    
    del duplicate_col, sort_col_ls
    
    ct = tft_newest_df['CreateTime'][0]
    OPID = tft_newest_df['OPID'][0]
    grade = tft_newest_df['Grade'][0]
    
    R_light_ID = pdi.get_specific_object_id(tft_newest_df, 'LightingCheck_2D', 'R', Ins_type)
    G_light_ID = pdi.get_specific_object_id(tft_newest_df, 'LightingCheck_2D', 'G', Ins_type)
    B_light_ID = pdi.get_specific_object_id(tft_newest_df, 'LightingCheck_2D', 'B', Ins_type)
    
    R_lum_ID = pdi.get_specific_object_id(tft_newest_df, 'Luminance_2D', 'R', Ins_type)
    G_lum_ID = pdi.get_specific_object_id(tft_newest_df, 'Luminance_2D', 'G', Ins_type)
    B_lum_ID = pdi.get_specific_object_id(tft_newest_df, 'Luminance_2D', 'B', Ins_type)
    
    R_pattern_ls = pdi.get_NGCNT_Yield(df=tft_newest_df, LED_TYPE='R')
    G_pattern_ls = pdi.get_NGCNT_Yield(df=tft_newest_df, LED_TYPE='G')
    B_pattern_ls = pdi.get_NGCNT_Yield(df=tft_newest_df, LED_TYPE='B')
    
    R_tft_x, R_tft_y = pdi.get_defect_coord(object_id=R_light_ID, fs=lum_fs, coc2=False)
    G_tft_x, G_tft_y = pdi.get_defect_coord(object_id=G_light_ID, fs=lum_fs, coc2=False)
    B_tft_x, B_tft_y = pdi.get_defect_coord(object_id=B_light_ID, fs=lum_fs, coc2=False)
    
    R_lum_arr, R_cmap, R_lum_max, R_lum_min = pdi.get_heat_map_imshow_params(object_id=R_lum_ID, LED_TYPE='R', fs=lum_fs)
    G_lum_arr, G_cmap, G_lum_max, G_lum_min = pdi.get_heat_map_imshow_params(object_id=G_lum_ID, LED_TYPE='G', fs=lum_fs)
    B_lum_arr, B_cmap, B_lum_max, B_lum_min = pdi.get_heat_map_imshow_params(object_id=B_lum_ID, LED_TYPE='B', fs=lum_fs)

    R_tft_defect_arr, R_tft_defect_cmp = pdi.get_defect_imshow_params(object_id=R_light_ID, LED_TYPE='R', fs=lum_fs, coc2=False)
    G_tft_defect_arr, G_tft_defect_cmp = pdi.get_defect_imshow_params(object_id=G_light_ID, LED_TYPE='G', fs=lum_fs, coc2=False)
    B_tft_defect_arr, B_tft_defect_cmp = pdi.get_defect_imshow_params(object_id=B_light_ID, LED_TYPE='B', fs=lum_fs, coc2=False)
    
    R_tft_defect_arr = flip_arr_bottom_right(R_tft_defect_arr)
    G_tft_defect_arr = flip_arr_bottom_right(G_tft_defect_arr)
    B_tft_defect_arr = flip_arr_bottom_right(B_tft_defect_arr)
    
    R_lum_arr = flip_arr_bottom_right(R_lum_arr)
    G_lum_arr = flip_arr_bottom_right(G_lum_arr)
    B_lum_arr = flip_arr_bottom_right(B_lum_arr)
    
    
    fig_tft, axs = plt.subplots(3, 3)
    fig_tft.set_figheight(8)
    fig_tft.set_figwidth(15)
    
    y ,x = R_lum_arr.shape
    
    mark_cluster = Mark_cluster(R_tft_defect_arr, threshold, axs[0, 1])
    
    axs[0, 0].set_xlim([x, 0])
    axs[0, 0].set_ylim([0, y])
    axs[0, 0].scatter(R_tft_x, R_tft_y, s=10, marker='.', edgecolors='lightcoral', facecolors='none')
    axs[0, 0].set_title('R Defect Scatter', y=bottom_gap)
    axs[0, 0].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    axs[0, 0].invert_xaxis()
    axs[0, 0].invert_yaxis()

    axs[0, 1] = mark_cluster(R_tft_defect_arr, threshold, axs[0, 1])
    axs[0, 1].imshow(R_tft_defect_arr, cmap=R_tft_defect_cmp)
    axs[0, 1].set_title('R Defect Cluster Detectation', y=bottom_gap)
    axs[0, 1].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

    axs[0, 2].imshow(R_lum_arr, cmap=R_cmap, vmin=R_lum_min, vmax=R_lum_max)
    axs[0, 2].set_title('R HeatMap', y=bottom_gap)
    axs[0, 2].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    plt.colorbar(axs[0, 2].imshow(R_lum_arr, cmap=R_cmap, vmin=R_lum_min, vmax=R_lum_max), ax=axs[0, 2])


    axs[1, 0].set_xlim([x, 0])
    axs[1, 0].set_ylim([0, y])
    axs[1, 0].scatter(G_tft_x, G_tft_y, s=10, marker='.', edgecolors='mediumseagreen', facecolors='none')
    axs[1, 0].set_title('G Defect Scatter', y=bottom_gap)
    axs[1, 0].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    axs[1, 0].invert_xaxis()
    axs[1, 0].invert_yaxis()

    axs[1, 1] = mark_cluster(G_tft_defect_arr, threshold, axs[1, 1])
    axs[1, 1].imshow(G_tft_defect_arr, cmap=G_tft_defect_cmp)
    axs[1, 1].set_title('G Defect Cluster Detectation', y=bottom_gap)
    axs[1, 1].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

    axs[1, 2].imshow(G_lum_arr, cmap=G_cmap, vmin=G_lum_min, vmax=G_lum_max)
    axs[1, 2].set_title('G HeatMap', y=bottom_gap)
    axs[1, 2].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    plt.colorbar(axs[1, 2].imshow(G_lum_arr, cmap=G_cmap, vmin=G_lum_min, vmax=G_lum_max), ax=axs[1, 2])


    axs[2, 0].set_xlim([x, 0])
    axs[2, 0].set_ylim([0, y])
    axs[2, 0].scatter(B_tft_x, B_tft_y, s=10, marker='.', edgecolors='blue', facecolors='none')
    axs[2, 0].set_title('B Defect Scatter', y=bottom_gap)
    axs[2, 0].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    axs[2, 0].invert_xaxis()
    axs[2, 0].invert_yaxis()

    axs[2, 1] = mark_cluster(B_tft_defect_arr, threshold, axs[2, 1])
    axs[2, 1].imshow(B_tft_defect_arr, cmap=B_tft_defect_cmp)
    axs[2, 1].set_title('B Defect Cluster Detectation', y=bottom_gap)
    axs[2, 1].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

    axs[2, 2].imshow(B_lum_arr, cmap=B_cmap, vmin=B_lum_min, vmax=B_lum_max)
    axs[2, 2].set_title('B HeatMap', y=bottom_gap)
    axs[2 ,2].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    plt.colorbar(axs[2, 2].imshow(B_lum_arr, cmap=B_cmap, vmin=B_lum_min, vmax=B_lum_max), ax=axs[2, 2])
    fig_tft.tight_layout()

    axs[0, 0].table(
        cellText = R_pattern_ls, 
        colLabels = ['DefectCnt', 'Yield'],
        rowLabels = ['Normal', 'Edge', 'L0', 'L10'],
        cellLoc='center',
        bbox = [3.5, 0., 0.5, 1],
    )

    axs[1, 0].table(
        cellText = G_pattern_ls, 
        colLabels = ['DefectCnt', 'Yield'],
        rowLabels = ['Normal', 'Edge', 'L0', 'L10'],
        cellLoc='center',
        bbox = [3.5, 0., 0.5, 1],
    )

    axs[2, 0].table(
        cellText = B_pattern_ls, 
        colLabels = ['DefectCnt', 'Yield'],
        rowLabels = ['Normal', 'Edge', 'L0', 'L10'],
        cellLoc='center',
        bbox = [3.5, 0., 0.5, 1],
    )
    
    fig_tft.suptitle(f'{ct} {sheet_ID} {OPID} {grade} {Ins_type} Defect Info', fontsize=font_size, y=1.05)
    

    fig_full, axs2 = plt.subplots(1, 2)
    # white TFT
    fig_full.set_figheight(5)
    fig_full.set_figwidth(17.5)
    tft_white = R_tft_defect_arr + G_tft_defect_arr + B_tft_defect_arr
    tft_white_cluster = np.where(tft_white > 0, 1, 0)

    axs2[0].set_xlim([x, 0])
    axs2[0].set_ylim([0, y])
    axs2[0].scatter(R_tft_x, R_tft_y, s=10, marker='.', edgecolors='lightcoral', facecolors='none')
    axs2[0].scatter(G_tft_x, G_tft_y, s=10, marker='.', edgecolors='mediumseagreen', facecolors='none')
    axs2[0].scatter(B_tft_x, B_tft_y, s=10, marker='.', edgecolors='blue', facecolors='none')
    axs2[0].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    axs2[0].set_title('TFT White Defect Map', y=bottom_gap)
    axs2[0].invert_xaxis()
    axs2[0].invert_yaxis()
    
    axs2[1] = mark_cluster(tft_white_cluster, threshold, axs2[1])
    axs2[1].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    axs2[1].set_title('Panel Cluster Info', y=bottom_gap)
    axs2[1].imshow(tft_white_cluster, cmap=cmap)
    
    fig_full.suptitle(f'{sheet_ID}_{OPID} Result', fontsize=font_size, y=1)
    fig_full.tight_layout()
    return [fig_tft, fig_full]


# plot shipping to M01 specific image
def main_forShipping(df:pd.DataFrame, SHEET_ID:str, Ins_type):
    df = df[df['SHEET_ID']==SHEET_ID].sort_values(['LED_TYPE'], ascending=False).reset_index(drop=True)
    OPID = df['OPID'].tolist()[0]
    CREATETIME = df['CreateTime'].tolist()[0]
    
    ngcnt_ls = df['NGCNT'].tolist()
    total_ng = df['NGCNT'].sum()
    ngcnt_ls.append(total_ng)
    # Create table info each element type should be list
    ng_info_ls = [[i] for i in ngcnt_ls]
    del total_ng, ngcnt_ls
    
    pdi = plot_defect_info(sheet_ID=SHEET_ID)
    COC2df = pdi.get_BONDING_MAP_df()
    
    if len(COC2df.index)!=0:
        COC2 = COC2df['coc2'].tolist()[0]
    else:
        COC2 = "COC2 NOT FOUND"
        
    del COC2df
    
    lum_fs = grid_fs(client=pdi.MongoDBClient, db_name=pdi.DB, collection='LUM_SummaryTable')
    R_light_ID = pdi.get_specific_object_id(df, 'LightingCheck_2D', 'R', Ins_type)
    G_light_ID = pdi.get_specific_object_id(df, 'LightingCheck_2D', 'G', Ins_type)
    B_light_ID = pdi.get_specific_object_id(df, 'LightingCheck_2D', 'B', Ins_type)
    
    R_tft_defect_arr, _ = pdi.get_defect_imshow_params(object_id=R_light_ID, LED_TYPE='R', fs=lum_fs, coc2=False)
    y, x = R_tft_defect_arr.shape
    del R_tft_defect_arr
    
    R_tft_x, R_tft_y = pdi.get_defect_coord(object_id=R_light_ID, fs=lum_fs, coc2=False)
    G_tft_x, G_tft_y = pdi.get_defect_coord(object_id=G_light_ID, fs=lum_fs, coc2=False)
    B_tft_x, B_tft_y = pdi.get_defect_coord(object_id=B_light_ID, fs=lum_fs, coc2=False)
    
    fig_full, axs = plt.subplots(1, 1)
    fig_full.set_figheight(5)
    fig_full.set_figwidth(13)
    axs.set_xlim([x, 0])
    axs.set_ylim([0, y])
    axs.scatter(R_tft_x, R_tft_y, s=20, marker='.', edgecolors='red', facecolors='none')
    axs.scatter(G_tft_x, G_tft_y, s=20, marker='.', edgecolors='green', facecolors='none')
    axs.scatter(B_tft_x, B_tft_y, s=20, marker='.', edgecolors='blue', facecolors='none')
    axs.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    axs.invert_xaxis()
    axs.invert_yaxis()
    
    axs.table(
        cellText = ng_info_ls, # such as [[50], [42], [19], [111]]
        colLabels = ['DefectCNT'],
        rowLabels = ['R', 'G', 'B', 'Total'],
        cellLoc='center',
        bbox = [1.05, 0., 0.2, 1],
    )
    
    temp_save_path = './modules/temp/temp_image.png'
    create_ppt = CreatePPT()
    
    fig_full.suptitle(f'{CREATETIME}_{SHEET_ID}_{COC2}_{OPID}_{Ins_type} Result', fontsize=font_size, y=1)
    fig_full.savefig(temp_save_path, facecolor=None, bbox_inches='tight')
    
    create_ppt(image_path=temp_save_path)
    
    return fig_full



def main(sheet_ID:str, threshold:int, option:str, Ins_type=None|str, TFT_df=None|pd.DataFrame, OPID=None|str):
    options = ["COC2", "TFT", "TFT+COC2"]
    
    if threshold is not isinstance(threshold, int):
        threshold = int(threshold)
        
    if option==options[1]:
        fig_list = plot_tft(sheet_ID, threshold, Ins_type, TFT_df=TFT_df, OPID=OPID)

    elif option==options[0]:
        fig_list = plot_coc2(sheet_ID, threshold, onlyCOC2=True)
    
    elif option==options[2]:
        fig_list = TFT_and_COC2(sheet_ID, threshold, Ins_type, TFT_df=TFT_df, OPID=OPID)
        
    return fig_list
    
    
        
        
if __name__ == '__main__':
  
    main(sheet_ID='VKV3457722A1812', threshold=20, option="TFT+COC2", Ins_type='L255', OPID='MT-ACL')
