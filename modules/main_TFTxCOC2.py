from modules.utils import Mark_cluster, plot_defect_info, grid_fs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np



font_size = 16
bottom_gap = -0.2


def flip_arr_bottom_right(arr):
    return np.flip(np.flip(arr, 0), 1)


def TFT_and_COC2(sheet_ID:str, threshold:int):
    pdi = plot_defect_info(sheet_ID)
    
    coc2_fs = grid_fs(client=pdi.MongoDBClient, db_name=pdi.DB, collection='COC2_AOI_ARRAY')
    colors = ['white', 'black']
    cmap = mcolors.LinearSegmentedColormap.from_list('CMAP', colors)
    
    lum_fs = grid_fs(client=pdi.MongoDBClient, db_name=pdi.DB, collection='LUM_SummaryTable')
    tft_newest_df = pdi.get_TFT_CreateTime_df()

    # tft params
    R_light_ID = pdi.get_specific_object_id(tft_newest_df, 'LightingCheck_2D', 'R', 'L255')
    G_light_ID = pdi.get_specific_object_id(tft_newest_df, 'LightingCheck_2D', 'G', 'L255')
    B_light_ID = pdi.get_specific_object_id(tft_newest_df, 'LightingCheck_2D', 'B', 'L255')
    
    R_lum_ID = pdi.get_specific_object_id(tft_newest_df, 'Luminance_2D', 'R', 'L255')
    G_lum_ID = pdi.get_specific_object_id(tft_newest_df, 'Luminance_2D', 'G', 'L255')
    B_lum_ID = pdi.get_specific_object_id(tft_newest_df, 'Luminance_2D', 'B', 'L255')
    
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
    fig_tft_coc2.set_figwidth(20)

    axs2[0,0].set_xlim([0, x])
    axs2[0,0].set_ylim([0, y])
    axs2[0,0].scatter(R_coc2_x, R_coc2_y, s=1, marker='.', edgecolors='lightcoral', facecolors='none')
    axs2[0,0].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    axs2[0,0].invert_yaxis()
    
    axs2[0,1].scatter(G_coc2_x, G_coc2_y, s=1, marker='.', edgecolors='mediumseagreen', facecolors='none')
    axs2[0,1].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    axs2[0,1].invert_yaxis()
    
    axs2[0,2].scatter(B_coc2_x, B_coc2_y, s=1, marker='.', edgecolors='blue', facecolors='none')
    axs2[0,2].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    axs2[0,2].invert_yaxis()
    
    axs2[0,3].scatter(R_coc2_x, R_coc2_y, s=1, marker='.', edgecolors='lightcoral', facecolors='none')
    axs2[0,3].scatter(G_coc2_x, G_coc2_y, s=1, marker='.', edgecolors='mediumseagreen', facecolors='none')
    axs2[0,3].scatter(B_coc2_x, B_coc2_y, s=1, marker='.', edgecolors='blue', facecolors='none')
    axs2[0,3].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    axs2[0,3].invert_yaxis()

    axs2[0,0].set_title('COC2 R Defect Map', y=bottom_gap)
    axs2[0,1].set_title('COC2 G Defect Map', y=bottom_gap)
    axs2[0,2].set_title('COC2 B Defect Map', y=bottom_gap)
    axs2[0,3].set_title('COC2 Full Defect Map', y=bottom_gap)


    axs2[1,0].scatter(R_tft_x, R_tft_y, s=1, marker='.', edgecolors='lightcoral', facecolors='none')
    axs2[1,1].scatter(G_tft_x, G_tft_y, s=1, marker='.', edgecolors='mediumseagreen', facecolors='none')
    axs2[1,2].scatter(B_tft_x, B_tft_y, s=1, marker='.', edgecolors='blue', facecolors='none')
    axs2[1,3].scatter(R_tft_x, R_tft_y, s=1, marker='.', edgecolors='lightcoral', facecolors='none')
    axs2[1,3].scatter(G_tft_x, G_tft_y, s=1, marker='.', edgecolors='mediumseagreen', facecolors='none')
    axs2[1,3].scatter(B_tft_x, B_tft_y, s=1, marker='.', edgecolors='blue', facecolors='none')

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
    
    
    fig_tft = plot_tft(sheet_ID, threshold)
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
    
    coc2_df = pdi.get_COC2_df()
    
    if onlyCOC2:
        coc2_df = pdi.get_COC2_df_without_TFT()
    
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

    
    coc2_df = pdi.get_COC2_df_without_TFT()
    mark_cluster = Mark_cluster(coc2_white_cluster, threshold, axs3[0,1])
    axs3[0, 0].set_xlim([0, x])
    axs3[0, 0].set_ylim([y, 0])
    axs3[0, 0].scatter(R_coc2_x, R_coc2_y, s=1, marker='.', edgecolors='lightcoral', facecolors='none')
    axs3[0, 0].scatter(G_coc2_x, G_coc2_y, s=1, marker='.', edgecolors='mediumseagreen', facecolors='none')
    axs3[0, 0].scatter(B_coc2_x, B_coc2_y, s=1, marker='.', edgecolors='blue', facecolors='none')
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
    axs3[1, 0].scatter(R_coc2_x, R_coc2_y, s=1, marker='.', edgecolors='lightcoral', facecolors='none')
    axs3[1, 0].set_title('R Defect Scatter', y=bottom_gap)
    axs3[1, 0].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    axs3[1, 0].invert_yaxis()

    axs3[1, 1] = mark_cluster(R_coc2_defect_arr, threshold, axs3[1, 1])
    axs3[1, 1].imshow(R_coc2_defect_arr, cmap=R_coc2_defect_cmp)
    axs3[1, 1].set_title('R Defect Cluster Detectation', y=bottom_gap)
    axs3[1, 1].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)


    axs3[2, 0].set_xlim([0, x])
    axs3[2, 0].set_ylim([0, y])
    axs3[2, 0].scatter(G_coc2_x, G_coc2_y, s=1, marker='.', edgecolors='mediumseagreen', facecolors='none')
    axs3[2, 0].set_title('G Defect Scatter', y=bottom_gap)
    axs3[2, 0].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    axs3[2, 0].invert_yaxis()

    axs3[2, 1] = mark_cluster(G_coc2_defect_arr, threshold, axs3[2, 1])
    axs3[2, 1].imshow(G_coc2_defect_arr, cmap=G_coc2_defect_cmp)
    axs3[2, 1].set_title('G Defect Cluster Detectation', y=bottom_gap)
    axs3[2, 1].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

    axs3[3, 0].set_xlim([0, x])
    axs3[3, 0].set_ylim([0, y])
    axs3[3, 0].scatter(B_coc2_x, B_coc2_y, s=1, marker='.', edgecolors='blue', facecolors='none')
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


def plot_tft(sheet_ID, threshold):
    pdi = plot_defect_info(sheet_ID)
    
    colors = ['white', 'black']
    cmap = mcolors.LinearSegmentedColormap.from_list('CMAP', colors)

    lum_fs = grid_fs(client=pdi.MongoDBClient, db_name=pdi.DB, collection='LUM_SummaryTable')
    tft_newest_df = pdi.get_TFT_CreateTime_df()
    ct = tft_newest_df['CreateTime'][0]
    OPID = tft_newest_df['OPID'][0]
    R_light_ID = pdi.get_specific_object_id(tft_newest_df, 'LightingCheck_2D', 'R', 'L255')
    G_light_ID = pdi.get_specific_object_id(tft_newest_df, 'LightingCheck_2D', 'G', 'L255')
    B_light_ID = pdi.get_specific_object_id(tft_newest_df, 'LightingCheck_2D', 'B', 'L255')
    
    R_lum_ID = pdi.get_specific_object_id(tft_newest_df, 'Luminance_2D', 'R', 'L255')
    G_lum_ID = pdi.get_specific_object_id(tft_newest_df, 'Luminance_2D', 'G', 'L255')
    B_lum_ID = pdi.get_specific_object_id(tft_newest_df, 'Luminance_2D', 'B', 'L255')
    
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
    axs[0, 0].scatter(R_tft_x, R_tft_y, s=1, marker='.', edgecolors='lightcoral', facecolors='none')
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
    axs[1, 0].scatter(G_tft_x, G_tft_y, s=1, marker='.', edgecolors='mediumseagreen', facecolors='none')
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
    axs[2, 0].scatter(B_tft_x, B_tft_y, s=1, marker='.', edgecolors='blue', facecolors='none')
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
    
    fig_tft.suptitle(f'{ct} {sheet_ID} {OPID} P L255 Defect Info', fontsize=font_size, y=1.05)
    

    fig_full, axs2 = plt.subplots(1, 2)
    # white TFT
    fig_full.set_figheight(5)
    fig_full.set_figwidth(15)
    tft_white = R_tft_defect_arr + G_tft_defect_arr + B_tft_defect_arr
    tft_white_cluster = np.where(tft_white > 0, 1, 0)

    axs2[0].set_xlim([x, 0])
    axs2[0].set_ylim([0, y])
    axs2[0].scatter(R_tft_x, R_tft_y, s=1, marker='.', edgecolors='lightcoral', facecolors='none')
    axs2[0].scatter(G_tft_x, G_tft_y, s=1, marker='.', edgecolors='mediumseagreen', facecolors='none')
    axs2[0].scatter(B_tft_x, B_tft_y, s=1, marker='.', edgecolors='blue', facecolors='none')
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
    

def main(sheet_ID:str, threshold:int, option:str):
    options = ["COC2", "TFT", "TFT+COC2"]
    
    if threshold is not isinstance(threshold, int):
        threshold = int(threshold)
        
    if option==options[1]:
        fig_list = plot_tft(sheet_ID, threshold)
        return fig_list
        
    elif option==options[0]:
        fig_list = plot_coc2(sheet_ID, threshold, onlyCOC2=True)
    
    elif option==options[2]:
        fig_list = TFT_and_COC2(sheet_ID, threshold)
        
    return fig_list
    
    
        
        
if __name__ == '__main__':
    
    main(sheet_ID='VKV3457722A1812', threshold=20, option="TFT+COC2")
