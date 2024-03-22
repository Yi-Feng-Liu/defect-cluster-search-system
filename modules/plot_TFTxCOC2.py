from modules.utils import Mark_cluster, plot_defect_info, CornerOffDot
from modules.utils import grid_fs, CreatePPT, connect_MongoDB, conv_cluster
from modules.utils import calculateExecutedTime
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.style as mplstyle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import pickle
import six
from bson import ObjectId
from scipy import signal

mplstyle.use('fast')

### Confing ###
font_size = 16
bottom_gap = -0.2

# matplot 
RED = 'red'
GREEN = 'mediumseagreen'
BLUE = 'blue'
DOT_SIZE = 10
NONE_FACECOLOR = 'none'
COC2_ARR_COL_NAME = 'arr_id'
TFT_LIGHT_ON_ARR_COL_NAME = 'LightingCheck_2D'
TFT_LUMINANCE_COL_NAME = 'Luminance_2D'
DOT = '.'

led_types = ['R', 'G', 'B']
color_dict = {
    'R': RED,
    'G': GREEN,
    'B': BLUE
}
### Confing ###


def flip_arr_bottom_right(arr):
    return np.flip(np.flip(arr, 0), 1)

    
def getWhitePanel(RGBdefect_arr_ls: list, useWhere: bool|None=True) -> np.ndarray:
    white = np.zeros_like(RGBdefect_arr_ls[0])
    for color_defect_arr in RGBdefect_arr_ls:
        white += color_defect_arr
        
    if useWhere:
        white_cluster = np.where(white > 0, 1, 0)
        return white_cluster
    return white


def plotConvCluster(sheetID, arr, kernelSize, threshold, pixelType):
    cc = conv_cluster(arr, kernelSize, threshold)
    convImage = cc.convImage()
    res = cc.checkOutofSpec(convImage)
    convFig = cc.plotConvClusterFrame(sheetID, convImage, pixelType)
    return res, convFig

### AT Charge Map ###

class Chargemap:
    
    def __init__(self, chip_id):
        
        # 以 chip 當根源找
        self.chip_id = chip_id
        # 連結資料庫
        self.collection_charge2d = connect_MongoDB(client='mongodb://wma:mamcb1@10.88.26.102:27017', db_name="AT", collection="4A_charge2d")  
        self.collection_defectinfo = connect_MongoDB(client='mongodb://wma:mamcb1@10.88.26.102:27017', db_name="AT", collection="4A_defectinfo")            
        self.fs_charge2d = grid_fs(client='mongodb://wma:mamcb1@10.88.26.102:27017', db_name="AT", collection="4A_charge2d")
        self.fs_defectinfo = grid_fs(client='mongodb://wma:mamcb1@10.88.26.102:27017', db_name="AT", collection="4A_defectinfo")

        # query 資料庫
        self.df_defectinfo = pd.DataFrame.from_records(self.collection_defectinfo.find({'chip_id':self.chip_id}))
        self.df_defectinfo = self.df_defectinfo.drop(columns=["_id"])
        self.df_defectinfo['lm_time'] = pd.to_datetime(self.df_defectinfo['lm_time'], format="%Y/%m/%d %H:%M:%S.%f")  
        self.df_charge2d = pd.DataFrame.from_records(self.collection_charge2d.find({'chip_id':self.chip_id}))
        self.df_charge2d = self.df_charge2d.drop(columns=["_id"])
        self.df_charge2d['lm_time'] = pd.to_datetime(self.df_charge2d['lm_time'], format="%Y/%m/%d %H:%M:%S.%f")  

        # 合併 2 db
        self.df_all = self.df_defectinfo.merge(self.df_charge2d,
                                            how='inner',
                                            left_on=['eqp_id',
                                                    'op_id',
                                                    'recipe_id',
                                                    'chip_id',
                                                    'chip_pos',
                                                    'ins_cnt']
                                            ,right_on=['eqp_id',
                                                    'op_id',
                                                    'recipe_id',
                                                    'chip_id',
                                                    'chip_pos',
                                                    'ins_cnt'])
        
        # self.df_all['lm_time_x'] = pd.to_datetime(self.df_all['lm_time_x'], format="%Y/%m/%d %H:%M:%S.%f")        

    def get_recipes(self):
        
        return list(set(self.df_all['recipe_id'].tolist()))  
    
    def get_steps(self, ins_cnt, recipe_id, start_date):
        
        df = self.df_all[(self.df_all['recipe_id'] == recipe_id) & (self.df_all['ins_cnt'] == ins_cnt) & (self.df_all['lm_time_x'].dt.date == pd.to_datetime(start_date).date())]
        temp = list(set(df['step'].tolist()))
        temp = [[s,int(s.split("_")[0][4:])] for s in temp]
        temp.sort(key=lambda x: x[1])
        
        return [i[0] for i in temp]

    def get_retest_and_time(self, recipe_id):

        df = self.df_all[self.df_all["recipe_id"]==recipe_id]
        
        if len(df) > 0:         
            temp = df["ins_cnt"] + " - " + df["chip_start_time"]
            temp = list(temp.unique())
        else:
            temp = []

        temp.sort(key = lambda x: int(x.split("-")[0][:-1]))
        
        return temp

    def plot_chargemap_img(self, step, ins_cnt, recipe_id, start_date, ct_rect_step_lst = [], ct_rect_retest_lst = [], ct_rect_recipe_lst = [], ct_rect_time_lst = []):
        
        if ct_rect_step_lst and ct_rect_retest_lst and ct_rect_recipe_lst and ct_rect_time_lst:
            
            #多片
            df_ct = self.df_charge2d[(self.df_charge2d['step'] == ct_rect_step_lst[0]) &\
                                (self.df_charge2d['ins_cnt'] == ct_rect_retest_lst[0]) &\
                                (self.df_charge2d['recipe_id'] == ct_rect_recipe_lst[0]) &\
                                (self.df_charge2d['lm_time'].dt.date == pd.to_datetime(ct_rect_time_lst[0]).date())]
            df_rect = self.df_charge2d[(self.df_charge2d['step'] == ct_rect_step_lst[1]) &\
                                (self.df_charge2d['ins_cnt'] == ct_rect_retest_lst[1]) &\
                                (self.df_charge2d['recipe_id'] == ct_rect_recipe_lst[1]) &\
                                (self.df_charge2d['lm_time'].dt.date == pd.to_datetime(ct_rect_time_lst[1]).date())]

            bgr_ary, avg_ary, max_ary, min_ary, rng_ary = [], [], [], [], []

            for i in range(3):
                
                if i == 0: color = "B: "
                elif i == 1: color = "G: "
                elif i == 2: color = "R: "
                
                arr_ct = self.fs_charge2d.get(ObjectId(df_ct[['2d_b_object_id','2d_g_object_id','2d_r_object_id']].iloc[0,i])).read()
                arr_ct = pickle.loads(arr_ct)

                arr_rect = self.fs_charge2d.get(ObjectId(df_rect[['2d_b_object_id','2d_g_object_id','2d_r_object_id']].iloc[0,i])).read()
                arr_rect = pickle.loads(arr_rect)  
                
                arr = np.subtract(arr_ct, arr_rect)

                # transform V160 90 degree
                if len(arr) == 540: 
                    arr = np.fliplr(arr)
                    arr = arr.T
                    
                bgr_ary.append(arr)
                avg_ary.append(color + str(round(arr.mean(axis=1).mean(),1)))
                max_ary.append(color + str(round(arr.max(axis=1).max(),1)))
                min_ary.append(color + str(round(arr.min(axis=1).min(),1)))
                rng_ary.append(color + str(round(arr.max(axis=1).max()-arr.min(axis=1).min(),1)))        
            
            # calculate the chargemap value in R/G/B
            data_dict = {'Average': avg_ary[::-1], 'Max': max_ary[::-1], 'Min': min_ary[::-1], 'Range': rng_ary[::-1]}
            df_chargemap = pd.DataFrame(data_dict, columns=['Average', 'Max', 'Min', 'Range'])

            fig_full,axs = plt.subplots(1,4,gridspec_kw={'width_ratios': [3,3,3,2]})
            fig_full.suptitle(f'Chargemap: {recipe_id}_{self.chip_id}', fontsize=20, fontweight='bold')
            fig_full.set_figwidth(25)
                                            
        else:
            # 單片
            df_charge2d = self.df_charge2d[(self.df_charge2d['step'] == step) & (self.df_charge2d['recipe_id'] == recipe_id) & (self.df_charge2d['ins_cnt'] == ins_cnt) & (self.df_charge2d['lm_time'].dt.date == pd.to_datetime(start_date).date())]
            
            df_defectinfo = self.df_defectinfo[(self.df_defectinfo['recipe_id'] == recipe_id) & (self.df_defectinfo['ins_cnt'] == ins_cnt) & (self.df_defectinfo['lm_time'].dt.date == pd.to_datetime(start_date).date())]            

            bgr_ary, avg_ary, max_ary, min_ary, rng_ary = [], [], [], [], []

            for i in range(3):
                
                if i == 0: color = "B: "
                elif i == 1: color = "G: "
                elif i == 2: color = "R: "
                
                arr = self.fs_charge2d.get(ObjectId(df_charge2d[['2d_b_object_id','2d_g_object_id','2d_r_object_id']].iloc[0,i])).read()
                arr = pickle.loads(arr)

                # transform V160 90 degree
                if len(arr) == 540: 
                    arr = np.fliplr(arr)
                    arr = arr.T
                    
                bgr_ary.append(arr)
                avg_ary.append(color + str(round(arr.mean(axis=1).mean(),1)))
                max_ary.append(color + str(round(arr.max(axis=1).max(),1)))
                min_ary.append(color + str(round(arr.min(axis=1).min(),1)))
                rng_ary.append(color + str(round(arr.max(axis=1).max()-arr.min(axis=1).min(),1)))
            
            # calculate the chargemap value in R/G/B
            data_dict = {'Average': avg_ary[::-1], 'Max': max_ary[::-1], 'Min': min_ary[::-1], 'Range': rng_ary[::-1]}
            df_chargemap = pd.DataFrame(data_dict, columns=['Average', 'Max', 'Min', 'Range'])

            # get BIN code
            BIN = df_defectinfo['BIN'].values[0]

            fig_full,axs = plt.subplots(1,4,gridspec_kw={'width_ratios': [3,3,3,2]})
            fig_full.suptitle(f'Chargemap: {recipe_id}_{self.chip_id}_{step}_BIN{BIN}_Retest{ins_cnt}', fontsize=20, fontweight='bold')
            fig_full.set_figwidth(25)

        if len(arr) in [270,600,720]:
            fig_full.set_figheight(4)
        elif len(arr) == 240:
            fig_full.set_figheight(3.5)
        elif len(arr) == 156:
            fig_full.set_figheight(2.75)                        
        
        # 繪製三張 cahrge 圖
        for i in range(3):
            
            if i == 0: color = "Reds"
            elif i == 1: color = "Greens"
            elif i == 2: color = "Blues"
        
            axs[i].xaxis.tick_top()
            m = axs[i].imshow(bgr_ary[2-i], cmap=color)
            divider = make_axes_locatable(axs[i])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig_full.colorbar(m, cax=cax, orientation='vertical')  

        arr_chargemap = df_chargemap.to_numpy()

        axs[3].table(cellText=arr_chargemap, 
                    colLabels=df_chargemap.columns,
                    colColours = ["#c1bebe"]*len(arr_chargemap[0]),
                    cellLoc = 'left', 
                    loc = 'center',
                    bbox = [0.05, 0.2, 1, 0.5]
                    )        
        
        axs[3].axis('off')
        
        return fig_full

    def plot_defectmap_img(self, step, ins_cnt, start_date, recipe_id):
        
        df = self.df_all[(self.df_all['step'] == step) & (self.df_all['recipe_id'] == recipe_id) & (self.df_all['ins_cnt'] == ins_cnt) & (self.df_all['lm_time_x'].dt.date == pd.to_datetime(start_date).date())]

        # get BIN code
        BIN = df['BIN'].values[0]    
        
        if len(df):
            
            # 依據產品別去做 width 和 height 的極值設定
            product = df['recipe_id'].values[0][:4]
            
            if product == "Y136":
                W = 1440/3
                H = 270
            elif product == "Y173":
                W = 3840/3
                H = 720
            elif product == "V160":
                W = 720/3
                H = 540   
            elif product == "Z300":
                W = 2070/3
                H = 156  
            elif product == "Z123":
                W = 4800/3
                H = 600                    
            
            # 讀取 defectmap 位置的 dataframe
            df_defect = self.fs_defectinfo.get(ObjectId(df['df_defect'].values[0])).read()
            df_defect = pickle.loads(df_defect)
            df_defect["Color"] = df_defect["LED_Type"].map(lambda x: x.lower())  
            df_defect = df_defect[df_defect["Step"]==step]
            df_defect["Source"] = df_defect["Source"].map(lambda x: x//3)

            # 畫圖
            fig_full,axs = plt.subplots(1,5,gridspec_kw={'width_ratios': [1.5,1.5,1.5,1.5,1]})
            fig_full.suptitle(f'Defectmap: {recipe_id}_{self.chip_id}_{step}_BIN{BIN}_Retest{ins_cnt}', fontsize=20, fontweight='bold', y=1.12)   
            fig_full.set_figwidth(25)
            fig_full.set_figheight(4)    

            # 畫 R/G/B 各一張
            for i in range(3):
                
                if i == 0: LED_TYPE = "R"
                elif i == 1: LED_TYPE = "G"
                elif i == 2: LED_TYPE = "B"
            
                axs[i].xaxis.tick_top()
                df_plot = df_defect[df_defect["LED_Type"]==LED_TYPE]
                
                if len(df_plot):
                    if product == "V160":
                        df_plot.plot.scatter(
                            x='Gate',
                            y='Source',
                            c=LED_TYPE.lower(),
                            ax=axs[i]
                        )
                        axs[i].set_ylim(W,0)
                        axs[i].set_xlim(0,H)                          
                    else:
                        df_plot.plot.scatter(
                            x='Source',
                            y='Gate',
                            c=LED_TYPE.lower(),
                            ax=axs[i]
                        )
                        axs[i].invert_yaxis()
                        axs[i].set_xlim(0,W)
                        axs[i].set_ylim(H,0)            
            
            # R/G/B 合成一張圖
            axs[3].xaxis.tick_top()
            if product == "V160":
                df_defect.plot.scatter(
                    x='Gate',
                    y='Source',
                    c=df_defect["Color"],
                    ax=axs[3]
                )
                axs[3].set_ylim(W,0)
                axs[3].set_xlim(0,H)    
            else:               
                df_defect.plot.scatter(
                    x='Source',
                    y='Gate',
                    c=df_defect["Color"],
                    ax=axs[3]
                )
                axs[3].invert_yaxis()
                axs[3].set_xlim(0,W)
                axs[3].set_ylim(H,0)
            
            # 產生 defect count 表
            df_rgb_cnt = df_defect[df_defect["Step"]==step].pivot_table(
                values='Value', 
                index='Defect_code', 
                columns='LED_Type', 
                aggfunc=len, 
                fill_value=0
            )
            df_rgb_cnt = df_rgb_cnt.reset_index()
            df_rgb_cnt = df_rgb_cnt.rename_axis(None, axis=1)
            df_rgb_cnt.loc["Total"] = df_rgb_cnt.sum()
            df_rgb_cnt.iat[len(df_rgb_cnt)-1,0] = "Total"

            # 繪製 defect count 表
            table = axs[4].table(
                cellText=df_rgb_cnt.values,
                colLabels=df_rgb_cnt.columns,
                colColours = ["#c1bebe"]*4,
                cellLoc = 'center',
                loc = 'center',
                bbox = [-0.1, 0.4, 1.2, 0.5]
            )
            
            # 把 Total 那列標底橘色
            for k, cell in six.iteritems(table._cells):
                if k[0] == len(df_rgb_cnt): cell.set_facecolor('#E56B51')            
            
            axs[4].axis('off')
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.auto_set_column_width(col=list(range(len(df_rgb_cnt.columns))))
            
            return fig_full    
        
    def get_total_defect_count(self, ins_cnt, start_date, recipe_id):
        
        df = self.df_all[(self.df_all['recipe_id'] == recipe_id) & (self.df_all['ins_cnt'] == ins_cnt) & (self.df_all['lm_time_x'].dt.date == pd.to_datetime(start_date).date())]
        
        df_defect = self.fs_defectinfo.get(ObjectId(df['df_defect'].values[0])).read()
        df_defect = pickle.loads(df_defect)

        data = [['', 'B', 'G', 'R'],
                ['Total Defect Count']]
                
        for color in ['B','G','R']:
            
            data[1].append(len(df_defect[df_defect["LED_Type"]==color].drop_duplicates(["Source","Gate"])))
        
        fig, ax = plt.subplots(figsize=(5, 0.4))

        table = plt.table(cellText=data, loc='center', cellLoc='center')
        table.set_fontsize(20)
        table.scale(1.5, 1.5)

        ax.axis('off')
        plt.subplots_adjust(left=0.1, right=0.9, top=1, bottom=0.1) 

        return fig  

    def at2mb(self, df, product):
        
        if product == "Y136":
            df["X(mb)"] = df["Source"].apply(lambda x: (x-1)//3+1)
            df["Y(mb)"] = df["Gate"].apply(lambda x: 271-x)
            
        elif product == "Y173":  
            df["X(mb)"] = df["Source"].apply(lambda x: (x-1)//3+1)
            df["Y(mb)"] = df["Gate"].apply(lambda x: 721-x)            
        
        elif product == "V160":
            df["X(mb)"] = df["Gate"]
            df["Y(mb)"] = df["Source"].apply(lambda x: 240-(x-1)//3)

        elif product == "Z300": 
            df["X(mb)"] = df["Source"].apply(lambda x: 690-(x-1)//3)
            df["Y(mb)"] = df["Gate"]        

        elif product == "Z123": 
            df["X(mb)"] = df["Source"].apply(lambda x: (x-1)//3+1)
            df["Y(mb)"] = df["Gate"] 
                
        return df

    def mb2at(self, product, mbx, mby):
        
        if product == "Y136":
            atx = mbx
            aty = 271 - mby              
            
        elif product == "Y173":  
            atx = mbx
            aty = 721 - mby                        
        
        elif product == "V160":
            atx = mby
            aty = 241 - mbx

        elif product == "Z300": 
            atx = mby
            aty = 691 - mbx 

        elif product == "Z123": 
            atx = mbx
            aty = mby            
            
        return atx, aty

    def gen_defect_summary(self, retest_lst, time_lst, recipe_lst, product = ""):
        
        if len(retest_lst) == 1:

            df = self.df_all[(self.df_all['recipe_id'] == recipe_lst[0]) & (self.df_all['ins_cnt'] == retest_lst[0]) & (self.df_all['lm_time_x'].dt.date == pd.to_datetime(time_lst[0]).date())]
            
            df = self.fs_defectinfo.get(ObjectId(df['df_defect'].values[0])).read()
            df = pickle.loads(df)
            
            df = self.at2mb(df, product)
            
            df_defect_cnt = df.groupby(['Step','Defect_code','LED_Type'])['Value'].count().reset_index(name='Count')
            
            df = df.reset_index(drop=True)
            df_defect_cnt = df_defect_cnt.reset_index(drop=True)
                        
            return [df,df_defect_cnt]            
        
        else:
            
            # CT
            df_ct = self.df_all[(self.df_all['recipe_id'] == recipe_lst[0]) & (self.df_all['ins_cnt'] == retest_lst[0]) & (self.df_all['lm_time_x'].dt.date == pd.to_datetime(time_lst[0]).date())]
            
            df_ct = self.fs_defectinfo.get(ObjectId(df_ct['df_defect'].values[0])).read()
            df_ct = pickle.loads(df_ct)

            df_ct_defect_cnt = df_ct.groupby(['Step','Defect_code','LED_Type'])['Value'].count().reset_index(name='Count')
            
            df_ct = df_ct.reset_index(drop=True)
            df_ct_defect_cnt = df_ct_defect_cnt.reset_index(drop=True)

            # RE-CT
            df_rect = self.df_all[(self.df_all['recipe_id'] == recipe_lst[1]) & (self.df_all['ins_cnt'] == retest_lst[1]) & (self.df_all['lm_time_x'].dt.date == pd.to_datetime(time_lst[1]).date())]
            
            df_rect = self.fs_defectinfo.get(ObjectId(df_rect['df_defect'].values[0])).read()
            df_rect = pickle.loads(df_rect)

            df_rect_defect_cnt = df_rect.groupby(['Step','Defect_code','LED_Type'])['Value'].count().reset_index(name='Count')
            
            df_rect = df_rect.reset_index(drop=True)
            df_rect_defect_cnt = df_rect_defect_cnt.reset_index(drop=True)

            # for defect 座標
            df_only_ct = df_ct.merge(df_rect,indicator = True, how='outer').query("_merge == 'left_only'").drop('_merge', axis=1)
            df_only_rect = df_ct.merge(df_rect,indicator = True, how='outer').query("_merge == 'right_only'").drop('_merge', axis=1)        
            df_both = df_ct.merge(df_rect,indicator = True, how='outer').query("_merge == 'both'").drop('_merge', axis=1)
            
            # 轉出 mb 座標
            df_only_ct = self.at2mb(df_only_ct, product)
            df_only_rect = self.at2mb(df_only_rect, product)
            df_both = self.at2mb(df_both, product)
            
            # for defect count
            df_ct_defect_cnt = df_only_ct.groupby(['Step','Defect_code','LED_Type'])['Value'].count().reset_index(name='Count')
            df_rect_defect_cnt = df_only_rect.groupby(['Step','Defect_code','LED_Type'])['Value'].count().reset_index(name='Count')
            df_both_defect_cnt = df_both.groupby(['Step','Defect_code','LED_Type'])['Value'].count().reset_index(name='Count')   
            
            df_only_ct = df_only_ct.reset_index(drop=True)
            df_only_rect = df_only_rect.reset_index(drop=True)
            df_both = df_both.reset_index(drop=True)
            df_ct_defect_cnt = df_ct_defect_cnt.reset_index(drop=True)
            df_rect_defect_cnt = df_rect_defect_cnt.reset_index(drop=True)
            df_both_defect_cnt = df_both_defect_cnt.reset_index(drop=True)
            
            return [df_only_ct,df_only_rect,df_both,df_ct_defect_cnt,df_rect_defect_cnt,df_both_defect_cnt]  

    def get_charge_value(self, lbj_id, at_x, at_y):
        
        arr = self.fs_charge2d.get(ObjectId(lbj_id)).read()
        arr = pickle.loads(arr)   
        
        return arr[at_x-1, at_y-1]
    
#####################
@calculateExecutedTime
def TFT_and_COC2(
    sheet_ID: str, 
    threshold: int, 
    Ins_type: str, 
    TFT_df = None|pd.DataFrame, 
    OPID = None|pd.DataFrame,
):
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
        fig_tft, _, _ = plot_tft(sheet_ID=sheet_ID, threshold=threshold, Ins_type=Ins_type, TFT_df=tft_newest_df)
        
    # for multiple sheet id. only select specific opid dataframe
    else:
        tft_newest_df = pdi.get_TFT_CreateTime_df() 
        sort_col_ls = ['CreateTime', 'LED_TYPE', 'Inspection_Type']
        tft_newest_df = tft_newest_df.sort_values(by=sort_col_ls, ascending=False)
        tft_newest_df = tft_newest_df.drop_duplicates(duplicate_col, keep='first').reset_index(drop=True)
        fig_tft, _, _ = plot_tft(sheet_ID=sheet_ID, threshold=threshold, Ins_type=Ins_type, OPID=OPID, recipe="")

    del duplicate_col, sort_col_ls
    
    # tft + coc2
    fig_col = 2
    fig_tft_coc2, axs2 = plt.subplots(fig_col, 5)
    fig_tft_coc2.set_figheight(5)
    fig_tft_coc2.set_figwidth(17.5)
    
    coc2_df = pdi.get_COC2_df()
    coc2_id = coc2_df['SHEET_ID'][0]
    
    lum_arr = None
    coc2_defect_arr_ls = []
    tft_defect_arr_ls = []
    for col in range(fig_col):
        for i in range(len(led_types)):
            # tft params 
            light_ID = pdi.get_specific_object_id(tft_newest_df, TFT_LIGHT_ON_ARR_COL_NAME, led_types[i], Ins_type)    
            lum_ID = pdi.get_specific_object_id(tft_newest_df, TFT_LUMINANCE_COL_NAME, led_types[i], Ins_type)
            tft_x, tft_y = pdi.get_defect_coord(object_id=light_ID, fs=lum_fs, coc2=False)
            lum_arr, _, _, _ = pdi.get_heat_map_imshow_params(object_id=lum_ID, LED_TYPE=led_types[i], fs=lum_fs)
            tft_defect_arr, _ = pdi.get_defect_imshow_params(object_id=light_ID, LED_TYPE=led_types[i], fs=lum_fs, coc2=False)
            tft_defect_arr = flip_arr_bottom_right(tft_defect_arr)
            lum_arr = flip_arr_bottom_right(lum_arr)
            tft_defect_arr_ls.append(tft_defect_arr)
            
            # COC2 params
            COC2_ID = pdi.get_specific_object_id(coc2_df, COC2_ARR_COL_NAME, led_types[i], Ins_tpye=None)
            coc2_x, coc2_y = pdi.get_defect_coord(object_id=COC2_ID, fs=coc2_fs, coc2=True)
            coc2_defect_arr, _ = pdi.get_defect_imshow_params(object_id=COC2_ID, LED_TYPE=led_types[i], fs=coc2_fs, coc2=True)
            
            coc2_defect_arr_ls.append(coc2_defect_arr)
            
            y, x = lum_arr.shape
            # plot COC2
            if col == 0:
                axs2[col, i].set_xlim([0, x])
                axs2[col, i].set_ylim([0, y])
                
                # plot RGB, repectively
                axs2[col, i].scatter(
                    coc2_x, coc2_y, s=DOT_SIZE, marker=DOT, edgecolors=color_dict.get(led_types[i]), facecolors=NONE_FACECOLOR
                )
                # plot RGB together 
                axs2[col, 3].scatter(
                    coc2_x, coc2_y, s=DOT_SIZE, marker=DOT, edgecolors=color_dict.get(led_types[i]), facecolors=NONE_FACECOLOR
                )
                
                axs2[col, i].set_title(f'COC2 {led_types[i]} Defect Map', y=bottom_gap)
                
            # plot TFT    
            elif col == 1:
                # plot RGB, repectively
                axs2[col, i].scatter(
                    tft_x, tft_y, s=DOT_SIZE, marker=DOT, edgecolors=color_dict.get(led_types[i]), facecolors=NONE_FACECOLOR
                )
                # plot RGB together 
                axs2[col, 3].scatter(
                    tft_x, tft_y, s=DOT_SIZE, marker=DOT, edgecolors=color_dict.get(led_types[i]), facecolors=NONE_FACECOLOR
                )
                
                axs2[col, i].set_title(f'TFT {led_types[i]} Defect Map', y=bottom_gap)
                axs2[col, i].set_xlim([x, 0])
                axs2[col, i].invert_xaxis()
                
            axs2[col, i].invert_yaxis()
            axs2[col, i].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)    
            axs2[col, 3].invert_yaxis()
            axs2[col, 3].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
            
    # white COC2
    coc2_white = getWhitePanel(RGBdefect_arr_ls=coc2_defect_arr_ls, useWhere=True)
    mark_cluster = Mark_cluster(coc2_white, threshold, axs2[0,4])
    
    axs2[0,4] = mark_cluster(coc2_white, threshold, axs2[0,4])
    axs2[0,4].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    axs2[0,4].set_title('Panel Defect Cluster Detectation', y=bottom_gap)
    axs2[0,4].set_title('COC2 White Defect Map', y=bottom_gap)
    axs2[0,4].imshow(coc2_white, cmap=cmap, aspect='auto')

    # white TFT
    tft_white = getWhitePanel(RGBdefect_arr_ls=tft_defect_arr_ls, useWhere=True)
    axs2[1,4] = mark_cluster(tft_white, threshold, axs2[1,4])
    axs2[1,4].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    axs2[1,4].set_title('Panel Defect Cluster Detectation', y=bottom_gap)
    axs2[1,4].imshow(tft_white, cmap=cmap, aspect='auto')
    axs2[1,4].set_title('TFT White Defect Map', y=bottom_gap)
    fig_tft_coc2.suptitle(f'{sheet_ID}_{coc2_id} Comparison', fontsize=font_size, y=1)
    fig_tft_coc2.tight_layout()
    
    fig_coc2 = plot_singleChip_coc2(sheet_ID, threshold, onlyCOC2=False, Chip=None)
    
    figlist = [fig_tft_coc2] + fig_tft + fig_coc2

    return figlist

# using np.vstack將Chip1跟Chip2的矩陣上下疊起來 形成一張圖
# 如果有coc2 有 chip-1 & chip-2之分 則使用這個function將兩個chip整合
def add_offset_for_coord_list(list1:list, offsetValue:int) -> list:
    coordList = [i + offsetValue for i in list1]
    return coordList
  
    
@calculateExecutedTime   
def plot_fullChip_coc2(sheet_ID:str, threshold:int, onlyCOC2:bool, layoutOffset: int):
    fig_coc2, axs3 = plt.subplots(1,2)
    fig_coc2.set_figheight(4)
    fig_coc2.set_figwidth(12)
    
    pdi = plot_defect_info(sheet_ID)
    
    colors = ['white', 'black']
    cmap = mcolors.LinearSegmentedColormap.from_list('CMAP', colors)
    
    coc2_fs = grid_fs(client=pdi.MongoDBClient, db_name=pdi.DB, collection='COC2_AOI_ARRAY')
    
    if onlyCOC2:
        coc2_df = pdi.get_COC2_df_without_TFT()
    else:
        coc2_df = pdi.get_COC2_df()
        
    chips = coc2_df['CHIP'].sort_values(ascending=True).unique()
    coc2_id = coc2_df['SHEET_ID'][0]
    coc2_opid = coc2_df['OPID'][0]
    
    total_xcoord, total_ycoord, coc2_chip_arr_ls = [], [], []
    complete_defect_arr_ls = []
    coc2_defect_arr = None
    
    imageOffset = 156 + layoutOffset
    for chip in chips:
        firstChipName = ['Chip-1', '11']
        coc2_spec_df = coc2_df[coc2_df['CHIP']==chip].reset_index(drop=True)
        
        for i in range(len(led_types)):
            COC2_ID = pdi.get_specific_object_id(coc2_spec_df, COC2_ARR_COL_NAME, led_types[i], Ins_tpye=None)
            coc2_x, coc2_y = pdi.get_defect_coord(object_id=COC2_ID, fs=coc2_fs, coc2=True)
            coc2_defect_arr, _ = pdi.get_defect_imshow_params(object_id=COC2_ID, LED_TYPE=led_types[i], fs=coc2_fs, coc2=True)

            if chip not in firstChipName:
                coc2_y = add_offset_for_coord_list(list1=coc2_y, offsetValue=imageOffset)
                
            total_xcoord += coc2_x
            total_ycoord += coc2_y
            
            # stack chips 
            coc2_chip_arr_ls.append(coc2_defect_arr)
            
            # patch offset
            x_size = coc2_defect_arr.shape[1]
            patchMatrix = np.zeros((layoutOffset, x_size), dtype=np.uint8) 
            coc2_chip_arr_ls.insert(1, patchMatrix)
            
            # plot RGB scatter
            axs3[0].scatter(total_xcoord, total_ycoord, s=DOT_SIZE, marker=DOT, edgecolors=color_dict.get(led_types[i]), facecolors=NONE_FACECOLOR)
            
            complete_defect_arr = np.vstack(coc2_chip_arr_ls)
            complete_defect_arr_ls.append(complete_defect_arr)
        
    
    complete_defect_arr = np.vstack(coc2_chip_arr_ls)
    y, x = complete_defect_arr.shape

    coc2_white = getWhitePanel(RGBdefect_arr_ls=complete_defect_arr_ls, useWhere=True)
    mark_cluster = Mark_cluster(coc2_white, threshold, axs3[1])
    axs3[0].set_xlim([0, x])
    axs3[0].set_ylim([y, 0])
    axs3[0].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    axs3[0].set_title('COC2 White Full Defect Map', y=bottom_gap)
    
    axs3[1].set_xlim([0, x])
    axs3[1].set_ylim([0, y])
    axs3[1] = mark_cluster(coc2_white, threshold, axs3[1])
    axs3[1].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    axs3[1].set_title('Panel Defect Cluster Detectation', y=bottom_gap)
    axs3[1].imshow(coc2_white, cmap=cmap, aspect='auto')
    axs3[1].invert_yaxis()
    
    if coc2_opid == 'empty':
        coc2_opid = 'C2-ATO'
    if Chip == None:
        Chip = '' 
        
    fig_coc2.suptitle(f'COC2 {coc2_id} {coc2_opid} Full Defect Info', fontsize=font_size)
    fig_coc2.tight_layout()
    return [fig_coc2]

    
@calculateExecutedTime    
def plot_singleChip_coc2(
    sheet_ID: str, 
    threshold: int, 
    onlyCOC2: bool, 
    Chip = None|str
):
    fig_coc2, axs3 = plt.subplots(4,2)
    fig_coc2.set_figheight(8)
    fig_coc2.set_figwidth(DOT_SIZE)

    pdi = plot_defect_info(sheet_ID)
    
    colors = ['white', 'black']
    cmap = mcolors.LinearSegmentedColormap.from_list('CMAP', colors)
    
    coc2_fs = grid_fs(client=pdi.MongoDBClient, db_name=pdi.DB, collection='COC2_AOI_ARRAY')
    
    if onlyCOC2:
        coc2_df = pdi.get_COC2_df_without_TFT()
    else:
        coc2_df = pdi.get_COC2_df()
    
    if Chip != None:
        coc2_df = coc2_df[coc2_df['CHIP']==Chip].sort_values(by='CHIP', ascending=True).reset_index(drop=True)
    
    coc2_id = coc2_df['SHEET_ID'][0]
    coc2_opid = coc2_df['OPID'][0]
    
    defect_arr_ls = []
    for i in range(len(led_types)):
        COC2_ID = pdi.get_specific_object_id(coc2_df, COC2_ARR_COL_NAME, led_types[i], Ins_tpye=None)
        coc2_x, coc2_y = pdi.get_defect_coord(object_id=COC2_ID, fs=coc2_fs, coc2=True)
        coc2_defect_arr, coc2_defect_cmp = pdi.get_defect_imshow_params(object_id=COC2_ID, LED_TYPE=led_types[i], fs=coc2_fs, coc2=True)
        y, x = coc2_defect_arr.shape
        
        mark_cluster = Mark_cluster(coc2_defect_arr, threshold, axs3[i+1, 1])
        axs3[i+1, 0].set_xlim([0, x])
        axs3[i+1, 0].set_ylim([0, y])
        axs3[i+1, 0].scatter(coc2_x, coc2_y, s=DOT_SIZE, marker=DOT, edgecolors=color_dict.get(led_types[i]), facecolors=NONE_FACECOLOR)
        axs3[i+1, 0].set_title(f'{led_types[i]} Defect Scatter', y=bottom_gap)
        axs3[i+1, 0].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        axs3[i+1, 0].invert_yaxis()

        axs3[i+1, 1] = mark_cluster(coc2_defect_arr, threshold, axs3[i+1, 1])
        axs3[i+1, 1].imshow(coc2_defect_arr, cmap=coc2_defect_cmp, aspect='auto')
        axs3[i+1, 1].set_title(f'{led_types[i]} Defect Cluster Detectation', y=bottom_gap)
        axs3[i+1, 1].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        
        axs3[0, 0].scatter(coc2_x, coc2_y, s=DOT_SIZE, marker=DOT, edgecolors=color_dict.get(led_types[i]), facecolors=NONE_FACECOLOR)
        defect_arr_ls.append(coc2_defect_arr)
        
    # white COC2
    coc2_white = getWhitePanel(RGBdefect_arr_ls=defect_arr_ls,  useWhere=True)
    mark_cluster = Mark_cluster(coc2_white, threshold, axs3[0,1])
    axs3[0, 0].set_xlim([0, x])
    axs3[0, 0].set_ylim([y, 0])
    axs3[0, 0].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    axs3[0, 0].set_title('COC2 White Defect Map', y=bottom_gap)

    axs3[0, 1].set_xlim([0, x])
    axs3[0, 1].set_ylim([0, y])
    axs3[0, 1] = mark_cluster(coc2_white, threshold, axs3[0,1])
    axs3[0, 1].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    axs3[0, 1].set_title('Panel Defect Cluster Detectation', y=bottom_gap)
    axs3[0, 1].imshow(coc2_white, cmap=cmap, aspect='auto')
    axs3[0, 1].invert_yaxis()
    
    if coc2_opid == 'empty':
        coc2_opid = 'C2-ATO'
    if Chip == None:
        Chip = '' 
        
    fig_coc2.suptitle(f'COC2 {coc2_id} {coc2_opid} {Chip} Defect Info', fontsize=font_size)
    fig_coc2.tight_layout()
    
    return [fig_coc2]


def combineRGBDefectMartrix(pdi: object, df: pd.DataFrame, fs:object, Ins_type: str):
    defect_arr_ls = []
    for i in range(len(led_types)):
        light_ID = pdi.get_specific_object_id(df, TFT_LIGHT_ON_ARR_COL_NAME, led_types[i], Ins_type)
        tft_defect_arr, _ = pdi.get_defect_imshow_params(object_id=light_ID, LED_TYPE=led_types[i], fs=fs, coc2=False)
        tft_defect_arr = flip_arr_bottom_right(tft_defect_arr)
        defect_arr_ls.append(tft_defect_arr)
        
    tft_white = getWhitePanel(RGBdefect_arr_ls=defect_arr_ls,  useWhere=False)
    
    return tft_white
        
        
def combineTwoInstypeMartix(
    sheet_ID: str,
    df: pd.DataFrame, 
    kernelSize=4, 
    conv_threshold=4,
    Ins_type='L255 + L0'
):
    firstInstype, secondInstype = Ins_type.split('+')
    firstInstype = firstInstype.strip()
    secondInstype = secondInstype.strip()
    
    fisrt_df = df[df['Inspection_Type']==firstInstype]
    second_df = df[df['Inspection_Type']==secondInstype]
    
    duplicate_col = ['OPID', 'LED_TYPE', 'Inspection_Type']
    fisrt_df = fisrt_df.drop_duplicates(duplicate_col, keep='first')
    second_df = second_df.drop_duplicates(duplicate_col, keep='first')
    
    pdi = plot_defect_info(sheet_ID)

    lum_fs = grid_fs(client=pdi.MongoDBClient, db_name=pdi.DB, collection='LUM_SummaryTable')
    firstWhite = combineRGBDefectMartrix(pdi, fisrt_df, lum_fs, firstInstype)
    secondWhite = combineRGBDefectMartrix(pdi, second_df, lum_fs, secondInstype)
    
    combinedMatrix = firstWhite + secondWhite 
    # convolution cluster
    resSubPixel, subPixelConvFig = plotConvCluster(sheet_ID, combinedMatrix, kernelSize, conv_threshold, "Sub Pixel")
    pixel_tft_white = np.where(combinedMatrix > 0, 1, 0)
    resPixel, pixelConvFig = plotConvCluster(sheet_ID, pixel_tft_white, kernelSize, conv_threshold, "Pixel")
    return [subPixelConvFig, pixelConvFig], resSubPixel, resPixel

   
@calculateExecutedTime
def plot_tft(
    sheet_ID: str, 
    threshold: int, 
    Ins_type: str, 
    TFT_df = None|pd.DataFrame, 
    OPID = None|str,
    kernelSize = 4, 
    conv_threshold = 4,
    recipe = ""
):
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
        if recipe != "":
            tft_newest_df = tft_newest_df[tft_newest_df['ACTUAL_RECIPE']==recipe]
        
        sort_col_ls = ['CreateTime','LED_TYPE', 'Inspection_Type']
        tft_newest_df = tft_newest_df.sort_values(by=sort_col_ls, ascending=False)
        duplicate_col = ['CreateTime', 'OPID', 'LED_TYPE', 'Inspection_Type']
        tft_newest_df = tft_newest_df[tft_newest_df['OPID'] == OPID]
        newest_ct = tft_newest_df['CreateTime'].unique()[0]
        tft_newest_df = tft_newest_df[(tft_newest_df['CreateTime'] == newest_ct)]
        tft_newest_df = tft_newest_df.drop_duplicates(duplicate_col, keep='first').reset_index(drop=True)

    del duplicate_col, sort_col_ls
    
    ct = tft_newest_df['CreateTime'][0]
    OPID = tft_newest_df['OPID'][0]
    # grade = tft_newest_df['Grade'][0]
    
    fig_tft,  axs  = plt.subplots(len(led_types), 3)
    fig_tft.set_figheight(8)
    fig_tft.set_figwidth(15)
    
    fig_full, axs2 = plt.subplots(1, 2)
    fig_full.set_figheight(5)
    fig_full.set_figwidth(17.5)
    
    tft_white = None
    defect_arr_ls = []
    for i in range(len(led_types)):
        light_ID = pdi.get_specific_object_id(tft_newest_df, TFT_LIGHT_ON_ARR_COL_NAME, led_types[i], Ins_type)
        lum_ID = pdi.get_specific_object_id(tft_newest_df, TFT_LUMINANCE_COL_NAME, led_types[i], Ins_type)
        pattern_ls = pdi.get_NGCNT_Yield(df=tft_newest_df, LED_TYPE=led_types[i])
        tft_x, tft_y = pdi.get_defect_coord(object_id=light_ID, fs=lum_fs, coc2=False)
        lum_arr, color_cmap, lum_max, lum_min = pdi.get_heat_map_imshow_params(object_id=lum_ID, LED_TYPE=led_types[i], fs=lum_fs)
        tft_defect_arr, tft_defect_cmp = pdi.get_defect_imshow_params(object_id=light_ID, LED_TYPE=led_types[i], fs=lum_fs, coc2=False)
        tft_defect_arr = flip_arr_bottom_right(tft_defect_arr)
        lum_arr = flip_arr_bottom_right(lum_arr)
        
        y, x = lum_arr.shape
        
        mark_cluster = Mark_cluster(tft_defect_arr, threshold, axs[i, 1])
        axs[i, 0].set_xlim([x, 0])
        axs[i, 0].set_ylim([0, y])
        axs[i, 0].scatter(tft_x, tft_y, s=DOT_SIZE, marker=DOT, edgecolors=color_dict.get(led_types[i]), facecolors=NONE_FACECOLOR)
        axs[i, 0].set_title(f'{led_types[i]} Defect Scatter', y=bottom_gap)
        axs[i, 0].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        axs[i, 0].invert_xaxis()
        axs[i, 0].invert_yaxis()
        
        axs[i, 1] = mark_cluster(tft_defect_arr, threshold, axs[i, 1])
        axs[i, 1].imshow(tft_defect_arr, cmap=tft_defect_cmp, aspect='auto')
        axs[i, 1].set_title(f'{led_types[i]} Defect Cluster Detectation', y=bottom_gap)
        axs[i, 1].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

        axs[i, 2].imshow(lum_arr, cmap=color_cmap, vmin=lum_min, vmax=lum_max, aspect='auto')
        axs[i, 2].set_title(f'{led_types[i]} HeatMap', y=bottom_gap)
        axs[i, 2].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        plt.colorbar(axs[i, 2].imshow(lum_arr, cmap=color_cmap, vmin=lum_min, vmax=lum_max), ax=axs[i, 2])
        
        new_pattern_ls = [x for x in pattern_ls if x]
        
        if len(new_pattern_ls) == 4:
            axs[i, 0].table(
                cellText = new_pattern_ls, 
                colLabels = ['DefectCnt', 'Yield'],
                rowLabels = ['Normal', 'Edge', 'L0', 'L10'],
                cellLoc='center',
                bbox = [3.65, 0., 0.5, 1],
            )
        else:
            axs[i, 0].table(
                cellText = new_pattern_ls, 
                colLabels = ['DefectCnt', 'Yield'],
                rowLabels = ['Normal'],
                cellLoc='center',
                bbox = [3.65, 0., 0.5, 1],
            )

        
        axs2[0].scatter(tft_x, tft_y, s=DOT_SIZE, marker=DOT, edgecolors=color_dict.get(led_types[i]), facecolors=NONE_FACECOLOR)
        defect_arr_ls.append(tft_defect_arr)
    
    tft_white = getWhitePanel(RGBdefect_arr_ls=defect_arr_ls,  useWhere=False)
    # convolution cluster
    resSubPixel, subPixelConvFig = plotConvCluster(sheet_ID, tft_white, kernelSize, conv_threshold, "Sub Pixel")
    pixel_tft_white = np.where(tft_white > 0, 1, 0)
    resPixel, pixelConvFig = plotConvCluster(sheet_ID, pixel_tft_white, kernelSize, conv_threshold, "Pixel")
    
    axs2[0].set_xlim([x, 0])
    axs2[0].set_ylim([0, y])
    axs2[0].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    axs2[0].set_title('TFT White Defect Map', y=bottom_gap)
    axs2[0].invert_xaxis()
    axs2[0].invert_yaxis()
    
    axs2[1] = mark_cluster(pixel_tft_white, threshold, axs2[1])
    axs2[1].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    axs2[1].set_title('Panel Cluster Info', y=bottom_gap)
    axs2[1].imshow(pixel_tft_white, cmap=cmap, aspect='auto')
    
    fig_tft.subplots_adjust(left=0.05, right=0.98, bottom=0.1, top=0.9, wspace=0.2, hspace=0.5)
    fig_tft.suptitle(f'{ct} {sheet_ID} {OPID} {Ins_type} Defect Info', fontsize=font_size, y=1)
    fig_full.subplots_adjust(left=0.01, right=0.99, bottom=0.1, top=0.9, wspace=0.05, hspace=0.5)
    fig_full.suptitle(f'{sheet_ID}_{OPID} Result', fontsize=font_size, y=1)
    return [fig_tft, fig_full, subPixelConvFig, pixelConvFig], resSubPixel, resPixel


# plot shipping to M01 specific image
@calculateExecutedTime
def main_forShipping(
    df: pd.DataFrame, 
    SHEET_ID: str, 
    Ins_type: str,
    shipping2Client = False,
    kernelSize = 4,
    conv_threshold = 4
):
    pdi = plot_defect_info(SHEET_ID)
    df = df[df['SHEET_ID']==SHEET_ID].sort_values(['LED_TYPE'], ascending=False).reset_index(drop=True)
    OPID = df['OPID'].tolist()[0]
    CREATETIME = df['CreateTime'].tolist()[0]
    lum_fs = grid_fs(client=pdi.MongoDBClient, db_name=pdi.DB, collection='LUM_SummaryTable')
    
    ngcnt_ls = df['NGCNT'].tolist()
    total_ng = df['NGCNT'].sum()
    ngcnt_ls.append(total_ng)
    # Create table info each element type should be list
    ng_info_ls = [[i] for i in ngcnt_ls]
    del total_ng, ngcnt_ls
    
    pdi = plot_defect_info(sheet_ID=SHEET_ID)
    COC2df = pdi.get_BONDING_MAP_df()
    
    if len(COC2df.index) != 0:
        COC2 = COC2df['coc2'].tolist()[0]
    else:
        COC2 = "COC2 NOT FOUND"
    
    defect_arr_ls = []
        
    del COC2df
    
    fig_full, axs = plt.subplots(1, 1)
    fig_full.set_figheight(5)
    fig_full.set_figwidth(13)
    
    defect_arr_ls = []
    for i in range(len(led_types)):
        light_ID = pdi.get_specific_object_id(df, TFT_LIGHT_ON_ARR_COL_NAME, led_types[i], Ins_type)
        tft_defect_arr, _ = pdi.get_defect_imshow_params(object_id=light_ID, LED_TYPE=led_types[i], fs=lum_fs, coc2=False)
        y, x = tft_defect_arr.shape
        tft_x, tft_y = pdi.get_defect_coord(object_id=light_ID, fs=lum_fs, coc2=False)
        axs.scatter(tft_x, tft_y, s=20, marker=DOT, edgecolors=color_dict.get(led_types[i]), facecolors=NONE_FACECOLOR)
        defect_arr_ls.append(tft_defect_arr)
    
    temp_save_path = './modules/temp/temp_image.png'
        
    axs.set_xlim([x, 0])
    axs.set_ylim([0, y])
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
    
    create_ppt = CreatePPT()
    
    fig_full.suptitle(f'{CREATETIME}_{SHEET_ID}_{COC2}_{OPID}_{Ins_type} Result', fontsize=font_size, y=1)
    fig_full.savefig(temp_save_path, facecolor=None, bbox_inches='tight')
    
    convfig = None
    resSubPixel = None
    resPixel = None
    if shipping2Client == True:
        white = getWhitePanel(defect_arr_ls, False)
        mainPixelWhite = flip_arr_bottom_right(white)
        resSubPixel, convfig = plotConvCluster(SHEET_ID, mainPixelWhite, kernelSize, conv_threshold, "Sub Pixel")
        byPixelWhite = np.where(mainPixelWhite > 0, 1, 0)
        resPixel, pixelConvFig = plotConvCluster(SHEET_ID, byPixelWhite, kernelSize, conv_threshold, "Pixel")
        pixelConvFig.savefig(temp_save_path, facecolor=None, bbox_inches='tight')
        
    create_ppt(image_path=temp_save_path)
    
    if convfig != None:
        return [fig_full, convfig, pixelConvFig], resSubPixel, resPixel
    return [fig_full]
    
    
@calculateExecutedTime
def main(
    sheet_ID: str, 
    threshold: int, 
    option: str, 
    Ins_type = None|str, 
    TFT_df = None|pd.DataFrame, 
    OPID = None|str, 
    **kargs
):

    options = ["COC2", "TFT", "TFT+COC2", "AT", "TFT+AT", "TFT+COC2+AT"]
    resSubPixel = None
    resPixel = None
    recipe = None
    
    if threshold is not isinstance(threshold, int):
        threshold = int(threshold)
        
    if option==options[1] or option==options[4]:
        kernelSize = kargs.get('kernelSize', 4)
        conv_threshold = kargs.get('conv_threshold', 4)
        recipe = kargs.get('recipe')
        
        fig_list, resSubPixel, resPixel = plot_tft(
            sheet_ID = sheet_ID, 
            threshold = threshold, 
            Ins_type = Ins_type, 
            TFT_df = TFT_df, 
            OPID = OPID,
            kernelSize = kernelSize,
            conv_threshold = conv_threshold,
            recipe = recipe
        )

    
    elif option==options[0]:
        Chip = kargs.get('Chip', None)
        fig_list = plot_singleChip_coc2(
            sheet_ID, 
            threshold, 
            onlyCOC2 = True, 
            Chip = Chip
        )
    
    elif option==options[2] or option==options[5]:
        fig_list = TFT_and_COC2(
            sheet_ID, 
            threshold, 
            Ins_type, 
            TFT_df = TFT_df, 
            OPID = OPID,
        )
        
    else:
        raise KeyError(f"{option} Not in option List")
    
    return fig_list, resSubPixel, resPixel
    
