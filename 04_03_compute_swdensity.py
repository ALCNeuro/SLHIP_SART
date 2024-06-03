#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 15:50:28 2024

@author: arthurlecoz

04_03_compute_swdensity.py
"""
# %% Paths

# %% Script

print(f"\n...Processing {sub_id}... Computing SW Features")  
    thisFeaturePath = os.path.join(
        swDataPath, "slow_waves", "1_sess_features", 
        f"{sub_id}_sw_features_freq_{1/slope_range[1]}_{1/slope_range[0]}.csv"
        )
    if os.path.exists(thisFeaturePath) :
        df_feature = pd.read_csv(thisFeaturePath)
        del df_feature['Unnamed: 0']
        dfFeatureList.append(df_feature)
    else : 
        # if 1/df_clean.pos_halfway_period.min() > 4 :
        #     df_clean = df_clean.loc[
        #         (df_clean['pos_halfway_period'] >= slope_range[0])
        #         & (df_clean['pos_halfway_period'] <= slope_range[1])
        #         ]
        subidlist = []
        densitylist = []
        ptplist = []
        dslopelist = []
        uslopelist = []
        frequencylist = []
        sw_threshlist = []
    
        chanlist = []
        difficultylist = []
        sessionlist = []
        epochlist = []
        for i_sess, n_sess in enumerate(df_clean.n_sess.unique()):
            print(f"...Processing session {n_sess}...")
            for i_epoch, n_epoch in enumerate(
                    df_clean.n_epoch.unique()) :
                for chan in df_clean.chan_name.unique() :
                    temp_df = df_clean.loc[
                        (df_clean["n_sess"] == n_sess)
                        & (df_clean["n_epoch"] == n_epoch)
                        & (df_clean["chan_name"] == chan)
                        ]

                    n_wave = temp_df.shape[0]
                    densitylist.append(n_wave)
                    subidlist.append(sub_id)
                    epochlist.append(n_epoch)
                    difficultylist.append(difficulty)
                    sessionlist.append(str(int(n_sess)))
                    chanlist.append(chan)
                    if n_wave == 0 :
                        ptplist.append(np.nan)
                        dslopelist.append(np.nan)
                        uslopelist.append(np.nan)
                        frequencylist.append(np.nan)
                    else : 
                        ptplist.append(np.nanmean(temp_df.PTP))
                        frequencylist.append(np.nanmean(
                            1/temp_df.pos_halfway_period))
                        dslopelist.append(np.nanmean(
                            temp_df.inst_neg_1st_segment_slope))
                        uslopelist.append(np.nanmean(
                            temp_df.max_pos_slope_2nd_segment
                               ))
                
        df_feature = pd.DataFrame(
            {
             "sub_id" : subidlist,
             "difficulty" : difficultylist,
             "n_session" : sessionlist,
             "n_epoch" : epochlist,
             "channel" : chanlist,
             "density" : densitylist,
             "frequency" : frequencylist,
             "ptp" : ptplist,
             "d_slope" : dslopelist,
             "u_slope" : uslopelist
             }
            )
        df_feature.to_csv(thisFeaturePath)
        dfFeatureList.append(df_feature)

df = pd.concat(dfFeatureList)
df.to_csv(f"{swDataPath}{os.sep}df_allsw_exgausscrit_S1_freq_{1/slope_range[1]}_{1/slope_range[0]}.csv")
mean_df = df.groupby(
    by = ["sub_id", "difficulty", "n_session", "channel"], 
    as_index = False).mean()

channel_category = pd.Categorical(
    mean_df['channel'], 
    categories = config.channels, 
    ordered=True
    )

mean_df = mean_df.loc[channel_category.argsort()]
mean_df.to_csv(f"{swDataPath}/df_meansw_exgausscrit_computedS1_freq_{1/slope_range[1]}_{1/slope_range[0]}.csv")


