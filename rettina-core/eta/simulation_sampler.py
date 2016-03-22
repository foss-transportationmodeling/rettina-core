"""
Created on May 27, 2015

@author: Asif.Rehan@engineer.uconn.edu
"""
import os
import pandas as pd
import pickle as pkl
from datetime import timedelta
from random import randint

def choose_offboard_ts(onboard_time_min, onboard_time_max, onboard_ts):
    """chooses time when trip stops"""
    rand_onboard_time = randint(onboard_time_min*60, onboard_time_max*60)
    offboard_ts = onboard_ts + timedelta(seconds=rand_onboard_time)
    return offboard_ts

def next_agent_start_ts(overlap_max_minute, overlap_dir, offboard_ts):
    '''
    randomly select the time when the next trip starts

    Input
    -----
    overlap_dir : if -1, next trip starts no earlier than the time when
                    the current one stops. Ensures no overlapping of trips
                  if +1, next trip starts no later than when the
                    current one stops. Ensures no sparsity of trips
    '''
    rand_ovrlp_time = randint(0, overlap_max_minute*60)
    rand_overlap_tdelta = timedelta(seconds=rand_ovrlp_time)
    onboard_ts = offboard_ts - overlap_dir * rand_overlap_tdelta
    return onboard_ts

def crowd_source_simu(rd_files_df, src_fldr, tod, dow, n_road,
                      onboard_time_min, onboard_time_max,
                      overlap_max_minute, overlap_dir):
    '''
    Randomly generates trips using the existing map-matched GPS traces and
    also calculates the travel times.

    Return
    ------
    len_indic_mat : link indicator matrix, i.e. Q matrix from these trips
                    trajectory travel time vector
    hop_time : pandas series providing the travel times for the trajectories in
                      len_indic_mat
    '''
    tod_dow_rd_files = rd_files_df.loc[(rd_files_df['DOW']==dow) &
                               (rd_files_df['TOD']==tod),
                               'road_file'].values
    trip_id = 0
    hop_time = []
    len_indic_mat = pd.DataFrame(index=[road_id for road_id in xrange(n_road)])
    for rd_file in tod_dow_rd_files:
        road_f = pd.read_csv(os.path.join(src_fldr, 'road_files', rd_file),
                             index_col=0, parse_dates=['ts_prev', 'ts_next'])
        start_idx, stop_idx = road_f.index[0], road_f.index[-1]
        data_strt_ts = road_f.loc[start_idx,'ts_prev']
        data_stop_ts = road_f.loc[stop_idx,'ts_next']
        onboard_ts = data_strt_ts
        go_live = True
        while go_live:
            offboard_ts = choose_offboard_ts(onboard_time_min,
                                             onboard_time_max,
                                             onboard_ts)
            if offboard_ts > data_stop_ts:
                offboard_ts = data_stop_ts
                go_live = False
            hop = road_f.loc[(road_f['ts_prev'] >= onboard_ts) &
                              (road_f['ts_next'] <= offboard_ts)]
            if len(hop.index) != 0:
                onbrd_experienced_time = hop.loc[hop.ts_idx_prev.unique(),
                                                        'ts_delta_sec'].sum()
                if onbrd_experienced_time > 0:
                    hop_time.append(onbrd_experienced_time)
                    len_indic_mat[trip_id] = 0
                    for hop_road_id in hop.road_id.unique():
                        hopped_len_on_road_id = hop.loc[hop['road_id'] ==   \
                                                    hop_road_id,'length'].sum()
                        len_indic_mat.loc[hop_road_id,trip_id] =   \
                                                        hopped_len_on_road_id
                    trip_id += 1
            onboard_ts = next_agent_start_ts(overlap_max_minute,
                                           overlap_dir, offboard_ts)
    return len_indic_mat, pd.Series(hop_time)

def main(n_road = 177, onboard_time_min = 4, onboard_time_max = 15,
         overlap_max_minute = 15, overlap_dir = 1,
         TOD_list=['af', 'ev', 'mo'], DOW_list=['thu', 'tue', 'wed']):
    '''
    main function for simulating the randomly sampled GPS trajectories
    It dumps the files in a folder (look at the last two "pkl.dump" folder)

    n_road  : number of links in the road network
    onboard_time_min : minimum duration of sampled trajectories
    onboard_time_max : maximum duration of sampled trajectories
    overlap_max_minute : maximum time by which the the next trajectory can
                            start before or after the current trajectory
    overlap_dir : if -1, next trip starts no earlier than the time when
                    the current one stops. Ensures no overlapping of trips
                  if +1, next trip starts no later than when the
                    current one stops. Ensures no sparsity of trips

    '''
    src_fldr = os.path.join(r'../../data/Relevant_files/files_for_ETA_simulation', r'60sec')
    road_files = [f for f in os.listdir(os.path.join(src_fldr,'road_files'))
                  if os.path.isfile(os.path.join(src_fldr, 'road_files', f))]
    store = [(f[12:14], f[15:18], f[19:21], f) for f in road_files]
    rd_files_df = pd.DataFrame(store,
                               columns=['route', 'DOW', 'TOD', 'road_file'])
    #route_set = set(['rd', 'bl', 'gr', 'yl', 'or', 'pl'])
    seg = [(TOD, DOW) for TOD in TOD_list for
        DOW in DOW_list]
    for tod, dow in seg:
        len_indic_mat, hop_time = crowd_source_simu(rd_files_df,
                                                    src_fldr,
                                                    tod, dow,
                                                    n_road,
                                                    onboard_time_min,
                                                    onboard_time_max,
                                                    overlap_max_minute,
                                                    overlap_dir)
        pkl.dump(len_indic_mat, open(os.path.join(src_fldr, dow + '_' + tod + "_len_indic_mat.p"), 'wb'))
        pkl.dump(hop_time, open(os.path.join(src_fldr, dow + '_' + tod + "_hop_time.p"), 'wb'))
    return None
if __name__ == "__main__":
    main()
