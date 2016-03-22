'''
Created on May 27, 2015

@author: Asif.Rehan@engineer.uconn.edu
'''
import os
import pandas as pd
import numpy as np
import pickle as pkl
from dateutil import parser
from scipy import stats

def fill_non_static_data(node_f, road_f):
    """ fills in the five columns for non-static portion
    ignores the speed in averaging if it equals inf    """
    ts_idx = node_f[node_f['timestamp'] != '0'].index
    delta_ts_ticks = zip(ts_idx[:-1], ts_idx[1:])
    for ts_idx_prev, ts_idx_next in delta_ts_ticks:
        includd_0_spd = road_f[ts_idx_prev:ts_idx_next]
        excludd_0_spd = includd_0_spd[includd_0_spd.length != 0]
        if len(excludd_0_spd.index) != 0:
            ts_next = parser.parse(node_f.timestamp[ts_idx_next])
            ts_prev = parser.parse(node_f.timestamp[ts_idx_prev])
            ts_delta = ts_next - ts_prev
            avg_speed = np.divide(excludd_0_spd.length.sum(),
                ts_delta.total_seconds())
            for indx in excludd_0_spd.index:
                road_f.set_value(indx, 'speed_mps', avg_speed)
                road_f.set_value(indx, 'ts_delta_sec', ts_delta.total_seconds())
                road_f.set_value(indx, 'ts_idx_prev', ts_idx_prev)
                road_f.set_value(indx, 'ts_idx_next', ts_idx_next)
                road_f.set_value(indx, 'ts_prev', ts_prev)
                road_f.set_value(indx, 'ts_next', ts_next)
    return road_f

def fill_static_data(node_f, road_f):
    """ fills in the five columns for static portion """
    #print road_f.head()
    #print node_f.head()
    static = road_f[road_f['length'] == 0]
    for idx in static.index:
        try:
            ts_idx_next = idx+1
            ts_next = parser.parse(node_f.loc[ts_idx_next, 'timestamp'])

        except ValueError:
            temp = node_f.loc[idx+2:]
            valid_next = temp[temp.timestamp!='0'].head(1)
            ts_idx_next = valid_next.index.values[0]
            ts_next = parser.parse(valid_next.timestamp.values[0])

        try:
            ts_idx_prev = idx
            ts_prev = parser.parse(node_f.loc[ts_idx_prev, 'timestamp'])

        except ValueError:
            temp = node_f.loc[:idx]
            valid_prev = temp[temp.timestamp!='0'].tail(1)
            ts_idx_prev = valid_prev.index.values[0]
            ts_prev = parser.parse(valid_prev.timestamp.values[0])
        ts_delta = ts_next - ts_prev
        road_f.set_value(idx, 'speed_mps', 0)
        road_f.set_value(idx, 'ts_delta_sec', ts_delta.total_seconds())
        road_f.set_value(idx, 'ts_idx_prev', ts_idx_prev)
        road_f.set_value(idx, 'ts_idx_next', ts_idx_next)
        road_f.set_value(idx, 'ts_prev', ts_prev)
        road_f.set_value(idx, 'ts_next', ts_next)
    return road_f


def sp_vec_mean_with_nan_replaced(n_road, speed_storage):
    mean_speed_vec = np.zeros((1, n_road))
    for i in xrange(n_road):
        mean_speed_vec[0][i] = np.mean(np.array(speed_storage[i]))

    nan_indx = np.where(np.isnan(mean_speed_vec))
    col_mean = stats.nanmean(mean_speed_vec, axis=1)
    mean_speed_vec[nan_indx] = np.take(col_mean, nan_indx[0])
    return mean_speed_vec.reshape(n_road, 1)

def speed_vector(src_fldr, nd_rd_pair_files, n_road, max_speed_limit):
    """post-processing tool following map-matching.
    Works on map-matched node and road output
    Calculates speed for each road when vehicle was not static
    Adds three new columns to the road output dataframe

    Speed:  Measured by the previous and next timestamp available and
            the distance covered within that time difference.
            Ignores static ones
    ts_idx_prev: Timestamp index from node file for immediate before as
                    vehicle is on or about to move
    ts_idx_next: Timestamp index immediately after as vehicle is on move or
                    about to stop
    ts_prev: Timestamp immediately before as vehicle is on or about to move
    ts_next: Timestamp immediately after as vehicle is on move or about to stop
    ts_delta: Time difference from ts_prev to ts_next
    """
    speed_storage = {}
    for i in xrange(n_road):
        speed_storage[i] = []


    for v, e in nd_rd_pair_files:
        node_f = pd.read_csv(os.path.join(src_fldr, 'node_files', v),
                             index_col=0, usecols=[0, 3, 4])
        road_f = pd.read_csv(os.path.join(src_fldr, 'road_files', e),
                             index_col=0)
        road_f['speed_mps']    = ""
        road_f['ts_delta_sec'] = ""
        road_f['ts_idx_prev']  = ""
        road_f['ts_idx_next']  = ""
        road_f['ts_prev']      = ""
        road_f['ts_next']      = ""

        road_f = fill_static_data(node_f, road_f)
        road_f = fill_non_static_data(node_f, road_f)
        road_f.to_csv(os.path.join(src_fldr, 'road_files', e))

        for idx in road_f.index:
            if road_f.speed_mps[idx] != np.inf:
                valid_speed = min(road_f.speed_mps[idx], max_speed_limit)
                speed_storage[road_f.road_id[idx]].append(valid_speed)
    mean_speed_vec = sp_vec_mean_with_nan_replaced(n_road, speed_storage)
    return mean_speed_vec, speed_storage


def main(seg, src_fldr):
    '''
    Uses speed_vector function for selected TOD and DOW combo for testing
    Input
    -----
    n_road : num of links in the network
    max_speed_limit : if GPS speed calculate from map-matching output is
                        larger than this practical speed limit (m/s), used this
    '''

    node_files = [f for f in os.listdir(os.path.join(src_fldr, 'node_files'))
                  if os.path.isfile(os.path.join(src_fldr, 'node_files', f))]
    road_files = [f for f in os.listdir(os.path.join(src_fldr, 'road_files'))
                  if os.path.isfile(os.path.join(src_fldr, 'road_files', f))]
    nd_rd_pair_files = zip(node_files, road_files)
    n_road = 177
    max_speed_limit = 30 #mps ~= 65mph
    store = [(f[0][12:14], f[0][15:18], f[0][19:21], f[0], f[1]) for f in
        nd_rd_pair_files]
    files_df = pd.DataFrame(store, columns=['route', 'DOW', 'TOD', 'node_file',
            'road_file'])

    for tod, dow in seg:
        #======================================================================
        #to obtain speed vector for across either of (all_dow && all tod)
        #change selection condition
        #======================================================================
        if dow == 'all':
            select = files_df.loc[files_df['TOD']==tod]
        else:
            select = files_df.loc[(files_df['DOW']==dow) &
                                  (files_df['TOD']==tod)]
        nd_rd_pair_files = zip(select.node_file, select.road_file)
        sp_vec, speed_stor = speed_vector(src_fldr, nd_rd_pair_files,
            n_road, max_speed_limit)
        pkl.dump(speed_stor, open(dow+'_'+tod+'_speed_storage.p', 'wb'))
        pkl.dump(sp_vec, open(dow + '_' + tod + '_speed_vector.p', 'wb'))
    return None

if __name__ == "__main__":
    data_folder = r'../../data/Relevant_files'
    src_fldr = os.path.join(data_folder, 'files_for_ETA_simulation', '60sec')
    seg = [(TOD, DOW) for TOD in ['af', 'ev', 'mo'] for
        DOW in ['thu', 'tue', 'wed']]
    main(seg, src_fldr)
