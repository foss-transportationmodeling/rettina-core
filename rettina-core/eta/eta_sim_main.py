'''
Created on Jun 6, 2015

@author: Asif.Rehan@engineer.uconn.edu
'''
import os
import numpy as np
import pandas as pd
import pickle as pkl
import process
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy import stats, linalg
from simulation_sampler import crowd_source_simu
from speed_vector import main as sp_vec_main
from mapmatching.validate.validation import Validate as mm_val

data_folder = r'../../data/Relevant_files'

def crowd_density(train_len_indic_mat, link_len_vec):
    """Measures redundancy measures of the crowd-sourced data

    Returns
    -------
    Redundancy : (How many times the link was observed - 1)
                Indicates excess information than needed to have it once
                If not observed at all, it is -1 for that link
    Redundant Length Present : (Total length covered in k traces - k*length)
                Assuming link was observed in k-traces
                Measures its redundant presence influencing least square calc
    """
    M= train_len_indic_mat.shape[0]
    mat = train_len_indic_mat.as_matrix()
    #cover = mat - link_len_vec
    link_count_arr = np.zeros((M,1))
    for i in range(M):
        row = mat[i]
        link_count_arr[i] = len(row[row > 0])
    avg_count_redundancy = link_count_arr.mean()
    link_cnt_minus_1_arr = link_count_arr-1
    return link_cnt_minus_1_arr, avg_count_redundancy

def get_metrics(test_pred_experience_time, test_experience_time,
                dow, tod, onboard_time_max, overlap_max_minute,
                speed_vec_df, optim_f_vec):
    '''
    Calculates the performance metrics
    '''
    test_rmse = process.calc_rmse(test_pred_experience_time,
                                  test_experience_time.as_matrix())
    test_exp_arr = test_experience_time.as_matrix()
    test_coeff_corr = stats.linregress(test_pred_experience_time.flatten(),
                                       test_exp_arr)[2]
    test_coeff_det = test_coeff_corr**2
    diff = test_pred_experience_time.flatten()-test_experience_time.as_matrix()
    abs_diff = np.absolute(diff)
    Mean_AD = abs_diff.mean()
    Max_AD = abs_diff.max()
    Min_AD = abs_diff.max()
    MAPE = abs(diff.astype('float')/test_experience_time).mean()*100
    metrics =  test_rmse, test_coeff_corr, test_coeff_det, \
               diff.min(), diff.max(), diff.mean(),  \
               Min_AD, Max_AD, Mean_AD, MAPE
    return metrics

def inner_loop(Freq,dow, tod, onboard_time_max, overlap_dir, val_tods,
               overlap_max_minute, speed_vec_dow, speed_vec_tod):
    '''
    '''
    train_link_indic_mat,train_experienced_time = crowd_source_simu(
                                                    rd_files_df,
                                                    src_fldr,
                                                    tod, dow,
                                                    M,
                                                    onboard_time_min,
                                                    onboard_time_max,
                                                    overlap_max_minute,
                                                    overlap_dir)
    N_train = int(train_link_indic_mat.shape[1])

    print speed_vec_files_df.head()
    print speed_vec_dow, speed_vec_tod
    speed_vec_file = speed_vec_files_df.loc[
                                (speed_vec_files_df['DOW']==speed_vec_dow) &
                                (speed_vec_files_df['TOD']==speed_vec_tod),
                                'speed_vec_file'].values[0]
    speed_vec_arr = pkl.load(open(speed_vec_file, 'rb'))
    built = process.build_model(train_link_indic_mat,
                                      train_experienced_time,
                                      speed_vec_arr,
                                      Lapl)
    optim_f_vec, opt_lambda, err_log, min_lambda, max_lambda = list((built))
    print optim_f_vec.shape
    print opt_lambda
    print err_log[:2] # list of tuples (lambda, LOOCV)
    print min_lambda
    print max_lambda
    # raw_input()
    #==========================================================================
    # make LOO CV plot here
    #==========================================================================
    plotting(Freq, dow, tod, opt_lambda, min_lambda, max_lambda,
              overlap_dir, err_log, onboard_time_max)
    #==========================================================================
    # make heatmap and scatterplot on train dataset
    #==========================================================================
    train_pred_experience_time = process.predict_travel_time(optim_f_vec,
                                                1.0 / speed_vec_arr,
                                            train_link_indic_mat.as_matrix())
    train_metrics = get_metrics(train_pred_experience_time,
                                train_experienced_time,
                                dow, tod, onboard_time_max, overlap_max_minute,
                                speed_vec_arr, optim_f_vec)
    train_avg_trace_len = round(train_link_indic_mat.sum(axis=0).mean(),2)
    train_metrics = train_metrics + (N_train,train_avg_trace_len)
    #==========================================================================
    #Test set
    #==========================================================================
    test_link_indic_mat, test_experience_time = crowd_source_simu(rd_files_df,
                                                        src_fldr,
                                                        tod, dow,
                                                        M,
                                                        onboard_time_min,
                                                        onboard_time_max,
                                                        overlap_max_minute,
                                                        1)
    N_test = int(test_link_indic_mat.shape[1])
    test_pred_experience_time = process.predict_travel_time(optim_f_vec,
                                                        1.0 / speed_vec_arr,
                                            test_link_indic_mat.as_matrix())
    test_metrics = get_metrics(test_pred_experience_time, test_experience_time,
                               dow, tod, onboard_time_max, overlap_max_minute,
                               speed_vec_arr, optim_f_vec)
    test_avg_trace_len = round(test_link_indic_mat.sum(axis=0).mean(),2)
    test_metrics = test_metrics + (N_test,test_avg_trace_len)
    #==========================================================================
    # make heatmap and scatterplot on test dataset
    #==========================================================================
    #==========================================================================
    # validation set
    #==========================================================================
    val_metrics_list= []
    if len(val_tods) != 0:
        for val_tod in val_tods:
            val_link_indic_mat_file = (''.join([dow, '_', val_tod,
                                               '_len_indic_mat.p']))
            val_link_indic_mat_loc = os.path.join(data_folder,
                                                  'files_for_ETA_simulation',
                                                  '{}sec'.format(Freq),
                                                  val_link_indic_mat_file)
            val_link_indic_mat = pkl.load(open(val_link_indic_mat_loc,'rb'))
            N_val = int(val_link_indic_mat.shape[1])
            val_experience_time_file = (''.join([dow, '_',
                                                val_tod, '_hop_time.p']))
            val_experience_time_loc = os.path.join(data_folder,
                                                  'files_for_ETA_simulation',
                                                  '{}sec'.format(Freq),
                                                   val_experience_time_file)
            val_experiece_time = pkl.load(open(val_experience_time_loc,'rb'))
            val_pred_experience_time = process.predict_travel_time(optim_f_vec,
                                                        1.0 / speed_vec_arr,
                                            val_link_indic_mat.as_matrix())
            val_metrics = get_metrics(val_pred_experience_time,
                                      val_experiece_time,
                                      dow, val_tod, onboard_time_max,
                                      overlap_max_minute, speed_vec_arr,
                                      optim_f_vec)
            val_avg_trace_len = round(
                                    train_link_indic_mat.sum(axis=0).mean(),2)
            val_metrics_list.append((val_tod,
                                     val_metrics+(N_val,val_avg_trace_len)))
    #==========================================================================
    # Redundancy
    #==========================================================================
    train_count_redunt, train_avg_redun = crowd_density(train_link_indic_mat,
                                                        link_len_vec)

    sparsity = lambda overlap_dir: 'Sparse' if overlap_dir==-1  \
                                                        else 'Continuous'
    speed_pred =  1.0/(1.0/speed_vec_arr + optim_f_vec).flatten()*2.236936
    pred_median_sp = np.median(speed_pred)
    pred_speed_over_pred = len(speed_pred[speed_pred > 100])
    pred_speed_under_pred = len(speed_pred[speed_pred < 0])
    plot_speed_hist(Freq, dow, tod, onboard_time_max, overlap_dir, sparsity,
                     speed_pred)
    #2.236936 to convert m/s to mph

    for clip in [None, 100]:
        congestion_heatmap(Freq, dow, tod, onboard_time_max,
                          val_tod, overlap_dir, speed_pred,
                         train_count_redunt.flatten(), clipped_upper=clip)

    return opt_lambda, train_avg_redun, train_metrics, \
            test_metrics, val_metrics_list,  \
            train_experienced_time.mean()/60,  \
            train_experienced_time.min()/60,  \
            train_experienced_time.max()/60,   \
            pred_median_sp, pred_speed_over_pred, pred_speed_under_pred,   \
            test_experience_time.as_matrix(),  \
            test_pred_experience_time.flatten(),  \
            train_pred_experience_time.flatten(),  \
            train_experienced_time.as_matrix().flatten()


def run_full_output(Freq,seg, max_onboard_time_conditions=[15,10,5],
                    speed_vec_dow='all',speed_vec_tod='af',
                    val_tods=['mo', 'ev'],repeat=1):
    columns = ['Model_ID', 'GPS_Freq','Dataset', 'TOD', 'DOW',
                'Lambda', 'Max_OnBoardTime_minute',
                'Actual_Mean_OnBoardTime_minute',
                'Actual_Min_OnBoardTime_minute',
                'Actual_Max_OnBoardTime_minute',
                'Median_Predicted_speed_mph',
                'Over_Predicted_Speed_count',
                'Under_Predicted_Speed_count',
                'Sparsity',
                'Avg_Count_Redunt',
                #get_metrics output below
                'RMSE', 'Pearson_r', 'R_Squared',
                'Min_Diff_sec', 'Max_Diff_sec','Mean_Diff_sec',
                'Min_Abs_Diff_sec', 'Max_Abs_Diff_sec','Mean_Abs_Diff_sec',
                'MAPE_sec',
                'Number_of_Traces', 'Avg_Trace_length_m']
    output_df = pd.DataFrame(columns = columns)
    model_id = 0
    for r in range(repeat):
        for tod, dow in seg:
            scat_plt_data = []
            for obd_max in  max_onboard_time_conditions:
                for overlap_dir in [1, -1]:
                    model_id += 1
                    overlap_max_minute = obd_max
                    out = inner_loop(Freq,dow, tod, obd_max, overlap_dir,
                                                        val_tods,
                                                        overlap_max_minute,
                                                        speed_vec_dow,
                                                        speed_vec_tod)
                    opt_lam             = out[0]
                    avg_cnt_redun       = out[1]
                    train_metrics       = out[2]
                    test_metrics        = out[3]
                    val_metrics_list    = out[4]
                    mean_actual_exp_time= out[5]
                    min_actual_exp_time = out[6]
                    max_actual_exp_time = out[7]
                    pred_median_sp      = out[8]
                    pred_speed_over_pred= out[9]
                    pred_speed_under_pred= out[10]
                    sparsity= lambda overlap_dir:'Sparse' if overlap_dir==-1  \
                                                    else 'Continuous'
                    for (metrics, dat) in [(train_metrics, 'Train'),
                                       (test_metrics, 'Test')]:
                        row = [model_id, Freq, dat, tod, dow, opt_lam,
                               obd_max, mean_actual_exp_time,
                               min_actual_exp_time,
                               max_actual_exp_time,
                               pred_median_sp,
                               pred_speed_over_pred,
                               pred_speed_under_pred,
                               sparsity(overlap_dir),
                               avg_cnt_redun] + list(metrics)
                        row_df = pd.DataFrame([row], columns=columns)
                        output_df = output_df.append(row_df, ignore_index=True)
                    for (val_tod, metrics) in val_metrics_list:
                        row = [model_id,Freq, 'Validation', val_tod,dow,
                               opt_lam,obd_max,  mean_actual_exp_time,
                               min_actual_exp_time,
                               max_actual_exp_time,
                               pred_median_sp,
                               pred_speed_over_pred,
                               pred_speed_under_pred,sparsity(overlap_dir),
                               avg_cnt_redun] + list(metrics)
                        row_df = pd.DataFrame([row], columns=columns)
                        output_df = output_df.append(row_df, ignore_index=True)
                    scat_plt_data.append((obd_max,sparsity(overlap_dir),
                                    out[-4:-2], out[-2:],
                                [test_metrics[0],
                                 test_metrics[1],
                                 test_metrics[2],
                                 test_metrics[8],
                                 test_metrics[9],
                                                 train_metrics[0],
                                                 train_metrics[1],
                                                 train_metrics[2],
                                                 train_metrics[8],
                                                 train_metrics[9],]))
            scatter_plots(Freq, dow, tod,scat_plt_data,sharexy=True)
            scatter_plots(Freq, dow, tod,scat_plt_data,sharexy=False)
    return output_df

def congestion_heatmap(Freq, dow, tod, obt, val_tod, overlap_dir_tag,
                       speed_pred, count_redunt,clipped_upper=None):
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
    fig.set_size_inches(7, 8, forward=True)
    mpl.rcParams.update({'font.size': 8})
    sparsity = lambda overlap_dir: 'Sparse' if overlap_dir==-1  \
                                                        else 'Continuous'
    fig.suptitle('Data Redundancy Vs Predicted Speed',
                 fontsize=16)
    axes[0].set_title('Predicted Speed (mph)', fontsize=12)
    axes[0].set_axis_bgcolor('k')
    if clipped_upper != None:
        speed_pred[speed_pred > clipped_upper] = clipped_upper
        speed_pred[speed_pred < 0] = 0
    mm_val('').plot_roadnetwork(axes[0], fig, select=False, heatmap=True,
                                heatmap_cmap=speed_pred, heat_label='mph')
    axes[1].set_title('Link Count Redundancy',fontsize=12)
    axes[1].set_axis_bgcolor('k')
    mm_val('').plot_roadnetwork(axes[1], fig, select=False, heatmap=True,
                heatmap_cmap=count_redunt, heat_label='Link Count - 1')

    fig.text(0.5, 0.05, 'Easting', ha='center', va='center', fontsize=10)
    fig.text(0.05, 0.5, 'Northing', ha='center', va='center',fontsize=10,
             rotation='vertical')
    plt.figtext(0.5, 0.925,
                 'Overlap:'+sparsity(overlap_dir_tag)+  \
                 '    '+'Day of Week: '+dow.upper()+  \
                 '    '+'Max Onboard Time (Min): '+str(obt)+  \
                 '    '+'GPS Frequency(sec): '+str(Freq),
                 ha='center', va='center')
    clip_tag = lambda clipped_upper:'True' if clipped_upper!= None else 'False'

    fig_file = ('heatmap{}_{}_{}_{}_clipped{}'.
                format(dow.upper(), tod.upper(), obt,
                       sparsity(overlap_dir_tag), clip_tag(clipped_upper)))
    fig_loc = os.path.join(data_folder, "results",
                           '{}sec'.format(Freq), fig_file)
    fig.savefig(fig_loc)

    plt.close()
    return None

def scatter_plots(Freq, dow, tod, scat_plt_data, sharexy=True):
    '''
    predicted vs actual trip travel time scatter plot

    Inputs
    ------
    Freq: used to note on the top of the graph
    dow : day of week
    tod : time of day, morning, afternoon and evening
    scat_plt_data : data to plot with metric data to write on the plot

    '''
    return # TODO: Short circuiting to avoid the error
    fig, axes = plt.subplots(nrows=len(scat_plt_data)/2,
                             ncols=2, sharex=sharexy, sharey=sharexy)
    fig.set_size_inches(8, 10.5, forward=True)
    mpl.rcParams.update({'font.size': 8})
    if sharexy == True:
        title_sec_line = 'Same Axes for All Plots'
    else:
        title_sec_line = 'Individual Axes for Each Plot'
    fig.suptitle('Predicted vs Actual Travel Time for Test Dataset\n'+  \
                 title_sec_line, fontsize=14)
    #leg = []
    for i in range(len(scat_plt_data)):
        row = i//2
        col = i%2
        axes[row,col].scatter(scat_plt_data[i][2][0],
                              scat_plt_data[i][2][1], label='Test Data',
                              s=20, c='r', marker='<', alpha=0.40)
        test_rmse       = round(scat_plt_data[i][4][0], 2)
        test_pearson    = round(scat_plt_data[i][4][1], 2)
        test_Rsq        = round(scat_plt_data[i][4][2], 2)
        test_MAD        = round(scat_plt_data[i][4][3], 2)
        test_MAPE       = round(scat_plt_data[i][4][4], 2)
        plot_info = "Coeff of Corr, r = {}\n".format(test_pearson)+  \
                    "Coeff of Det, R-sq = {}\n".format(test_Rsq)+   \
                    "RMSE = {}(sec)\n".format(test_rmse)+  \
                    "MAD = {}(sec)\n".format(test_MAD)+   \
                    "MAPE = {}(sec)".format(test_MAPE)
        axes[row,col].text(0.05, 0.95, plot_info, ha='left', va='top',
                           transform=axes[row,col].transAxes)
        #tst = axes[row,col].scatter(scat_plt_data[i][3][0],
        #                      scat_plt_data[i][3][1], label='Train Data',
        #                      s=10, c='b', marker='>', alpha=0.6)
        #if leg == []:
        #    leg = [trn, tst]
        if col==1:
            axes[row,col].yaxis.set_label_position("right")
            axes[row,col].set_ylabel(str(scat_plt_data[i][0])+' Minutes',
                                 rotation=270, labelpad= 12)
    #fig.legend(leg, ['Test Data','Train Data'],
    #              loc='upper center', bbox_to_anchor=(0.5, 0.5), ncol=2)
    fig.text(0.50, 0.04, 'Actual Time (sec)',
             ha='center', va='center',fontsize=10)
    fig.text(0.30, 0.06, scat_plt_data[0][1],
             ha='center', va='center',fontsize=9)
    fig.text(0.70, 0.06, scat_plt_data[1][1],
             ha='center', va='center',fontsize=9)
    fig.text(0.05, 0.5, 'Predicted Time (sec)', ha='center', va='center',
             rotation='vertical',fontsize=10)
    fig.text(0.95, 0.5, 'GPS Trace Duration Max Limit',
             ha='center', va='center',
             rotation=270,fontsize=10)
    plt.figtext(0.5, 0.925, 'Day of Week: '+dow.upper(),
                 ha='center', va='center')

    fig_file = ('scat_sec_{}_{}_sharexy_{}'.format(dow.upper(),
                                                   tod.upper(),
                                                   sharexy))
    fig_loc = os.path.join(data_folder, "results",
                           '{}sec'.format(Freq), fig_file)
    fig.savefig(fig_loc)
    plt.tight_layout()
    plt.close()
    return None

def plotting(Freq, dow, tod, opt_lambda, min_lambda, max_lambda,
             overlap_dir_tag, err_log, onboard_time_max):
    '''
    makes the Error vs LOO CV plot

    Input
    -----
    opt_lambda : optimum lambda which produces minimum error
    '''
    fig = plt.figure()
    ax = plt.axes()
    lambda_values, errors = zip(*err_log)
    plt.plot(lambda_values, errors)

    sparsity = lambda overlap_dir: 'Sparse' if overlap_dir==-1  \
                                                        else 'Continuous'
    ttl = 'LOOCV Error versus Lambda'
    plt.suptitle(ttl)
    plt.yscale('log')
    plt.xlabel('lambda')
    plt.ylabel('LOOCV Error')
    plt.axhline(min([err[1] for err in err_log]), color='r', ls='--')
    plt.axvline(opt_lambda, color='r', ls='--')
    plt.figtext(0.5, 0.925,
                 'Overlap:'+sparsity(overlap_dir_tag)+  \
                 '    '+'Day of Week: '+dow.upper()+  \
                 '    '+'Max Onboard Time (Min): '+str(onboard_time_max)+   \
                 '    '+'GPS Frequency(sec): '+str(Freq),
                 ha='center', va='center')
    plot_info = "Lambda* = {}\n".format(opt_lambda)+   \
                "Min Lambda = Min EigValue = {}\n".format(min_lambda)+   \
                "Max Lambda = Max EigValue = {}\n".format(max_lambda)
    ax.text(0.05, 0.95, plot_info, ha='left', va='top',
                           transform=ax.transAxes)
    fig_file = ('{}_{}_{}_{}_{}_clipped'.format(ttl, dow.upper(),
                                                tod.upper(), onboard_time_max,
                                                sparsity(overlap_dir_tag)))
    fig_loc = os.path.join(data_folder, "results",
                           '{}sec'.format(Freq), fig_file)

    plt.savefig(fig_loc)
    plt.close()
    return None

#disagg(seg[0], 1).to_csv(r'../_files/eta_krr_plots/disagg_summary_all.csv')
#==============================================================================
# for all-dow, afternoon
#==============================================================================

def plot_speed_hist(Freq, dow, tod, onboard_time_max, overlap_dir,
                    sparsity, speed_pred):
    plt.hist(speed_pred, label='Predicted Speed')
    plt.suptitle('Predicted Speed Histogram')
    plt.figtext(0.5, 0.925,
                 'Overlap:'+sparsity(overlap_dir)+  \
                 '    '+'Day of Week: '+dow.upper()+  \
                 '    '+'Max Onboard Time (Min): '+str(onboard_time_max)+  \
                 '    '+'GPS Frequency(sec): '+str(Freq),
                 ha='center', va='center')
    plt.xlabel('Predicted Speed (mph)')
    plt.ylabel('Count')
    fig_file = ('speed_hist_{}_{}_{}_{}'.format(dow.upper(), tod.upper(),
                                                onboard_time_max,
                                                sparsity(overlap_dir)))
    fig_loc = os.path.join(data_folder, "results",
                           '{}sec'.format(Freq), fig_file)
    plt.savefig(fig_loc)
    plt.close()

if __name__ == '__main__':
    this_dir =  os.path.dirname(__file__)
    speed_vec_files = [f for f in os.listdir(data_folder)
                           if f[7:-2] == 'speed_vector']
    speed_vec_store = [(f[:3], f[4:6], f) for f in speed_vec_files]
    speed_vec_files_df = pd.DataFrame(speed_vec_store,
                                columns=['DOW', 'TOD', 'speed_vec_file'])
    LinkIdLenSer=os.path.join(data_folder,
                              r'LinkIdLenSer.p')
    id_len_series = pkl.load(open(LinkIdLenSer,'rb'))
    link_len_vec = id_len_series.as_matrix().reshape(len(id_len_series),1)
    LaplacianMatrix = os.path.join(data_folder,
                                   r'Laplacian_matrix.p')
    Lapl = pkl.load(open(LaplacianMatrix, 'rb'))

    M = 177
    onboard_time_min = 2
    onboard_time_max = 15
    overlap_max_minute = 15
    overlap_dir = 1    #or -1 for sparse
    lamb_min = 1
    lamb_max= 10000
    lamb_step = 10
    for Freq in [60]:
        src_fldr = os.path.join(data_folder, 'files_for_ETA_simulation',
                                '{}sec'.format(Freq))
        road_files = [f for f in os.listdir(os.path.join(src_fldr,'road_files'))
                      if os.path.isfile(os.path.join(src_fldr,
                                                     'road_files', f))]
        store = [(f[12:14], f[15:18], f[19:21], f) for f in road_files]
        rd_files_df = pd.DataFrame(store,
                                   columns=['route', 'DOW', 'TOD', 'road_file'])
        #seg = [(TOD, DOW) for TOD in ['af'] for DOW in ['wed', 'tue','thu']]
        seg = [(TOD, DOW) for TOD in ['af'] for DOW in ['wed']] # just run for afteroon wed as opposed to all tod for all day of weeks as in the previous line
        sp_vec_main([('af', 'wed')], src_fldr)
        allout = run_full_output(Freq,seg,
                                max_onboard_time_conditions=[15],
                                speed_vec_dow='wed', speed_vec_tod='af')#,
                                #val_tods=['mo', 'ev'], repeat=10)
        #==========================================================================
        #allout.to_csv(
        #    '../_files/eta_krr_plots/{0}sec/ALLOUTPUT_{0}sec_10x_3.csv'.format(Freq))
