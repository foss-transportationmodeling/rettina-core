'''
Created on Jun 5, 2015

@author: Asif.Rehan@engineer.uconn.edu
'''
import numpy as np
from scipy import linalg
from sklearn.metrics import mean_squared_error

def solve_f(Q_arr, Lapl, y_vec_arr, reg_lambda):
    '''
    solve for the f-vector where the eqn is Af = b
    '''
    A = Q_arr.dot(Q_arr.T) + reg_lambda*Lapl
    b = Q_arr.dot(y_vec_arr)
    f_vec = linalg.solve(A, b) 
    return f_vec

def calc_rmse(y, y_pred):
    """inputs as arrays"""
    assert len(y) == len(y_pred)
    RMSE = mean_squared_error(y, y_pred)**0.5
    return  RMSE

def xfrange(start, stop, step):
    while start < stop:
        yield start
        start += step
        
def optimize_lambda(N, Q_arr, y_vec_arr, Lapl, fast=True):
    '''
    Input
    -----
    N : number of trips, length of y_vec_arr
    Lapl : Laplacian matrix of the line graph of the network
    fast : chooses between two functions to calculate LOOCV error
            if True : uses fast_LOOCV_cost()
            if False : uses 
    
    returns
    -------
    LOOCV_argmin_lambda : optimum lambda creating least LOOCV error, 
    error_log : returns a array whose rows are lambda and LOOCV error it causes
    '''
    eig_vals = linalg.eigvalsh(np.dot(Q_arr,Q_arr.T))
    min_lambda = max(min(eig_vals), 0.1)
    max_lambda = max(eig_vals)
    
    error_threshold = np.inf
    error_log = []
    for lambda_now in np.linspace(min_lambda, max_lambda, 10000):
        if fast:
            try:
                error = fast_LOOCV_cost(N, Q_arr, y_vec_arr, Lapl, lambda_now)
            except:
                error = slow_LOOCV_cost(N, Q_arr, y_vec_arr, Lapl, lambda_now)
        else:
            error = slow_LOOCV_cost(N, Q_arr, y_vec_arr, Lapl, lambda_now)
        error_log.append((lambda_now, error))
        if error < error_threshold:
            LOOCV_argmin_lambda = lambda_now
            error_threshold = error
    return LOOCV_argmin_lambda, error_log, min_lambda, max_lambda 

def fast_LOOCV_cost(N, Q_arr, y_vec_arr, Lapl, reg_lambda):
    '''
    calculate LOOCV cost using analytical solution, so calculates it fast 
    '''
    I = np.identity(N)
    inverted = linalg.inv(Q_arr.dot(Q_arr.T) + reg_lambda*Lapl)
    H = Q_arr.T.dot(inverted.dot(Q_arr))
    I_minus_H = I - H
    mat= linalg.inv(linalg.block_diag(I_minus_H)).dot(I_minus_H.dot(y_vec_arr))
    LOOCV_cost = mat.T.dot(mat)[0][0]
    avg_LOOCV_cost = LOOCV_cost/float(N)
    return avg_LOOCV_cost

def slow_LOOCV_cost(N, Q_arr, y_vec_arr, Lapl, reg_lambda):
    '''
    calculate LOOCV cost using leave-one-out approach repeatedly
    So calculation is slow 
    '''
    LOOCV_cost = 0
    for n in xrange(N): 
        _Q_leave_n = np.delete(Q_arr, n, 1)
        _y_leave_n = np.delete(y_vec_arr, n, 0)
        _f_vec_leave_n = solve_f(_Q_leave_n, Lapl, _y_leave_n, reg_lambda)
        LOOCV_cost += (y_vec_arr[n] - Q_arr[:, n].T.dot(_f_vec_leave_n))**2
    avg_LOOCV_cost = LOOCV_cost/float(N)
    return avg_LOOCV_cost 

def predict_travel_time(optim_f_vec, speed_vec_arr, Q_test_arr):
    """
    Input
    -----
    optim_f_vec : optimal solution from the solve_f()    
    speed_vec_arr : vector of inverse avg speed
    Q_test_arr : Q matrix for the test trajectories 
    """
    state_inv_speed = optim_f_vec + speed_vec_arr
    return Q_test_arr.T.dot(state_inv_speed)

def validate(y_test_vec_arr, pred_y_vec_arr):
    '''
    returns difference between predicted trip travel time and 
    predicted ones for the test trips'''
    diff = pred_y_vec_arr - y_test_vec_arr
    return diff

def build_model(Q_df, y_vec_df, speed_vec_arr, Lapl):
    """
    Q_df : pandas.DF of training set of trajectories, which is the Q-matrix
    y_vec_df : onbrd_experienced_time vector in pandas.DF
    speed_vec_arr : array with links speed values, the Phi^0 vector
    y_dev_arr : vector after subtracting avg_onboard_experience_time
            avg_onboard_experience_time = sum(links involved/avg link speed)
    """
    Q_arr = Q_df.as_matrix()
    N = len(y_vec_df)
    assert Q_arr.shape[1] == y_vec_df.shape[0]
    assert Q_arr.shape[0] == speed_vec_arr.shape[0]
    y_vec_arr = y_vec_df.as_matrix().reshape(N,1)
    
    inv_speed_vec = 1.0 / speed_vec_arr
    y_dev_vec_arr = y_vec_arr - Q_arr.T.dot(inv_speed_vec)
    
    optim_lambda, error_log, min_lambda, max_lambda = optimize_lambda(N, Q_arr,
                                                        y_dev_vec_arr, Lapl)
    
    optim_f_vec = solve_f(Q_arr, Lapl, y_dev_vec_arr, optim_lambda)
    return optim_f_vec, optim_lambda, error_log, min_lambda, max_lambda
    
def main(Q_arr, y_vec_df, speed_vec_files_df, Lapl, 
         min_lambda, max_lambda, increment, optim, 
         Q_test_arr, y_test_vec_arr):
    '''
    This function is not used in process. Was written for testing purpose.
    It builds the model and predicts the travel time for the test trajectories
    
    Input
    -----
    Q_arr : training trajectories (Q matrix) as an array 
    y_vec_df : trajectory travel times for the training trajectories
    speed_vec_arr : array with links speed values, the Phi^0 vector
    Lapl : Laplacian matrix for the line graph of the network
    min_lambda, max_lambda : search range for solution of the objective funcn
    increment : steps within the search range
    Q_test_arr : testing trajectories (Q matrix) as an array
    y_test_vec_arr : trajectory travel times for the testing trajectories
    
    returns
    -------
    returns the diff of the testing and predicted travel time
    '''
    optim_f_vec = build_model(Q_arr, y_vec_df, speed_vec_files_df, Lapl, 
                              min_lambda, max_lambda, increment)[0]
    speed_vec_arr = 1.0/ speed_vec_files_df
    pred = predict_travel_time(optim_f_vec, speed_vec_arr, Q_test_arr)
    diff = validate(y_test_vec_arr, pred)
    return diff

if __name__ == "__main__":
    main()