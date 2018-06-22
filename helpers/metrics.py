import scipy.stats

def perform_ztest(p1, p2, n, bonferonni):
    '''
    Performs pairwise Z-test on two accuracies for the test sets. 
    
    :param p1: Accuracy of model 1
    :param p2: Accuracy of model 2
    :param n: Number of samples for model 1 and model 2 (same in our case)
    :param bonferonni: Positive integer determining number of pairwise z-tests perormed
    '''
    # Determine significance level
    alpha = 0.05 / bonferonni
    
    # Statistics
    Z = (p1 - p2) / np.sqrt(((p1 * (1 - p1)) / n ) + ((p2 * (1 - p2)) / n))
    p = scipy.stats.norm.sf(abs(Z))

    if p > alpha:
        print('There is no significant performance in accuracy between model 1 (p1 = {}) and model 2 ({}).\n'
        'z = {:.3f} with p-value: {:.3f} > {}'.format(p1, p2, Z, p, alpha))
    else:
         print('There is a significant difference in accuracy between model 1 (p2 = {}) and model 2 ({}).\n'
       'z = {:2f} with p-value: {:.3f} < {}'.format(p1, p2, Z, p, alpha))
    
    
# Example of performing z-test. 
perform_ztest(0.9104, 0.9080, 804, 1)