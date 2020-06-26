import numpy as np

def max_entropy(num_classes):
    p = np.array([[1 for i in range(num_classes)]])/num_classes
    e = entropy(p)
    return e

def expected_calibration_error(uncertainty, acc_preds):
    calibration_error = 0.
    conf = 1 - uncertainty
    bins = np.linspace(0, 1, 11)
    
    for i in range(len(bins)-1):
        sublist = []
        indices = []
        
        for j in range(len(conf)):
            if conf[j] > bins[i] and conf[j] <= bins[i+1]:
                sublist.append(conf[j])
                indices.append(j)
        
        if len(sublist) !=0:
            conf_bin = np.mean(np.array(sublist))
            acc_bin = np.mean(acc_preds[indices])
            calibration_error += (acc_bin - conf_bin)**2 * len(sublist)/len(uncertainty) 
    print("Expected Calibration Error: {}".format(calibration_error))
    return calibration_error

def entropy(probs):
    entropy = -np.sum(probs * np.log(probs + 1e-20), axis=1)
    return entropy
