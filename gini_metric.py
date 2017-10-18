# create Gini as custom metric
import numpy as np

def gini(actual,pred,cmpcol=0,sortcol=1):
    assert(len(actual)==len(pred))
    all=np.asarray(np.c_[actual,pred,np.arange(len(actual))],dtype=np.float)
    all=all[np.lexsort((all[:,2],-1*all[:,1]))]
    totalLosses=all[:,0].sum()
    giniSum=all[:,0].cumsum().sum()/totalLosses
    giniSum-=(len(actual)+1)/2.
    return giniSum/len(actual)
 
# this is compatible with sklearn
def gini_normalized(a,p):
    return gini(a,p)/gini(a,a)

# create an XGBoost-compatible metric from Gini
def gini_xgb(preds,dtrain):
    labels=dtrain.get_label()
    gini_score=gini_normalized(labels,preds)
    return [('gini',gini_score)]