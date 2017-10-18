from sklearn.preprocessing import LabelEncoder,OneHotEncoder

# one-hot encoding
def one_hot_encode(data):
    ohe=OneHotEncoder(sparse=False)
    col_names=[]
    cats_df=df()
    for col in data.columns:
        if '_cat' in col:
            col_names.append(col)
            col_name=len(np.unique(data[col].values))*[col]
            cat_ind=list(np.unique(data[col],axis=0))
            for i,item in enumerate(cat_ind):
                cat_ind[i]='.'+str(item)
            new_col_names=(np.core.defchararray.add(col_name,cat_ind))
            cat_col=data[col].values.reshape(data.shape[0],1)
            cat_array=ohe.fit_transform(cat_col)
            cat_df=df(cat_array,columns=new_col_names)
            cats_df=pd.concat([cats_df,cat_df],axis=1,join_axes=[cat_df.index])
    data_encoded=data.drop(col_names,axis=1)
    data_encoded=pd.concat([data_encoded,cats_df],axis=1,join_axes=[data_encoded.index])
    return data_encoded
