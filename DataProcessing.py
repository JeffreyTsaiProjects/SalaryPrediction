import pandas as pd

class Data:
    def __init__(self,fname_Xtrain=None,fname_ytrain=None,fname_Xtest=None,
                 columns_cat=None,columns_num=None,ylabel=None,joinId=None,
                 ordinal_dict=None):
        self.ylabel = ylabel
        self.columns_cat = columns_cat
        self.columns_num = columns_num
        self.columns_all = (columns_num + columns_cat)
        self.id = joinId         
        # train set
        Xtrain_df = self._readData(fname=fname_Xtrain)
        ytrain_df = self._readData(fname=fname_ytrain)
        # test set
        self.Xtest_df = self._readData(fname_Xtest)
        # all train
        self.train_df = self._join_df(Xtrain_df,ytrain_df,col_id=self.id,
                                      clean=True,drop_id=False)
        self.train_id = self.train_df[self.id]
        # category groups
        self.ordinal_dict = ordinal_dict
        self.group_train_cat = self.train_df.groupby(self.columns_cat)
        self.group_test_cat = self.Xtest_df.groupby(self.columns_cat)

    def _readData(self,fname=None):
        return pd.read_csv(fname)
            
    
    def _join_df(self,left_df,right_df,col_id=None,clean=True,drop_id=True):
        return pd.merge(left_df,right_df,on=col_id,how='left')
    
    def _clean_df(self,df,subset_id=None):
        clean_df = df[df[self.ylabel]>0]
        clean_df = df.drop_duplicates(subset=subset_id)
        return clean_df

    
    def _encode_categorical_features(self,input_df,ordinal_dict=None,
                                     nominal_cols=['companyId']):
        '''
        encode categorical features to numeric values
        '''
        nominal_df = self._labelEncoder(input_df,featuresList=nominal_cols) 
        if ordinal_dict:
            columns_ordinal = list(ordinal_dict.keys())
            ordinal_df = input_df[columns_ordinal].copy()
            ordinal_df.replace(ordinal_dict,inplace=True)
            ordinal_df[self.id] = input_df[self.id]
            encoded_df = self._join_df(nominal_df,ordinal_df,col_id=self.id)
            if self.ylabel in input_df.columns:
                encoded_df = self._join_df(input_df[self.columns_num+[self.id,self.ylabel]],encoded_df,
                                     col_id=self.id)
            else:
                encoded_df = self._join_df(input_df[self.columns_num+[self.id]],encoded_df,
                                     col_id=self.id)
            return encoded_df
        else:
            encoded_df = self._labelEncoder(input_df,
                                            featuresList=self.columns_cat)
            if self.ylabel in input_df.columns:
                encoded_df = self._join_df(input_df[self.columns_num+[self.id,self.ylabel]],encoded_df,
                                     col_id=self.id)
            else:
                encoded_df = self._join_df(input_df[self.columns_num+[self.id]],encoded_df,
                                     col_id=self.id)
            return encoded_df
        
    
    def _labelEncoder(self,input_df,featuresList=None):
        from sklearn.preprocessing import LabelEncoder
        df = pd.DataFrame()
        df[self.id] = input_df[self.id]    
        for f in featuresList:
            le = LabelEncoder()                      
            df[f] = le.fit_transform(input_df[f])
        return df    
    

    def _preprocessData(self,input_df,clean=True,encode=True,nominal_cols=None):
        '''
        drop zero valued response rows and duplicated id rows
        '''
        df = input_df.copy()
        if clean:
            if self.ylabel in df.columns:
                df = df[df[self.ylabel]>0]
            df.drop_duplicates(subset=self.id,inplace=True)
        if encode:
            df = self._encode_categorical_features(df,
                                                   ordinal_dict=self.ordinal_dict,
                                                   nominal_cols=nominal_cols)
        return df    
            
                        
