import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

class Data:
    def __init__(self,fname_features,fname_response,columns_cat,
                 columns_num,ylabel,joinId,ordinal_dict,test_sz=.3):
        '''
        fname_features  : filename of training features, str
        fname_response  : filename of training response, str
        columns_cat     : names of categorical features, list of str
        columns_num     : names of numeric features, list of str
        ylabel          : name of response, str
        joinId          : name of feature unique Id to merge tables on, str
        ordinal_dict    : ordinal keys to int values mappings for caregorical feature encoding, dict of str:int
        '''
        self.ylabel = ylabel
        self.columns_cat = columns_cat
        self.columns_num = columns_num
        self.id = joinId         
        self.ordinal_dict = ordinal_dict
        # train set
        features_df = pd.read_csv(fname_features)
        response_df = pd.read_csv(fname_response)
        # combine training features with response
        self.raw_data_df = pd.merge(features_df,response_df,
                                    on=self.id,how='right')
        self.raw_data_df = shuffle(self.raw_data_df).reset_index()
        # assign held-out dataset
        self.train_df,self.test_df = train_test_split(self.raw_data_df,
                                                      test_size=test_sz,
                                                      random_state=100)

    
    def _clean_df(self,df,subset_id):
        '''
        method which removes zero-valued response values 
        and drops duplicated id rows from training set
        df        : table to process, DataFrame
        subset_id : column consider duplicates, str
        returns   : DataFrame
        '''
        clean_df = df[df[self.ylabel]>0]
        clean_df = df.drop_duplicates(subset=subset_id)
        return clean_df

    
    def _encode_categorical_features(self,input_df,ordinal_dict,nominal_cols):
        '''
        encode categorical features to numeric values
        input_df     : data to encode categoricals to ordinal numeric values, DataFrame
        ordinal_dict : mappings of ordinal categorical features to ints, dict of str keys to int codes
        nominal_cols : list of nominal categorical features, list
        returns      : returns DataFrame
        '''
        nominal_df = self._labelEncoder(input_df,nominal_cols) 
        if ordinal_dict:
            columns_ordinal = list(ordinal_dict.keys())
            ordinal_df = input_df[columns_ordinal].copy()
            ordinal_df.replace(ordinal_dict,inplace=True)
            ordinal_df[self.id] = input_df[self.id]
            encoded_df = pd.merge(nominal_df,ordinal_df,
                                  on=self.id,how='left')
            if self.ylabel in input_df.columns:
                encoded_df = pd.merge(input_df[self.columns_num+[self.id,self.ylabel]],
                                      encoded_df,on=self.id,how='left')
            else:
                encoded_df = pd.merge(input_df[self.columns_num+[self.id]],
                                     encoded_df,on=self.id,how='left')
            return encoded_df
        else:
            encoded_df = self._labelEncoder(input_df,self.columns_cat)
            if self.ylabel in input_df.columns:
                encoded_df = pd.merge(input_df[self.columns_num+[self.id,
                                                                 self.ylabel]],
                                      encoded_df,on=self.id,how='left')
            else:
                encoded_df = pd.merge(input_df[self.columns_num+[self.id]],
                                      encoded_df,on=self.id,how='left')
            return encoded_df
        
    
    def _labelEncoder(self,input_df,featuresList):
        '''
        encodes nominal categorical features to numeric integers
        input_df        : data to process categoricals to numeric, DataFrame
        featuresList    : nominal features to encode, list of str
        returns         : DataFrame
        '''
        from sklearn.preprocessing import LabelEncoder
        df = pd.DataFrame()
        df[self.id] = input_df[self.id]    
        for f in featuresList:
            le = LabelEncoder()                      
            df[f] = le.fit_transform(input_df[f])
        return df    
    

    def _preprocessData(self,input_df,clean,encode,nominal_cols):
        '''
        drop zero valued response rows and duplicated id rows
        input_df     : table to process, DataFrame
        clean        : binary to control cleaning zero-valued response, bool  
        encode       : binary to control categorical feature encoding, bool  
        nominal_cols : nominal categorical features, list of str 
        returns      : DataFrame
        '''
        df = input_df.copy()
        if clean:
            if self.ylabel in df.columns:
                df = df[df[self.ylabel]>0]
            df.drop_duplicates(subset=self.id,inplace=True)
        if encode:
            df = self._encode_categorical_features(df,self.ordinal_dict,
                                                   nominal_cols)
        return df    