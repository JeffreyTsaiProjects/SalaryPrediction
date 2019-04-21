import pandas as pd

class FeatureEng:
    def __init__(self,data_object=None):
        self.data = data_object
        
    def _cat_group_agg(self,vec_train_df=None,input_df=None,cat_columns=None):
        '''
        compute training grouped data object response aggregations by categorical feature 
        combinations as new features for train and test data subsets
        train_vec_df    : prevectorized training set, DataFrame
        input_df        : table to generate/add new grouped reponse features, DataFrame
        cat_columns     : categorical features to consider in groupby, list 
        '''
        # compute grouped object on pre-categorical-vectorized training data features 
        group_obj = vec_train_df.groupby(cat_columns)
        # generate new features on grouped obj w.r.t response variable  
        df = pd.DataFrame()
        df['cat_group_min'] = group_obj[self.data.ylabel].min()  
        df['cat_group_median'] = group_obj[self.data.ylabel].median() 
        df['cat_group_mean'] = group_obj[self.data.ylabel].mean() 
        df['cat_group_max'] = group_obj[self.data.ylabel].max()
        df['cat_group_std'] = group_obj[self.data.ylabel].std() 
        # select return rows
        df = df[['cat_group_min','cat_group_median','cat_group_mean','cat_group_max','cat_group_std']]
        df = df.reset_index() # collapse grouped levels
        result = self.data._join_df(input_df,df,col_id=cat_columns)
        return result


    def _compute_train_feature_quantile(self,vec_train_df=None,input_df=None,num_col=None,cuts=4):
        '''
        compute training set numeric feature quantile series
        '''
        df = vec_train_df[[num_col]].copy()
        df[num_col+'_quantile'] = pd.qcut(vec_train_df[num_col],q=cuts,labels=False)
        return df

    
    def _quantile_agg(self,vec_train_df=None,input_df=None,num_cols=None,cuts=4):
        '''
        compute new response stat features based on numeric quantile groups 
        '''
        # add quantile to training set
        q_cols=[]
        train_quantile_df = vec_train_df.copy()
        for numX in num_cols:
            # compute train-set-quantile for ea num_feature
            q_df = self._compute_train_feature_quantile(vec_train_df=vec_train_df,input_df=vec_train_df,num_col=numX,cuts=cuts)
            # join to df=input_df
            train_quantile_df = self.data._join_df(train_quantile_df,q_df.drop_duplicates(),col_id=numX)
            # update quantile column names for use in groupby
            q_cols.append(numX+'_quantile')
        # reduce to relevant join columns, add response for groupby agg w.r.t. response
        # and must drop duplicates to do merge, else will throw MemoryError 
        train_quantile_df = train_quantile_df[num_cols+q_cols+[self.data.ylabel]].drop_duplicates()
        # merge train_quantiles to input_df
        result_df = self.data._join_df(input_df,train_quantile_df[num_cols+q_cols].drop_duplicates(),col_id=num_cols)
        # compute grouped response stats on train_quantile_df
        group_obj = train_quantile_df.groupby(q_cols)
        stats_df = pd.DataFrame()
        # add response stats w.r.t. quantile groups
        stats_df['numeric_quantile_min'] = group_obj[self.data.ylabel].min()
        stats_df['numeric_quantile_median'] = group_obj[self.data.ylabel].median()
        stats_df['numeric_quantile_mean'] = group_obj[self.data.ylabel].mean()
        stats_df['numeric_quantile_max'] = group_obj[self.data.ylabel].max()
        stats_df['numeric_quantile_std'] = group_obj[self.data.ylabel].std()
        stats_df.reset_index(inplace=True)
        # merge response-stats-df on id quantiles 
        result_df = self.data._join_df(result_df,stats_df,col_id=q_cols)
        return result_df    

    def _compute_new_features(self,vec_train_df,input_df,cat_columns=None,num_columns=None,qcuts=4):
        '''
        consolidate feature engineering and data cleaning
        '''
        # compute response stats on categorical quantile groups
        df = self._cat_group_agg(vec_train_df=vec_train_df,input_df=input_df,cat_columns=self.data.columns_cat)
        # compute response stats on numeric quantile groups
        df = self._quantile_agg(vec_train_df=vec_train_df,input_df=df,num_cols=self.data.columns_num,cuts=qcuts)
        return df        