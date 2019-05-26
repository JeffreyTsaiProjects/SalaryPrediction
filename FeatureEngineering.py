import pandas as pd


class FeatureEng:
    def __init__(self,data_object=None):
        self.data = data_object

        
    def _cat_group_agg(self,vec_train_df,input_df,cat_columns):
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
        df = df[['cat_group_min','cat_group_median',
                 'cat_group_mean','cat_group_max','cat_group_std']]
        df = df.reset_index() # collapse grouped levels
        result = pd.merge(input_df,df,on=cat_columns,how='left')
        return result


    def _compute_train_feature_quantile(self,vec_train_df,input_df,
                                        num_col,cuts=4):
        '''
        compute training set numeric feature quantile series
        vec_train_df    : pre-vectorized training set data (without new numeric features), DataFrame
        input_df        : training data to process, DataFrame
        num_col         : numeric feature name to compute quantile, str
        cuts            : number of evenly-spaced quantiles to generate as new feature, int
        returns:        : DataFrame
        '''
        df = vec_train_df[[num_col]].copy()
        df[num_col+'_quantile'] = pd.qcut(vec_train_df[num_col],
                                          q=cuts,labels=False)
        return df

    
    def _quantile_agg(self,vec_train_df,input_df,num_cols,cuts=4):
        '''
        compute new response stat features based on numeric quantile groups
        vec_train_df    : pre-vectorized training set data, DataFrame
        input_df        : training data to process, DataFrame
        num_cols        : numeric feature names to compute quantiles, list of str
        cuts            : number of evenly-spaced quantiles to generate as new feature, int
        returns         : DataFrame
        '''
        # add quantile to training set
        q_cols=[]
        train_quantile_df = vec_train_df.copy()
        for numX in num_cols:
            # compute train-set-quantile for ea num_feature
            q_df = self._compute_train_feature_quantile(vec_train_df=vec_train_df,
                                                        input_df=vec_train_df,
                                                        num_col=numX,cuts=cuts)
            # join to df=input_df
            train_quantile_df = pd.merge(train_quantile_df,
                                         q_df.drop_duplicates(),on=numX,
                                         how='left')
            # update quantile column names for use in groupby
            q_cols.append(numX+'_quantile')
        # reduce to relevant join columns, add response for groupby agg w.r.t. response
        # and must drop duplicates to do merge, else will throw MemoryError 
        train_quantile_df = train_quantile_df[num_cols+q_cols+[self.data.ylabel]].drop_duplicates()
        # merge train_quantiles to input_df
        result_df = pd.merge(input_df,
                             train_quantile_df[num_cols+q_cols].drop_duplicates(),
                             on=num_cols,how='left')
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
        result_df = pd.merge(result_df,stats_df,on=q_cols,how='left')
        return result_df    

    
    def _compute_new_features(self,vec_train_df,input_df,
                              cat_columns,num_columns,qcuts=4):
        '''
        runs feature engineering and data cleaning procedures in single function call
        vec_train_df    : pre-vectorized training set data, DataFrame
        input_df        : training data to process, DataFrame
        cat_columns     : categorical features to consider in groupby for _cat_group_agg(), list of str
        num_columns     : numeric feature names to compute quantiles, list of str
        qcuts           : number of evenly-spaced quantiles to generate as new numeric features, int
        returns         : DataFrame
        '''
        # compute response stats on categorical quantile groups
        df = self._cat_group_agg(vec_train_df,input_df,
                                 self.data.columns_cat)
        # compute response stats on numeric quantile groups
        df = self._quantile_agg(vec_train_df,df,
                                self.data.columns_num,cuts=qcuts)
        return df        