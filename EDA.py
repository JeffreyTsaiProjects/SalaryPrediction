class EDA:
    def __init__(self,train_df=None,num_df=None,cat_df=None,num_column_list=None,cat_column_list=None):
        '''
        init eda class
        '''
        self.df = train_df # contains both training X features and y response
        self.num_df = num_df
        self.cat_df = cat_df
        self.num_column_list = num_column_list
        self.cat_column_list = cat_column_list
        
    def _separate_feature_dtypes(self,numomitlist=None,catomitlist=None):
        '''
        create numeric, categorical dataframes and respective column name lists
        '''
        self.num_df = self.df.select_dtypes(include=[np.number]).copy()
        if numomitlist:
            self.num_df.drop(numomitlist,axis=1,inplace=True)
        self.cat_df = self.df.select_dtypes(exclude=[np.number]).copy()
        if catomitlist:
            self.cat_df.drop(catomitlist,axis=1,inplace=True)
        self.num_colnames = [x for x in list(self.num_df.columns)]
        self.cat_colnames = [x for x in list(self.cat_df.columns)]
        
    def _describe_info(self,rows=0):
        '''
        calls pandas dataframe describe and info methods for dataset overview
        rows   : number of rows to display in df.head(rows)
        '''
        shape = self.df.shape
        print('\nnumber of rows: {:,.0f}'.format(shape[0]))
        print('\nnumber of columns: {}\n'.format(shape[1]))
        self.num_des = self.df.describe()
        self.cat_des = self.df.select_dtypes(include='object').describe()
        self.info = self.df.info()
        display(self.num_des)
        print('\n')
        display(self.cat_des)
        print('\n')
        display(self.info)
        if rows>0:
            display(self.df.head(rows))

    def _find_missing_values(self,colname_list=None):
        '''
        flag missing (NaN) values
        '''
        missing_rows = self.df.isnull()
        if not missing_rows.values.any():
            print('\nNo missing values in train set\n')
        else: print('\nNaN values exist in train set\n')    
        # count missing in ea col
        print('-------------------------Missing data count-------------------------')
        display(missing_rows.sum())
        # percent missing in ea col
        print('-------------------------Missing data percent-----------------------')
        display(missing_rows.mean())
        
    def _drop_missing(self,colnameList=None):
        '''
        drops rows where missing values exist for features in colnameList
        '''
        rowlen0 = len(self.df)
        print('\nshape before dropna:',self.df.shape)
        if colnameList:
            self.df.dropna(subset=colnameList,inplace=True)
        else:
            self.df.dropna(inplace=True)
        print('shape after dropna:',self.df.shape)
        print('rows dropped:',rowlen0-len(self.df))
            
    def _find_duplicates(self):
        '''
        method which flags duplicated rows
        '''
        # keep=first considers first row as orig, next rows as duplicated 
        bool_duplicates,duplicate_count = self.df.duplicated(),self.df.duplicated().sum()
        print('Duplicated rows:',duplicate_count)
        dupe_df = self.df[bool_duplicates]
        if len(dupe_df)>0:
            display(dupe_df)
        else:
            print('\nThere are no duplicated rows')
        print('-'*70)
        
    def drop_columns(self,df=None,column_to_drop=None):
        '''
        drop a column from a dataframe
        '''
        if column_to_drop in df.columns:
            df.drop(column_to_drop,axis=1,inplace=True)
        else:
            print(column_to_drop,'not in DataFrame.')
            
    def _iqr_outlier(self,df=None,Xname=None):
        '''
        computes feature IQR outlier tails as Q1-(IQR*1.5), Q3+(IQR*1.5)
        df       : data to compute df.describe(), DataFrame 
        Xname    : X feature name to compute IQR 1.5 outliers, str
        '''
        des = df.describe()
        q1,q3 = des[Xname]['25%'], des[Xname]['75%']  
        iqr = q3-q1
        # compute so-called IQR outliers 
        iqr_tail = iqr*1.5
        outlier_low,outlier_high = q1-iqr_tail, q3+iqr_tail
        return outlier_low,outlier_high
    
    def _compute_skewtest(self,col_name):
        '''
        compute statistical test for skewness of feature 
        H0: the population that the sample was drawn from is the same as that of
            a corresponding normal distribution 
        '''
        zscore,pvalue = skewtest(self.df[col_name])
        print('\nscipy.skewtest result:\nz-score={:.3f}\npvalue={:.3f}\n'.format(zscore,pvalue))
        if pvalue<.05:
            print('At alpha=.05, we can reject the null hypothesis that the skewness of the population that the sample was drawn is the same as that of a corresponding normal distribution.\n')
        else:
            print('At alpha=.05, fail to reject the null hypothesis that the skewness of the population that the sample was drawn is the same as that of a corresponding normal distribution.\n')

    def _viz_response(self,response_name='',figsz=(20,6),xrotate=45):
        '''
        eda on response variable
        '''
        f,ax = plt.subplots(nrows=1,ncols=3,figsize=figsz)
        ax[0] = sns.distplot(self.df[response_name].dropna(),fit=stats.norm,norm_hist=False,ax=ax[0])
        # plot 1.5*IQR outliers indicators
        if np.issubdtype(self.df[response_name].dtype,np.number): # numeric features
            iqr_low,iqr_high = self._iqr_outlier(df=self.df,Xname=response_name)
            ax[0].axvline(iqr_low,linestyle=':',color='gray')
            ax[0].axvline(iqr_high,linestyle=':',color='gray')
        ax[0].set(title='{} vs. normal distribution overlay'.format(response_name))    
        ax[1] = sns.boxplot(self.df[response_name],ax=ax[1])
        ax[1].axvline(iqr_low,linestyle=':',color='gray')
        ax[1].axvline(iqr_high,linestyle=':',color='gray')
        ax[2] = sns.heatmap(self.df.corr(),ax=ax[2],cmap="YlGnBu")
        for tick in ax[2].get_xticklabels():
            tick.set_rotation(xrotate)
        for tick in ax[2].get_yticklabels():
            tick.set_rotation(xrotate)
        
        
    def _plot_series(self,refdf,col_name='salary',figsz=(20,6)):
        '''
        viz distribution of data series 
        '''
        fig,ax = plt.subplots(nrows=1,ncols=2,figsize=figsz)
        # show IQR*1.5 outliers
        iqr_low,iqr_high = self._iqr_outlier(df=refdf,Xname=col_name)
        # scipy.skew
        skew_value = skew(self.df[col_name])
        # histogram
        ax[0] = sns.distplot(self.df[col_name].dropna(),kde=False,norm_hist=False,fit=stats.norm,ax=ax[0])
        ax[0].axvline(iqr_low,linestyle=':',color='gray')
        ax[0].axvline(iqr_high,linestyle=':',color='gray')
        ax[0].set(title='Q1-1.5*Q1={:.3f}, Q3+1.5*Q3={:.3f}, scipy.skew={:.3f}'.format(iqr_low,iqr_high,skew_value))
        # boxplot
        ax[1] = sns.boxplot(self.df[col_name],ax=ax[1])
        # zero indicator
        ax[1].axvline(0,linestyle='--',color='k')
        # iqr outlier indicators
        ax[1].axvline(iqr_low,linestyle=':',color='gray')
        ax[1].axvline(iqr_high,linestyle=':',color='gray')
        ax[1].set(title='Q1-1.5*Q1={:.3f},  Q3+1.5*Q3={:.3f}'.format(iqr_low,iqr_high))
        
    def _plot_feature(self,refdf=None,feature_name=None,response_name='salary',plot_type=None,sns_sample_n=None,xrotate=45,figsz=(20,5)):
        '''
        plot linear relationship between feature and response variables
        
        refdf    : unprocessed reference table used to compute IQR Q1,Q3 tails 
        '''
        if np.issubdtype(self.df[feature_name].dtype,np.number): # numeric features
            try:
                iqr_low,iqr_high = self._iqr_outlier(df=refdf,Xname=feature_name)
            except: print('{} in refdf remains categorical, feature_name must be transformed to numeric type'.format(feature_name))    
            fig,ax = plt.subplots(nrows=1,ncols=4,figsize=figsz)
            # random sample 10,000 instances for sns.kde to compute in reasonable time
            if sns_sample_n:
                # boxplot distribution
                ax[0] = sns.boxplot(self.df[feature_name],ax=ax[0])
                # density distribution
                ax[1] = sns.distplot(self.df[feature_name],fit=stats.norm,kde=False,norm_hist=False,ax=ax[1])
                # scatter feature,response
                sample_df = shuffle(self.df[:sns_sample_n].dropna())
                ax[2] = sns.kdeplot(sample_df[feature_name],sample_df[response_name], shade=True, ax=ax[2])
                ax[2].set(title='Random Sample n={} from Train Set'.format(sns_sample_n))
                ax[3].hexbin(sample_df[feature_name],sample_df[response_name],cmap='inferno') 
                # overlay regression line onto scatter
                ax[3] = sns.regplot(sample_df[feature_name],sample_df[response_name],scatter=False,color='white',ax=ax[3])
                ax[3].set(title='Random Sample n={} from Train Set'.format(sns_sample_n),
                         xlabel=feature_name,ylabel=response_name)
        else:
            fig,ax = plt.subplots(nrows=1,ncols=2,figsize=figsz)
            # means by category group
            grp = self.df.groupby([feature_name])
            # sorted category mean salary
            response_means_by_category = grp[response_name].mean().sort_values().dropna().index # .index to get the ordered cat name
            # categorical value_counts percentage
            cat_value_counts_pct = (self.df[feature_name].value_counts())/len(self.df)
            if plot_type:
                ax[0] = sns.barplot(x=cat_value_counts_pct,y=response_means_by_category,label=cat_value_counts_pct,ax=ax[0])
            else:
                ax[0] = sns.barplot(x=cat_value_counts_pct,y=response_means_by_category,label=cat_value_counts_pct,ax=ax[0])
            # overlay line for trend    
            ax[0].set(title='Value Counts Percent by {}'.format(feature_name),ylabel='value_counts (%)')
            for tick in ax[0].get_xticklabels():
                tick.set_rotation(xrotate)
            ax[1] = sns.boxplot(self.df[feature_name],self.df[response_name],order=response_means_by_category,ax=ax[1]) 
            ax[1].set(title='Mean salary by {}'.format(feature_name),xlabel=None)
            # set x_axis xticks rotation
            ax[1].set_xticklabels(ax[1].get_xticklabels(),rotation=xrotate)
            for tick in ax[1].get_xticklabels():
                tick.set_rotation(xrotate)
                
    def correl_matrix(self,df=None,response_name='salary',figsz=(10,10),xrotate=45):
        '''
        computes correlations between features
        this is a public method
        '''
        corr_matrix = df.corr()
        fig,ax = plt.subplots(figsize=figsz)
        sns.heatmap(corr_matrix)#,cmap="YlGnBu",ax=ax)
        for tick in ax.get_xticklabels():
            tick.set_rotation(xrotate)
        for tick in ax.get_yticklabels():
            tick.set_rotation(xrotate)
        corr_df = pd.DataFrame(corr_matrix,columns=list(corr_matrix.index))            
        display(corr_df)    

    def _cat_value_counts(self,response_name='salary',feature_list=[]):
        '''
        compute value counts by feature 
        '''
        for f in feature_list:
            val_counts = self.df[f].value_counts().sort_values(ascending=False)
            val_counts_df = pd.DataFrame(val_counts)
            val_counts_df['count_pct'] = val_counts_df[f]/val_counts_df[f].sum()
            # add grouped category mean salary col
            grp = self.df.groupby(f)
            val_counts_df['salary_min'] = grp[response_name].apply(lambda x: x.min()) 
            val_counts_df['salary_median'] = grp[response_name].apply(lambda x: x.median()) 
            val_counts_df['salary_mean'] = grp[response_name].apply(lambda x: x.mean()) 
            val_counts_df['salary_max'] = grp[response_name].apply(lambda x: x.max()) 
            val_counts_df['salary_std'] = grp[response_name].apply(lambda x: x.std()) 
            display(val_counts_df.sort_values('salary_mean',ascending=False))
            
    def feature_heatmap(self,df,response_name='salary',excludelist=None,catorderdict=None,deg=2,figsz=(10,10),xrotate=45,yrotate=45,wspace=.5):
        '''
        visualize feature interactions
        '''
        # compute interactions only
        poly = PolynomialFeatures(interaction_only=True,include_bias=False,degree=deg)
        # drop response to exclude it from polynomial/interaction features
        Xdf = df.drop('salary',axis=1)
        # separate categoricals from numeric
        cat_df = df.select_dtypes(exclude=[np.number]).copy()
        num_df = Xdf.select_dtypes(include=[np.number]).copy()
        catcols = [x for x in list(cat_df.columns) if x not in excludelist]        
        for col in catcols:
            num_df[col+'_code'] = cat_df[col].map(catorderdict[col])
        # compute polynomials on real value numeric only 
        Xpoly = poly.fit_transform(num_df)
        # dataframe with poly features 
        Xcolnames = num_df.columns        
        poly_df = pd.DataFrame(Xpoly,columns=poly.get_feature_names(Xcolnames))
        # compute correl netween features only for multicollinearity
        corr_multicollinearity = poly_df.corr()
        # add response back to df for correl vs response
        poly_df[response_name] = df[response_name].values
        # compute correl matrix, now with response included
        corr_matrix = poly_df.corr()
        corr_response = pd.DataFrame(corr_matrix[response_name].sort_values(ascending=False))
        fig,ax = plt.subplots(nrows=1,ncols=2,figsize=figsz)
        ax[0] = sns.heatmap(corr_response,annot=True,cbar=False,ax=ax[0])
        ax[0].set(title='X feature vs. y reponse correlations')
        ax[1] = sns.heatmap(corr_multicollinearity,annot=True,fmt='.1f',cbar=False,ax=ax[1])
        ax[1].set(title='X feature correlations')
        for tick in ax[1].get_xticklabels():
            tick.set_rotation(xrotate)
        for tick in ax[1].get_yticklabels():
            tick.set_rotation(yrotate)
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=None)    
            
    def _hue_regplot(self,x=None,y='salary',xhue='degree',xsize=None,figsz=(10,6),n_sample=None):
        '''
        seaborn numeric X vs. y scatterplot with categorical hue
        '''
        fig,ax = plt.subplots(figsize=figsz)
        # descrease plot density with sample
        if n_sample:
            df = shuffle(self.df)
            ax = sns.scatterplot(x=x,y=y,hue=xhue,size=xsize,data=df[:n_sample])
            ax = sns.regplot(df[x],df[y],scatter=False,color='black',ax=ax)
        else:
            ax = sns.scatterplot(x=x,y=y,hue=xhue,size=xsize,data=self.df)
            ax = sns.regplot(self.df[x],self.df[y],scatter=False,color='black',ax=ax)
        ax.set(title='train data n={:,.0f} random sample'.format(n_sample))    
            
    def _interaction_plot(self,x=None,y='salary',hue=None,kind='violin',aggfunc='mean',n_samples=100,figsz=(10,6)):
        '''
        compute seaborn catplot, cat x with cat hue vs. y salary 
        '''
        fig,ax= plt.subplots(figsize=figsz)
        # groupby to order x axis by category aggregation values
        cat_response_agg = self.df.groupby(x).agg({y:aggfunc})
        xorder = cat_response_agg.sort_values(y).index
        # groupby to order hue order in x axis 
        hue_response_agg = self.df.groupby(hue).agg({y:aggfunc})
        hueorder = hue_response_agg.sort_values(y).index
        df = shuffle(self.df)
        if kind=='violinplot':
            ax = sns.violinplot(x=x,y=y,hue=hue,data=df[:n_samples],order=xorder,hue_order=hueorder,ax=ax)
        elif kind=='stripplot':
            ax = sns.stripplot(x=x,y=y,hue=hue,data=df[:n_samples],order=xorder,hue_order=hueorder,ax=ax)
        else:
            print('Error: valid "kind" parameters are violinplot or stripplot.')
            
    def company_groups(self,df,x=None,y='salary',cat_columns=None,aggfunc='mean',hue=None,figsz=(10,6),legcols=8,table=False):
        '''
        compute aggregations by company
        '''
        group_obj = df.groupby(cat_columns)
        df = pd.DataFrame()
        # agg(response) by [companyId,degree,jobType,industry,major]
        group_agg_df = group_obj.agg({y:aggfunc}).reset_index()
        y_order = group_agg_df.groupby(x)[y].mean().sort_values().index
        hue_order = group_agg_df.groupby(hue)[y].mean().sort_values().index
        if table:
            display(group_agg_df.head())
        f,ax = plt.subplots(figsize=figsz)
        ax = sns.stripplot(x=x,y=y,hue=hue,data=group_agg_df,order=y_order,hue_order=hue_order,jitter=.2,palette='inferno',ax=ax)
        ax.legend(ncol=legcols)

    def numeric_quantiler(self,input_df,column=None,cuts=4):
        '''
        compute evenly spaced quantiles of numeric feature
        '''
        df = pd.DataFrame()
        df[column+'_quantile'] = pd.qcut(input_df[column],q=cuts,labels=False)
        df['jobId'] = input_df['jobId']
        return pd.merge(input_df,df,on='jobId')

    def quantile_agg(self,input_df,Xgroup=None,x='Xnum',cuts=4,y=None,aggfunc='mean',kind='violinplot',figsz=(10,6),legcols=8):
        '''
        compute x feature evenly spaced quantiles and then compute response 
        stat aggregations on cut number of quantile groups
        '''
        df = input_df.copy()
        # add quantile column
        df = self.numeric_quantiler(df,column=x,cuts=cuts)
        if type(Xgroup)!=list:
            group_obj = df.groupby([Xgroup]+[x+'_quantile'],as_index=False)
        else:
            group_obj = df.groupby(Xgroup+[x+'_quantile'],as_index=False)
        g = group_obj.agg({y:'mean'})
        display(g.head())
        Xorder = g[x]
        hue_order = g[x+'_quantile']
        fig,ax = plt.subplots(figsize=figsz)
        if kind=='violinplot':
            ax = sns.violinplot(x=x,y=y,hue=x+'_quantile',order=Xorder,hue_order=hue_order,data=df,ax=ax)
        else:
            ax = sns.stripplot(x=x,y=y,hue=x+'_quantile',order=Xorder,hue_order=hue_order,data=df,ax=ax)
        ax.legend(ncol=legcols)
    
