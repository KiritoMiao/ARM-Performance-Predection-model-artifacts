ARMFastTestList= [    
    ...
    ]
ARMSlowTestList = [
    ...
    ]
ARMSimilarTestList = [
    ...
    ]

# Generate the individual arm-fast, arm-slow, arm-similar rate for each experiment, arm fast is arm 15% faster than x86, arm-slow is arm 15% slower than x86, arm-similar is arm within 15% range to x86
testExperimentList = ARMFastTestList + ARMSlowTestList + ARMSimilarTestList
df_test = df[df['experiment'].isin(testExperimentList)]
df_test = df_test[['experiment','runtime','architecture', 'runtimeControlVarUnified']]

# Group the data by experiment and architecture and runtimeControlVarUnified, sort the data by runtime and pair x86 with arm64
df_test = df_test.groupby(['experiment','architecture','runtimeControlVarUnified']).apply(lambda x: x.sort_values('runtime')).reset_index(drop=True)
df_test['runtime'] = df_test['runtime'].astype(int)

# Pair the x86 with arm64 by the runtimeControlVarUnified and experiment
df_test_x86 = df_test[df_test['architecture'] == 'x86'].copy()
df_test_arm = df_test[df_test['architecture'] == 'arm64'].copy()
df_test_x86['runtime'] = df_test_x86['runtime'].astype(int)
df_test_arm['runtime'] = df_test_arm['runtime'].astype(int)
# reinex the index
df_test_x86.reset_index(drop=True, inplace=True)
df_test_arm.reset_index(drop=True, inplace=True)
display(df_test_x86)
display(df_test_arm)
current_experiment = ""
current_rcv = 0
current_arm_index = 0
for item in df_test_x86.iterrows():
    rcv = item[1]['runtimeControlVarUnified']
    exp = item[1]['experiment']
    # get the arm64 data with the same runtimeControlVarUnified and experiment
    df_arm_datepeace = df_test_arm[(df_test_arm['runtimeControlVarUnified'] == rcv) & (df_test_arm['experiment'] == exp)]
    # check if the experiment and runtimeControlVarUnified is the same as the last one
    if current_experiment != exp or current_rcv != rcv:
        current_experiment = exp
        current_rcv = rcv
        current_arm_index = 0
    try:
        # add runtime_arm to x86 data
        df_test_x86.at[item[0],'runtime_arm'] = df_arm_datepeace.iloc[current_arm_index]['runtime']
        current_arm_index += 1
    except:
        continue
# drop na
df_test_x86.dropna(inplace=True)
display(df_test_x86)

df_test = df_test_x86.copy()
# drop architecture and runtimeControlVarUnified
df_test.drop(['architecture','runtimeControlVarUnified'], axis=1, inplace=True)
# calculate the rate of arm64 to x86
#df_cost_ratio['ratio'] = (df_cost_ratio['arm64']/df_cost_ratio['x86'] - 1 )* 100
df_test['rate'] = (df_test['runtime_arm']/df_test['runtime'] - 1) * 100
# add the label to the data
df_test['label'] = 'arm-similar'
# if the rate is greater than 15, then it is arm-slow
df_test.loc[df_test['rate'] > 15, 'label'] = 'arm-slow'
# if the rate is less than -15, then it is arm-fast
df_test.loc[df_test['rate'] < -15, 'label'] = 'arm-fast'
display(df_test)
# calculate the min and max rate for each experiment
df_rate = df_test.groupby(['experiment'])['rate'].agg(['min','max'])
# add distance for rate min and max
df_rate['distance'] = abs(df_rate['max'] - df_rate['min'])
# add avg and median rate 
df_rate['avg'] = df_test.groupby(['experiment'])['rate'].mean()
df_rate['median'] = df_test.groupby(['experiment'])['rate'].median()

display(df_rate)
# calculate the number of arm-fast, arm-slow, arm-similar
df_rate = df_test.groupby(['experiment'])['label'].value_counts().unstack(fill_value=0)
display(df_rate)
