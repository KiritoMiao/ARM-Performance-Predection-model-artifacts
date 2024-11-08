# Label the data by ARMFastList, ARMSlowList, ARMSimilarList
df_train['label'] = '0'
df_train.loc[df_train['experiment'].isin(ARMFastList), 'label'] = '1'
df_train.loc[df_train['experiment'].isin(ARMSlowList), 'label'] = '2'
df_train.loc[df_train['experiment'].isin(ARMSimilarList), 'label'] = '3'


# Train the model use RandomForestClassifier
# Using the x86 data to estimate the arm64 label
model = sklearn.ensemble.AdaBoostClassifier()
df_train_x86 = df_train[df_train['architecture'] == 'x86'].copy()
df_train_x86.drop(['architecture','functionName','function','experiment'], axis=1, inplace=True)

model.fit(df_train_x86.drop(['label'], axis=1), df_train_x86['label'])
    


testExperimentList = ARMFastTestList + ARMSlowTestList + ARMSimilarTestList
df_test = df[df['experiment'].isin(testExperimentList)]
df_test = df_test[df_test['newcontainer'] == '0'].copy()
df_test['label'] = '0'
df_test.loc[df_test['experiment'].isin(ARMFastTestList), 'label'] = '1'
df_test.loc[df_test['experiment'].isin(ARMSlowTestList), 'label'] = '2'
df_test.loc[df_test['experiment'].isin(ARMSimilarTestList), 'label'] = '3'
# Drop columns that are not used
df_test.drop(['version','lang','startTime','invocations','initializationTime','uuid','newcontainer',
               'platform','containerID','functionRegion','functionMemory','vmID','linuxVersion',
               'bootTime','openssl_version','fio_version','payload.method','payload.link','payload.events',
               'payload.locks','payload.accessmode','payload.operation','payload.blocksize','sysbench_version',
               'sysbench_output','graph_generating_time','process_time','maxprime','rounds','size','yields',
               'payload.iodepth','payload.ioengine','payload.numjobs','payload.seed','payload.password',
               'payload.threads','payload.rw','payload.bs','payload.size','payload.buffer_size',
               'cpuCores','cpuUser','cpuNice','cpuKernel','cpuIdle','cpuIOWait','cpuIrq','cpuSoftIrq',
               'cpuSteal','cpuGuest','cpuGuestNice','cpuType','cpuModel','cpuInfo','payload','runtimeControlVarUnified','totalsize','timeControlVarUnit','timeControlVar',
               'pageFaults', 'cpuNiceDelta', 'cpuIrqDelta', 'cpuGuestDelta', 'cpuGuestNiceDelta', 'availableCPUs',
               'recommendedMemory', 'endTime'], axis=1, inplace=True)

df_test_x86 = df_test[df_test['architecture'] == 'x86'].copy()
# Predict the label of test data, and add to the dataframe as a new column, named 'predictLabel'
df_test_x86['predictLabel'] = model.predict(df_test_x86.drop(['architecture','functionName','function','experiment', 'label'], axis=1))

# Add a column to show if the prediction is correct
df_test_x86['correct'] = '0'
df_test_x86.loc[df_test_x86['predictLabel'] == df_test_x86['label'], 'correct'] = '1'
display(df_test_x86[['experiment','predictLabel', 'label', 'correct']])
# Calculate the accuracy pre experiment
df_test_x86['correct'] = df_test_x86['correct'].astype(int)
df_test_x86['label'] = df_test_x86['label'].astype(int)
df_test_x86['predictLabel'] = df_test_x86['predictLabel'].astype(int)

# Calculate the accuracy 
accuracy = df_test_x86.groupby(['experiment'])['correct'].sum() / df_test_x86.groupby(['experiment'])['correct'].count()
print("Accuracy of each experiment:")
display(accuracy)
print("Average accuracy of all experiments:")
display(accuracy.mean())
cm = confusion_matrix(df_test_x86['label'], df_test_x86['predictLabel'], labels=[1,2,3])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1,2,3])
disp.plot()
