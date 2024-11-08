# filter out experiments, and keep only newcontainer = 0
df_train = df[df['experiment'].isin(CombinedTrainningList)]
df_train = df_train[df_train['newcontainer'] == '0'].copy()
# Drop columns that are not used
df_train.drop(['version','lang','startTime','invocations','initializationTime','uuid','newcontainer',
               'platform','containerID','functionRegion','functionMemory','vmID','linuxVersion',
               'bootTime','openssl_version','fio_version','payload.method','payload.link','payload.events',
               'payload.locks','payload.accessmode','payload.operation','payload.blocksize','sysbench_version',
               'sysbench_output','graph_generating_time','process_time','maxprime','rounds','size','yields',
               'payload.iodepth','payload.ioengine','payload.numjobs','payload.seed','payload.password',
               'payload.threads','payload.rw','payload.bs','payload.size','payload.buffer_size',
               'cpuCores','cpuUser','cpuNice','cpuKernel','cpuIdle','cpuIOWait','cpuIrq','cpuSoftIrq',
               'cpuSteal','cpuGuest','cpuGuestNice','cpuType','cpuModel','cpuInfo','payload','runtimeControlVarUnified','totalsize','timeControlVarUnit','timeControlVar',
               'pageFaults', 'cpuNiceDelta', 'cpuIrqDelta', 'cpuGuestDelta', 'cpuGuestNiceDelta', 'availableCPUs',
               'recommendedMemory', 'endTime', 'latency'], axis=1, inplace=True)
