def find_min_distance(lst):
    min_distance = 0
    sorted_lst = sorted(lst)
    for i in range(len(sorted_lst)-1):
        distance = sorted_lst[i+1] - sorted_lst[i]
        if distance < min_distance or min_distance == 0:
            min_distance = distance
    return min_distance

def find_step(lst):
    step = None
    sorted_lst = sorted(lst)
    # Find the largest distance between two consecutive elements
    for i in range(len(sorted_lst)-1):
        distance = sorted_lst[i+1] - sorted_lst[i]
        if step == None or distance > step:
            step = distance
    return step

def plot_graph(config, df):
    # Select experiment
    df_graph = df.copy()
    df_graph = df_graph[df_graph['experiment'].str.contains(config["experiment"])]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # x-axis title is the time control variable(e.g. rounds, size, etc.), if df have time control variable unit, add it to the title in the format of (unit).
    # y-axis title is the runtime in ms
    # title is the experiment name

    x_title = df_graph['timeControlVar'].unique()[0]
    if df_graph['timeControlVarUnit'].unique()[0] != None:
        x_title += " (" + df_graph['timeControlVarUnit'].unique()[0] + ")"
    fig.update_layout(
        barmode='stack',
        title= df_graph['function'].unique()[0].replace("_", " ").title() + '<br><sup>' + config["experiment"] + '</sup>',
        xaxis_title=x_title,
        yaxis_title="Runtime (ms)",
        yaxis2_title="Variation (%)",
        font=dict(size=16),
        height=config["height"],
        width=config["width"],
    )
    if "enableFixedYAxis2Range" in config["flags"]:
        fig.update_layout(yaxis2_range=[-100,100])

    if "enableSecondTimer" in config["flags"]:
        fig.update_layout(yaxis_title="Runtime (s)")

    if "enableWarmFilter" in config["flags"]:
        df_graph = df_graph[df_graph['newcontainer'] == '0']
    df_graph.sort_values(by=[df_graph['timeControlVar'].unique()[0]], inplace=True, ascending=True)
    # Create a net df with only the columns we need
    finalData = df_graph[['architecture', 'runtimeControlVarUnified'] + list(config["timings"].keys())]
    # Detatach from original df
    finalData = finalData.copy()


    # Convert all timings to float64
    for timing in config["timings"]:
        finalData = finalData.astype({timing: np.float64})
        if "enableSecondTimer" in config["flags"]:
            finalData[timing] = finalData[timing] / 1000
    # Group by architecture and time control variable, and calculate the mean of each column
    finalDataGrouped1 = finalData.groupby(['architecture', 'runtimeControlVarUnified']).mean().reset_index()

    # Calculate the difference between arm and x86 for each timing
    finalDataGrouped2 = finalData.groupby(['runtimeControlVarUnified', 'architecture']).mean().reset_index()
    finalDataGrouped2 = finalDataGrouped2.pivot(index='runtimeControlVarUnified', columns='architecture', values=list(config["timings"].keys()))
    # Cleanup NaN values by removing row
    finalDataGrouped2 = finalDataGrouped2.dropna()

    finalDataGrouped3 = pd.DataFrame()
    for timing in config["timings"]:
        finalDataGrouped3[timing + 'diff'] = (finalDataGrouped2[timing]['arm64'] - finalDataGrouped2[timing]['x86']) / finalDataGrouped2[timing]['x86'] * 100
    # Add raw data traces to the graph
    for timing in config["timings"]:
        if "raw" in config["modes"]:
            if "enableRawNearZeroFilter" and "enableSecondTimer" in config["flags"]:
                if finalDataGrouped1[finalDataGrouped1['architecture'] == 'x86'][timing].max() < 0.2:
                    if not "disableInfo" in config["flags"]: print("[Info][RAW] Skipping " + timing + " due to low values")
                    continue
            if "enableRawNearZeroFilter" in config["flags"] and not "enableSecondTimer" in config["flags"]:
                if finalDataGrouped1[finalDataGrouped1['architecture'] == 'x86'][timing].max() < 200:
                    if not "disableInfo" in config["flags"]: print("[Info][RAW] Skipping " + timing + " due to low values")
                    continue    
            fig.add_trace(
                go.Scatter(
                    x=finalDataGrouped1[finalDataGrouped1['architecture'] == 'x86']['runtimeControlVarUnified'],
                    y=finalDataGrouped1[finalDataGrouped1['architecture'] == 'x86'][timing],
                    name=config["timings"][timing][0] + '(x86)', line=dict(color=config["timings"][timing][1], width=1)),
                secondary_y=False,)
            fig.add_trace(
                go.Scatter(
                    x=finalDataGrouped1[finalDataGrouped1['architecture'] == 'arm64']['runtimeControlVarUnified'],
                    y=finalDataGrouped1[finalDataGrouped1['architecture'] == 'arm64'][timing],
                    name=config["timings"][timing][0] + '(arm)', line=dict(color=config["timings"][timing][2], width=1)),
                secondary_y=False,)
    # Add diff data traces to the graph
    for timing in config["timings"]:
        if "diff" in config["modes"]:
            # Skip if diff is too big, or too small, notice if the diff is too big, this filter will also filter out the correct data
            # In this case, we can use the "enableNoDiffFilter" flag to disable this filter
            if "enableNoDiffFilter" in config["flags"]:
                if finalDataGrouped3[timing + 'diff'].max() > 100:
                    if not "disableInfo" in config["flags"]: print("[Info][DIFF] Skipping " + timing + " due to high values")
                    continue
                if finalDataGrouped3[timing + 'diff'].min() < -100:
                    if not "disableInfo" in config["flags"]: print("[Info][DIFF] Skipping " + timing + " due to low values")
                    continue
                if finalDataGrouped3[timing + 'diff'].unique().size == 1:
                    if not "disableInfo" in config["flags"]: print("[Info][DIFF] Skipping " + timing + " due to no difference")
                    continue

            fig.add_trace(
                go.Scatter(
                    x=finalDataGrouped3.index.values,
                    y=finalDataGrouped3[timing + 'diff'],
                    name=config["timings"][timing][0] + '(diff)', line=dict(color=config["timings"][timing][3], width=1)),
                secondary_y=True,)
        
    display(fig)
    if config["save"]:
        fig.write_image(config["savePath"] + config["experiment"] + '.png')
        fig.write_html(config["savePath"] + config["experiment"] + '.html')
        if not "disableInfo" in config["flags"]: print("[Info] Saved to " + config["savePath"] + config["experiment"] + '.png')

def plot_ruDelta_graph(config, df):
    df_graph = df.copy()
    # Filter out experiments in config
    df_graph = df_graph[df_graph['experiment'].isin(config['experiments'])]
    # Keep only timing parameters in config
    finalData = df_graph[['architecture', 'experiment', 'function'] + list(config["timings"].keys())]
    finalData = finalData.copy()
    # Convert to int
    for timing in config["timings"]:
        finalData[timing] = finalData[timing].astype(np.float64)
    finalData = finalData.groupby(['architecture', 'experiment', 'function']).mean()
    finalData = finalData.reset_index()
    # Calculate the total runtime of each experiment(sum of all timings)
    finalData['runtime'] = finalData[config["timings"].keys()].sum(axis=1)
    # Calculate the percentage of each timing
    for timing in config["timings"]:
        finalData[timing + "Percent"] = finalData[timing] / finalData['runtime'] * 100

    # Plot 
    fig = go.Figure()
    fig.update_layout(
        barmode='stack',
        title= 'Ru Delta for ' + str(len(config['experiments'])) + ' experiments',
        xaxis_title="Experiment",
        yaxis_title="Percentage(%)",
        xaxis = dict( tickfont = dict(size=8)),
        width=config['width'],
        height=config['height'],
    )
    # Shorten experiment name
    if 'enableDate' in config['flags']:
        finalData['experiment'] = finalData['experiment'].str[:19] + '<br>' + finalData['experiment'].str[20:]
    else:
        finalData['experiment'] = finalData['experiment'].str[20:]
  
    finalData['experiment'] = finalData['experiment'].str.replace('_static', ' ')
    # Sort by cpuUser
    finalData = finalData.sort_values(by=['cpuUserDeltaPercent'], ascending=False)
    
    for timing in config["timings"]:
        fig.add_trace(
            go.Bar(x=[finalData[finalData['architecture'] == 'x86']['experiment'], ['x86']*len(finalData.experiment)], 
                   y=finalData[finalData['architecture'] == 'x86'][timing + "Percent"], name=config["timings"][timing][0], marker_color=config["timings"][timing][1])
        )
        fig.add_trace(
            go.Bar(x=[finalData[finalData['architecture'] == 'arm64']['experiment'], ['arm64']*len(finalData.experiment)], 
                     y=finalData[finalData['architecture'] == 'arm64'][timing + "Percent"], name=config["timings"][timing][0], marker_color=config["timings"][timing][2],showlegend=False)
        )    

    display(fig)
    if config["save"]:
        fig.write_image(config["savePath"] + 'ruDelta' + '.png')
        fig.write_html(config["savePath"] + 'ruDelta' + '.html')
        if not "disableInfo" in config["flags"]: print("[Info] Saved to " + config["savePath"] + 'ruDelta' + '.png')

def display_exp_status(config, df):
    # Copy dataframe
    df_stats = df.copy()
    if "enableWarmFilter" in config["flags"]:
        df_stats = df_stats[df_stats['newcontainer'] == '0']
    # Filter out experiments with less than 10 runs
    df_stats = df_stats.groupby(['experiment', 'architecture']).filter(lambda x: len(x) > 10)
    aggconfig = {
        'timeControlVar': 'first',
        'runtime': ['mean', 'std', 'min', 'max',('step', lambda x: find_step(x)), ('cv', lambda x: np.std(x) / np.mean(x))],
        'runtimeControlVarUnified': ['min', 'max', ('unique', pd.Series.nunique ), ('step', lambda x: find_min_distance(x.unique()))],
        'newcontainer' : ['sum', ('percent', lambda x: x.sum() / len(x))],
    }
    for timing in config['timings']:
        aggconfig[timing] = ['mean', 'std', 'min', 'max', ('cv', lambda x: np.std(x) / np.mean(x))]
        # Convert to float64
        df_stats[timing] = df_stats[timing].astype(np.float64)
        if "enableSecondTimer" in config['flags']: df_stats[timing] = df_stats[timing] / 1000
    df_stats["runtime"] = df_stats["runtime"].astype(np.float64)
    df_stats["newcontainer"] = df_stats["newcontainer"].astype(np.float64)

    # Filter out experiments in config, if config is empty, keep all experiments, otherwise filter out experiments not in config,
    if len(config['experiments']) > 0: df_stats = df_stats[df_stats['experiment'].isin(config['experiments'])]
    # Filter out experimentType in config, if config is "default", keep all experimentType, otherwise filter out experiments does not contain experimentType
    if config['experimentType'] != "default": df_stats = df_stats[df_stats['experiment'].str.contains(config['experimentType'])]
    if "enableSecondTimer" in config['flags']: df_stats['runtime'] = df_stats['runtime'] / 1000
    df_stats = df_stats.groupby(['experiment', 'architecture']).agg(aggconfig)
    # This block is used to prepar data to csv
    if config['save']:
        csv = df_stats.copy()
        csv.columns = csv.columns.map('_'.join)
        csv = csv.reset_index()
        csv = csv.rename(columns={"runtime_step": "runtime_step_size", "runtimeControlVarUnified_step": "runtimeControlVarUnified_step_size"})
        csv.to_csv("stats.csv", index=False)
        
    return (HTML(df_stats.to_html()))

def cleanup_df(df, config, experiment, flagWarmFilter, flagFilterTimeOver1Min, flagFilter3GCPU = True):
    df_liner = df.copy()
    # Filter out experiments in config
    df_liner = df_liner[df_liner['experiment'].str.contains(experiment)]
    # If enableWarmFilter, filter out warm runs
    if flagWarmFilter: df_liner = df_liner[df_liner['newcontainer'] == '0']
    # If enableFilter3GCPU, filter out 3Ghz CPU for x86 (Remove "Intel(R) Xeon(R) Processor @ 3.00GHz" CPU in cpuType)
    if flagFilter3GCPU: 
        # Print How many 3G CPU in the data
        print("[Info] 3G CPU in the data: ", df_liner[df_liner['architecture'] == 'x86'][df_liner['cpuType'] == 'Intel(R) Xeon(R) Processor @ 3.00GHz'].shape[0])
        df_liner = df_liner[df_liner['cpuType'] != 'Intel(R) Xeon(R) Processor @ 3.00GHz']
    # Keep only timing parameters in config
    df_liner = df_liner[['architecture', 'runtime', 'runtimeControlVarUnified'] + list(config["timings"].keys())]
    df_liner = df_liner.copy()

    # flagFilterTimeOver1Min
    #print("1Min Limit Enable:", flagFilterTimeOver1Min)
    if flagFilterTimeOver1Min == True:
        df_liner = df_liner[df_liner['runtime'] < 60000].copy()


    # Sort by runtimeControlVarUnified
    df_liner = df_liner.sort_values(by=['runtimeControlVarUnified'])
    df_liner = df_liner.copy()
    # Conver timings to float
    for k, v in config["timings"].items():
        df_liner[k] = df_liner[k].astype(np.float64)
    df_liner['runtime'] = df_liner['runtime'].astype(np.float64)
    df_liner = df_liner.copy()
    
    unique_runtimeControlVarUnified = df_liner['runtimeControlVarUnified'].unique()
    runs_per_runtimeControlVarUnified_arm = df_liner[df_liner['architecture'] == 'arm64'].groupby(['runtimeControlVarUnified']).size()
    runs_per_runtimeControlVarUnified_x86 = df_liner[df_liner['architecture'] == 'x86'].groupby(['runtimeControlVarUnified']).size()
    # Keep runtimeControlVarUnified that have least runs in both arm and x86
    for rtcvu in unique_runtimeControlVarUnified:
        try:
            min_runs = min(runs_per_runtimeControlVarUnified_arm[rtcvu], runs_per_runtimeControlVarUnified_x86[rtcvu])
        except:
            min_runs = 0
        finally:
            # Remove extra runs for both arm and x86
            try:
                df_liner = df_liner.drop(df_liner[(df_liner['runtimeControlVarUnified'] == rtcvu) & (df_liner['architecture'] == 'arm64')].tail(runs_per_runtimeControlVarUnified_arm[rtcvu] - min_runs).index)
            except:
                pass
            try: 
                df_liner = df_liner.drop(df_liner[(df_liner['runtimeControlVarUnified'] == rtcvu) & (df_liner['architecture'] == 'x86')].tail(runs_per_runtimeControlVarUnified_x86[rtcvu] - min_runs).index)
            except: 
                pass
    # Return to ungrouped dataframe
    df_liner = df_liner.reset_index(drop=True)
    df_liner = df_liner.copy()


    return df_liner
 
def liner_regression_model(config, df, experiment, flagWarmFilter = True, flagRuntimeMatch = False, flagFilterTimeOver1Min = False, flagFilter3GCPU = True):
    def generate_model(x, y):
        x = x.values.reshape(-1, 1)
        y = y.values.reshape(-1, 1)
        model = LinearRegression().fit(x, y)
        return model
    df_liner = cleanup_df(df, config, experiment, flagWarmFilter, flagFilterTimeOver1Min)

    if flagRuntimeMatch:
        # Sort by rcv and runtime
        df_liner = df_liner.sort_values(by=[ 'runtimeControlVarUnified', 'runtime'], ascending=True)
        df_liner = df_liner.copy().reset_index(drop=True)

            
    # Group by architecture
    df_liner = df_liner.groupby(['architecture'])
    
    models = {}
    models['runtime'] = generate_model(df_liner.get_group('x86')['runtime'], df_liner.get_group('arm64')['runtime'])
    models['multireg'] = LinearRegression().fit(df_liner.get_group('x86')[list(config["timings"].keys())], df_liner.get_group('arm64')['runtime'])
    for timing in config["timings"].keys():
        models[timing] = generate_model(df_liner.get_group('x86')[timing], df_liner.get_group('arm64')[timing])
    return models



def eval_liner_regression_model(models, config, df, experiment, flagWarmFilter = True, flagPlotGraph = False, flagPlotDetailedTimming = False, flagRuntimeMatch = False, flagFilterTimeOver1Min = False, flagFilter3GCPU = True):
    def compute_r2(x, y):
        x = x.values.reshape(-1, 1)
        y = y.values.reshape(-1, 1)
        ssr = np.sum((y - x)**2)
        sst = np.sum((y - np.mean(y))**2)
        return 1 - (ssr / sst)
    def compute_mape(x, y):
        x = x.values.reshape(-1, 1)
        y = y.values.reshape(-1, 1)
        return np.mean(np.abs((y - x) / y)) * 100
    def compute_model_score(x, y, model):
        x = x.values.reshape(-1, 1)
        y = y.values.reshape(-1, 1)
        return model.score(x, y)
    
    def pred_linux_timing_accounting(models, df, config, div_ra=2):
        y_pred = np.zeros(len(df.get_group('x86')))
        for timing in config["timings"].keys():
            y_pred += models[timing].predict(df.get_group('x86')[timing].values.reshape(-1, 1)).flatten()
        y_pred = y_pred / div_ra
        return y_pred
    def plot_graph(points, config, experiment, flagPlotDetailedTimming = False):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        # Update title
        fig.update_layout(title_text="Runtime vs. Runtime Control Variable for Measured Runs and Predicted Runs")
        # Update X axis
        fig.update_xaxes(title_text="Runtime Control Variable")
        # Update Y axis
        fig.update_yaxes(title_text="Runtime (ms)", secondary_y=False)
        # Update Y2 axis
        fig.update_yaxes(title_text="Differance between measure and predect(%)", secondary_y=True)
        fig.add_trace(go.Scatter(x=points['rcv'], y=points['x86_runtime'], mode='lines', name='x-runtime'), secondary_y=False)
        fig.add_trace(go.Scatter(x=points['rcv'], y=points['arm_runtime'], mode='lines', name='a-runtime'), secondary_y=False)
        fig.add_trace(go.Scatter(x=points['rcv'], y=points['pred_runtime'], mode='lines', name='p-runtime'), secondary_y=False)
        fig.add_trace(go.Scatter(x=points['rcv'], y=points['pred_multireg'], mode='lines', name='p-multireg'), secondary_y=False)
        fig.add_trace(go.Scatter(x=points['rcv'], y=points['pred_linux_timing_accounting'], mode='lines', name='p-lta'), secondary_y=False)
        fig.add_trace(go.Scatter(x=points['rcv'], y=abs(((points['pred_runtime'] / points['arm_runtime'])-1)*100), mode='lines', name='d-runtime'), secondary_y=True)
        fig.add_trace(go.Scatter(x=points['rcv'], y=abs(((points['pred_multireg'] / points['arm_runtime'])-1)*100), mode='lines', name='d-multireg'), secondary_y=True)
        fig.add_trace(go.Scatter(x=points['rcv'], y=abs(((points['pred_linux_timing_accounting'] / points['arm_runtime'])-1)*100), mode='lines', name='d-lta'), secondary_y=True)
        if flagPlotDetailedTimming:
            for timing in config["timings"].keys():
                fig.add_trace(go.Scatter(x=points['rcv'], y=points['x86_' + timing], mode='lines', name='x-' + timing), secondary_y=False)
                fig.add_trace(go.Scatter(x=points['rcv'], y=points['arm_' + timing], mode='lines', name='a-' + timing), secondary_y=False)
                fig.add_trace(go.Scatter(x=points['rcv'], y=points['pred_' + timing], mode='lines', name='p-' + timing), secondary_y=False)
                fig.add_trace(go.Scatter(x=points['rcv'], y=abs(((points['pred_' + timing] / points['arm_' + timing])-1)*100), mode='lines', name='d-' + timing), secondary_y=True)
        fig.write_html("graph/" + experiment + '_graph.html')
        fig.show()

    df_liner = cleanup_df(df, config, experiment, flagWarmFilter, flagFilterTimeOver1Min)

    if flagRuntimeMatch:
        # Sort by rcv and runtime
        df_liner = df_liner.sort_values(by=[ 'runtimeControlVarUnified', 'runtime'], ascending=True)
        df_liner = df_liner.copy().reset_index(drop=True)


    # Group by architecture
    df_liner = df_liner.groupby(['architecture'])
    points = {}
    points['rcv'] = df_liner.get_group('x86')['runtimeControlVarUnified'].tolist()
    points['x86_runtime'] = df_liner.get_group('x86')['runtime'].tolist()
    points['arm_runtime'] = df_liner.get_group('arm64')['runtime'].tolist()
    points['pred_runtime'] = models['runtime'].predict(df_liner.get_group('x86')['runtime'].values.reshape(-1, 1)).flatten()
    points['pred_multireg'] = models['multireg'].predict(df_liner.get_group('x86')[list(config["timings"].keys())]).flatten()
    points['pred_linux_timing_accounting'] = pred_linux_timing_accounting(models, df_liner, config).flatten()
    for timing in config["timings"].keys():
        points['x86_' + timing] = df_liner.get_group('x86')[timing].tolist()
        points['arm_' + timing] = df_liner.get_group('arm64')[timing].tolist()
        points['pred_' + timing] = models[timing].predict(df_liner.get_group('x86')[timing].values.reshape(-1, 1)).flatten()
    points = pd.DataFrame(points)

    # Calculate R2 and MAPE for each model
    status = {}
    status['r2'] = {}
    status['mape'] = {}
    status['percent'] = {}
    status['r2']['runtime'] = compute_r2(points['arm_runtime'], points['pred_runtime'])
    status['mape']['runtime'] = compute_mape(points['arm_runtime'], points['pred_runtime'])
    
    status['r2']['multireg'] = compute_r2(points['arm_runtime'], points['pred_multireg'])
    status['mape']['multireg'] = compute_mape(points['arm_runtime'], points['pred_multireg'])

    status['r2']['linux_timing_accounting'] = compute_r2(points['arm_runtime'], points['pred_linux_timing_accounting'])
    status['mape']['linux_timing_accounting'] = compute_mape(points['arm_runtime'], points['pred_linux_timing_accounting'])

    for timing in config["timings"].keys():
        status['r2'][timing] = compute_r2(points['arm_' + timing], points['pred_' + timing])
        status['mape'][timing] = compute_mape(points['arm_' + timing], points['pred_' + timing])
        # Calculate the timing percentage in the total runtime by sum(timing) / sum(runtime) /2
        status['percent'][timing] = (points['x86_' + timing].sum() + points['arm_' + timing].sum()) / (points['x86_runtime'].sum() + points['arm_runtime'].sum()) / 2 * 100
    status = pd.DataFrame(status)
    scores = {}
    scores['runtime'] = compute_model_score(points['x86_runtime'], points['arm_runtime'], models['runtime'])
    scores['multireg'] = models['multireg'].score(df_liner.get_group('x86')[list(config["timings"].keys())], df_liner.get_group('arm64')['runtime'])
    for timing in config["timings"].keys():
        scores[timing] = compute_model_score(points['x86_' + timing], points['arm_' + timing], models[timing])
    if flagPlotGraph:
        plot_graph(points, config, experiment, flagPlotDetailedTimming)
    return points, status, scores

def rand_forest_model(config, df, experiment, flagWarmFilter = True, flagRuntimeMatch = False, flagFilterTimeOver1Min = False, flagFilter3GCPU = True):
    def generate_model(x, y):
        x = x.values.reshape(-1, 1)
        #y = y.values.reshape(-1, 1)
        model = RandomForestRegressor(n_estimators = 1000, random_state = 42)
        model = model.fit(x, y)
        return model
    df_liner = cleanup_df(df, config, experiment, flagWarmFilter, flagFilterTimeOver1Min)
    if flagRuntimeMatch:
        # Sort by rcv and runtime
        df_liner = df_liner.sort_values(by=[ 'runtimeControlVarUnified', 'runtime'], ascending=True)
        df_liner = df_liner.copy().reset_index(drop=True)

    # Group by architecture
    df_liner = df_liner.groupby(['architecture'])
    models = {}
    models['runtime'] = generate_model(df_liner.get_group('x86')['runtime'], df_liner.get_group('arm64')['runtime'])
    models['multireg'] = RandomForestRegressor(n_estimators = 1000, random_state = 42).fit(df_liner.get_group('x86')[list(config["timings"].keys())], df_liner.get_group('arm64')['runtime'])

    for timing in config["timings"].keys():
        models[timing] = generate_model(df_liner.get_group('x86')[timing], df_liner.get_group('arm64')[timing])
    return models

def eval_rand_forest_model(models, config, df, experiment, flagWarmFilter = True, flagPlotGraph = False, flagPlotDetailedTimming = False, flagRuntimeMatch = False, flagFilterTimeOver1Min = False, flagFilter3GCPU = True):
    def compute_r2(x, y):
        x = x.values.reshape(-1, 1)
        y = y.values.reshape(-1, 1)
        ssr = np.sum((y - x)**2)
        sst = np.sum((y - np.mean(y))**2)
        return 1 - (ssr / sst)
    def compute_mape(x, y):
        x = x.values.reshape(-1, 1)
        y = y.values.reshape(-1, 1)
        return np.mean(np.abs((y - x) / y)) * 100
    def compute_model_score(x, y, model):
        x = x.values.reshape(-1, 1)
        y = y.values.reshape(-1, 1)
        return model.score(x, y)
    
    def pred_linux_timing_accounting(models, df, config, div_ra=2):
        y_pred = np.zeros(len(df.get_group('x86')))
        for timing in config["timings"].keys():
            y_pred += models[timing].predict(df.get_group('x86')[timing].values.reshape(-1, 1)).flatten()
        y_pred = y_pred / div_ra
        return y_pred
    def plot_graph(points, config, experiment, flagPlotDetailedTimming = False):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        # Update title
        fig.update_layout(title_text="Runtime vs. Runtime Control Variable for Measured Runs and Predicted Runs")
        # Update X axis
        fig.update_xaxes(title_text="Runtime Control Variable")
        # Update Y axis
        fig.update_yaxes(title_text="Runtime (ms)", secondary_y=False)
        # Update Y2 axis
        fig.update_yaxes(title_text="Differance between measure and predect(%)", secondary_y=True)
        fig.add_trace(go.Scatter(x=points['rcv'], y=points['x86_runtime'], mode='lines', name='x-runtime'), secondary_y=False)
        fig.add_trace(go.Scatter(x=points['rcv'], y=points['arm_runtime'], mode='lines', name='a-runtime'), secondary_y=False)
        fig.add_trace(go.Scatter(x=points['rcv'], y=points['pred_runtime'], mode='lines', name='p-runtime'), secondary_y=False)
        fig.add_trace(go.Scatter(x=points['rcv'], y=points['pred_linux_timing_accounting'], mode='lines', name='p-lta'), secondary_y=False)
        fig.add_trace(go.Scatter(x=points['rcv'], y=abs(((points['pred_runtime'] / points['arm_runtime'])-1)*100), mode='lines', name='d-runtime'), secondary_y=True)
        fig.add_trace(go.Scatter(x=points['rcv'], y=abs(((points['pred_linux_timing_accounting'] / points['arm_runtime'])-1)*100), mode='lines', name='d-lta'), secondary_y=True)
        if flagPlotDetailedTimming:
            for timing in config["timings"].keys():
                fig.add_trace(go.Scatter(x=points['rcv'], y=points['x86_' + timing], mode='lines', name='x-' + timing), secondary_y=False)
                fig.add_trace(go.Scatter(x=points['rcv'], y=points['arm_' + timing], mode='lines', name='a-' + timing), secondary_y=False)
                fig.add_trace(go.Scatter(x=points['rcv'], y=points['pred_' + timing], mode='lines', name='p-' + timing), secondary_y=False)
                fig.add_trace(go.Scatter(x=points['rcv'], y=abs(((points['pred_' + timing] / points['arm_' + timing])-1)*100), mode='lines', name='d-' + timing), secondary_y=True)
        fig.write_html("graph/" + experiment + '_graph.html')
        fig.show()
    
    df_liner = cleanup_df(df, config, experiment, flagWarmFilter, flagFilterTimeOver1Min)
    if flagRuntimeMatch:
        # Sort by rcv and runtime
        df_liner = df_liner.sort_values(by=[ 'runtimeControlVarUnified', 'runtime'], ascending=True)
        df_liner = df_liner.copy().reset_index(drop=True)

            
    # Group by architecture
    df_liner = df_liner.groupby(['architecture'])
    points = {}
    points['rcv'] = df_liner.get_group('x86')['runtimeControlVarUnified'].tolist()
    points['x86_runtime'] = df_liner.get_group('x86')['runtime'].tolist()
    points['arm_runtime'] = df_liner.get_group('arm64')['runtime'].tolist()
    points['pred_runtime'] = models['runtime'].predict(df_liner.get_group('x86')['runtime'].values.reshape(-1, 1)).flatten()
    points['pred_linux_timing_accounting'] = pred_linux_timing_accounting(models, df_liner, config).flatten()
    for timing in config["timings"].keys():
        points['x86_' + timing] = df_liner.get_group('x86')[timing].tolist()
        points['arm_' + timing] = df_liner.get_group('arm64')[timing].tolist()
        points['pred_' + timing] = models[timing].predict(df_liner.get_group('x86')[timing].values.reshape(-1, 1)).flatten()
    points = pd.DataFrame(points)

    # Calculate R2 and MAPE for each model
    status = {}
    status['r2'] = {}
    status['mape'] = {}
    status['percent'] = {}
    status['r2']['runtime'] = compute_r2(points['arm_runtime'], points['pred_runtime'])
    status['mape']['runtime'] = compute_mape(points['arm_runtime'], points['pred_runtime'])
    status['r2']['linux_timing_accounting'] = compute_r2(points['arm_runtime'], points['pred_linux_timing_accounting'])
    status['mape']['linux_timing_accounting'] = compute_mape(points['arm_runtime'], points['pred_linux_timing_accounting'])

    for timing in config["timings"].keys():
        status['r2'][timing] = compute_r2(points['arm_' + timing], points['pred_' + timing])
        status['mape'][timing] = compute_mape(points['arm_' + timing], points['pred_' + timing])
        # Calculate the timing percentage in the total runtime by sum(timing) / sum(runtime) /2
        status['percent'][timing] = (points['x86_' + timing].sum() + points['arm_' + timing].sum()) / (points['x86_runtime'].sum() + points['arm_runtime'].sum()) / 2 * 100
    status = pd.DataFrame(status)
    scores = {}
    scores['runtime'] = compute_model_score(points['x86_runtime'], points['arm_runtime'], models['runtime'])
    for timing in config["timings"].keys():
        scores[timing] = compute_model_score(points['x86_' + timing], points['arm_' + timing], models[timing])
    if flagPlotGraph:
        plot_graph(points, config, experiment, flagPlotDetailedTimming)
    return points, status, scores


def get_experiment_list(DataFrame, experimentType = 'ALL' ,doNotIgnoreBrokenExperiments = False):
    temp_df = DataFrame.copy()

    if not doNotIgnoreBrokenExperiments:
        # If the experiment contain less than 10 rows, it's probably broken, so we ignore it
        temp_df = temp_df.groupby('experiment').filter(lambda x: len(x) > 10)
    if experimentType == 'ALL':
        return temp_df['experiment'].unique().tolist()
    else:
        # Only contain experiments that contain the experimentType string in their name
        return temp_df['experiment'].unique()[[experimentType in x for x in temp_df['experiment'].unique()]].tolist()
    
def get_function_list(DataFrame):
    return DataFrame['function'].unique().tolist()

def buildModel(experimentList, modelConfig):
    f = widgets.IntProgress(min=0, max=3) 
    h = widgets.HTMLMath(value="<p style=\"color:#FFFFFF\">Now building model... <br><br> Initialization: 0/3</p>")
    display(f, h, clear = False) # display the bar
    # Filter out experiments in config
    h.value = "<p style=\"color:#FFFFFF\">Now building model... <br><br> Cleanup Datas: 1/3</p>"
    df_model = df[df['experiment'].isin(experimentList)]

    f.value += 1
    h.value = "<p style=\"color:#FFFFFF\">Now building model... <br><br> Liner Regression: 2/3</p>"
    LRModel = liner_regression_model(modelConfig, df_model, '', flagRuntimeMatch = True)
    f.value += 1
    h.value = "<p style=\"color:#FFFFFF\">Now building model... <br><br> Random Forest: 3/3</p>"
    RFModel = rand_forest_model(modelConfig, df_model, '', flagRuntimeMatch = True)
    f.value += 1
    h.value = "<p style=\"color:#FFFFFF\">Model build complete.</p>"
    return LRModel, RFModel

def evaluateModel(modelsConfig, experimentList, LRModel, RFModel):
    f = widgets.IntProgress(min=0, max=(len(experimentList)*2+2))
    h = widgets.HTMLMath(value="<p style=\"color:#FFFFFF\">Now evaluating model... <br><br> Initialization: 0/" + str(len(experimentList)*2+2) + "</p>")
    display(f, h, clear=True) # display the bar
    statusGroup = {}
    for i in range(len(experimentList)):
        #print("Processing experiment: " + experimentList[i])
        f.value += 1
        h.value = "<p style=\"color:#FFFFFF\">Now evaluating model... <br><br> Liner Regression(" + experimentList[i] + "): " + str(f.value) + "/" + str(len(experimentList)*2+2) + "</p>"
        points, status, scores = eval_liner_regression_model(LRModel, modelsConfig, df, experimentList[i], flagPlotGraph=False, flagPlotDetailedTimming=True, flagRuntimeMatch = True)

        statusGroup[experimentList[i][20:-12]] = status
    # Convert to dataframe
    f.value += 1
    h.value = "<p style=\"color:#FFFFFF\">Now evaluating model... <br><br> Liner Regression(Building Graph): " + str(f.value) + "/" + str(len(experimentList)*2+2) + "</p>"
    statusGroup = pd.concat(statusGroup, axis=1)
    statusGroupWork = statusGroup.copy()
    statusGroupWork = statusGroupWork.transpose()
    statusGroupWork = statusGroupWork.reset_index()
    statusGroupWork = statusGroupWork.rename(columns={"level_0": "Experiment"})
    statusGroupWork = statusGroupWork.rename(columns={"level_1": "metric"})



    statusGroupWorkR2 = statusGroupWork[statusGroupWork['metric'] == 'r2'].drop(columns=['metric']).set_index('Experiment')
    statusGroupWorkR2['dist-runtime/LTA'] = abs(statusGroupWorkR2['runtime'] - statusGroupWorkR2['linux_timing_accounting'])
    statusGroupWorkR2['dist-runtime/multireg'] = abs(statusGroupWorkR2['runtime'] - statusGroupWorkR2['multireg'])
    statusGroupWorkR2['dist-LTA/multireg'] = abs(statusGroupWorkR2['linux_timing_accounting'] - statusGroupWorkR2['multireg'])
    # Average of the distance between the winner and the other two
    statusGroupWorkR2['dist-best'] = statusGroupWorkR2[['dist-runtime/LTA', 'dist-runtime/multireg', 'dist-LTA/multireg']].min(axis=1)
    statusGroupWorkR2['dist-avg'] = (statusGroupWorkR2[['dist-runtime/LTA', 'dist-runtime/multireg', 'dist-LTA/multireg']].sum(axis=1) - statusGroupWorkR2['dist-best']) / 2
    statusGroupWorkR2.drop(columns=['dist-best'], inplace=True)



    statusGroupWorkMAPE = statusGroupWork[statusGroupWork['metric'] == 'mape'].drop(columns=['metric']).set_index('Experiment')
    statusGroupWorkMAPE['dist-runtime/LTA'] = abs(statusGroupWorkMAPE['runtime'] - statusGroupWorkMAPE['linux_timing_accounting'])
    statusGroupWorkMAPE['dist-runtime/multireg'] = abs(statusGroupWorkMAPE['runtime'] - statusGroupWorkMAPE['multireg'])
    statusGroupWorkMAPE['dist-LTA/multireg'] = abs(statusGroupWorkMAPE['linux_timing_accounting'] - statusGroupWorkMAPE['multireg'])
    statusGroupWorkMAPE['dist-best'] = statusGroupWorkMAPE[['dist-runtime/LTA', 'dist-runtime/multireg', 'dist-LTA/multireg']].max(axis=1)
    statusGroupWorkMAPE['dist-avg'] = (statusGroupWorkMAPE[['dist-runtime/LTA', 'dist-runtime/multireg', 'dist-LTA/multireg']].sum(axis=1) - statusGroupWorkMAPE['dist-best']) / 2
    statusGroupWorkMAPE.drop(columns=['dist-best'], inplace=True)
    statusRFMAPE = statusGroupWorkMAPE.copy()

    # AVG MAPE for all workloads in a additional line
    statusGroupWorkR2.loc['mean'] = statusGroupWorkR2.mean()
    statusGroupWorkMAPE.loc['mean'] = statusGroupWorkMAPE.mean()

    # Highlight the best result in "runtime", "multireg" and "linux_timing_accounting" for each experiment
    LRR2 = statusGroupWorkR2.style.highlight_max(axis=1, subset=['runtime', 'multireg', 'linux_timing_accounting'], color='red')

    LRMAPE = statusGroupWorkMAPE.style.highlight_min(axis=1, subset=['runtime', 'multireg', 'linux_timing_accounting'], color='red')
    


    statusGroup = {}
    for i in range(len(experimentList)):
        #print("Processing experiment: " + experimentList[i])
        f.value += 1
        h.value = "<p style=\"color:#FFFFFF\">Now evaluating model... <br><br> Random Forest(" + experimentList[i] + "): " + str(f.value) + "/" + str(len(experimentList)*2+2) + "</p>"
        points, status, scores = eval_liner_regression_model(RFModel, modelsConfig, df, experimentList[i], flagPlotGraph=False, flagPlotDetailedTimming=True, flagRuntimeMatch = True)
        statusGroup[experimentList[i][20:-12]] = status
    # Convert to dataframe
    f.value += 1
    h.value = "<p style=\"color:#FFFFFF\">Now evaluating model... <br><br> Random Forest(Building Graph): " + str(f.value) + "/" + str(len(experimentList)*2+2) + "</p>"
    statusGroup = pd.concat(statusGroup, axis=1)
    statusGroupWork = statusGroup.copy()
    statusGroupWork = statusGroupWork.transpose()
    statusGroupWork = statusGroupWork.reset_index()
    statusGroupWork = statusGroupWork.rename(columns={"level_0": "Experiment"})
    statusGroupWork = statusGroupWork.rename(columns={"level_1": "metric"})



    statusGroupWorkR2 = statusGroupWork[statusGroupWork['metric'] == 'r2'].drop(columns=['metric']).set_index('Experiment')
    statusGroupWorkR2['dist-runtime/LTA'] = abs(statusGroupWorkR2['runtime'] - statusGroupWorkR2['linux_timing_accounting'])
    statusGroupWorkR2['dist-runtime/multireg'] = abs(statusGroupWorkR2['runtime'] - statusGroupWorkR2['multireg'])
    statusGroupWorkR2['dist-LTA/multireg'] = abs(statusGroupWorkR2['linux_timing_accounting'] - statusGroupWorkR2['multireg'])
    # Average of the distance between the winner and the other two
    statusGroupWorkR2['dist-best'] = statusGroupWorkR2[['dist-runtime/LTA', 'dist-runtime/multireg', 'dist-LTA/multireg']].min(axis=1)
    statusGroupWorkR2['dist-avg'] = (statusGroupWorkR2[['dist-runtime/LTA', 'dist-runtime/multireg', 'dist-LTA/multireg']].sum(axis=1) - statusGroupWorkR2['dist-best']) / 2
    statusGroupWorkR2.drop(columns=['dist-best'], inplace=True)



    statusGroupWorkMAPE = statusGroupWork[statusGroupWork['metric'] == 'mape'].drop(columns=['metric']).set_index('Experiment')
    statusGroupWorkMAPE['dist-runtime/LTA'] = abs(statusGroupWorkMAPE['runtime'] - statusGroupWorkMAPE['linux_timing_accounting'])
    statusGroupWorkMAPE['dist-runtime/multireg'] = abs(statusGroupWorkMAPE['runtime'] - statusGroupWorkMAPE['multireg'])
    statusGroupWorkMAPE['dist-LTA/multireg'] = abs(statusGroupWorkMAPE['linux_timing_accounting'] - statusGroupWorkMAPE['multireg'])
    statusGroupWorkMAPE['dist-best'] = statusGroupWorkMAPE[['dist-runtime/LTA', 'dist-runtime/multireg', 'dist-LTA/multireg']].max(axis=1)
    statusGroupWorkMAPE['dist-avg'] = (statusGroupWorkMAPE[['dist-runtime/LTA', 'dist-runtime/multireg', 'dist-LTA/multireg']].sum(axis=1) - statusGroupWorkMAPE['dist-best']) / 2
    statusGroupWorkMAPE.drop(columns=['dist-best'], inplace=True)
    statusRFMAPE = statusGroupWorkMAPE.copy()

    # AVG MAPE for all workloads in a additional line
    statusGroupWorkR2.loc['mean'] = statusGroupWorkR2.mean()
    statusGroupWorkMAPE.loc['mean'] = statusGroupWorkMAPE.mean()

    # Highlight the best result in "runtime", "multireg" and "linux_timing_accounting" for each experiment
    RFR2 = statusGroupWorkR2.style.highlight_max(axis=1, subset=['runtime', 'multireg', 'linux_timing_accounting'], color='red')

    RFMAPE = statusGroupWorkMAPE.style.highlight_min(axis=1, subset=['runtime', 'multireg', 'linux_timing_accounting'], color='red')
    h.value = "<p style=\"color:#FFFFFF\">Model evaluation complete."

    display(
        widgets.HTMLMath(value="<p style=\"color:#FFFFFF\">Liner Regression:R^2</p>"),
        LRR2, 
        widgets.HTMLMath(value="<p style=\"color:#FFFFFF\">Liner Regression:MAPE</p>"),
        LRMAPE, 
        widgets.HTMLMath(value="<p style=\"color:#FFFFFF\">Random Forest:R^2</p>"),
        RFR2, 
        widgets.HTMLMath(value="<p style=\"color:#FFFFFF\">Random Forest:MAPE</p>"),
        RFMAPE)
    
modelsConfig = {
    "timings": {
        "cpuUserDelta": ["CPU User", "blue", "darkblue"],
        "cpuKernelDelta": ["CPU Kernel", "cyan", "darkcyan"],
        "cpuIdleDelta": ["CPU Idle", "green", "darkgreen"],
        #"cpuIOWaitDelta": ["CPU IO Wait", "yellow", "greenyellow"],
        #"cpuIrqDelta": ["CPU IRQ", "pink", "hotpink"],
        #"cpuSoftIrqDelta": ["CPU SoftIRQ", "red", "darkred"],
    },
}

