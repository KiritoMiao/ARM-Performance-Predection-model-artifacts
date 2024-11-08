def sebs_501(request, context): 
    from SAAF import Inspector
    import datetime, igraph
    req_size = int(request['size'])
    inspector = Inspector()
    inspector.inspectAll()
    graph_generating_begin = datetime.datetime.now()
    graph = igraph.Graph.Barabasi(req_size, 10)
    graph_generating_end = datetime.datetime.now()

    process_begin = datetime.datetime.now()
    result = graph.pagerank()
    process_end = datetime.datetime.now()
    inspector.addAttribute("graph_generating_time", (graph_generating_end - graph_generating_begin).total_seconds())
    inspector.addAttribute("process_time", (process_end - process_begin).total_seconds())
    inspector.inspectAllDeltas()

    return inspector.finish()