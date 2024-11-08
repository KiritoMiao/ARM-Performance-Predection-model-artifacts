def float_operation_python(request, context): 
    from SAAF import Inspector
    import math
    req_rounds= int(request['rounds'])
    inspector = Inspector()
    inspector.inspectAll()
    for i in range(req_rounds):
        sin_i = math.sin(i)
        cos_i = math.cos(i)
        sqrt_i = math.sqrt(i)

    inspector.inspectAllDeltas()

    return inspector.finish()