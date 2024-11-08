def cpu_crypto_loader_openssl(request, context):
    from SAAF import Inspector 
    import subprocess
    import os
    import string
    import random   
    blocksize=8192000 # 8MB
    cleartext = ''.join(random.choices(string.ascii_uppercase +
                             string.digits, k=blocksize))
    # write to file
    with open('/tmp/cleartext', 'w') as f:
        f.write(cleartext)
    # Disable CPU Crypto
    os.environ["OPENSSL_armcap"] = "0"
    os.environ["OPENSSL_ia32cap"] = "~0x20000000"
    req_method = request.get('method')
    req_rounds = request.get('rounds')
    req_password = request.get('password')
    random.seed(request.get('seed'))
    inspector = Inspector()
    inspector.inspectAll()
    # write openssl version
    inspector.addAttribute("openssl_version", subprocess.run(['openssl','version'], check=True, stdout=subprocess.PIPE).stdout.decode('utf-8'))
    # Run Benchmark
    subprocess.run(['openssl','enc', req_method ,'-salt','-pbkdf2','-iter', req_rounds,'-in', '/tmp/cleartext','-out','/tmp/ciphertext',
                    '-pass','pass:' + str(req_password)], check=True, stdout=subprocess.PIPE)
    inspector.inspectAllDeltas()
    return inspector.finish()