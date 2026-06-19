import json, os, signal, subprocess, sys, time, urllib.request
VENV="/tmp/senpai-venvs/20f658587e8a6643/bin/python"
MODEL="google/gemma-4-E4B-it-qat-w4a16-ct"
backend=sys.argv[1] if len(sys.argv)>1 else "FLASHINFER"
port=8137
argv=[VENV,"-m","vllm.entrypoints.openai.api_server","--model",MODEL,
 "--served-model-name","gemma-4-e4b-it","--quantization","compressed-tensors",
 "--dtype","bfloat16","--max-model-len","4096","--max-num-seqs","1",
 "--gpu-memory-utilization","0.90","--trust-remote-code","--no-enable-log-requests",
 "--host","127.0.0.1","--port",str(port)]
env=dict(os.environ)
env.update({"CUDA_VISIBLE_DEVICES":"0","VLLM_BATCH_INVARIANT":"1",
 "VLLM_ATTENTION_BACKEND":backend,"VLLM_USE_FLASHINFER_SAMPLER":"0",
 "HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1",
 "PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True","VLLM_LOGGING_LEVEL":"INFO"})
log=open("research/validity/partial_bi_identity_pareto_760/runs/boot_%s.log"%backend,"w")
print(f"[smoke] backend={backend} booting...",flush=True)
p=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
deadline=time.time()+360
ok=False
try:
    while time.time()<deadline:
        if p.poll() is not None:
            print(f"[smoke] PROC EXITED rc={p.returncode}",flush=True); break
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health",timeout=5) as r:
                if r.status==200: ok=True; print("[smoke] HEALTHY",flush=True); break
        except Exception: pass
        time.sleep(5)
    else:
        print("[smoke] TIMEOUT not healthy in 360s",flush=True)
finally:
    if p.poll() is None:
        os.killpg(os.getpgid(p.pid),signal.SIGINT)
        try: p.wait(timeout=25)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(p.pid),signal.SIGKILL); p.wait(timeout=20)
    log.close()
print("[smoke] RESULT:",("BOOTS" if ok else "FAILS"),flush=True)
sys.exit(0 if ok else 1)
