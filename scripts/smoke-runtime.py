"""Authenticated runtime check. Credentials come from the container environment; output is sanitized."""
import concurrent.futures
import json
import os
import time
import httpx

base='http://127.0.0.1:8090'
client=httpx.Client(base_url=base, timeout=210)
health=client.get('/health'); assert health.status_code==200
assert health.json()['auth']=={'required': True, 'configured': True}
payload={'message':'Jordan, Our deployment workflow gives engineering managers a clear view of failed releases. Your team can compare the change history, find the owner and restore the last working version from one screen. Would a short demonstration next Tuesday help you evaluate it?', 'persona':'An engineering manager who needs reliable releases and a practical demonstration before buying.', 'platform':'email'}
assert client.post('/score',json=payload).status_code==401
assert client.post('/refine',content=b' '*(128*1024+1)).status_code==413
login=client.post('/auth/login',json={'username':os.environ['PITCHSERVER_AUTH_SEED_USERNAME'], 'password':os.environ['PITCHSERVER_AUTH_SEED_PASSWORD']})
assert login.status_code==200, 'Seed credentials do not match persisted account'
client.headers['Authorization']='Bearer '+login.json()['token']
assert client.post('/score',json={**payload,'message':'short'}).status_code==422
started=time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
    future=pool.submit(client.post,'/score',json=payload)
    latencies=[]; observed_busy=False
    while not future.done():
        now=time.monotonic()
        h=client.get('/health')
        latencies.append(time.monotonic()-now)
        assert h.status_code==200
        if h.json()['pipeline']['active_scores']:
            unload=client.post('/runtime/unload').json()
            if not unload['ok']:
                assert unload['reason']=='score_in_progress'
                observed_busy=True
        time.sleep(.1)
    response=future.result()
assert response.status_code==200, f'Score HTTP {response.status_code}'
report=response.json()['report']; assert 0<=report['persuasion_score']<=100
assert report['fmri_output']['voxel_count']==20484
assert report['robustness']['llm_model'], 'Provider fell back to neural-only report'
assert observed_busy, 'Busy unload protection was not exercised'
score_seconds=round(time.monotonic()-started,3)
started=time.monotonic()
refine=client.post('/refine',json={**payload,'forceRewrite':True,'suggestions':['Keep a simple CTA and preserve the verified facts.']})
assert refine.status_code==200, f'Refine HTTP {refine.status_code}'
data=refine.json(); assert data['refined_message'] or data['questions']
refine_seconds=round(time.monotonic()-started,3)
unload=client.post('/runtime/unload').json(); assert unload['ok'] and not unload['model_loaded']
result={'login':True,'unauthorized_status':401,'oversized_status':413,'invalid_status':422,
        'score_status':response.status_code,'score_seconds':score_seconds,'llm_model':report['robustness']['llm_model'],
        'voxel_count':report['fmri_output']['voxel_count'],'health_max_seconds':round(max(latencies),3),
        'busy_unload_blocked':observed_busy,'refine_status':refine.status_code,'refine_seconds':refine_seconds,
        'runtime_unloaded':True}
print(json.dumps(result,indent=2))
