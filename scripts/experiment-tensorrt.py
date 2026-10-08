"""Isolated real TensorRT encoder/voxel comparison. Never used by the service."""
import gc
import os
import importlib.metadata
import ctypes
import json
import random
import resource
import sys
import time
import traceback
from pathlib import Path

if not Path('/.dockerenv').exists() or os.getenv('PITCHCHECK_TRT_EXPERIMENT') != '1':
    raise RuntimeError('Run only in a disposable Docker container with PITCHCHECK_TRT_EXPERIMENT=1')
assert importlib.metadata.version('torch-tensorrt') == '2.7.0'
assert importlib.metadata.version('tensorrt') == '10.9.0.34'

import numpy as np
import torch

converter = Path('/usr/local/lib/python3.11/site-packages/torch_tensorrt/dynamo/conversion/aten_ops_converters.py')
source = converter.read_text()
old = '            torch.float16,\n        }'
assert source.count(old) == 1
converter.write_text(source.replace(old, '            torch.float16,\n            torch.bfloat16,\n        }'))
# Default contexts reserve maximum-profile scratch separately for every block.
# ponytail: one workspace supports serial calls only; use per-context memory for concurrency.
# Runtime stream fences order each block after the previous one.
runtime = converter.parent.parent / 'runtime/_PythonTorchTensorRTModule.py'
source = runtime.read_text()
context_creation = 'self.engine.create_execution_context()'
caller_stream = '                self._caller_stream = torch.cuda.current_stream()'
assert source.count(context_creation) == 3 and source.count(caller_stream) == 2
source = source.replace(context_creation,
    'self.engine.create_execution_context(trt.ExecutionContextAllocationStrategy.USER_MANAGED)')
source = source.replace(caller_stream,
    '                _pitchcheck_workspace_for_context(self.context)\n'
    '                self._caller_stream = torch.cuda.current_stream()')
source += '''
_pitchcheck_shared_workspace = None
def _pitchcheck_workspace_for_context(context):
    global _pitchcheck_shared_workspace
    required = context.update_device_memory_size_for_shapes()
    budget = 768 * 1024**2
    if required > budget:
        raise MemoryError(f"TensorRT workspace needs {required} bytes, budget is {budget}")
    if _pitchcheck_shared_workspace is None:
        _pitchcheck_shared_workspace = torch.empty(budget, dtype=torch.uint8, device="cuda")
    context.device_memory = _pitchcheck_shared_workspace.data_ptr()
'''
runtime.write_text(source)
import torch_tensorrt
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPast

ROOT = Path('/audit')
MODEL = 'NousResearch/Hermes-3-Llama-3.2-3B'
torch.manual_seed(0)
torch.cuda.set_per_process_memory_fraction(8 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)


class Decoder(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, hidden, mask, cos, sin):
        return self.block(hidden, attention_mask=mask, position_embeddings=(cos, sin), use_cache=False)


class CompiledDecoder(torch.nn.Module):
    def __init__(self, compiled):
        super().__init__()
        self.compiled = compiled

    def forward(self, hidden, attention_mask, position_embeddings, **kwargs):
        return self.compiled(hidden, attention_mask, *position_embeddings)


class Encoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config

    def forward(self, input_ids, attention_mask, **kwargs):
        hidden = self.model.embed_tokens(input_ids)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        position_ids = positions.unsqueeze(0)
        mask = (positions[:,None]>=positions[None,:])[None,None] & attention_mask[:,None,None,:].bool()
        embeddings = self.model.rotary_emb(hidden, position_ids=position_ids)
        states = [hidden]
        for layer in self.model.layers:
            hidden = layer(hidden, attention_mask=mask, position_embeddings=embeddings, use_cache=False)
            states.append(hidden)
        states[-1] = self.model.norm(hidden)
        return BaseModelOutputWithPast(last_hidden_state=states[-1], hidden_states=tuple(states))


result = {'model': MODEL, 'torch': torch.__version__, 'torch_tensorrt': torch_tensorrt.__version__,
          'scope': 'all_28_decoder_blocks_and_real_tribe_predictions', 'compiled': False,
          'bf16_cast_validator_patch': True, 'eligible_for_deployment': False,
          'blocks': [], 'rows': []}


def save():
    result['max_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    (ROOT/'probe-result.json').write_text(json.dumps(result,indent=2)+'\n')


try:
    save()
    print('Loading real encoder and calibration tensors', flush=True)
    model = AutoModel.from_pretrained(MODEL, local_files_only=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(['Our deployment workflow gives managers a clear view of failed releases.',
                        'Your team can restore the last working version from one screen.',
                        'Would a demonstration help?', 'Find the owner and compare changes.'],
                       return_tensors='pt', padding=True).to('cuda')
    length = inputs.input_ids.shape[1]
    with torch.no_grad():
        hidden = model.embed_tokens(inputs.input_ids)
        positions = torch.arange(length, device='cuda')
        cos, sin = model.rotary_emb(hidden, position_ids=positions.unsqueeze(0))
        mask = (positions[:,None]>=positions[None,:])[None,None] & inputs.attention_mask[:,None,None,:].bool()
        args = (hidden, mask, cos, sin)
    batch = torch.export.Dim('batch', min=1, max=4)
    sequence = torch.export.Dim('sequence', min=1, max=1024)
    profiles = [
        torch_tensorrt.Input(min_shape=(1,1,3072),opt_shape=(4,128,3072),max_shape=(4,1024,3072),dtype=torch.bfloat16),
        torch_tensorrt.Input(min_shape=(1,1,1,1),opt_shape=(4,1,128,128),max_shape=(4,1,1024,1024),dtype=torch.bool),
        torch_tensorrt.Input(min_shape=(1,1,128),opt_shape=(1,128,128),max_shape=(1,1024,128),dtype=torch.bfloat16),
        torch_tensorrt.Input(min_shape=(1,1,128),opt_shape=(1,128,128),max_shape=(1,1024,128),dtype=torch.bfloat16)]
    model.cpu()
    torch.cuda.empty_cache()
    for index, block in enumerate(model.layers):
        start = time.perf_counter()
        wrapper = Decoder(block.cuda())
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            exported = torch.export.export(wrapper,args,dynamic_shapes=(
                {0:batch,1:sequence},{0:batch,2:sequence,3:sequence},
                {1:sequence},{1:sequence}))
        wrapper.cpu()
        torch.cuda.empty_cache()
        compiled = torch_tensorrt.dynamo.compile(exported,inputs=profiles,
            enabled_precisions={torch.float32},use_explicit_typing=True,use_fp32_acc=False,
            require_full_compilation=True,min_block_size=1,workspace_size=512*1024**2,
            use_python_runtime=True,optimization_level=1,disable_tf32=True,lazy_engine_init=True,
            max_aux_streams=0)
        model.layers[index] = CompiledDecoder(compiled)
        del wrapper, exported, block
        torch._dynamo.reset()
        gc.collect()
        ctypes.CDLL('libc.so.6').malloc_trim(0)
        torch.cuda.empty_cache()
        row = {'block':index,'compile_seconds':round(time.perf_counter()-start,3),
               'remaining_torch_parameters':sum(p.numel() for p in compiled.parameters()),
               'rss_mib':int(next(x for x in Path('/proc/self/status').read_text().splitlines()
                                  if x.startswith('VmRSS:')).split()[1])/1024}
        result['blocks'].append(row)
        save()
        print(json.dumps(row),flush=True)
    del args, hidden, mask, cos, sin, inputs, compiled
    model.embed_tokens.cuda()
    model.norm.cuda()
    model.rotary_emb.cuda()
    encoder = Encoder(model)
    result['compiled'] = True
    result['stage'] = 'real_tribe_predictions'
    save()
    sys.path.insert(0,'/app')
    from tribe_service import engine
    from neuralset.extractors.text import HuggingFaceText
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    engine.get_model()  # Apply the existing loader/features patches first.
    HuggingFaceText._load_model = lambda self,**kwargs: encoder
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    base = ('Our deployment workflow gives engineering managers a clear view of failed releases. '
            'Your team can compare the change history, find the owner and restore the last working '
            'version from one screen. Would a short demonstration next Tuesday help you evaluate it?')
    messages = [f'Jordan, {base}',f'Morgan, {base} '*3,f'Taylor, {base} '*9]
    for index in [0,1,2,1]:
        start = time.perf_counter()
        predictions = engine.score_text(messages[index])
        assert predictions.shape[1] == 20484 and np.isfinite(predictions).all()
        reference = np.load(Path('/baseline')/f'case-{index}.npy')
        assert reference.shape == predictions.shape
        row = {'case':index,'seconds':round(time.perf_counter()-start,4),
               'max_abs_delta':float(np.abs(reference-predictions).max()),
               'mean_abs_delta':float(np.abs(reference-predictions).mean()),
               'relative_l2':float(np.linalg.norm(reference-predictions)/np.linalg.norm(reference)),
               'strict_parity':bool(np.allclose(reference,predictions,rtol=1e-4,atol=1e-5)),
               'feature_cache_entries':len(engine.get_model().data.text_feature.infra.cache_dict),
               'metrics':engine.last_score_metrics()}
        np.save(ROOT/f'tensorrt-case-{index}.npy',predictions)
        result['rows'].append(row)
        save()
        print(json.dumps(row),flush=True)
    engine.unload_model()
    result['stage'] = 'complete'
except Exception as exc:
    result.update(error_type=type(exc).__name__,error=str(exc))
    traceback.print_exc()
    raise
finally:
    save()
    print(json.dumps({k:v for k,v in result.items() if k not in ['blocks','rows']}),flush=True)
