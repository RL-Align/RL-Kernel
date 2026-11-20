# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import pytest

from rl_engine.models.deepseek_v41.config import DeepSeekV41Config

FIXTURE = Path(__file__).parent / "fixtures/config.json"


def test_released_config_and_original_layers():
    c = DeepSeekV41Config.from_json(FIXTURE)
    t = c.text
    assert (t['hidden_size'], t['num_hidden_layers'], t['rms_norm_eps']) == (5120, 40, 1e-20)
    plans = c.plan_layers((0, 1, 2, 3))
    assert [p.original_layer_id for p in plans] == [0, 1, 2, 3]
    assert [p.engram for p in plans] == [False, True, False, False]
    assert [p.kv_source_layer_id for p in plans] == [None, None, 2, 2]
    assert [p.index_source_layer_id for p in plans] == [None, None, 2, 2]
    assert plans[2].owns_kv and not plans[3].owns_kv
    for i in (8, 14, 20):
        assert c.layer_plan(i).owns_kv
    for i in (24, 28, 32, 36):
        assert c.layer_plan(i).owns_index and not c.layer_plan(i).owns_kv
    assert c.layer_plan(14).engram and c.layer_plan(20).compress_ratio == 1
    assert c.layer_plan(39).candidate_source_layer_id == 20
    t['rms_norm_eps'] = 1e-6
    assert c.text['rms_norm_eps'] == 1e-20


def test_official_inference_config_agrees():
    c = DeepSeekV41Config.from_json(FIXTURE)
    infer = json.loads(FIXTURE.with_name('inference_config.json').read_text())
    aliases = {'dim': 'hidden_size', 'n_layers': 'num_hidden_layers', 'norm_eps': 'rms_norm_eps',
               'kv_source_layers': 'kv_source_layer_ids', 'index_source_layers': 'index_source_layer_ids',
               'n_activated_experts': 'num_experts_per_tok', 'moe_inter_dim': 'moe_intermediate_size'}
    for key, target in aliases.items():
        assert infer[key] == c.text[target]
    assert infer['compress_ratios'] == c.text['compress_ratios']


@pytest.mark.parametrize('ids', [(3,), (24,), (2, 2), (40,), ()])
def test_incomplete_layer_selections_fail(ids):
    with pytest.raises(ValueError):
        DeepSeekV41Config.from_json(FIXTURE).plan_layers(ids)


@pytest.mark.parametrize('mutate', [
    lambda c: c.update(model_type='deepseek_v4'),
    lambda c: c['text_config'].pop('rms_norm_eps'),
    lambda c: c['text_config'].update(compress_ratios=[0]*40),
    lambda c: c['text_config'].update(kv_source_layer_ids=[8, 2]),
    lambda c: c['text_config'].update(index_source_layer_ids=[8, 14, 20]),
    lambda c: c['quantization_config'].update(weight_block_size=[128, 128]),
])
def test_v4_defaults_and_bad_sources_rejected(mutate):
    raw = json.loads(FIXTURE.read_text())
    mutate(raw)
    with pytest.raises(ValueError):
        DeepSeekV41Config(raw)


@pytest.mark.parametrize('tp,pp,dp', [(1,1,1), (1,2,1), (2,1,1), (2,2,1),
                                      (1,2,2), (4,2,1), (2,2,2)])
def test_requested_topology_arithmetic(tp, pp, dp):
    DeepSeekV41Config.from_json(FIXTURE).validate_topology(tp=tp, pp=pp, dp=dp, world_size=tp*pp*dp)
