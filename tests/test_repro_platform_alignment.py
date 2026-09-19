# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from rl_engine import repro
from examples.vime_rocm_attention_ablation.run import MatrixConfig, build_arm_environment


def profile_file(tmp_path, backend):
    path = tmp_path / 'profile.json'
    path.write_text(json.dumps({
        'schema_version': 'rlkernel.repro.profile.v1',
        'requirements': {'backend': backend},
        'paths': {'workspace': str(tmp_path)},
        'modes': {'consistency': {}},
        'defaults': {'tp': 4, 'cp': 2, 'rollouts': 200},
    }))
    return str(path)


@pytest.mark.parametrize('backend', ['cuda', 'rocm'])
@pytest.mark.parametrize('tp,cp', [(1,8), (2,4), (4,2), (8,1)])
def test_same_short_command_overrides_stale_profile_cp(tmp_path, capsys, backend, tp, cp):
    argv = ['plan', '--profile', profile_file(tmp_path, backend),
            '--tp', str(tp), '--rollout-tp', '2', '--temperature', '1.3',
            '--top-p', '0.8', '--steps', '3', '--max-response-len', '256',
            '--lr', '2e-6', '--weight-decay', '0.03', '--kl-coef', '0.001']
    assert repro.main(argv) == 0
    command = json.loads(capsys.readouterr().out)['runner_command']
    expected = {'--tp-size': str(tp), '--cp-size': str(cp), '--rollout-tp-size': '2',
                '--rollout-temperature': '1.3', '--rollout-top-p': '0.8',
                '--lr': '2e-06', '--weight-decay': '0.03',
                '--rollouts' if backend == 'rocm' else '--num-rollout': '3',
                '--kl-coef' if backend == 'rocm' else '--kl-loss-coef': '0.001',
                '--max-response-length' if backend == 'rocm' else '--max-response-len': '256'}
    for flag, value in expected.items():
        assert command[command.index(flag)+1] == value
    assert '--use-rollout-logprobs' not in command


def test_verify_uses_profile_backend_instead_of_accidentally_launching_cuda(tmp_path, capsys):
    assert repro.main(['verify', '--profile', profile_file(tmp_path, 'rocm')]) == 2
    assert 'ROCm does not yet implement' in capsys.readouterr().err


def test_cuda_verify_preserves_short_verification_defaults(tmp_path, monkeypatch):
    plans = []
    monkeypatch.setattr(repro, '_print_plan', lambda paths, profile, args: plans.append(args))
    assert repro.main(['verify', '--dry-run', '--profile', profile_file(tmp_path, 'cuda')]) == 0
    assert plans[0].rollouts == 2
    assert plans[0].max_response_len == 512


@pytest.mark.parametrize('option', [['--top-k','128'], ['--temperature','0']])
def test_rocm_unsupported_capabilities_fail_in_plan(tmp_path, option):
    assert repro.main(['plan', '--profile', profile_file(tmp_path, 'rocm'), *option]) == 2


def test_rollout_cp_is_forwarded_for_both_backends(tmp_path, capsys):
    for backend in ('cuda', 'rocm'):
        assert repro.main(['plan', '--profile', profile_file(tmp_path, backend),
                           '--tp', '4', '--cp', '2', '--rollout-tp', '2',
                           '--rollout-cp', '2']) == 0
        command = json.loads(capsys.readouterr().out)['runner_command']
        assert command[command.index('--rollout-tp-size') + 1] == '2'
        assert command[command.index('--rollout-cp-size') + 1] == '2'


def test_rocm_optimizer_parameters_reach_manifest_and_shell_environment(tmp_path):
    paths = {key: tmp_path/key for key in (
        'vime_root','rl_kernel_root','megatron_root','model_root','reference_checkpoint',
        'prompt_data','run_dir','launcher')}
    config = MatrixConfig(**paths, learning_rate=2e-6, weight_decay=0.03, kl_coef=0.002)
    config.validate(require_paths=False)
    frozen = config.frozen_parameters()
    assert frozen['optimizer']['lr'] == 2e-6
    assert frozen['optimizer']['weight_decay'] == 0.03
    assert frozen['reference_kl'] == {'enabled': True, 'coefficient': 0.002}
    env = build_arm_environment(config, 'R/R', tmp_path/'arm', arm_index=0)
    assert env['RLK_ABLATION_LR'] == '2e-06'
    assert env['RLK_ABLATION_WEIGHT_DECAY'] == '0.03'
    assert env['RLK_ABLATION_USE_KL_LOSS'] == '1'
    assert env['RLK_ABLATION_KL_LOSS_COEF'] == '0.002'
