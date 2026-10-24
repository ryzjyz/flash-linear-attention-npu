import pytest
import torch
import torch_npu  # noqa: F401
from unittest.mock import patch

from fla_npu.ops.ascendc._kda_policy import (
    _kda_bwd_metadata_tuple,
    _kda_bwd_optimized_legacy_outputs,
    _kda_bwd_saved_staged_supported,
    _select_kda_bwd_optimized,
)


def _args(**overrides):
    q = torch.zeros(1, 8, 128, 128, dtype=torch.bfloat16)
    scalar = q.shape[:-1]
    args = {
        "q": q, "k": q.clone(), "v": q.clone(), "d_o": q.clone(),
        "beta": torch.zeros(scalar, dtype=torch.bfloat16),
        "gk": q.clone().float(),
        "Aqk": torch.zeros((*scalar, 64), dtype=torch.bfloat16),
        "Akk": torch.zeros((*scalar, 64), dtype=torch.bfloat16),
        "w": q.clone(), "qg": q.clone(), "kg": q.clone(), "v_new": q.clone(),
        "h": torch.zeros(1, 2, 8, 128, 128, dtype=torch.bfloat16),
        "raw_g": q.clone().float(), "A_log": torch.zeros(8), "dt_bias": None,
        "scale": 128 ** -0.5, "lower_bound": -5.0,
        "q_rstd": None, "k_rstd": None, "disable_recompute": True,
        "chunk_size": 64, "safe_gate": True, "use_gate_in_kernel": True,
        "use_exp2": True, "state_v_first": False,
        "initial_state": None, "dht": None,
        "cu_seqlens": None, "chunk_indices": None,
    }
    args.update(overrides)
    return args


def _npu_args(**overrides):
    args = _args(**overrides)
    return {name: value.to("npu:0") if isinstance(value, torch.Tensor) else value
            for name, value in args.items()}


def _is_ascend950():
    try:
        return "Ascend950" in torch.npu.get_device_name(torch.device("npu:0"))
    except Exception:
        return False


requires_a5 = pytest.mark.skipif(not _is_ascend950(), reason="requires Ascend950")


def test_saved_staged_probe_rejects_prepare_failure():
    with patch("fla_npu.ops.ascendc._kda_policy._prepare_kda_bwd_optimized",
               side_effect=ValueError("unsupported")):
        assert not _kda_bwd_saved_staged_supported(_args())


@patch("fla_npu.ops.ascendc._kda_policy._prepare_kda_bwd_optimized", return_value={})
def test_saved_staged_probe_accepts_supported_shape(_prepare):
    assert _kda_bwd_saved_staged_supported(_args())
    assert _kda_bwd_saved_staged_supported(_args(lower_bound=None))
    assert not _kda_bwd_saved_staged_supported(_args(lower_bound=-5.1))


def test_saved_staged_requires_npu():
    assert not _kda_bwd_saved_staged_supported(_args())


@requires_a5
def test_saved_staged_rejects_non_a5_npu():
    with patch("torch.npu.get_device_name", return_value="Ascend910B"):
        assert not _kda_bwd_saved_staged_supported(_npu_args())


@requires_a5
def test_saved_staged_rejects_k_or_v_256():
    assert not _kda_bwd_saved_staged_supported(
        _npu_args(q=torch.zeros(1, 8, 128, 256, dtype=torch.bfloat16, device="npu:0")))
    assert not _kda_bwd_saved_staged_supported(
        _npu_args(v=torch.zeros(1, 8, 128, 256, dtype=torch.bfloat16, device="npu:0")))


def test_select_kda_bwd_optimized_saved_staged_flag():
    assert _select_kda_bwd_optimized("auto", None, None, True, saved_staged=True)
    assert not _select_kda_bwd_optimized("auto", None, None, True, saved_staged=False)
    assert _select_kda_bwd_optimized("optimized", None, None, True)
    assert not _select_kda_bwd_optimized("legacy", None, None, True, saved_staged=True)
    assert _select_kda_bwd_optimized("auto", object(), object(), True)
    assert _select_kda_bwd_optimized("auto", None, None, False)


@pytest.mark.parametrize("implementation,q_rstd,expect_cast", [
    ("auto", None, True),
    ("optimized", None, False),
    ("auto", torch.ones(2), False),
])
def test_optimized_legacy_outputs_only_cast_new_auto_path(implementation, q_rstd, expect_cast):
    outputs = tuple(torch.zeros(4, dtype=torch.bfloat16) for _ in range(4)) + (
        torch.zeros(4), None, torch.zeros(4), torch.zeros(4))
    args = {"implementation": implementation, "q_rstd": q_rstd, "disable_recompute": True}
    result = _kda_bwd_optimized_legacy_outputs(args, outputs)
    for index in (0, 1, 3):
        assert result[index].dtype == (torch.float32 if expect_cast else torch.bfloat16)
    assert result[2].dtype == torch.bfloat16
    assert result[4].dtype == torch.float32


def test_metadata_iterator_is_materialized_once():
    values = _kda_bwd_metadata_tuple(iter([0, 128]))
    assert values == (0, 128)
    assert tuple(values) == (0, 128)


def _forward_saved(tokens, cu):
    from fla_npu.ops.ascendc import chunk_kda_fwd
    h = 8
    gen = torch.Generator(device="cpu").manual_seed(20261024)
    def make(shape, dtype=torch.bfloat16):
        return torch.randn(*shape, generator=gen, dtype=torch.float32).to(dtype).reshape(
            tokens, h, shape[-1]).permute(1, 0, 2).contiguous().to("npu:0")
    q = make((tokens, h, 128))
    k = make((tokens, h, 128))
    v = make((tokens, h, 128))
    raw_g = make((tokens, h, 128), torch.float32)
    beta = torch.rand(tokens, h, generator=gen, dtype=torch.float32).to(
        torch.bfloat16).reshape(tokens, h).permute(1, 0).contiguous().to("npu:0")
    a_log = torch.zeros(h, device="npu:0")
    dt_bias = torch.zeros(h * 128, device="npu:0")
    q_hat = torch.empty_like(q)
    k_hat = torch.empty_like(k)
    q_rstd = torch.empty(h, tokens, dtype=torch.float32, device="npu:0")
    k_rstd = torch.empty_like(q_rstd)
    beta_eff = torch.empty_like(q_rstd)
    out = chunk_kda_fwd(
        q, k, v, raw_g, beta, 128 ** -0.5, 64, layout="NTD",
        cu_seqlens=cu,
        safe_gate=True, lower_bound=-5.0, use_gate_in_kernel=True,
        A_log=a_log, dt_bias=dt_bias, disable_recompute=True,
        return_intermediate_states=False, use_qk_l2norm_in_kernel=True,
        use_beta_sigmoid_in_kernel=False, use_exp2=True,
        q_hat_out=q_hat, k_hat_out=k_hat, q_rstd_out=q_rstd,
        k_rstd_out=k_rstd, beta_eff_out=beta_eff)
    return out, (q_hat, k_hat, v, raw_g, beta, a_log, dt_bias)


def _bwd_args(fwd):
    (o, _, gk, aqk, akk, w, _, qg, kg, v_new, h, _), (q_hat, k_hat, v, raw_g, beta, a_log, dt_bias) = fwd
    d_o = torch.ones_like(v)
    return (q_hat, k_hat, v, beta, gk, aqk, akk, w, qg, kg, v_new, h, d_o, 128 ** -0.5), {
        "raw_g": raw_g, "A_log": a_log, "dt_bias": dt_bias.view(8, 128),
        "chunk_size": 64, "safe_gate": True, "lower_bound": -5.0,
        "use_gate_in_kernel": True, "disable_recompute": True,
        "use_exp2": True, "state_v_first": False,
    }


@requires_a5
def test_auto_saved_accepts_one_shot_metadata_iterators():
    from fla_npu.ops.ascendc import chunk_kda_bwd
    cu = (0, 128)
    indices = (0, 0, 0, 1)
    fwd = _forward_saved(128, cu)
    args, kwargs = _bwd_args(fwd)
    auto = chunk_kda_bwd(*args, implementation="auto",
                         cu_seqlens=iter(cu), chunk_indices=iter(indices), **kwargs)
    optimized = chunk_kda_bwd(*args, implementation="optimized",
                              cu_seqlens=cu, chunk_indices=indices, **kwargs)
    for index in range(8):
        if auto[index] is not None:
            assert torch.equal(auto[index], optimized[index].to(auto[index].dtype))


@requires_a5
def test_explicit_legacy_packed_tail_never_probes_or_upgrades():
    from fla_npu.ops.ascendc import chunk_kda_bwd
    cu = (0, 65)
    indices = (0, 0, 0, 1)
    fwd = _forward_saved(65, cu)
    args, kwargs = _bwd_args(fwd)
    with patch("fla_npu.ops.ascendc._kda_policy._kda_bwd_saved_staged_supported",
               side_effect=AssertionError("probe must not run for legacy")), \
            patch("fla_npu.ops.ascendc._stable._kda_bwd_saved_staged_supported",
                  side_effect=AssertionError("probe must not run for legacy"), create=True), \
            patch("fla_npu.ops.ascendc._aclnn_ctypes._kda_bwd_saved_staged_supported",
                  side_effect=AssertionError("probe must not run for legacy")):
        outputs = chunk_kda_bwd(*args, implementation="legacy",
                                cu_seqlens=cu, chunk_indices=indices, **kwargs)
    for output in outputs:
        if output is not None:
            assert torch.isfinite(output).all()
