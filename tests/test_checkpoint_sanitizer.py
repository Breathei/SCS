"""Mode-matrix tests for main.sanitize_checkpoint_state_dict.

Covers legacy router (``router.*``) and v2 router (``router_v2.*``)
namespaces across none / asr / asr_scan / asr_scan_v2 targets, plus stale
key filtering and strict non-router checks.  Uses a tiny mock module — no
dataset, checkpoint, CUDA, or training required.
"""
import os
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import sanitize_checkpoint_state_dict  # noqa: E402


class Mini(nn.Module):
    def __init__(self, legacy_router=False, v2_router=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2))
        if legacy_router:
            self.router = nn.Linear(2, 4)
        if v2_router:
            self.router_v2 = nn.Linear(2, 4)


def sd_of(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


class TestCheckpointSanitizer(unittest.TestCase):
    def test_none_checkpoint_loads_into_v2_with_fresh_router(self):
        source = sd_of(Mini())
        target = Mini(v2_router=True)
        ignored, allowed = sanitize_checkpoint_state_dict(
            source, target, scan_routing='asr_scan_v2'
        )
        self.assertEqual(ignored, [])
        self.assertEqual(
            sorted(allowed), ['router_v2.bias', 'router_v2.weight']
        )
        self.assertTrue(torch.equal(target.weight, torch.ones(2)))

    def test_v2_checkpoint_loads_into_none_by_dropping_router(self):
        source = sd_of(Mini(v2_router=True))
        target = Mini()
        ignored, allowed = sanitize_checkpoint_state_dict(
            source, target, scan_routing='none'
        )
        self.assertEqual(sorted(ignored), ['router_v2.bias', 'router_v2.weight'])
        self.assertEqual(allowed, [])

    def test_partial_v2_router_state_rejected(self):
        source = sd_of(Mini(v2_router=True))
        del source['router_v2.bias']
        target = Mini(v2_router=True)
        with self.assertRaisesRegex(RuntimeError, 'incomplete v2 router'):
            sanitize_checkpoint_state_dict(
                source, target, scan_routing='asr_scan_v2'
            )

    def test_none_checkpoint_loads_into_asr_with_fresh_router(self):
        source = sd_of(Mini())
        target = Mini(legacy_router=True)
        ignored, allowed = sanitize_checkpoint_state_dict(
            source, target, scan_routing='asr'
        )
        self.assertEqual(ignored, [])
        self.assertEqual(sorted(allowed), ['router.bias', 'router.weight'])

    def test_asr_checkpoint_loads_into_none_by_dropping_router(self):
        source = sd_of(Mini(legacy_router=True))
        target = Mini()
        ignored, allowed = sanitize_checkpoint_state_dict(
            source, target, scan_routing='none'
        )
        self.assertEqual(sorted(ignored), ['router.bias', 'router.weight'])
        self.assertEqual(allowed, [])

    def test_partial_legacy_router_state_rejected(self):
        source = sd_of(Mini(legacy_router=True))
        del source['router.weight']
        target = Mini(legacy_router=True)
        with self.assertRaisesRegex(RuntimeError, 'incomplete legacy router'):
            sanitize_checkpoint_state_dict(
                source, target, scan_routing='asr_scan'
            )

    def test_v2_target_drops_legacy_router_and_allows_missing_v2(self):
        source = sd_of(Mini(legacy_router=True))
        target = Mini(v2_router=True)
        ignored, allowed = sanitize_checkpoint_state_dict(
            source, target, scan_routing='asr_scan_v2'
        )
        self.assertEqual(sorted(ignored), ['router.bias', 'router.weight'])
        self.assertEqual(
            sorted(allowed), ['router_v2.bias', 'router_v2.weight']
        )

    def test_legacy_target_drops_v2_router_and_allows_missing_legacy(self):
        source = sd_of(Mini(v2_router=True))
        target = Mini(legacy_router=True)
        ignored, allowed = sanitize_checkpoint_state_dict(
            source, target, scan_routing='asr'
        )
        self.assertEqual(sorted(ignored), ['router_v2.bias', 'router_v2.weight'])
        self.assertEqual(sorted(allowed), ['router.bias', 'router.weight'])

    def test_full_v2_checkpoint_loads_strictly(self):
        source_model = Mini(v2_router=True)
        with torch.no_grad():
            source_model.weight.fill_(3.0)
            source_model.router_v2.weight.fill_(0.5)
        target = Mini(v2_router=True)
        ignored, allowed = sanitize_checkpoint_state_dict(
            sd_of(source_model), target, scan_routing='asr_scan_v2'
        )
        self.assertEqual(ignored, [])
        self.assertEqual(allowed, [])
        self.assertTrue(torch.equal(
            target.router_v2.weight, source_model.router_v2.weight
        ))

    def test_stale_keys_dropped(self):
        source = sd_of(Mini())
        source['backbone.trap_gate_logit'] = torch.tensor(0.5)
        source['backbone.last_route_w'] = torch.zeros(4)
        target = Mini()
        ignored, _ = sanitize_checkpoint_state_dict(
            source, target, scan_routing='none'
        )
        self.assertIn('backbone.trap_gate_logit', ignored)
        self.assertIn('backbone.last_route_w', ignored)

    def test_unexpected_non_router_key_rejected(self):
        source = sd_of(Mini())
        source['bogus.weight'] = torch.ones(2)
        target = Mini()
        with self.assertRaisesRegex(RuntimeError, 'unexpected'):
            sanitize_checkpoint_state_dict(
                source, target, scan_routing='none'
            )

    def test_missing_non_router_key_rejected(self):
        source = sd_of(Mini())
        del source['weight']
        target = Mini()
        with self.assertRaisesRegex(RuntimeError, 'missing'):
            sanitize_checkpoint_state_dict(
                source, target, scan_routing='none'
            )

    def test_shape_mismatch_rejected(self):
        source = sd_of(Mini())
        source['weight'] = torch.ones(3)
        target = Mini()
        with self.assertRaisesRegex(RuntimeError, 'shape mismatch'):
            sanitize_checkpoint_state_dict(
                source, target, scan_routing='none'
            )

    def test_source_state_dict_not_mutated(self):
        source = sd_of(Mini(legacy_router=True))
        keys_before = set(source.keys())
        target = Mini()
        sanitize_checkpoint_state_dict(source, target, scan_routing='none')
        self.assertEqual(set(source.keys()), keys_before)


if __name__ == '__main__':
    unittest.main()
