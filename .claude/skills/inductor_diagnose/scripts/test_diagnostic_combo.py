from pathlib import Path

from diagnostic_combo import ComboTrace

from torch._inductor import config
from torch._inductor.scheduler import ForeachKernelSchedulerNode
from torch.testing._internal.common_utils import run_tests, TestCase


class ProbeNode:
    def __init__(self, name, template=False, strict=False):
        self.name = name
        self.template = template
        self.strict = strict
        self.strict_calls = 0

    def is_template(self):
        return self.template

    def has_strict_sum(self):
        self.strict_calls += 1
        return self.strict


def serialize(value):
    if isinstance(value, ProbeNode):
        return value.name
    if isinstance(value, (list, tuple)):
        return [serialize(x) for x in value]
    if isinstance(value, dict):
        return {str(k): serialize(v) for k, v in value.items()}
    return str(value)


class TestComboTrace(TestCase):
    def test_filter_stage_and_no_predicate_replay(self):
        nodes = [
            ProbeNode("keep"),
            ProbeNode("template", template=True),
            ProbeNode("strict", strict=True),
        ]
        rows, errors = [], []
        import torch

        source = Path(torch.__file__).resolve().parent.parent
        with config.patch(
            benchmark_combo_kernel=False,
            combo_kernel_per_subkernel_blocks=False,
            combo_kernels_pointwise_only=False,
        ):
            with ComboTrace(source, rows.append, serialize, errors, lambda frame: None):
                result = ForeachKernelSchedulerNode.combinable_nodes(nodes)
        self.assertEqual(errors, [])
        self.assertEqual([n.name for n in result], ["keep"])
        self.assertEqual([n.strict_calls for n in nodes], [1, 0, 1])
        filters = [r for r in rows if r["event"] == "combo_filter"]
        template = next(r for r in filters if "template_nodes" in r["condition"])
        strict = next(r for r in filters if "has_strict_sum" in r["condition"])
        self.assertEqual(template["before"], ["keep", "template", "strict"])
        self.assertEqual(template["after"], ["keep", "strict"])
        self.assertEqual(strict["before"], ["keep", "strict"])
        self.assertEqual(strict["after"], ["keep"])


if __name__ == "__main__":
    run_tests()
