# SPDX-License-Identifier: Apache-2.0
"""Instance-local packed wo_a boundary for the locked, unmodified native MCore."""

import ast
import inspect
import textwrap
import types


class _GroupedBoundary(ast.NodeTransformer):
    def __init__(self):
        self.views = self.calls = 0

    def visit_Assign(self, node):
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "wo_a_weight"
        ):
            expected = (
                "self.linear_o_group_proj.view(self.output_projection_local_groups, "
                "self.config.output_projection_lora_rank, -1)"
            )
            if ast.dump(node.value) != ast.dump(ast.parse(expected, mode="eval").body):
                raise ValueError("unsupported native grouped weight view ABI")
            self.views += 1
            return ast.copy_location(ast.Pass(), node)
        return self.generic_visit(node)

    def visit_Call(self, node):
        expected = 'torch.einsum("...gd,grd->...gr", core_attn_out, wo_a_weight)'
        if ast.dump(node) == ast.dump(ast.parse(expected, mode="eval").body):
            self.calls += 1
            return ast.copy_location(
                ast.parse("self.group_output_projection(core_attn_out)", mode="eval").body, node
            )
        return self.generic_visit(node)


def grouped_forward(native_forward):
    """Change exactly the frozen grouped-weight projection, retaining native scheduling."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(native_forward)))
    transform = _GroupedBoundary()
    tree = transform.visit(tree)
    if (transform.views, transform.calls) != (1, 1):
        raise ValueError(
            f"unsupported native grouped forward ABI: {transform.views}/{transform.calls}"
        )
    namespace = dict(native_forward.__globals__)
    exec(compile(ast.fix_missing_locations(tree), __file__, "exec"), namespace)
    result = namespace[native_forward.__name__]
    result.__module__ = __name__
    result.__qualname__ = "RLKDSv4PackedGroup.forward"
    return result


def bind_grouped_projection(attention, projection):
    if hasattr(attention, "_rlk_grouped_projection_bound"):
        raise ValueError("native grouped projection already bound")
    attention.group_output_projection = projection
    # The old diagnostic image had an upstream helper. It is accepted only as a
    # baseline; the clean dependency audit rejects it for pure-RLK acceptance.
    if not hasattr(type(attention), "_apply_grouped_output_projection"):
        attention.forward = types.MethodType(grouped_forward(type(attention).forward), attention)
    attention._rlk_grouped_projection_bound = True
    return attention
