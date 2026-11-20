# SPDX-License-Identifier: Apache-2.0
"""Local absolute-layer adapter; framework/PP code supplies global layer numbers."""

from types import MappingProxyType

from rl_engine.ops.moe.hash_route import HashRouteTable, hash_route


class DeepSeekV4HashRouter:
    def __init__(self, tables):
        owned = dict(tables)
        if not owned or any(
            type(layer) is not int
            or not isinstance(table, HashRouteTable)
            or layer != table.absolute_layer
            for layer, table in owned.items()
        ):
            raise ValueError("Hash router maps absolute layer numbers to their prepared tables")
        self.tables = MappingProxyType(owned)

    def __call__(self, logits, token_ids, *, absolute_layer, **kwargs):
        if type(absolute_layer) is not int or absolute_layer not in self.tables:
            raise ValueError("absolute layer has no configured Hash table on this stage")
        return hash_route(
            logits, token_ids, self.tables[absolute_layer], absolute_layer=absolute_layer, **kwargs
        )
