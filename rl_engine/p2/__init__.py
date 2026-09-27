# SPDX-License-Identifier: Apache-2.0
"""P2 development ABI and explicitly synthetic reference tooling.

Nothing in this package registers or silently substitutes a production kernel.
"""

from .contract import ABI_VERSION, CONTRACT_VERSION, ContractError, Status

__all__ = ["ABI_VERSION", "CONTRACT_VERSION", "ContractError", "Status"]
