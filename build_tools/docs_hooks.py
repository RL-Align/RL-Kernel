# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Rewrite GitHub-relative image paths so the built site serves same-origin assets.

Docs pages embed images with the path form that renders on GitHub: one ``../``
per directory below ``docs/`` (``docs/community/wechat.md`` uses
``../assets/``, ``docs/blog/2026/post.md`` would use ``../../assets/``). Under
``use_directory_urls`` the built site serves those pages deeper than their
source directories, where the same relative paths would escape the site root.
This hook derives the source prefix from each page's own source directory and
rewrites it to the depth of the page's site URL, keeping a single copy of
every image under ``docs/assets/`` while GitHub and the built site both render
it.
"""

from __future__ import annotations

import posixpath


def on_page_content(html: str, page, config, **kwargs) -> str:
    """Rewrite ``src="<source-relative>assets/..."`` to the page's site depth."""
    src_dir = posixpath.dirname(page.file.src_uri)
    src_up = "../" * (len(src_dir.split("/")) if src_dir else 0)
    url_up = "../" * page.url.count("/")
    return html.replace(f'src="{src_up}assets/', f'src="{url_up}assets/')
