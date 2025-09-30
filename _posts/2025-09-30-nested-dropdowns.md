---
layout: post
title: Nested dropdowns
date: 2025-09-30 12:00:00
description: Example post showing nested dropdowns inside normal Markdown.
tags: ui dropdowns
categories: examples
published: false
---

{% include dd_styles.liquid %}

Intro text in regular Markdown. Below are nested dropdowns composed inline:

{% include dd_parent_open.liquid title="Section A" %}

Some Markdown body for Section A with a list:

- item one
- item two

{% include dd_child_open.liquid title="A.1 Concepts" %}

Concept details here. You can add code:

```python
def add(a: int, b: int) -> int:
    return a + b
```

{% include dd_child_close.liquid %}

{% include dd_child_open.liquid title="A.2 Notes" %}

More notes with links and formatting. Here's a third level nested dropdown using native details/summary:

{% include dd_child_open.liquid title="Deep Dive" %}

  This is level 3 content. You can still write Markdown here,
  including lists and code:

  - subpoint
  - another subpoint

  ```bash
  echo "hello from level 3"
  ```

{% include dd_child_close.liquid %}

{% include dd_child_close.liquid %}

{% include dd_parent_close.liquid %}

{% include dd_parent_open.liquid title="Section B" %}

Section B overview paragraph.

{% include dd_child_open.liquid title="Tips" %}

1. Keep titles short
2. Use Markdown as usual

{% include dd_child_close.liquid %}

{% include dd_parent_close.liquid %}


