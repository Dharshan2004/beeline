"""Test package: force deterministic offline agents.

The Agent's "auto" semantic-ranker default connects to OpenAI when a key is
available. Tests must stay offline, free, and deterministic regardless of the
developer's environment, so the auto path is disabled here before any test
module constructs an Agent. Tests that exercise LLM stages pass fakes
explicitly, which bypasses the auto path anyway.
"""
import os

os.environ["BEELINE_OFFLINE"] = "1"
