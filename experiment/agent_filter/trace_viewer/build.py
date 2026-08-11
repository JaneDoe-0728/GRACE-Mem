"""Compatibility wrapper for the Agent Filter trace viewer builder."""

from tools.agent_filter_trace_viewer.build import *  # noqa: F401,F403
from tools.agent_filter_trace_viewer.build import main


if __name__ == "__main__":
    main()
