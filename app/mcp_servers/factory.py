"""
app/mcp_servers/factory.py — MCP Toolkit Factory for Agno Agents
=================================================================
Provisions Agno MCPTools connected to FastMCP servers for each pipeline stage:
- 'eda'      → app/mcp_servers/eda_server.py
- 'cleaning' → app/mcp_servers/cleaning_server.py
- 'feature'  → app/mcp_servers/feature_server.py
- 'training' → app/mcp_servers/training_server.py
"""

import sys
import logging
from typing import Optional
from agno.tools.mcp import MCPTools

logger = logging.getLogger(__name__)

# Valid MCP stage servers
SERVER_MODULES = {
    "eda": "app.mcp_servers.eda_server",
    "cleaning": "app.mcp_servers.cleaning_server",
    "feature": "app.mcp_servers.feature_server",
    "training": "app.mcp_servers.training_server",
}


def get_mcp_toolkit(server_type: str) -> Optional[MCPTools]:
    """
    Instantiates an Agno MCPTools instance connected to the specified FastMCP server.
    
    Args:
        server_type: 'eda', 'cleaning', 'feature', or 'training'.
        
    Returns:
        Agno MCPTools object or None if server_type is invalid.
    """
    if server_type not in SERVER_MODULES:
        logger.error(f"Unknown MCP server_type: '{server_type}'. Valid types: {list(SERVER_MODULES.keys())}")
        return None

    module_path = SERVER_MODULES[server_type]
    try:
        from mcp import StdioServerParameters
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", module_path],
        )
        toolkit = MCPTools(server_params=params)
        logger.info(f"Initialized Agno MCPTools for stage='{server_type}' (module: {module_path})")
        return toolkit
    except Exception as exc:
        logger.error(f"Failed to initialize MCPTools for stage='{server_type}': {exc}")
        return None
