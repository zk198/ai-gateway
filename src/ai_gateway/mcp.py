from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_headers
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.server.providers.openapi import MCPType, RouteMap

from .api import app
from .auth import reset_mcp_authorization, set_mcp_authorization

class AuthorizationContextMiddleware(Middleware):
    async def on_request(self, context: MiddlewareContext, call_next):
        headers = get_http_headers(include={"authorization"})
        token = set_mcp_authorization(headers.get("authorization"))
        try:
            return await call_next(context)
        finally:
            reset_mcp_authorization(token)

mcp = FastMCP.from_fastapi(
    app=app,
    name="AI Gateway",
    route_maps=[
        RouteMap(tags={"llm"}, mcp_type=MCPType.TOOL),
        RouteMap(mcp_type=MCPType.EXCLUDE),
    ],
)
mcp.add_middleware(AuthorizationContextMiddleware())
mcp_app = mcp.http_app(path="/", transport="streamable-http", stateless_http=True)
app.router.lifespan_context = mcp_app.lifespan
app.mount("/mcp", mcp_app)

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8200)
