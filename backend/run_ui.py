import os
import uvicorn


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    # Import the local app object directly to avoid any module resolution ambiguity.
    from backend.app import app

    uvicorn.run(app, host="127.0.0.1", port=port, reload=False)

