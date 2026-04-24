"""FastAPI server for pipeline execution and UI.

Provides:
- REST API for pipeline execution
- WebSocket for real-time updates
- Serving the HTML UI
"""

from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import asyncio
import json
from pathlib import Path
from typing import Optional

from engine import PipelineEngine
from pipeline import get_pipeline_dag, STEPS_BY_ID


app = FastAPI()
engine = PipelineEngine()

# Store active WebSocket connections for broadcasting
connections: list[WebSocket] = []


@app.get("/api/pipeline/dag")
async def get_dag():
    """Get the DAG structure."""
    return get_pipeline_dag()


@app.post("/api/pipeline/run")
async def run_pipeline(
    page_id: str,
    context: dict,
    final_step: Optional[str] = None,
    force: bool = False,
):
    """Run the pipeline for a page."""
    try:
        state = engine.load_or_create_state(page_id, context)
        result = engine.execute_pipeline(final_step_id=final_step, force=force)

        # Broadcast update
        await broadcast({"type": "pipeline_complete", "state": result.to_dict()})

        return result.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/step/execute")
async def execute_step(step_id: str, force: bool = False):
    """Execute a single step."""
    if not engine.state:
        raise HTTPException(status_code=400, detail="No pipeline loaded")

    try:
        exec = engine.execute_step(step_id, force=force)

        # Broadcast update
        await broadcast({"type": "step_complete", "step": exec.to_dict()})

        return exec.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/step/override")
async def override_step_param(step_id: str, param: str, value):
    """Override a step parameter (e.g., change prompt)."""
    if not engine.state:
        raise HTTPException(status_code=400, detail="No pipeline loaded")

    try:
        engine.override_step_param(step_id, param, value)

        # Broadcast update
        await broadcast(
            {
                "type": "param_override",
                "step_id": step_id,
                "param": param,
                "value": value,
            }
        )

        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/state")
async def get_state():
    """Get current pipeline state."""
    if not engine.state:
        return {"page_id": None, "steps": {}, "context": {}}

    return engine.state.to_dict()


@app.get("/api/step/{step_id}")
async def get_step(step_id: str):
    """Get details of a single step."""
    if not engine.state:
        raise HTTPException(status_code=400, detail="No pipeline loaded")

    if step_id not in engine.state.steps:
        raise HTTPException(status_code=404, detail=f"Step {step_id} not found")

    step_exec = engine.state.steps[step_id]
    step_def = STEPS_BY_ID[step_id]

    return {
        "execution": step_exec.to_dict(),
        "definition": {
            "id": step_def.id,
            "label": step_def.label,
            "depends_on": step_def.depends_on,
            "prompt": step_def.prompt,
            "model": step_def.model,
        },
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates."""
    await websocket.accept()
    connections.append(websocket)

    try:
        # Send initial state
        if engine.state:
            await websocket.send_json({"type": "initial_state", "state": engine.state.to_dict()})

        # Keep connection open
        while True:
            data = await websocket.receive_text()
            # Echo back (clients can send commands if needed)
            await websocket.send_json({"type": "pong"})
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        connections.remove(websocket)


async def broadcast(message: dict):
    """Broadcast a message to all connected WebSocket clients."""
    for conn in connections:
        try:
            await conn.send_json(message)
        except Exception as e:
            print(f"Failed to broadcast: {e}")


@app.get("/")
async def serve_ui():
    """Serve the HTML UI."""
    ui_path = Path(__file__).parent / "ui.html"
    if ui_path.exists():
        return FileResponse(ui_path, media_type="text/html")
    return {"error": "UI not found"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
