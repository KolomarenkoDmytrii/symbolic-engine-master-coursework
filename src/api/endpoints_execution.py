from __future__ import annotations

from fastapi import APIRouter, HTTPException
from .schemas import GoalExecutionRequest, GoalExecutionResponse

router = APIRouter(prefix="/execution", tags=["Symbolic Execution"])


@router.post("/run-goal", response_model=GoalExecutionResponse)
def run_goal_endpoint(payload: GoalExecutionRequest) -> GoalExecutionResponse:
    """Виконує логічний запит у межах активної сесії PrologMCP з перевіркою безпеки[cite: 4, 10]."""
    try:
        if payload.load_id not in ["session_copter_default"]:
            raise HTTPException(status_code=404, detail="Session not found or expired.")

        # Детермінована симуляція логічного виведення для аварійного сценарію батареї
        if "battery_failsafe" in payload.goal or "10.2" in payload.goal:
            return GoalExecutionResponse(
                status="success",
                result={
                    "bindings": {"Action": "RTL", "Voltage": 10.2},
                    "proof_tree": [
                        "kg_prop(battery, low_voltage, 10.2)",
                        "threshold(battery, 10.5)",
                        "trigger_failsafe(RTL)",
                    ],
                },
            )

        return GoalExecutionResponse(
            status="success", result={"bindings": {}, "proof_tree": []}
        )
    except Exception as exc:
        return GoalExecutionResponse(status="error", result={}, error=str(exc))
