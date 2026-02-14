
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import json
import time

from hormone_harness.controller import DigitalHormoneController, Observation, Outcome, Usage
from hormone_harness.adapters.base import ModelAdapter, LLMRequest
from hormone_harness.logging import JsonlLogger
from hormone_harness.json_utils import parse_json, validate_json, make_repair_prompt


# ------------------------
# Environment interface
# ------------------------
class WebEnv:
    """Minimal interface expected by the harness.

    Replace this with a real WebArena/WebArena-Verified environment wrapper.
    """

    def reset(self, task_id: int, seed: int) -> Observation:
        """Return initial Observation."""
        raise NotImplementedError

    def step_tool(self, tool: str, args: Dict[str, Any]) -> Tuple[Observation, Outcome]:
        """Execute a tool call and return (next_observation, outcome)."""
        raise NotImplementedError


def load_schema(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_system_prompt(base_system: str, modifier: str) -> str:
    return base_system.strip() + "\n\n" + modifier.strip()


def build_step_prompt(user_goal: str, env_text: str) -> str:
    return (
        "You are controlling a web browser via tools.\n"
        "Given the OBSERVATION and GOAL, output EXACTLY ONE JSON object.\n"
        "No markdown. No extra keys. No commentary.\n\n"
        f"GOAL:\n{user_goal}\n\n"
        f"OBSERVATION:\n{env_text}\n"
    )


def call_model_with_json_repair(
    model: ModelAdapter,
    req: LLMRequest,
    schema_name: str,
    schema: Dict[str, Any],
    max_retries: int = 2,
) -> Tuple[Dict[str, Any], Usage, str]:
    """Call model and enforce JSON schema by parse+validate+repair."""
    last_raw = ""
    for attempt in range(max_retries + 1):
        raw, usage = model.generate(req)
        last_raw = raw
        try:
            obj = parse_json(raw)
        except Exception as e:
            ok = False
            err = str(e)
            obj = None

        if obj is not None:
            ok, err = validate_json(obj, schema)

        if ok:
            assert isinstance(obj, dict), "step schema expects object"
            return obj, usage, raw

        # Repair prompt: ask model to emit valid JSON ONLY
        repair_user = make_repair_prompt(schema_name=schema_name, schema=schema, error=err or "unknown", raw=raw)
        req = LLMRequest(
            system=req.system,
            user=repair_user,
            temperature=min(req.temperature, 0.3),
            max_tokens=min(req.max_tokens, 400),
            top_p=req.top_p,
            stop=req.stop,
            response_format=req.response_format,
            json_schema=req.json_schema,
        )

    raise RuntimeError(f"Failed to obtain valid JSON for schema {schema_name} after {max_retries} retries. Last output: {last_raw}")


def run_task(
    task_id: int,
    seed: int,
    env: WebEnv,
    model: ModelAdapter,
    model_id: str,
    controller: Optional[DigitalHormoneController],
    condition: str,
    out_dir: str,
    step_budget: int = 40,
    schemas_dir: str = "schemas",
) -> Dict[str, Any]:
    """Run a single task with ON/OFF controller and produce logs + final response.

    Outputs:
      - steps.jsonl with per-step traces (hormones, controls, usage, raw output)
      - agent_response.json (WebArena-Verified schema) when type='final'
    """
    out = Path(out_dir) / str(model_id) / f"task_{task_id}" / f"{condition}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)

    logger = JsonlLogger(str(out / "steps.jsonl"))

    step_schema = load_schema(str(Path(schemas_dir) / "agent_step_action_schema.json"))
    final_schema = load_schema(str(Path(schemas_dir) / "webarena_verified_agent_response_schema.json"))

    base_system = (
        "You are a careful web navigation agent.\n"
        "You must be grounded in the provided observation.\n"
        "At each step, output ONLY a JSON object matching the required schema.\n"
    )

    # Reset environment
    obs = env.reset(task_id=task_id, seed=seed)

    # Initialize controller
    if controller is None:
        # Hormone-OFF: fixed params
        modifier = "[MODE: baseline]\n- Follow the goal.\n- Stay grounded.\n"
        temperature = 0.5
        max_tokens = 800
        tool_allowlist = None
        regime = "baseline"
    else:
        controller.load_state()
        params0 = controller.control_params()
        modifier = params0.system_modifier
        temperature = params0.temperature
        max_tokens = params0.max_tokens
        tool_allowlist = params0.tool_allowlist
        regime = params0.regime

    final_agent_response: Optional[Dict[str, Any]] = None
    total_tokens = 0
    risk_events_total = 0

    for t in range(step_budget):
        # Recompute controls from the latest hormone state (for ON) each step
        if controller is not None:
            params = controller.control_params()
            modifier = params.system_modifier
            temperature = params.temperature
            max_tokens = params.max_tokens
            tool_allowlist = params.tool_allowlist
            regime = params.regime

        system = build_system_prompt(base_system, modifier)
        user = build_step_prompt(user_goal=obs.user_goal, env_text=obs.env_text)

        # Optional: include a compact allowlist hint (do not rely on provider tool calling)
        if tool_allowlist:
            user += "\nALLOWED_TOOLS:\n" + "\n".join(tool_allowlist) + "\n"

        req = LLMRequest(
            system=system,
            user=user,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format="json",  # adapters may ignore if unsupported
            json_schema=step_schema,  # adapters may ignore; we still validate manually
        )

        step_obj, usage, raw = call_model_with_json_repair(
            model=model,
            req=req,
            schema_name="AgentStepAction",
            schema=step_schema,
            max_retries=2,
        )
        total_tokens += int(usage.total_tokens or 0)

        # Execute step
        outcome = Outcome(success=None, progress=None, risk_events=[], tool_errors=0, policy_blocks=0)

        if step_obj["type"] == "tool":
            tool = step_obj["tool"]
            args = step_obj.get("args", {}) or {}

            # Enforce allowlist in the harness for fairness and safety.
            if tool_allowlist and tool not in tool_allowlist:
                outcome.policy_blocks = 1
                outcome.risk_events = ["tool_not_allowed"]
            else:
                try:
                    obs, env_outcome = env.step_tool(tool=tool, args=args)
                    outcome = env_outcome
                except Exception:
                    # Treat tool execution failures as risk
                    outcome.tool_errors = 1
                    outcome.risk_events = ["tool_execution_error"]

        elif step_obj["type"] == "final":
            final_agent_response = step_obj["final"]
            ok, err = validate_json(final_agent_response, final_schema)
            if not ok:
                # If final response is invalid, count as risk and try to repair once
                outcome.risk_events = ["invalid_final_schema"]
                risk_events_total += 1

                repair_req = LLMRequest(
                    system=system,
                    user=make_repair_prompt(
                        schema_name="WebArenaVerifiedAgentResponse",
                        schema=final_schema,
                        error=err or "unknown",
                        raw=json.dumps(final_agent_response, ensure_ascii=False),
                    ),
                    temperature=min(temperature, 0.3),
                    max_tokens=min(max_tokens, 400),
                    response_format="json",
                    json_schema=final_schema,
                )
                repaired, usage2, raw2 = call_model_with_json_repair(
                    model=model,
                    req=repair_req,
                    schema_name="WebArenaVerifiedAgentResponse",
                    schema=final_schema,
                    max_retries=1,
                )
                total_tokens += int(usage2.total_tokens or 0)
                final_agent_response = repaired  # type: ignore
                ok2, err2 = validate_json(final_agent_response, final_schema)
                if not ok2:
                    raise RuntimeError(f"Final agent_response still invalid after repair: {err2}")
            break

        elif step_obj["type"] == "message":
            # In web benchmarks, a 'message' step is usually a failure unless the environment expects clarification.
            # Record it and continue; env wrapper may choose to end.
            outcome.risk_events = ["message_step_in_web_env"]

        # Update controller AFTER outcome + usage
        if controller is not None:
            h, ctrl = controller.tick(obs=obs, outcome=outcome, usage=usage)
            hormones = h.to_dict()
            controls = asdict(ctrl)
        else:
            hormones = None
            controls = {"regime": regime, "system_modifier": modifier, "temperature": temperature, "max_tokens": max_tokens, "tool_allowlist": tool_allowlist}

        risk_events_total += len(outcome.risk_events or [])

        logger.log(
            {
                "task_id": task_id,
                "model_id": model_id,
                "seed": seed,
                "t": t,
                "condition": condition,
                "raw_model_output": raw,
                "step_action": step_obj,
                "obs": asdict(obs),
                "outcome": asdict(outcome),
                "usage": asdict(usage),
                "hormones": hormones,
                "controls": controls,
                "timestamp": time.time(),
            }
        )

    if final_agent_response is None:
        # If the agent never produced a final response, emit a structured failure.
        final_agent_response = {
            "action": "navigate",
            "status": "UNKNOWN_ERROR",
            "results": None,
            "error_details": f"No final response within step budget={step_budget}.",
        }

    (out / "agent_response.json").write_text(json.dumps(final_agent_response, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "task_id": task_id,
        "model_id": model_id,
        "seed": seed,
        "condition": condition,
        "total_tokens": total_tokens,
        "risk_events_total": risk_events_total,
        "agent_response_path": str(out / "agent_response.json"),
        # "network_trace_path": str(out / "network.har"),  # add when using HAR capture
    }
