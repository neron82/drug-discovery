#!/usr/bin/env bash
# Simple operator control for the Drug-Discovery API and research loop.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE="dd-api.service"
API="http://127.0.0.1:8020"
DEFAULT_WS="f8f894ac-aac9-478c-8467-0ac2ea3ebfc8"

cd "$ROOT"

usage() {
  cat <<'EOF'
Usage: scripts/dd-service.sh COMMAND [WORKSPACE_ID]

API service:
  start             Start the API/UI service and wait for health
  stop              Stop research cleanly, then stop the API/UI service
  restart           Restart the API/UI service; enabled research resumes
  status            Show service, API, workspace, and queue state

Autonomous research:
  research-start    Enable and start one workspace (default: MPB workspace)
  research-stop     Pause and stop one workspace (default: MPB workspace)
  cycle             Run one forced manual cycle for one workspace

Environment:
  DD_WORKSPACE_ID   Override the default workspace ID

The API service and autonomous research are intentionally separate:
`start` does not begin scientific work; use `research-start` explicitly.
EOF
}

health() {
  curl -fsS --max-time 5 "$API/api/status" >/dev/null
}

wait_for_health() {
  for _ in $(seq 1 30); do
    if health; then
      return 0
    fi
    sleep 1
  done
  echo "API did not become healthy at $API" >&2
  systemctl --user status "$SERVICE" --no-pager >&2 || true
  return 1
}

service_active() {
  systemctl --user is-active --quiet "$SERVICE"
}

start_service() {
  systemctl --user start "$SERVICE"
  wait_for_health
  echo "Drug-Discovery API is running at $API"
}

workspace_id() {
  if [[ -n "${DD_WORKSPACE_ID:-}" ]]; then
    printf '%s\n' "$DD_WORKSPACE_ID"
    return
  fi
  if ! service_active; then
    echo "API service is not running; set DD_WORKSPACE_ID or start the service first" >&2
    return 1
  fi
  curl -fsS --max-time 10 "$API/api/workspaces" \
    | uv run python -c 'import json,sys; rows=json.load(sys.stdin); hit=next((r for r in rows if "androgen" in (r.get("disease") or "").lower()), rows[0] if rows else None); print(hit["id"] if hit else "")'
}

workspace_ids() {
  curl -fsS --max-time 10 "$API/api/workspaces" \
    | uv run python -c 'import json,sys; print("\n".join(r["id"] for r in json.load(sys.stdin)))'
}

research_action() {
  local action="$1"
  local ws="$2"
  curl -fsS --max-time 30 -X POST \
    -H 'Content-Type: application/json' \
    -d "{\"action\":\"$action\"}" \
    "$API/api/workspaces/$ws/research"
  printf '\n'
}

start_research() {
  start_service >/dev/null
  local ws
  ws="$(workspace_id)"
  [[ -n "$ws" ]] || { echo "No workspace found" >&2; return 1; }
  echo "Starting autonomous research for workspace $ws"
  research_action start "$ws"
}

stop_research() {
  if ! service_active; then
    echo "API service is already stopped; no research loop can be active"
    return 0
  fi
  local ws="${1:-}"
  if [[ -n "$ws" ]]; then
    echo "Stopping autonomous research for workspace $ws"
    research_action stop "$ws"
    return
  fi
  local id
  while IFS= read -r id; do
    [[ -n "$id" ]] || continue
    research_action stop "$id"
  done < <(workspace_ids)
  echo "Stopped autonomous research for all workspaces"
}

status() {
  echo "== systemd =="
  printf 'service: '; systemctl --user is-active "$SERVICE" || true
  printf 'enabled: '; systemctl --user is-enabled "$SERVICE" 2>/dev/null || true
  if ! service_active; then
    return 0
  fi
  echo "== API =="
  curl -fsS --max-time 10 "$API/api/status"
  printf '\n== workspaces ==\n'
  curl -fsS --max-time 10 "$API/api/workspaces" \
    | uv run python -c 'import json,sys; rows=json.load(sys.stdin); [print("{id}  {name}  research_enabled={enabled}  state={state}".format(id=r["id"], name=r["name"], enabled=r.get("research_enabled", False), state=r["research_state"])) for r in rows]'
}

cmd="${1:-status}"
case "$cmd" in
  start)
    start_service
    ;;
  stop)
    stop_research
    systemctl --user stop "$SERVICE"
    echo "Drug-Discovery API stopped"
    ;;
  restart)
    systemctl --user restart "$SERVICE"
    wait_for_health
    echo "Drug-Discovery API restarted at $API"
    ;;
  status)
    status
    ;;
  research-start)
    start_research
    ;;
  research-stop)
    ws="${2:-}"
    if [[ -n "$ws" ]]; then stop_research "$ws"; else stop_research; fi
    ;;
  cycle)
    start_service >/dev/null
    ws="${2:-$(workspace_id)}"
    [[ -n "$ws" ]] || { echo "No workspace found" >&2; exit 1; }
    research_action cycle "$ws"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "Unknown command: $cmd" >&2
    usage >&2
    exit 2
    ;;
esac
