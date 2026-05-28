"""CLI entry point: python -m forge.proxy"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time

from forge.proxy.proxy import ProxyServer
from forge.server import BudgetMode


def _load_env_file(path: str) -> None:
    """Parse a .env file and set vars via os.environ.setdefault().

    Handles KEY=value, KEY="value", KEY='value', comments, blanks.
    Uses setdefault() so existing env vars take priority.
    """
    env_path = os.path.expanduser(path)
    if not os.path.isfile(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip("\"'")
            os.environ.setdefault(k, v)


def main() -> None:
    # Load .env file before parse_args() so os.environ.get() defaults work.
    # Check ENV_FILE env var, then scan sys.argv for --env-file.
    _env_file = os.environ.get("ENV_FILE")
    if not _env_file:
        try:
            idx = sys.argv.index("--env-file")
            _env_file = sys.argv[idx + 1]
        except (ValueError, IndexError):
            pass
    if _env_file:
        _load_env_file(_env_file)

    parser = argparse.ArgumentParser(
        description="forge proxy — OpenAI-compatible proxy with guardrails",
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help="Path to .env file to load (env: ENV_FILE). Uses setdefault() so existing vars win.",
    )

    # Mode selection
    parser.add_argument(
        "--backend-url",
        default=os.environ.get("BACKEND_URL"),
        help="URL of externally managed backend (external mode), or Anthropic-compatible endpoint with --backend anthropic. Env: BACKEND_URL",
    )
    parser.add_argument(
        "--backend",
        choices=["llamaserver", "llamafile", "ollama", "anthropic"],
        default=os.environ.get("BACKEND"),
        help="Backend type (managed mode). Env: BACKEND",
    )

    # Managed mode options
    parser.add_argument("--model", default=os.environ.get("MODEL"), help="Model name (required for ollama). Env: MODEL")
    parser.add_argument("--gguf", default=os.environ.get("GGUF"), help="Path to GGUF file (llamaserver/llamafile). Env: GGUF")
    parser.add_argument("--backend-port", type=int, default=int(os.environ.get("BACKEND_PORT", "8080")), help="Backend port (default: 8080). Env: BACKEND_PORT")
    parser.add_argument(
        "--budget-mode",
        choices=["backend", "manual", "forge-full", "forge-fast"],
        default=os.environ.get("BUDGET_MODE", "backend"),
        help="Context budget mode (default: backend). Env: BUDGET_MODE",
    )
    parser.add_argument("--budget-tokens", type=int, default=os.environ.get("BUDGET_TOKENS", None), help="Manual token budget. Env: BUDGET_TOKENS")
    parser.add_argument("--extra-flags", nargs="*", default=os.environ.get("EXTRA_FLAGS").split() if os.environ.get("EXTRA_FLAGS") else None, help="Additional backend CLI flags. Env: EXTRA_FLAGS")

    # Proxy options
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"), help="Proxy listen host (default: 127.0.0.1). Env: HOST")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8081")), help="Proxy listen port (default: 8081). Env: PORT")
    parser.add_argument("--serialize", action="store_true", default=None, help="Force request serialization. Env: SERIALIZE")
    parser.add_argument("--no-serialize", action="store_true", help="Disable request serialization")
    parser.add_argument("--max-retries", type=int, default=int(os.environ.get("MAX_RETRIES", "3")), help="Max retries per request (default: 3). Env: MAX_RETRIES")
    parser.add_argument("--no-rescue", action="store_true", default=os.environ.get("RESCUE", "true").lower() == "false", help="Disable rescue parsing. Env: RESCUE (set to 'false' to disable)")
    parser.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "INFO"),
        type=lambda v: v.upper(),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO). Env: LOG_LEVEL",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        default=os.environ.get("VERBOSE", "").lower() in ("1", "true"),
        help="Verbose logging (shortcut for --log-level DEBUG). Env: VERBOSE",
    )

    args = parser.parse_args()

    # Validate backend mode (allow both None for env-var-only usage)
    if not args.backend_url and not args.backend:
        parser.error("Provide either --backend-url / BACKEND_URL or --backend / BACKEND")

    # Logging — verbose overrides LOG_LEVEL
    if args.verbose:
        level = logging.DEBUG
    else:
        level = getattr(logging, args.log_level, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resolve serialize flag
    serialize = None
    if args.serialize:
        serialize = True
    elif args.no_serialize:
        serialize = False

    proxy = ProxyServer(
        backend_url=args.backend_url,
        backend=args.backend,
        model=args.model,
        gguf=args.gguf,
        backend_port=args.backend_port,
        budget_mode=BudgetMode(args.budget_mode),
        budget_tokens=args.budget_tokens,
        extra_flags=args.extra_flags,
        host=args.host,
        port=args.port,
        serialize=serialize,
        max_retries=args.max_retries,
        rescue_enabled=not args.no_rescue,
    )

    def _shutdown(sig: int, _frame: object) -> None:
        print("\nShutting down...", file=sys.stderr, flush=True)
        try:
            proxy.stop()
        except Exception as e:
            print(f"Stop error (ignored): {e}", file=sys.stderr, flush=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _shutdown)

    proxy.start()
    print(f"forge proxy running at {proxy.url}")
    print(f"  Point your client at {proxy.url}/v1/chat/completions")
    print("  Ctrl+C to stop")

    # Block main thread. Use a timed loop so Python can deliver
    # signals between iterations (Event.wait() without timeout
    # blocks signal handling on Windows).
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        _shutdown(0, None)


if __name__ == "__main__":
    main()
