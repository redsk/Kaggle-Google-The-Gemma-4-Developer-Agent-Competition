"""Compatibility entry point for the supervised local runner (no smoke-test caps)."""

from run_supervised import main

if __name__ == "__main__":
    raise SystemExit(main())
