#!/bin/bash
# Fake engine: emits a canned startup, echoes an error for each command received.
# The echoed line's double quotes are backslash-escaped so the emitted event is
# valid JSON (the client's decoder drops undecodable lines by design, so an
# unescaped echo would silently vanish instead of reaching the test).
printf '%s\n' '{"event": "ready", "version": 1, "config": {}, "schema": []}'
printf '%s\n' '{"event": "engines_ready"}'
while IFS= read -r line; do
  case "$line" in
    *shutdown*) printf '%s\n' '{"event": "state", "state": "idle"}'; exit 0 ;;
    *) esc=${line//\"/\\\"}; printf '%s\n' "{\"event\": \"error\", \"message\": \"echo: $esc\"}" ;;
  esac
done
