#!/bin/sh
# Slack CLI get-manifest hook: ignore args (e.g. --source=...), print the static manifest
cat "$(dirname "$0")/manifest.json"
