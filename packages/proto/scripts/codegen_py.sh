#!/bin/bash

# Fail if any command fails
set -e

PROTO_DIR=$(dirname "$0")/..
echo "$PROTO_DIR"
OUT_DIR=$PROTO_DIR/gen/py
PYTHON=$PROTO_DIR/.venv/bin/python
PY_ACT=$PROTO_DIR/.venv/bin/activate

mkdir -p "$OUT_DIR"
source "$PY_ACT"

$PYTHON -m grpc_tools.protoc \
  -I "$PROTO_DIR" \
  --python_out="$OUT_DIR" \
  --grpc_python_out="$OUT_DIR" \
  "$PROTO_DIR/songs.proto"
