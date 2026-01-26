#!/bin/bash
# Generate Python gRPC stubs from proto files
#
# Usage: ./scripts/generate_protos.sh
#
# Prerequisites:
#   pip install grpcio-tools googleapis-common-protos mypy-protobuf

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_BIN="$PROJECT_ROOT/.venv/bin"
OUTPUT_DIR="$PROJECT_ROOT/athyr/_grpc"

# Find the site-packages directory for googleapis protos
SITE_PACKAGES=$("$VENV_BIN/python" -c "import site; print(site.getsitepackages()[0])")

echo "Generating Python gRPC stubs..."
echo "  Proto source: $PROJECT_ROOT/proto"
echo "  Output: $OUTPUT_DIR"
echo "  Google protos: $SITE_PACKAGES"

# Add venv bin to PATH so protoc can find the plugins
export PATH="$VENV_BIN:$PATH"

# Generate Python stubs
"$VENV_BIN/python" -m grpc_tools.protoc \
    --proto_path="$PROJECT_ROOT/proto" \
    --proto_path="$SITE_PACKAGES" \
    --python_out="$OUTPUT_DIR" \
    --grpc_python_out="$OUTPUT_DIR" \
    --mypy_out="$OUTPUT_DIR" \
    --mypy_grpc_out="$OUTPUT_DIR" \
    "$PROJECT_ROOT/proto/athyr/v1/athyr.proto"

# Flatten the directory structure (proto package creates athyr/v1/)
if [ -d "$OUTPUT_DIR/athyr/v1" ]; then
    mv "$OUTPUT_DIR/athyr/v1/"* "$OUTPUT_DIR/"
    rm -rf "$OUTPUT_DIR/athyr"
fi

# Fix imports in generated files
cd "$OUTPUT_DIR"

# Fix the grpc file to use relative import
if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' 's/from athyr\.v1 import athyr_pb2 as athyr_dot_v1_dot_athyr__pb2/from . import athyr_pb2 as athyr_dot_v1_dot_athyr__pb2/' athyr_pb2_grpc.py
else
    sed -i 's/from athyr\.v1 import athyr_pb2 as athyr_dot_v1_dot_athyr__pb2/from . import athyr_pb2 as athyr_dot_v1_dot_athyr__pb2/' athyr_pb2_grpc.py
fi

echo ""
echo "Done! Generated files:"
ls -la "$OUTPUT_DIR/"*.py 2>/dev/null | grep -v __init__
echo ""
ls -la "$OUTPUT_DIR/"*.pyi 2>/dev/null
