#!/bin/sh
# rampart-build-ext.sh <build-dir> -- build onnxruntime-extensions (tokenizers only,
# static, CPU-only) and fold everything into ONE relocatable object, onnxext_all.o,
# with the bundled protobuf/re2 symbols LOCALIZED.
#
# Why the localize step: ORT's libonnxruntime_deps.a already bundles protobuf + re2,
# and extensions bundles its own (protobuf-lite + re2) -- ~1655 symbols overlap, so
# statically linking both into rampart-onnx.so would fail with "multiple definition".
# We give extensions a PRIVATE, hidden copy: ld -r merges its archives, then objcopy
# demotes the protobuf/re2 symbols to local. Extensions' tokenizers bind to their own
# (version-matched) protobuf during the ld -r; ORT keeps its own. No clash, no protobuf
# version-check abort. Ortx C API + the BertTokenizer class symbols stay global so
# onnx_shim.cc can call them.
#
# Driven by extern.cmake's onnxext_ep ExternalProject (BUILD_BYPRODUCTS=onnxext_all.o).
set -e

BUILD="${1:?usage: rampart-build-ext.sh <build-dir>}"

# macOS: match rampart-langtools' 11.0 floor (see rampart-build-cpu.sh).
if [ "$(uname)" = "Darwin" ]; then
  MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-11.0}"
  export MACOSX_DEPLOYMENT_TARGET
fi
SRC="$(cd "$(dirname "$0")" && pwd)"
OUT="$BUILD/onnxext_all.o"
CMAKE="${CMAKE:-cmake}"

mkdir -p "$BUILD"
"$CMAKE" -S "$SRC" -B "$BUILD" -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DOCOS_ENABLE_C_API=ON \
    -DOCOS_ENABLE_BERT_TOKENIZER=ON \
    -DOCOS_ENABLE_WORDPIECE_TOKENIZER=ON \
    -DOCOS_ENABLE_SPM_TOKENIZER=ON \
    -DOCOS_ENABLE_GPT2_TOKENIZER=ON \
    -DOCOS_ENABLE_TRIE_TOKENIZER=OFF \
    -DOCOS_ENABLE_BLINGFIRE=OFF \
    -DOCOS_ENABLE_VISION=OFF -DOCOS_ENABLE_AUDIO=OFF \
    -DOCOS_ENABLE_DLIB=OFF -DOCOS_ENABLE_MATH=OFF \
    -DOCOS_ENABLE_CV2=OFF -DOCOS_ENABLE_OPENCV_CODECS=OFF \
    -DOCOS_USE_CUDA=OFF -DOCOS_ENABLE_PYTHON=OFF -DOCOS_ENABLE_CTEST=OFF \
    -DOCOS_BUILD_SHARED_LIB=OFF -DOCOS_ENABLE_STATIC_LIB=ON
NPROC=$( (command -v nproc >/dev/null 2>&1 && nproc) || sysctl -n hw.ncpu 2>/dev/null || echo 4)
"$CMAKE" --build "$BUILD" -j"$NPROC"

# Collect the archives (paths differ slightly across cmake versions -> find them).
# NB: we deliberately do NOT bundle extensions' protobuf-LITE. ORT already links
# FULL protobuf (a superset), and shipping a second protobuf makes both copies run
# their static initializers at .so load -> SIGSEGV in InitProtobufDefaults. Leaving
# protobuf undefined here lets sentencepiece bind to ORT's single copy at the final
# link.  (GPU flavor differs -- see ONNXEXT_BUNDLE_PROTOBUF below.)
OCOS="$BUILD/lib/libortcustomops.a"
OPS="$BUILD/lib/libocos_operators.a"
NOEXC="$BUILD/lib/libnoexcep_operators.a"
SPM="$(find "$BUILD" -name libsentencepiece.a | head -1)"
RE2="$(find "$BUILD" -name libre2.a | head -1)"
PBLITE="$(find "$BUILD" -name libprotobuf-lite.a | head -1)"
for a in "$OCOS" "$OPS" "$NOEXC" "$SPM" "$RE2" "$PBLITE"; do
    [ -f "$a" ] || { echo "rampart-build-ext: missing archive $a" >&2; exit 1; }
done

# Merge the extensions archives into one relocatable object. No symbol
# localization; the module version-script (local:*) hides everything from other
# modules. protobuf handling is PER-FLAVOR (ONNXEXT_BUNDLE_PROTOBUF, set by
# extern.cmake):
#  - CPU flavor (0, default): ORT's FULL protobuf is statically linked into the
#    same module -> do NOT bundle protobuf-lite (two protobufs in one .so both
#    run their static initializers at dlopen -> SIGSEGV in InitProtobufDefaults);
#    sentencepiece binds to ORT's copy at the final link.
#  - GPU flavor (1): ORT is the SHARED libonnxruntime.so.1 whose protobuf is
#    hidden -> the module has no other protobuf, so bundle protobuf-lite here.
#    (Separate .so's each with their own hidden protobuf is the normal safe case.)
#  - re2 is always bundled and kept GLOBAL-in-object: static-ORT builds reference
#    re2 but bundle none of their own, so ext's re2 must satisfy that reference.
EXTRA_A=""
[ "${ONNXEXT_BUNDLE_PROTOBUF:-0}" = "1" ] && EXTRA_A="$PBLITE"
if [ "$(uname)" = "Darwin" ]; then
  # Apple ld: -all_load is the --whole-archive equivalent (pulls every member).
  # Direct ld (no clang driver) does not infer the arch -> it must be explicit,
  # or every input is "ignored" and the objc pass asserts on 'unknown objc arch'.
  ld -r -arch "$(uname -m)" -all_load "$OCOS" "$OPS" "$NOEXC" "$SPM" "$RE2" $EXTRA_A -o "$OUT"
else
  ld -r --whole-archive "$OCOS" "$OPS" "$NOEXC" "$SPM" "$RE2" $EXTRA_A \
      --no-whole-archive -o "$OUT"
fi

echo "==> rampart-build-ext: $OUT ($(du -h "$OUT" | cut -f1)); protobuf $( [ -n "$EXTRA_A" ] && echo bundled || echo '-> ORT'\''s' ), re2 bundled global"
