#!/bin/bash

die() {
    echo "Build/Install for $1 failed"
    exit 1;
}

# cu11 is x86-only.  An ARM cu11 build bakes SASS for sm_72/sm_87 (Xavier and
# Orin -- Jetson iGPUs) but compiles against CUDA 11.8, and no JetPack ever
# shipped 11.8 (JP4=10.2, JP5=11.4, JP6=12.6 -> cu12, JP7=13 -> cu13), so the
# driver floor can never be met on the only hardware it targets: the artifact
# has no possible host.  ARM Jetson/Grace boxes use cu12/cu13.
case "$(uname -m)" in
    x86_64) variants="cpu cpu_2_28 cu11_2_28 cu12 cu13" ;;
    *)      variants="cpu cpu_2_28 cu12 cu13" ;;
esac

echo "==> building variants for $(uname -m): $variants"

for i in $variants; do
    ./build.sh build $i || die $i;
    ./build.sh install $i || die $i;
done;
