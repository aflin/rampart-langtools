#!/bin/bash

die() {
    echo "Build/Install for $1 failed"
    exit 1;
}

for i in cpu cpu_2_28 cu11_2_28 cu12 cu13; do
    ./build.sh build $i || die $i;
    ./build.sh install $i || die $i;
done;
