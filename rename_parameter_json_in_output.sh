#!/bin/bash

pushd synth-output
for d in */
do
    mv $d/{$d,input}.json
done
popd
