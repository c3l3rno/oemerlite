# oemerlite
Oemer with Tensorflow lite

This is the tensorflowlite version of [oemer](https://github.com/BreezeWhite/oemer) (MIT License). Oemer is an Optical Music Recognition system build on top of deep learning models and machine learning techniques. Able to transcribe on skewed and phone taken photos. The models were trained to identify Western Music Notation, which could mean the system will probably not work on transcribing hand-written scores or othernotation types.

_How to use oemerlite?_

download, make a package, run oemerlite.recognize()
pip installation will be added

_Why oemerlite?_

Oemerlite is based on Tensorflowlite which is particularly made for edge devices (e.g Raspberry Pi, mobile phones).

_What are the advantages?_

It uses way less RAM (~1GB) than the normal oemer (~6GB) and runs on 8 threads (optimal for current mobile SOCs).

_What are the disadvantages?_

It is slower and less powerefficent (probably because the tensorflowlite-kenel is optimized for ARM). Tested on AMD Ryzen 5 5600. Oemerlite: 135s@25W Oemer: 30s@55W
