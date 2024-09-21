# oemerlite
Oemer with Tensorflow lite

This is the tensorflowlite version of oemer. Oemer is an Optical Music Recognition system build on top of deep learning models and machine learning techniques. Able to transcribe on skewed and phone taken photos. The models were trained to identify Western Music Notation, which could mean the system will probably not work on transcribing hand-written scores or othernotation types.

Why oemerlite?

Oemerlite is based on Tensorflowlite which is particularly made for edge devices (e.g Raspberry Pi, mobile phones).

What are the advantages?

It uses way less RAM (~1GB) than the normal oemer (~6GB) and runs on 8 threads (optimal for current mobile SOCs).

What are the disadvantages?

It is slower and less powerefficent (probably because the tensorflowlite-kenel is optimized for ARM). Tested on AMD Ryzen 5 5600. Oemerlite: 135s@25W Oemer: 30s@55W

I did not optimize oemerlite.
