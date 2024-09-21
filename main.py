import os
import pickle
import urllib.request
import pathlib
from typing import Tuple

from PIL import Image
from numpy import ndarray
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np

from oemerlite import MODULE_PATH
from oemerlite import layers
from oemerlite.inference import inference
from oemerlite.dewarp import estimate_coords, dewarp
from oemerlite.staffline_extraction import extract as staff_extract
from oemerlite.notehead_extraction import extract as note_extract
from oemerlite.note_group_extraction import extract as group_extract
from oemerlite.symbol_extraction import extract as symbol_extract
from oemerlite.rhythm_extraction import extract as rhythm_extract
from oemerlite.build_system import MusicXMLBuilder
from oemerlite.draw_teaser import teaser


#add location for tflite files in github
CHECKPOINTS_URL = {
    "1st_model.tflite": "https://github.com/c3l3rno/oemerlite/releases/download/v0.0.1/1st_model.tflite",
    "2nd_model.tflite": "https://github.com/c3l3rno/oemerlite/releases/download/v0.0.1/2nd_model.tflite"
}


def clear_data() -> None:
    lls = layers.list_layers()
    for l in lls:
        layers.delete_layer(l)


def generate_pred(img_path: str) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
    print("Extracting staffline and symbols")
    staff_symbols_map, _ = inference(
        os.path.join(MODULE_PATH, "checkpoints/unet_big"),
        img_path
    )
    staff = np.where(staff_symbols_map==1, 1, 0)
    symbols = np.where(staff_symbols_map==2, 1, 0)

    print("Extracting layers of different symbols")
    #symbol_thresholds = [0.5, 0.4, 0.4]
    sep, _ = inference(
        os.path.join(MODULE_PATH, "checkpoints/seg_net"),
        img_path,
        manual_th=None
    )
    stems_rests = np.where(sep==1, 1, 0)
    notehead = np.where(sep==2, 1, 0)
    clefs_keys = np.where(sep==3, 1, 0)
    # stems_rests = sep[..., 0]
    # notehead = sep[..., 1]
    # clefs_keys = sep[..., 2]

    return staff, symbols, stems_rests, notehead, clefs_keys


def polish_symbols(rgb_black_th=300):
    img = layers.get_layer('original_image')
    sym_pred = layers.get_layer('symbols_pred')

    img = Image.fromarray(img).resize((sym_pred.shape[1], sym_pred.shape[0]))
    arr = np.sum(np.array(img), axis=-1)
    arr = np.where(arr < rgb_black_th, 1, 0)  # Filter background
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    arr = cv2.dilate(cv2.erode(arr.astype(np.uint8), ker), ker)  # Filter staff lines
    mix = np.where(sym_pred+arr>1, 1, 0)
    return mix


def register_notehead_bbox(bboxes):
    symbols = layers.get_layer('symbols_pred')
    layer = layers.get_layer('bboxes')
    for (x1, y1, x2, y2) in bboxes:
        yi, xi = np.where(symbols[y1:y2, x1:x2]>0)
        yi += y1
        xi += x1
        layer[yi, xi] = np.array([x1, y1, x2, y2])
    return layer


def register_note_id() -> None:
    symbols = layers.get_layer('symbols_pred')
    layer = layers.get_layer('note_id')
    notes = layers.get_layer('notes')
    for idx, note in enumerate(notes):
        x1, y1, x2, y2 = note.bbox
        yi, xi = np.where(symbols[y1:y2, x1:x2]>0)
        yi += y1
        xi += x1
        layer[yi, xi] = idx
        notes[idx].id = idx


def extract(img_path, use_tf, no_dewarp, save_cache, out_path) -> str:
    #removes datatype
    f_name = os.path.splitext(img_path)[0]
    print(f"picture_name {f_name}")

    pkl_path = pathlib.Path(os.path.splitext(f_name)[0] + ".pkl")
    if pkl_path.exists():
        # Load from cache
        pred = pickle.load(open(pkl_path, "rb"))
        notehead = pred["note"]
        symbols = pred["symbols"]
        staff = pred["staff"]
        clefs_keys = pred["clefs_keys"]
        stems_rests = pred["stems_rests"]
    else:
        # Make predictions
        if use_tf:
            ori_inf_type = os.environ.get("INFERENCE_WITH_TF", None)
            os.environ["INFERENCE_WITH_TF"] = "true"
        staff, symbols, stems_rests, notehead, clefs_keys = generate_pred(str(img_path))
        if use_tf and ori_inf_type is not None:
            os.environ["INFERENCE_WITH_TF"] = ori_inf_type
        
        #we dont need to save cache
        if save_cache:
            data = {
                'staff': staff,
                'note': notehead,
                'symbols': symbols,
                'stems_rests': stems_rests,
                'clefs_keys': clefs_keys
            }
            pickle.dump(data, open(pkl_path, "wb"))

    # Load the original image, resize to the same size as prediction.

    image_pil = Image.open(str(img_path))
    if "GIF" != image_pil.format:
        image = cv2.imread(str(img_path))
    else:
        gif_image = image_pil.convert('RGB')
        gif_img_arr = np.array(gif_image)
        image = gif_img_arr[:, :, ::-1].copy()
    
    print(f"reisze{staff.shape[1], staff.shape[0]}")

    image = cv2.resize(image, (staff.shape[1], staff.shape[0]))

    if not no_dewarp:
        print("Dewarping")
        coords_x, coords_y = estimate_coords(staff)
        staff = dewarp(staff, coords_x, coords_y)
        symbols = dewarp(symbols, coords_x, coords_y)
        stems_rests = dewarp(stems_rests, coords_x, coords_y)
        clefs_keys = dewarp(clefs_keys, coords_x, coords_y)
        notehead = dewarp(notehead, coords_x, coords_y)
        for i in range(image.shape[2]):
            image[..., i] = dewarp(image[..., i], coords_x, coords_y)

    # Register predictions
    symbols = symbols + clefs_keys + stems_rests
    symbols[symbols>1] = 1
    layers.register_layer("stems_rests_pred", stems_rests)
    layers.register_layer("clefs_keys_pred", clefs_keys)
    layers.register_layer("notehead_pred", notehead)
    layers.register_layer("symbols_pred", symbols)
    layers.register_layer("staff_pred", staff)
    layers.register_layer("original_image", image)

    # ---- Extract staff lines and group informations ---- #
    print("Extracting stafflines")
    staffs, zones = staff_extract()
    layers.register_layer("staffs", staffs)  # Array of 'Staff' instances
    layers.register_layer("zones", zones)  # Range of each zones, array of 'range' object.

    # ---- Extract noteheads ---- #
    print("Extracting noteheads")
    notes = note_extract()

    # Array of 'NoteHead' instances.
    layers.register_layer('notes', np.array(notes))

    # Add a new layer (w * h), indicating note id of each pixel.
    layers.register_layer('note_id', np.zeros(symbols.shape, dtype=np.int64)-1)
    register_note_id()

    # ---- Extract groups of note ---- #
    print("Grouping noteheads")
    groups, group_map = group_extract()
    layers.register_layer('note_groups', np.array(groups))
    layers.register_layer('group_map', group_map)

    # ---- Extract symbols ---- #
    print("Extracting symbols")
    barlines, clefs, sfns, rests = symbol_extract()
    layers.register_layer('barlines', np.array(barlines))
    layers.register_layer('clefs', np.array(clefs))
    layers.register_layer('sfns', np.array(sfns))
    layers.register_layer('rests', np.array(rests))

    # ---- Parse rhythm ---- #
    print("Extracting rhythm types")
    rhythm_extract()

    # ---- Build MusicXML ---- #
    print("Building MusicXML document")
    basename = os.path.basename(img_path).replace(".jpg", "").replace(".png", "")
    builder = MusicXMLBuilder(title=basename.capitalize())
    raw_data = builder.build()
    #sort(information)
    #mark_unsure
    xml = builder.to_musicxml()

    # ---- Write out the MusicXML ---- #
    if not out_path.endswith(".musicxml"):
        # Take the output path as the folder.
        out_path = os.path.join(out_path, basename+".musicxml")

    with open(out_path, "wb") as ff:
        ff.write(xml)

    return out_path, raw_data


def download_file(title: str, url: str, save_path: str) -> None:
    resp = urllib.request.urlopen(url)
    length = int(resp.getheader("Content-Length", -1))

    chunk_size = 2**9
    total = 0
    with open(save_path, "wb") as out:
        while True:
            print(f"{title}: {total*100/length:.1f}% {total}/{length}", end="\r")
            data = resp.read(chunk_size)
            if not data:
                break
            total += out.write(data)
        print(f"{title}: 100% {length}/{length}"+" "*20)



def recognize(img_path, use_tf, dewarp, use_cache, out_path) -> None:
    if not os.path.exists(img_path):
        print(f"The given image path doesn't exists: {img_path}")

    # Check there are checkpoints
    chk_path = os.path.join(MODULE_PATH, "checkpoints/unet_big/model.tflite")
    if not os.path.exists(chk_path):
        print(f"No checkpoint found in {chk_path}")

        for idx, (title, url) in enumerate(CHECKPOINTS_URL.items()):
            print(f"Downloading checkpoints ({idx+1}/{len(CHECKPOINTS_URL)})")
            save_dir = "unet_big" if title.startswith("1st") else "seg_net"
            save_dir = os.path.join(MODULE_PATH, "checkpoints", save_dir)
            save_path = os.path.join(save_dir, title.split("_")[1])
            download_file(title, url, save_path)

    clear_data()
    mxl_path, raw_data = extract(img_path, use_tf, dewarp, use_cache, out_path)
    img = teaser()
    img.save(mxl_path.replace(".musicxml", "_teaser.png"))
    return(raw_data, mxl_path)