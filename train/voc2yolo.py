"""Convert DUT Anti-UAV (Pascal VOC XML) to YOLO format; every object -> class 0 'drone'.
Expects <src>/{train,val,test}/{img,xml}/ and writes <dst>/data.yaml.  Usage: python voc2yolo.py DUT-Anti-UAV dut_yolo"""
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image


def voc_to_yolo(xmin, ymin, xmax, ymax, w, h):
    return (xmin + xmax) / 2 / w, (ymin + ymax) / 2 / h, (xmax - xmin) / w, (ymax - ymin) / h


def convert(src, dst, split):
    out_img, out_lbl = dst / "images" / split, dst / "labels" / split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    n = 0
    for xml in sorted((src / split / "xml").glob("*.xml")):
        imgs = list((src / split / "img").glob(xml.stem + ".*"))
        if not imgs:
            print("no image for", xml)
            continue
        w, h = Image.open(imgs[0]).size  # real size; some XML <size> tags are wrong
        lines = []
        for obj in ET.parse(xml).getroot().iter("object"):
            b = obj.find("bndbox")
            box = voc_to_yolo(*(float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")), w, h)
            lines.append("0 " + " ".join(f"{v:.6f}" for v in box))
        shutil.copy(imgs[0], out_img / imgs[0].name)
        (out_lbl / f"{xml.stem}.txt").write_text("\n".join(lines))
        n += 1
    print(split, n, "images")


assert voc_to_yolo(10, 20, 30, 60, 100, 200) == (0.2, 0.2, 0.2, 0.2)

if __name__ == "__main__":
    src, dst = Path(sys.argv[1]), Path(sys.argv[2]).resolve()
    splits = [s for s in ("train", "val", "test") if (src / s).exists()]
    for split in splits:
        convert(src, dst, split)
    yaml = [f"path: {dst.as_posix()}"] + [f"{s}: images/{s}" for s in splits] + ["names:", "  0: drone"]
    (dst / "data.yaml").write_text("\n".join(yaml) + "\n")
    print("wrote", dst / "data.yaml")
