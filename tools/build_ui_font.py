"""Subset the locally installed OFL Noto Sans SC for this project's UI text."""
from pathlib import Path
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
from fontTools import subset

root = Path(__file__).resolve().parents[1]
source = Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf")
target_dir = root / "Assets" / "Resources" / "Fonts"
target_dir.mkdir(parents=True, exist_ok=True)

characters = set(range(32, 127))
for script in (root / "Assets" / "Scripts").glob("*.cs"):
    characters.update(ord(char) for char in script.read_text(encoding="utf-8") if ord(char) > 127)

font = TTFont(source)
font = instantiateVariableFont(font, {"wght": 400}, inplace=True)
options = subset.Options()
options.name_IDs = [0, 1, 2, 3, 4, 5, 6, 13, 14, 16, 17]
options.layout_features = ["*"]
subsetter = subset.Subsetter(options=options)
subsetter.populate(unicodes=characters)
subsetter.subset(font)

for record in font["name"].names:
    rename = {1: "QHOP CJK", 3: "QHOP CJK Regular", 4: "QHOP CJK Regular",
              6: "QHOPCJK-Regular", 16: "QHOP CJK", 17: "Regular"}.get(record.nameID)
    if rename is not None:
        record.string = rename.encode(record.getEncoding())

output = target_dir / "QHOP-CJK.ttf"
font.save(output)

template = (root / "Assets" / "TextMesh Pro" / "Fonts" / "LiberationSans - OFL.txt").read_text(encoding="utf-8")
license_body = template[template.index("-----------------------------------------------------------"):]
notice = "Copyright © 2014-2021 Adobe (http://www.adobe.com/), with Reserved Font Name 'Source'.\n"
notice += "QHOP CJK is a subset of Noto Sans SC with a changed family name.\n\n"
(target_dir / "OFL.txt").write_text(notice + license_body, encoding="utf-8")
print(f"Built {output}: {output.stat().st_size} bytes, {len(characters)} requested characters")
