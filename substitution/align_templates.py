"""Print a nucleotide template in its local frame (P at the origin, P/C4'/C1' triangle in the xy plane).

    python align_templates.py templates/Adenine_template.cif
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Monomer import Monomer, TEMPLATE_FILES


def main(paths):
    for path in paths:
        name = os.path.basename(path)
        letter = next((k for k, v in TEMPLATE_FILES.items() if v == name), None)
        if letter is None:
            sys.exit(f"{path}: not one of the known templates {sorted(TEMPLATE_FILES.values())}")
        m = Monomer("nucleotide")
        m.load_template(letter, template_dir=os.path.dirname(os.path.abspath(path)))
        m.add_name(letter)
        print(m.align_triangle_to_xy())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
