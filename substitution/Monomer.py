import copy
import os
import shlex

import numpy as np

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
TEMPLATE_FILES = {
    "A": "Adenine_template.cif",
    "C": "Cytosine_template.cif",
    "G": "Guanine_template.cif",
    "U": "Uracil_template.cif",
}


def rotation_between(a, b):
    """Rotation matrix taking unit vector a onto unit vector b (Rodrigues' formula)."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    axis = np.cross(a, b)
    s = np.linalg.norm(axis)
    c = float(np.dot(a, b))
    if s < 1e-10:
        if c > 0:
            return np.eye(3)
        # Antiparallel: rotate 180 degrees about any axis perpendicular to a
        perp = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < abs(a[1]) else [0.0, 1.0, 0.0])
        perp = perp / np.linalg.norm(perp)
        return 2 * np.outer(perp, perp) - np.eye(3)
    axis = axis / s
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + s * K + (1 - c) * (K @ K)


class Monomer:
    """A single nucleotide: a dict of atom name -> xyz position. Atom names are stored without
    the quotes mmCIF puts around names containing a prime (C1', O5', ...)."""

    def __init__(self, macro="nucleotide"):
        self.atoms = dict()
        self.name = ""
        self.macro = macro

    def add_atom(self, x, y, z, e):
        self.atoms[e.strip('"')] = np.array([x, y, z], dtype=float)

    def get_atom_coordinates(self, e):
        return self.atoms[e.strip('"')]

    def element(self, e):
        return e.strip('"')[0]

    def get_atoms(self):
        return self.atoms

    def add_name(self, name):
        self.name = name

    def apply_transformation(self, x, y, z):
        """Return a copy translated by (x, y, z)."""
        out = copy.deepcopy(self)
        for atom in out.atoms:
            out.atoms[atom] = out.atoms[atom] + np.array([x, y, z])
        return out

    def _rotate(self, R, origin):
        out = copy.deepcopy(self)
        for atom in out.atoms:
            out.atoms[atom] = R @ (out.atoms[atom] - origin) + origin
        return out

    def calculate_normal(self):
        """Unit normal of the P, C4', C1' triangle."""
        p = self.get_atom_coordinates("P")
        normal = np.cross(self.get_atom_coordinates("C4'") - p, self.get_atom_coordinates("C1'") - p)
        return normal / np.linalg.norm(normal)

    def align_triangle_to_xy(self):
        """Return a copy in a local frame with P at the origin, the P/C4'/C1' triangle in the xy
        plane (normal along +z) and the P->C4' direction along +x."""
        p = self.get_atom_coordinates("P")
        v = self.get_atom_coordinates("C4'") - p
        z = self.calculate_normal()
        x = v - z * np.dot(v, z)
        if np.linalg.norm(x) < 1e-10:
            raise ValueError("Colinear points cannot form a triangle")
        x = x / np.linalg.norm(x)
        frame = np.stack([x, np.cross(z, x), z])  # rows are the new axes
        out = copy.deepcopy(self)
        for atom in out.atoms:
            out.atoms[atom] = frame @ (out.atoms[atom] - p)
        return out

    def align_to_normal(self, target_normal):
        """Return a copy rotated about P so the P/C4'/C1' normal points along target_normal."""
        R = rotation_between(self.calculate_normal(), np.asarray(target_normal, dtype=float))
        return self._rotate(R, self.get_atom_coordinates("P"))

    def load_template(self, n, template_dir=TEMPLATE_DIR):
        """Load the atoms of nucleotide n (A, C, G or U) from its template mmCIF file."""
        if n not in TEMPLATE_FILES:
            raise ValueError(f"Unknown nucleotide {n!r}; expected one of A, C, G, U")
        self.atoms = dict()
        with open(os.path.join(template_dir, TEMPLATE_FILES[n])) as f:
            for line in f:
                if not line.startswith("ATOM"):
                    continue
                fields = shlex.split(line)  # keeps "O5'" together and strips its quotes
                # ATOM id type atom comp alt asym entity seq ins x y z
                self.add_atom(float(fields[-3]), float(fields[-2]), float(fields[-1]), fields[3])

    def _atom_lines(self, start, number):
        lines = []
        for c, (atom, xyz) in enumerate(self.atoms.items(), start=start):
            label = f'"{atom}"' if "'" in atom else atom
            lines.append(f"ATOM {c} {self.element(atom)} {label} {self.name} . A 1 {number} ? "
                         f"{xyz[0]:.3f} {xyz[1]:.3f} {xyz[2]:.3f}\n")
        return "".join(lines)

    def __str__(self, start=1):
        """What this monomer would look like in an mmCIF file."""
        return self._atom_lines(start, 1)

    def print(self, start=1, number=1):
        """Same as str(), with an explicit residue number."""
        return self._atom_lines(start, number)

    def __len__(self):
        return len(self.atoms)
