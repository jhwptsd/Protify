import numpy as np
import copy
from gemmi import *


class Monomer:
    def __init__(self, macro):
        self.atoms = dict()
        self.name = ""
        self.macro = macro
        
    def add_atom(self, x, y, z, e):
        self.atoms[e] = np.array([x, y, z])
        
    def element(self, e):
        if e[0]=="\"":
            return e[1]
        else:
            return e[0]
        
    def get_atoms(self):
        return self.atoms
    
    def add_name(self, name):
        self.name = name
    
    def apply_transformation(self, x, y, z):
        out = self
        for atom in out.atoms:
            out.atoms[atom] += [x,y,z]
        return out
    
    def calculate_normal(self):
        # Get the triangle vertices
        c4_pos = np.array(self.atoms['"C4\'"'])
        c1_pos = np.array(self.atoms['"C1\'"'])
        translation = -np.array(self.atoms['P'])
            
        # Calculate triangle vectors
        p_to_c4 = c4_pos + translation  # Vector from P to C4'
        p_to_c1 = c1_pos + translation  # Vector from P to C1'
        
        # Calculate normal to triangle
        normal = np.cross(p_to_c4, p_to_c1)
        normal = normal / np.linalg.norm(normal)
        return normal
    
    def align_triangle_to_xy(self):
        """
        Aligns the triangle formed by C4', C1', and N1/N9 atoms to the positive xy plane.
        """
        out = copy.deepcopy(self)
        # Get the coordinates of the three atoms forming the triangle
        c4_coords = np.array(out.get_atom_coordinates('\"C4\'\"'))
        c1_coords = np.array(out.get_atom_coordinates('\"C1\'\"'))
        base_coords = np.array(out.get_atom_coordinates("P"))

        if c4_coords is None or c1_coords is None or base_coords is None:
            raise ValueError("Could not find required atoms for alignment")

        # Create vectors from C4' to C1' and C4' to N1/N9
        v1 = c1_coords - c4_coords
        v2 = base_coords - c4_coords

        # Calculate the normal vector of the triangle
        normal = np.cross(v1, v2)
        normal_magnitude = np.linalg.norm(normal)
        
        if normal_magnitude < 1e-10:
            raise ValueError("Colinear points cannot form a triangle")
            
        normal = normal / normal_magnitude

        # Calculate rotation matrix to align normal vector with z-axis
        z_axis = np.array([0, 0, 1])
        rotation_axis = np.cross(normal, z_axis)
        rotation_axis_magnitude = np.linalg.norm(rotation_axis)

        if rotation_axis_magnitude < 1e-10:
            # If vectors are parallel, no rotation needed or rotate 180° if antiparallel
            if normal[2] < 0:
                # If normal points in negative z, rotate 180° around x-axis
                rotation_matrix = np.array([
                    [1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1]
                ])
            else:
                return  # Already aligned correctly
        else:
            rotation_axis = rotation_axis / rotation_axis_magnitude
            angle = np.arccos(np.clip(np.dot(normal, z_axis), -1.0, 1.0))
            
            # Create rotation matrix using Rodrigues' rotation formula
            K = np.array([
                [0, -rotation_axis[2], rotation_axis[1]],
                [rotation_axis[2], 0, -rotation_axis[0]],
                [-rotation_axis[1], rotation_axis[0], 0]
            ])
            rotation_matrix = (np.eye(3) + np.sin(angle) * K + 
                            (1 - np.cos(angle)) * np.matmul(K, K))

        # Apply rotation to all atoms
        for atom in out.atoms.keys():
            coords = np.array(out.atoms[atom]) - c4_coords  # Center around C4'
            rotated_coords = np.dot(rotation_matrix, coords)
            atom.set_coordinates(rotated_coords + c4_coords)  # Move back to original position

        # After first rotation, calculate the angle in xy plane between C4'-C1' vector and x-axis
        c4_coords = np.array(out.get_atom_coordinates('C4\''))
        c1_coords = np.array(out.get_atom_coordinates('C1\''))
        v1_xy = c1_coords[:2] - c4_coords[:2]  # Only consider x and y components
        v1_xy_magnitude = np.linalg.norm(v1_xy)
        
        if v1_xy_magnitude < 1e-10:
            return  # Vector is vertical, no need for xy rotation
            
        cos_theta = np.clip(np.dot(v1_xy, [1, 0]) / v1_xy_magnitude, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        
        # Determine if we need to rotate clockwise or counterclockwise
        if v1_xy[1] < 0:
            theta = -theta

        # Create rotation matrix around z-axis
        rotation_matrix_z = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])

        # Apply second rotation to all atoms
        for atom in out.atoms:
            coords = np.array(atom.get_coordinates()) - c4_coords
            rotated_coords = np.dot(rotation_matrix_z, coords)
            atom.set_coordinates(rotated_coords + c4_coords)

        # Final check to ensure the molecule is in the positive xy plane
        # If the base atom is in the negative x region, rotate 180° around y-axis
        base_coords = np.array(out.get_atom_coordinates(out.base_atom))
        if base_coords[0] - c4_coords[0] < 0:
            rotation_matrix_y = np.array([
                [-1, 0, 0],
                [0, 1, 0],
                [0, 0, -1]
            ])
            for atom in out.atoms:
                coords = np.array(atom.get_coordinates()) - c4_coords
                rotated_coords = np.dot(rotation_matrix_y, coords)
                atom.set_coordinates(rotated_coords + c4_coords)


    def align_to_normal(self, target_normal):
        """
        Rotates the monomer so that the normal vector of its P-C1'-C4' triangle 
        aligns with the given target normal vector.
        
        Args:
            target_normal (np.ndarray): The target normal vector to align with (should be normalized)
            
        Returns:
            Monomer: A new Monomer instance with the rotated coordinates
        """
        out = copy.deepcopy(self)
        
        try:
            # Get current normal vector
            current_normal = self.calculate_normal()
            
            # Normalize target vector
            target_normal = target_normal / np.linalg.norm(target_normal)
            
            # Calculate rotation axis and angle
            rotation_axis = np.cross(current_normal, target_normal)
            
            # If vectors are parallel (or anti-parallel), rotation axis will be zero
            if np.linalg.norm(rotation_axis) < 1e-10:
                # If normals are anti-parallel, rotate 180° around any perpendicular axis
                if np.dot(current_normal, target_normal) < 0:
                    # Find a perpendicular vector to rotate around
                    if abs(current_normal[0]) < abs(current_normal[1]):
                        rotation_axis = np.cross(current_normal, [1, 0, 0])
                    else:
                        rotation_axis = np.cross(current_normal, [0, 1, 0])
                    angle = np.pi
                else:
                    # Vectors are already aligned
                    return out
            else:
                # Calculate rotation angle
                angle = np.arccos(np.clip(np.dot(current_normal, target_normal), -1.0, 1.0))
            
            # Normalize rotation axis
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            
            # Create rotation matrix using Rodrigues' rotation formula
            cos_theta = np.cos(angle)
            sin_theta = np.sin(angle)
            K = np.array([
                [0, -rotation_axis[2], rotation_axis[1]],
                [rotation_axis[2], 0, -rotation_axis[0]],
                [-rotation_axis[1], rotation_axis[0], 0]
            ])
            R = np.eye(3) + sin_theta * K + (1 - cos_theta) * np.dot(K, K)
            
            # Apply rotation to all atoms
            for atom in out.atoms:
                out.atoms[atom] = np.dot(R, out.atoms[atom])
                
            return out
            
        except KeyError as e:
            raise KeyError(f"Required atom {e} not found in this monomer")

    
    def load_template(self, n):
        if n=="A": path = "templates/Adenine_template.cif"
        elif n=="C": path = "templates/Cytosine_template.cif"
        elif n=="G": path = "templates/Guanine_template.cif"
        elif n=="U": path = "templates/Uracil_template.cif"
        atoms = []
        atom_xs = []
        atom_ys = []
        atom_zs = []
        try:
            doc = cif.read_file(path)  # copy all the data from mmCIF file
            block = doc.sole_block()  # mmCIF has exactly one block
            for element in block.find_loop("_atom_site.label_atom_id"):
                atoms.append(element)
            for element in block.find_loop("_atom_site.Cartn_x"):
                atom_xs.append(float(element))
            for element in block.find_loop("_atom_site.Cartn_y"):
                atom_ys.append(float(element))
            for element in block.find_loop("_atom_site.Cartn_z"):
                atom_zs.append(float(element))
        except Exception as e:
            print("Oops. %s" % e)
        for i in range(len(atoms)):
            self.add_atom(atom_xs[i], atom_ys[i], atom_zs[i], atoms[i])
        
        
    def __str__(self, start=1):
        #Return what this monomer would look like in an mmCIF file#
        out = ""
        c = start
        for i in self.atoms:
            out += f"ATOM {c}\t{self.element(i)}\t{i}\t{self.name}\t. A 1 1\t?\t{round(self.atoms[i][0],3)}\t{round(self.atoms[i][1],3)}\t{round(self.atoms[i][2],3)}\n"
            c += 1
        out += "\b\b"
        return out
    
    def print(self, start=1, number=1):
        #Return what this monomer would look like in an mmCIF file#
        out = ""
        c = start
        for i in self.atoms:
            out += f"ATOM {c}\t{self.element(i)}\t{i}\t{self.name}\t. A 1 {number}\t?\t{round(self.atoms[i][0],3)}\t{round(self.atoms[i][1],3)}\t{round(self.atoms[i][2],3)}\n"
            c += 1
        return out
    
    def __len__(self):
        return len(self.atoms)