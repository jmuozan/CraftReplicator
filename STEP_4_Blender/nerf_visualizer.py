import bpy
import numpy as np
import json
import os
from mathutils import Vector

class NeRFVisualizer:
    def __init__(self, nerf_data_path):
        self.nerf_data_path = nerf_data_path
        self.density_threshold = 0.5  # Adjustable density threshold
        self.point_size = 0.01  # Size of points in the visualization
        
    def load_nerf_data(self):
        """
        Load NeRF data from JSON file containing density and color information
        Expected format:
        {
            "points": [
                {
                    "position": [x, y, z],
                    "density": float,
                    "color": [r, g, b]
                },
                ...
            ]
        }
        """
        with open(self.nerf_data_path, 'r') as f:
            data = json.load(f)
        return data['points']
    
    def create_point_cloud(self, points):
        """Create a point cloud mesh from NeRF points"""
        # Create a new mesh object
        mesh = bpy.data.meshes.new(name="NeRFPointCloud")
        obj = bpy.data.objects.new("NeRFPointCloud", mesh)
        
        # Link object to scene
        bpy.context.collection.objects.link(obj)
        
        # Create vertices for points above density threshold
        vertices = []
        colors = []
        
        for point in points:
            if point['density'] > self.density_threshold:
                vertices.append(Vector(point['position']))
                colors.append(point['color'])
        
        # Create mesh from vertices
        mesh.from_pydata(vertices, [], [])
        mesh.update()
        
        # Add vertex colors
        if not mesh.vertex_colors:
            mesh.vertex_colors.new()
        
        color_layer = mesh.vertex_colors.active
        
        for i, color in enumerate(colors):
            color_layer.data[i].color = color + [1.0]  # Add alpha channel
        
        return obj
    
    def setup_material(self, obj):
        """Set up material for point cloud visualization"""
        material = bpy.data.materials.new(name="NeRFPointMaterial")
        material.use_nodes = True
        
        # Clear default nodes
        material.node_tree.nodes.clear()
        
        # Create nodes
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        
        # Add vertex color node
        vert_color = nodes.new('ShaderNodeVertexColor')
        
        # Add emission shader
        emission = nodes.new('ShaderNodeEmission')
        emission.inputs[1].default_value = 1.0  # Strength
        
        # Add material output
        material_output = nodes.new('ShaderNodeOutputMaterial')
        
        # Link nodes
        links.new(vert_color.outputs[0], emission.inputs[0])
        links.new(emission.outputs[0], material_output.inputs[0])
        
        # Assign material to object
        if obj.data.materials:
            obj.data.materials[0] = material
        else:
            obj.data.materials.append(material)
    
    def setup_point_cloud_display(self, obj):
        """Set up point cloud display settings"""
        # Enable point display
        obj.data.display_type = 'POINT'
        obj.data.display.use_point_size = True
        obj.data.display.point_size = self.point_size
    
    def visualize(self):
        """Main function to create NeRF visualization"""
        # Clear existing objects
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        
        # Load NeRF data
        points = self.load_nerf_data()
        
        # Create point cloud
        point_cloud_obj = self.create_point_cloud(points)
        
        # Setup material and display
        self.setup_material(point_cloud_obj)
        self.setup_point_cloud_display(point_cloud_obj)
        
        # Set up camera and lighting
        bpy.ops.object.camera_add(location=(0, -5, 0))
        bpy.ops.object.light_add(type='SUN', location=(5, 5, 5))
        
        # Set up viewport
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                area.spaces[0].shading.type = 'RENDERED'
                break

def main():
    # Example usage
    nerf_visualizer = NeRFVisualizer("path/to/nerf_data.json")
    nerf_visualizer.visualize()

if __name__ == "__main__":
    main()