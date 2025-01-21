import bpy
import os
import math
from mathutils import Vector, Euler
import contextlib
import sys
import time

@contextlib.contextmanager
def stdout_redirected(to=os.devnull):
    '''
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''
    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout) # restore stdout.
                                    

# ------------------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------------------
# Directory containing all GLB files
input_root = os.path.join('.cache', 'TRELLIS')

# Root for output images
output_root = os.path.join('.cache')

# Ensure the output directory exists
if not os.path.exists(output_root):
    os.makedirs(output_root)

# ------------------------------------------------------------------------------
# UTILITY FUNCTIONS
# ------------------------------------------------------------------------------

def clean_scene():
    """Remove all objects, meshes, and materials (except camera, light, world)."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    # Remove leftover data blocks
    for data_iter in (bpy.data.meshes, bpy.data.materials, bpy.data.images):
        for block in list(data_iter):
            data_iter.remove(block, do_unlink=True)

def setup_world_white_background():
    """Set the world background to pure white."""
    bpy.context.scene.world.use_nodes = True
    world_nodes = bpy.context.scene.world.node_tree.nodes
    bg_node = world_nodes.get('Background')
    if bg_node:
        bg_node.inputs[0].default_value = (1, 1, 1, 1)  # white
        bg_node.inputs[1].default_value = 1.0

def setup_camera():
    """
    Create a new camera object in the scene and make it the active camera.
    Returns the camera object.
    """
    cam_data = bpy.data.cameras.new("Camera")
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    return cam_obj

def setup_camera_light(parent_camera):
    """
    Create a light, parent it to the camera, and position it behind the camera.
    This light will move with the camera.
    """
    # Create a point light (you can also use SUN, AREA, etc.)
    light_data = bpy.data.lights.new(name="CameraLight", type='POINT')
    light = bpy.data.objects.new(name="CameraLight", object_data=light_data)
    bpy.context.scene.collection.objects.link(light)
    
    # Parent the light to the camera so they move together
    light.parent = parent_camera
    
    # Position the light slightly behind the camera along its local -Z axis
    # Note: In Blender, a camera looks down its local -Z axis, so placing
    # the light along the local +Z axis of the camera will put it behind the camera.
    light.location = (0, 0, 0)  # start at camera position
    # Move the light along the camera's local Z axis (backwards from view)
    light.location.z += 0.5  # Adjust as necessary for distance behind camera
    
    # Optionally adjust light properties
    light.data.energy = 1000  # Increase/decrease for brightness
    return light

def setup_light():
    """
    Create a sun light in the scene. Returns the light object.
    """
    light_data = bpy.data.lights.new(name="Sun", type='SUN')
    light_obj = bpy.data.objects.new(name="Sun", object_data=light_data)
    bpy.context.scene.collection.objects.link(light_obj)
    light_obj.location = (0, 0, 10)
    return light_obj

def adjust_camera_for_object(cam, obj):
    """
    Reposition and orient the camera so the object is nicely framed.
    We'll compute the object's bounding box center and radius, then place
    the camera at an angle above it, looking down.
    """
    # First, ensure the object's rotation is zero when measuring bounding box.
    obj.rotation_euler = (0, 0, 0)
    
    # Calculate bounding box in world coordinates
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    center = sum(bbox_corners, Vector()) / 8.0
    max_dist = max((corner - center).length for corner in bbox_corners)
    
    # Field of view (radians) for the camera
    fov = cam.data.angle
    # Distance needed so that the object fits in the camera's view
    distance = max_dist / math.tan(fov / 2.0)
    
    # Place the camera above and behind the object, angled downward
    # Example: from negative Y and positive Z, looking down at the center
    # We'll add a little extra multiplier so the object isn't too close to the frame edges
    camera_multiplier = 1.4
    cam.location = (center.x, center.y - distance * camera_multiplier, center.z + distance * 0.8)
    cam.data.clip_end = distance * 10.0
    
    # Aim the camera at the center
    direction = center - cam.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()

def render_object_at_rotations(obj, output_dir, base_name, samples=4096):
    """
    Render 8 images of 'obj' at different Z-rotations and save to output_dir.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Re-adjust camera every time in case bounding box changed
    adjust_camera_for_object(cam, obj)
    
    # Set render samples
    bpy.context.scene.cycles.samples = samples  # Set to a lower value, e.g., 512
    bpy.context.scene.cycles.use_adaptive_sampling = True  # Enable adaptive sampling for efficiency
    
    # For each of 8 rotations around Z-axis
    for i in range(8):
        angle_degrees = i * (360.0 / 8)
        print("Angle degrees:", angle_degrees)
        obj.rotation_mode = 'XYZ'
        obj.rotation_euler = (0, 0, math.radians(angle_degrees))
        
        # Set output filename
        filename = f"{base_name}-{i+1}.jpeg"
        bpy.context.scene.render.filepath = os.path.join(output_dir, filename)
        
        # Render
        with stdout_redirected():
            bpy.ops.render.render(write_still=True)
    
    # Save the scene as a Blender file
    # bpy.ops.wm.save_as_mainfile(filepath=os.path.join(output_dir, f"{base_name}.blend"))
    

# ------------------------------------------------------------------------------
# INITIAL SETUP
# ------------------------------------------------------------------------------
clean_scene()
setup_world_white_background()
cam = setup_camera()
light = setup_camera_light(cam)

# Set render settings (1024x1024, JPEG, Cycles)
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.render.film_transparent = False  # White background
scene.render.resolution_x = 1024
scene.render.resolution_y = 1024
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'JPEG'

# ------------------------------------------------------------------------------
# MAIN LOOP: Import each GLB and render
# ------------------------------------------------------------------------------
for root, dirs, files in os.walk(input_root):
    for file in files:
        if file.lower().endswith('.glb'):
            glb_path = os.path.join(root, file)
            
            # Derive subdirectory and base name for outputs
            dir_name = os.path.basename(root)  # e.g. "TRELLIS" or subfolder
            base_name = os.path.splitext(file)[0]
            output_dir = os.path.join(output_root, dir_name, base_name)
            
            # Skip if already rendered
            if os.path.exists(os.path.join(output_dir, f"{base_name}-8.jpeg")):
                print("Skipping already rendered:", base_name)
                continue
            
            # Clean up objects (except camera, light, world) before importing
            for obj in [o for o in bpy.data.objects if o.type in {'MESH', 'EMPTY', 'CURVE', 'SURFACE', 'META', 'FONT'}]:
                bpy.data.objects.remove(obj, do_unlink=True)
            for mesh in list(bpy.data.meshes):
                bpy.data.meshes.remove(mesh, do_unlink=True)
            
            # Import the GLB
            bpy.ops.import_scene.gltf(filepath=glb_path)
            
            # Get the imported mesh objects
            imported_objs = [o for o in bpy.context.selected_objects if o.type == 'MESH']
            if not imported_objs:
                continue
            
            # If multiple meshes, join them into a single object
            bpy.ops.object.select_all(action='DESELECT')
            for o in imported_objs:
                o.select_set(True)
            if len(imported_objs) > 1:
                bpy.context.view_layer.objects.active = imported_objs[0]
                bpy.ops.object.join()
                main_obj = bpy.context.active_object
            else:
                main_obj = imported_objs[0]
            
            # Move the newly imported object to origin
            main_obj.location = (0, 0, 0)
            
            t = time.time()
            
            print("Rendering:", base_name)
            
            # Render 8 images from different angles
            render_object_at_rotations(main_obj, output_dir, base_name, samples=4096)
            print("Time taken:", time.time() - t)

print("Rendering completed.")