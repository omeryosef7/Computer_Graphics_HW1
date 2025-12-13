import argparse
from PIL import Image
import numpy as np

from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere


def parse_scene_file(file_path):
    objects = []
    camera = None
    scene_settings = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            obj_type = parts[0]
            params = [float(p) for p in parts[1:]]
            if obj_type == "cam":
                camera = Camera(params[:3], params[3:6], params[6:9], params[9], params[10])
            elif obj_type == "set":
                scene_settings = SceneSettings(params[:3], params[3], params[4])
            elif obj_type == "mtl":
                material = Material(params[:3], params[3:6], params[6:9], params[9], params[10])
                objects.append(material)
            elif obj_type == "sph":
                sphere = Sphere(params[:3], params[3], int(params[4]))
                objects.append(sphere)
            elif obj_type == "pln":
                plane = InfinitePlane(params[:3], params[3], int(params[4]))
                objects.append(plane)
            elif obj_type == "box":
                cube = Cube(params[:3], params[3], int(params[4]))
                objects.append(cube)
            elif obj_type == "lgt":
                light = Light(params[:3], params[3:6], params[6], params[7], params[8])
                objects.append(light)
            else:
                raise ValueError("Unknown object type: {}".format(obj_type))
    return camera, scene_settings, objects


def save_image(image_array):
    image = Image.fromarray(np.uint8(image_array))

    # Save the image to a file
    image.save("scenes/Spheres.png")


def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, help='Path to the scene file')
    parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, objects = parse_scene_file(args.scene_file)

    # Ray tracer implementation

    def normalize(v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def reflect(incident, normal):
        return incident - 2 * np.dot(incident, normal) * normal

    def generate_ray(camera, x, y, width, height):
        # Setup camera coordinate system
        forward = normalize(np.array(camera.look_at) - np.array(camera.position))
        right = normalize(np.cross(forward, np.array(camera.up_vector)))
        up = normalize(np.cross(right, forward))

        # Calculate screen coordinates
        aspect_ratio = width / height
        screen_height = camera.screen_width / aspect_ratio

        # Map pixel to world coordinates
        pixel_x = (x / width - 0.5) * camera.screen_width
        pixel_y = (0.5 - y / height) * screen_height

        # Get world point on screen
        screen_center = np.array(camera.position) + camera.screen_distance * forward
        world_point = screen_center + pixel_x * right + pixel_y * up

        # Ray direction from camera to world point
        ray_direction = normalize(world_point - np.array(camera.position))

        return np.array(camera.position), ray_direction

    def intersect_sphere(ray_origin, ray_direction, sphere):
        # Ray-sphere intersection using quadratic formula
        oc = ray_origin - np.array(sphere.position)
        a = np.dot(ray_direction, ray_direction)
        b = 2.0 * np.dot(oc, ray_direction)
        c = np.dot(oc, oc) - sphere.radius * sphere.radius

        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return False, float('inf'), None, None

        sqrt_discriminant = np.sqrt(discriminant)
        t1 = (-b - sqrt_discriminant) / (2.0 * a)
        t2 = (-b + sqrt_discriminant) / (2.0 * a)

        # Use closest positive intersection
        t = t1 if t1 > 0.001 else t2
        if t > 0.001:
            point = ray_origin + t * ray_direction
            normal = normalize(point - np.array(sphere.position))
            return True, t, point, normal

        return False, float('inf'), None, None

    def intersect_plane(ray_origin, ray_direction, plane):
        normal = np.array(plane.normal)
        denom = np.dot(normal, ray_direction)

        if abs(denom) < 1e-6:  # Ray parallel to plane
            return False, float('inf'), None, None

        t = (plane.offset - np.dot(normal, ray_origin)) / denom

        if t > 0.001:
            point = ray_origin + t * ray_direction
            return True, t, point, normal

        return False, float('inf'), None, None

    def intersect_cube(ray_origin, ray_direction, cube):
        # AABB intersection using slab method
        cube_min = np.array(cube.position) - cube.scale / 2
        cube_max = np.array(cube.position) + cube.scale / 2

        t_min = -float('inf')
        t_max = float('inf')
        hit_normal = None

        for i in range(3):  # Check each axis
            if abs(ray_direction[i]) < 1e-6:  # Ray parallel to slab
                if ray_origin[i] < cube_min[i] or ray_origin[i] > cube_max[i]:
                    return False, float('inf'), None, None
            else:
                t1 = (cube_min[i] - ray_origin[i]) / ray_direction[i]
                t2 = (cube_max[i] - ray_origin[i]) / ray_direction[i]

                if t1 > t2:
                    t1, t2 = t2, t1

                if t1 > t_min:
                    t_min = t1
                    hit_normal = np.zeros(3)
                    hit_normal[i] = -1 if ray_direction[i] > 0 else 1

                if t2 < t_max:
                    t_max = t2

                if t_min > t_max:
                    return False, float('inf'), None, None

        if t_min > 0.001:
            point = ray_origin + t_min * ray_direction
            return True, t_min, point, hit_normal

        return False, float('inf'), None, None

    def find_closest_intersection(ray_origin, ray_direction, surfaces):
        closest_t = float('inf')
        closest_point = None
        closest_normal = None
        closest_material_index = None

        for surface in surfaces:
            hit = False
            t = float('inf')
            point = None
            normal = None

            if hasattr(surface, 'radius'):  # Sphere
                hit, t, point, normal = intersect_sphere(ray_origin, ray_direction, surface)
            elif hasattr(surface, 'normal'):  # Plane
                hit, t, point, normal = intersect_plane(ray_origin, ray_direction, surface)
            elif hasattr(surface, 'scale'):  # Cube
                hit, t, point, normal = intersect_cube(ray_origin, ray_direction, surface)

            if hit and t < closest_t:
                closest_t = t
                closest_point = point
                closest_normal = normal
                closest_material_index = surface.material_index

        return closest_t < float('inf'), closest_t, closest_point, closest_normal, closest_material_index

    def is_ray_blocked(ray_origin, ray_direction, max_distance, surfaces):
        hit, t, _, _, _ = find_closest_intersection(ray_origin, ray_direction, surfaces)
        return hit and t < max_distance - 0.001

    def cast_shadow_rays(intersection_point, light, surfaces, scene_settings):
        n = int(scene_settings.root_number_shadow_rays)
        total_rays = n * n
        hit_count = 0

        # Calculate perpendicular plane basis for light-to-surface ray
        light_to_surface = normalize(intersection_point - np.array(light.position))

        # Find perpendicular vectors to light_to_surface
        if abs(light_to_surface[0]) > 0.1:
            temp = np.array([0, 1, 0])
        else:
            temp = np.array([1, 0, 0])

        right = normalize(np.cross(light_to_surface, temp))
        up = normalize(np.cross(light_to_surface, right))

        for i in range(n):
            for j in range(n):
                # Random point in grid cell
                u = (i + np.random.random()) / n - 0.5
                v = (j + np.random.random()) / n - 0.5

                # Sample light area using perpendicular plane basis
                light_sample = np.array(light.position) + light.radius * (u * right + v * up)
                shadow_ray_dir = normalize(light_sample - intersection_point)
                light_distance = np.linalg.norm(light_sample - intersection_point)

                # Move origin slightly towards the light to avoid self-shadowing
                shadow_ray_origin = intersection_point + 0.001 * shadow_ray_dir

                if not is_ray_blocked(shadow_ray_origin, shadow_ray_dir, light_distance, surfaces):
                    hit_count += 1

        visibility = hit_count / total_rays
        return visibility

    def calculate_phong_lighting(intersection_point, normal, material, lights, surfaces, camera_pos, scene_settings):
        total_color = np.array([0.0, 0.0, 0.0])

        for light in lights:
            # Light direction and distance
            light_dir = np.array(light.position) - intersection_point
            light_distance = np.linalg.norm(light_dir)
            light_dir = normalize(light_dir)

            # View direction
            view_dir = normalize(np.array(camera_pos) - intersection_point)

            # Calculate soft shadows
            visibility = cast_shadow_rays(intersection_point, light, surfaces, scene_settings)

            # Shadow factor from assignment
            shadow_factor = (1 - light.shadow_intensity) + light.shadow_intensity * visibility

            # Diffuse component
            diffuse_intensity = max(0, np.dot(normal, light_dir))
            diffuse = diffuse_intensity * np.array(material.diffuse_color) * np.array(light.color)

            # Specular component
            reflect_dir = reflect(-light_dir, normal)
            specular_intensity = max(0, np.dot(reflect_dir, view_dir)) ** material.shininess
            specular = specular_intensity * np.array(material.specular_color) * np.array(light.color) * light.specular_intensity

            # Apply shadow and add to total
            total_color += (diffuse + specular) * shadow_factor

        return total_color

    def trace_ray(ray_origin, ray_direction, surfaces, materials, lights, scene_settings, depth=0):
        # Check recursion depth
        if depth >= scene_settings.max_recursions:
            return np.array(scene_settings.background_color)

        # Find closest intersection
        hit, t, point, normal, material_index = find_closest_intersection(ray_origin, ray_direction, surfaces)

        if not hit:
            return np.array(scene_settings.background_color)

        material = materials[material_index - 1]  # 1-indexed

        # Calculate local Phong lighting
        local_color = calculate_phong_lighting(point, normal, material, lights, surfaces, ray_origin, scene_settings)

        # Calculate reflection
        reflection_color = np.array([0.0, 0.0, 0.0])
        if any(np.array(material.reflection_color) > 0) and depth < scene_settings.max_recursions:
            reflection_dir = reflect(ray_direction, normal)
            reflection_origin = point + 0.001 * normal

            reflected_color = trace_ray(reflection_origin, reflection_dir, surfaces, materials, lights, scene_settings, depth + 1)
            reflection_color = reflected_color * np.array(material.reflection_color)

        # Calculate transparency
        background_color = np.array(scene_settings.background_color)
        if material.transparency > 0 and depth < scene_settings.max_recursions:
            # Continue ray through the object (no refraction)
            transmission_origin = point - 0.001 * normal
            transmitted_color = trace_ray(transmission_origin, ray_direction, surfaces, materials, lights, scene_settings, depth + 1)
            background_color = transmitted_color

        # Apply color blending formula from assignment
        final_color = (background_color * material.transparency +
                      local_color * (1 - material.transparency) +
                      reflection_color)

        return final_color

    def render_scene(camera, scene_settings, objects, width, height):
        # Separate objects by type
        surfaces = []
        materials = []
        lights = []

        for obj in objects:
            if hasattr(obj, 'material_index'):  # Surface objects
                surfaces.append(obj)
            elif isinstance(obj, Material):
                materials.append(obj)
            elif isinstance(obj, Light):
                lights.append(obj)

        # Create image array
        image = np.zeros((height, width, 3))

        # Render each pixel
        for y in range(height):
            for x in range(width):
                ray_origin, ray_direction = generate_ray(camera, x, y, width, height)
                color = trace_ray(ray_origin, ray_direction, surfaces, materials, lights, scene_settings)
                image[y, x] = np.clip(color, 0, 1)

        return image

    # Render the scene
    image_array = render_scene(camera, scene_settings, objects, args.width, args.height)

    # Convert from [0,1] to [0,255] for PIL
    image_array = (image_array * 255).astype(np.uint8)

    # Save the output image
    image = Image.fromarray(np.uint8(image_array))
    image.save(args.output_image)


if __name__ == '__main__':
    main()
