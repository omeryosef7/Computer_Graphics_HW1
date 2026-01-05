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

EPSILON = 1e-6

TRANSPARENCY_EPSILON = 1e-4

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


def save_image(image_array, output_path="scenes/Spheres.png"):
    image = Image.fromarray(np.uint8(image_array))
    image.save(output_path)


def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, help='Path to the scene file')
    parser.add_argument('output_image', nargs='?', type=str, default="scenes/Spheres.png", help='Name of the output image file (optional, default: scenes/Spheres.png)')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()

    camera, scene_settings, objects = parse_scene_file(args.scene_file)

    def normalize(v):
        norm_sq = np.dot(v, v)
        if norm_sq < 1e-20:
            return v
        return v / np.sqrt(norm_sq)

    def reflect(incident, normal):
        return incident - 2 * np.dot(incident, normal) * normal

    def generate_ray(camera, x, y, width, height):
        forward = normalize(camera.look_at - camera.position)
        right = normalize(np.cross(forward, camera.up_vector))
        up = normalize(np.cross(right, forward))

        aspect_ratio = width / height
        screen_height = camera.screen_width / aspect_ratio

        pixel_x = (0.5 - (x + 0.5) / width) * camera.screen_width
        pixel_y = (0.5 - (y + 0.5) / height) * screen_height

        screen_center = camera.position + camera.screen_distance * forward
        world_point = screen_center + pixel_x * right + pixel_y * up

        ray_direction = normalize(world_point - camera.position)

        return camera.position.copy(), ray_direction

    def intersect_sphere(ray_origin, ray_direction, sphere):
        oc = ray_origin - sphere.position
        a = np.dot(ray_direction, ray_direction)
        b = 2.0 * np.dot(oc, ray_direction)
        c = np.dot(oc, oc) - sphere.radius * sphere.radius

        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return False, float('inf'), None, None

        sqrt_discriminant = np.sqrt(discriminant)
        t1 = (-b - sqrt_discriminant) / (2.0 * a)
        t2 = (-b + sqrt_discriminant) / (2.0 * a)

        t = t1 if t1 > EPSILON else t2
        if t > EPSILON:
            point = ray_origin + t * ray_direction
            normal = normalize(point - sphere.position)
            return True, t, point, normal

        return False, float('inf'), None, None

    def intersect_plane(ray_origin, ray_direction, plane):
        normal = plane.normal
        denom = np.dot(normal, ray_direction)

        if abs(denom) < EPSILON:
            return False, float('inf'), None, None

        t = (plane.offset - np.dot(normal, ray_origin)) / denom

        if t > EPSILON:
            point = ray_origin + t * ray_direction
            return True, t, point, normal

        return False, float('inf'), None, None

    def intersect_cube(ray_origin, ray_direction, cube):
        cube_min = cube.position - cube.scale / 2
        cube_max = cube.position + cube.scale / 2

        t_near = -float('inf')
        t_far  =  float('inf')
        n_near = None
        n_far  = None

        for i in range(3):
            d = ray_direction[i]
            o = ray_origin[i]

            if abs(d) < EPSILON:
                if o < cube_min[i] or o > cube_max[i]:
                    return False, float('inf'), None, None
                continue

            inv_d = 1.0 / d
            t1 = (cube_min[i] - o) * inv_d
            t2 = (cube_max[i] - o) * inv_d

            n1 = np.zeros(3); n2 = np.zeros(3)
            n1[i] = -1.0; n2[i] = 1.0

            if t1 > t2:
                t1, t2 = t2, t1
                n1, n2 = n2, n1

            if t1 > t_near:
                t_near = t1
                n_near = n1
            if t2 < t_far:
                t_far = t2
                n_far = n2
            if t_near > t_far:
                return False, float('inf'), None, None

        if t_far <= EPSILON:
            return False, float('inf'), None, None

        if t_near > EPSILON:
            t_hit = t_near
            hit_normal = n_near
        else:
            t_hit = t_far
            hit_normal = n_far

        point = ray_origin + t_hit * ray_direction
        return True, t_hit, point, hit_normal

    def find_closest_intersection(ray_origin, ray_direction, surfaces, excluded_surface=None):
        closest_t = float('inf')
        closest_point = None
        closest_normal = None
        closest_material_index = None
        closest_surface = None

        for surface in surfaces:
            # Skip the surface we are explicitly excluding (the one we just stepped back into)
            if surface is excluded_surface:
                continue

            hit = False
            t = float('inf')
            point = None
            normal = None

            if hasattr(surface, 'radius'):
                hit, t, point, normal = intersect_sphere(ray_origin, ray_direction, surface)
            elif hasattr(surface, 'normal'):
                hit, t, point, normal = intersect_plane(ray_origin, ray_direction, surface)
            elif hasattr(surface, 'scale'):
                hit, t, point, normal = intersect_cube(ray_origin, ray_direction, surface)

            if hit and t < closest_t:
                closest_t = t
                closest_point = point
                closest_normal = normal
                closest_material_index = surface.material_index
                closest_surface = surface

        return closest_t < float('inf'), closest_t, closest_point, closest_normal, closest_material_index, closest_surface

    def is_ray_blocked(ray_origin, ray_direction, max_distance, surfaces):
        hit, t, _, _, _, _ = find_closest_intersection(ray_origin, ray_direction, surfaces)
        return hit and t < max_distance - EPSILON

    def cast_shadow_rays(intersection_point, light, surfaces, scene_settings):
        n = int(scene_settings.root_number_shadow_rays)
        total_rays = n * n
        hit_count = 0
        light_to_surface = normalize(intersection_point - light.position)
        
        if abs(light_to_surface[0]) > 0.1:
            temp = np.array([0, 1, 0], dtype=np.float64)
        else:
            temp = np.array([1, 0, 0], dtype=np.float64)
            
        right = normalize(np.cross(light_to_surface, temp))
        up = normalize(np.cross(light_to_surface, right))

        for i in range(n):
            for j in range(n):
                u = (i + np.random.random()) / n - 0.5
                v = (j + np.random.random()) / n - 0.5
                light_sample = light.position + light.radius * (u * right + v * up)
                shadow_ray_dir = normalize(light_sample - intersection_point)
                light_distance = np.linalg.norm(light_sample - intersection_point)
                
                shadow_ray_origin = intersection_point + EPSILON * shadow_ray_dir
                
                if not is_ray_blocked(shadow_ray_origin, shadow_ray_dir, light_distance, surfaces):
                    hit_count += 1

        return hit_count / total_rays

    def calculate_phong_lighting(intersection_point, normal, material, lights, surfaces, ray_direction, scene_settings):
        total_color = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        view_dir = normalize(-ray_direction)

        for light in lights:
            light_dir = light.position - intersection_point
            light_distance = np.linalg.norm(light_dir)
            light_dir = normalize(light_dir)
            visibility = cast_shadow_rays(intersection_point, light, surfaces, scene_settings)
            shadow_factor = (1 - light.shadow_intensity) + light.shadow_intensity * visibility

            diffuse_intensity = max(0, np.dot(normal, light_dir))
            diffuse = diffuse_intensity * material.diffuse_color * light.color
            reflect_dir = reflect(-light_dir, normal)
            specular_intensity = max(0, np.dot(reflect_dir, view_dir)) ** material.shininess
            specular = specular_intensity * material.specular_color * light.color * light.specular_intensity
            total_color += (diffuse + specular) * shadow_factor

        return total_color

    def trace_ray(ray_origin, ray_direction, surfaces, materials, lights, scene_settings, depth=0, excluded_surface=None):
        if depth >= scene_settings.max_recursions:
            return scene_settings.background_color

        hit, t, point, normal, material_index, hit_surface = find_closest_intersection(ray_origin, ray_direction, surfaces, excluded_surface)

        if not hit:
            return scene_settings.background_color

        material = materials[material_index - 1]
        local_color = calculate_phong_lighting(point, normal, material, lights, surfaces, ray_direction, scene_settings)
        
        reflection_color = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        if material.has_reflection and depth < scene_settings.max_recursions:
            reflection_dir = reflect(ray_direction, normal)
            reflection_origin = point + EPSILON * normal
            
            reflected_color = trace_ray(reflection_origin, reflection_dir, surfaces, materials, lights, scene_settings, depth + 1)
            reflection_color = reflected_color * material.reflection_color

        background_color = scene_settings.background_color
        if material.transparency > 0 and depth < scene_settings.max_recursions:
            # Entering/Exiting Logic (The one that worked best)
            entering = np.dot(ray_direction, normal) < 0
            
            if entering:
                # Entering: Move forward slightly (using stronger epsilon 1e-4 as requested)
                transmission_origin = point + TRANSPARENCY_EPSILON * ray_direction
                transmitted_color = trace_ray(transmission_origin, ray_direction, surfaces, materials, lights, scene_settings, depth + 1, excluded_surface=None)
            else:
                # Exiting: The "Salt and Pepper" Fix
                # Move BACKWARDS (into the object) by 1e-4 and EXCLUDE the object.
                # This ensures we don't hit the object again but we DO hit the floor at dist=0.
                transmission_origin = point - TRANSPARENCY_EPSILON * ray_direction
                transmitted_color = trace_ray(transmission_origin, ray_direction, surfaces, materials, lights, scene_settings, depth + 1, excluded_surface=hit_surface)
            
            background_color = transmitted_color

        final_color = (background_color * material.transparency +
                       local_color * (1 - material.transparency) +
                       reflection_color)

        return final_color

    def render_scene(camera, scene_settings, objects, width, height):
        surfaces = []
        materials = []
        lights = []

        for obj in objects:
            if hasattr(obj, 'material_index'):
                surfaces.append(obj)
            elif isinstance(obj, Material):
                materials.append(obj)
            elif isinstance(obj, Light):
                lights.append(obj)

        camera.position = np.array(camera.position, dtype=np.float64)
        camera.look_at = np.array(camera.look_at, dtype=np.float64)
        camera.up_vector = np.array(camera.up_vector, dtype=np.float64)
        scene_settings.background_color = np.array(scene_settings.background_color, dtype=np.float64)
        
        for light in lights:
            light.position = np.array(light.position, dtype=np.float64)
            light.color = np.array(light.color, dtype=np.float64)
        
        for surface in surfaces:
            if hasattr(surface, 'position'):
                surface.position = np.array(surface.position, dtype=np.float64)
            if hasattr(surface, 'normal'):
                surface.normal = np.array(surface.normal, dtype=np.float64)
        
        for material in materials:
            material.diffuse_color = np.array(material.diffuse_color, dtype=np.float64)
            material.specular_color = np.array(material.specular_color, dtype=np.float64)
            material.reflection_color = np.array(material.reflection_color, dtype=np.float64)
            material.has_reflection = np.any(material.reflection_color > 0)

        image = np.zeros((height, width, 3))

        for y in range(height):
            if y % 10 == 0: print(f"Rendering row {y}/{height}")
            for x in range(width):
                ray_origin, ray_direction = generate_ray(camera, x, y, width, height)
                color = trace_ray(ray_origin, ray_direction, surfaces, materials, lights, scene_settings)
                image[y, x] = np.clip(color, 0, 1)

        return image

    image_array = render_scene(camera, scene_settings, objects, args.width, args.height)
    image_array = (image_array * 255).astype(np.uint8)
    save_image(image_array, args.output_image)

if __name__ == '__main__':
    main()