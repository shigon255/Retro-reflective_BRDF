import math
import os
import numpy as np
import math
import mitsuba as mi
import drjit as dr
from drjit.auto import Float
import imageio
from tqdm import tqdm

# --- Choose a variant (GPU AD) ---
mi.set_variant('cuda_ad_rgb')
# mi.set_variant("scalar_rgb")

import brdf.retro_brdf

def save_folder_to_vid(folder_path):
    vid_name = folder_path + ".mp4"
    images = []
    file_names = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
    for filename in file_names:
        img_path = os.path.join(folder_path, filename)
        images.append(imageio.imread(img_path))
    imageio.mimwrite(vid_name, images, fps=30)
    print(f"Saved video to {vid_name}")

def _rotation_from_z_to(n):
    """Return a ScalarTransform4f that rotates +Z to the given (normalized) direction n."""
    n = np.array(n, dtype=np.float64)
    n = n / np.linalg.norm(n)
    z = np.array([0.0, 0.0, 1.0])
    c = np.clip(np.dot(z, n), -1.0, 1.0)
    if c > 1.0 - 1e-8:
        # already aligned
        return mi.ScalarTransform4f()
    if c < -1.0 + 1e-8:
        # opposite; rotate 180° around any axis orthogonal to z (use X)
        return mi.ScalarTransform4f.rotate([1, 0, 0], 180.0)
    axis = np.cross(z, n)
    axis_norm = np.linalg.norm(axis)
    axis = axis / axis_norm
    angle_deg = math.degrees(math.acos(c))
    return mi.ScalarTransform4f.rotate(axis.tolist(), angle_deg)


def _rect_on_plane_with_normal(normal, H, scale_xy):
    """
    Create a rectangle transform whose *front face* has normal 'normal'
    and lies on plane at distance H/||normal|| from the origin.
    """
    n = np.array(normal, dtype=np.float64)
    n_hat = n / np.linalg.norm(n)
    d = H / np.linalg.norm(n)  # signed distance to plane

    T = mi.ScalarTransform4f.translate(mi.ScalarVector3f((n_hat * d).tolist()))
    R = _rotation_from_z_to(n_hat)  # default rect faces +Z; rotate +Z -> n_hat
    S = mi.ScalarTransform4f.scale([scale_xy, scale_xy, 1.0])
    return T @ R @ S


def forward_to_up(forward, up):
    """Make 'up' orthogonal to 'forward' and normalize both."""
    f = np.array(forward, dtype=np.float64)
    f = f / np.linalg.norm(f)
    u = np.array(up, dtype=np.float64)
    u = u - np.dot(u, f) * f  # remove component along f
    u = u / np.linalg.norm(u)
    return u

def build_base_scene(
    H=0.45,
    H_back=-0.45,
    sheet_size=0.4,
    D_cam=5.0,
    D_light=6.2,
    light_strength=50.0,
    env_strength=0.0,
    diffuse_rgb=(1.0, 0.0, 0.0),
    img_res=640,
    spp=256,
    fov=22.0
):
    """
    Build a Mitsuba-3 scene once (we'll animate it later via params).
    """
    # Diffuse BSDF (for backing plane or debugging)
    diffuse_bsdf = {
        'type': 'diffuse',
        'reflectance': {'type': 'rgb', 'value': diffuse_rgb}
    }

    # Rough dielectric for front incident sheet
    rough_dielectric_bsdf = {
        'type': 'roughdielectric',
        'distribution': 'beckmann',
        'alpha': 0.01,
        'int_ior': 1.56,
        'ext_ior': 1.0,
    }
    
    dielectric_bsdf = {
        'type': 'dielectric',
        'int_ior': 1.56,
        'ext_ior': 1.0,
    }
    retroreflective_brdf = mi.load_dict({
        'type': 'retro_brdf',
        'id': 'my_retro_material',
        'alpha_m': 0.01, 
        'k_s': {
            'type': 'rgb',
            'value': [1.0, 1.0, 1.0]
        },
        'k_d': {
            'type': 'rgb',
            'value': [0.1, 0.1, 0.8]  # Blue substrate
        },
        'k_retro': 1.0
    })

    # Perfect mirror-like conductor for cube-corner interior
    mirror_bsdf = {
        'type': 'conductor',
        'material': 'Al',
    }

    # Incident plane x+y+z = H (front sheet), normal (1,1,1)
    n111 = (1.0, 1.0, 1.0)
    sheet_to_world = _rect_on_plane_with_normal(
        normal=n111,
        H=H,
        scale_xy=sheet_size
    )
    sheet = {
        'type': 'rectangle',
        'to_world': sheet_to_world,
        'bsdf': retroreflective_brdf
        # 'bsdf': dielectric_bsdf
    }


    # Initial camera & light along normal n111 direction
    n = np.array(n111, dtype=np.float64)
    n = n / np.linalg.norm(n)
    cam_origin = (n * D_cam)
    light_origin = (n * D_light)

    forward = np.array([0.0, 0.0, 0.0]) - cam_origin
    up_vec = forward_to_up(forward, (0.0, 0.0, 1.0))
    up = tuple(up_vec.tolist())

    light = {
        'type': 'point',
        'position': tuple(light_origin.tolist()),
        'intensity': {
            'type': 'rgb',
            'value': [light_strength, light_strength, light_strength]
        }
    }

    sensor = {
        'type': 'perspective',
        'to_world': mi.ScalarTransform4f.look_at(
            origin=tuple(cam_origin.tolist()),
            target=(0.0, 0.0, 0.0),
            up=up,
        ),
        'fov': fov,
        'film': {
            'type': 'hdrfilm',
            'width': img_res,
            'height': img_res,
            'pixel_format': 'rgb',
            'rfilter': {'type': 'gaussian'}
        },
        'sampler': {'type': 'independent', 'sample_count': spp}
    }

    scene_dict = {
        'type': 'scene',
        'integrator': {'type': 'path'},
        'sensor': sensor,


        # Front sheet (rough dielectric)
        'incident_sheet': sheet,

        # Light + env
        'area_light': light,
        'env': {
            'type': 'constant',
            'radiance': {'type': 'rgb', 'value': [0, 0, env_strength]}
        }
    }

    scene = mi.load_dict(scene_dict)
    return scene

def compute_basis():
    """Return retro-reflector normal n and two orthonormal tangents u, v."""
    n = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    n = n / np.linalg.norm(n)

    u = np.array([1.0, -1.0, 0.0], dtype=np.float64)
    u = u - np.dot(u, n) * n
    u_norm = np.linalg.norm(u)
    if u_norm < 1e-6:
        u = np.array([1.0, 0.0, -1.0], dtype=np.float64)
        u = u - np.dot(u, n) * n
    u = u / np.linalg.norm(u)

    v = np.cross(n, u)
    v = v / np.linalg.norm(v)
    return n, u, v


def compute_cam_light(mode, t, n, u, v, D_cam, D_light):
    """
    Compute camera and light positions for parameter t in [0,1].
    mode: one of the paths below.
    """
    if mode == "coaxial_small":  # Path 1: small tilt, cam=light
        theta_min = math.radians(-10.0)
        theta_max = math.radians(10.0)
        theta = theta_min + (theta_max - theta_min) * t
        dir_vec = n + math.tan(theta) * u
        dir_vec = dir_vec / np.linalg.norm(dir_vec)
        cam = dir_vec * D_cam
        light = dir_vec * D_light

    elif mode == "cam_offset":   # Path 2: light fixed, camera moves sideways
        offset_max = 1.5
        s = -1.0 + 2.0 * t   # s ∈ [-1, 1]
        cam = n * D_cam + u * (offset_max * s)
        light = n * D_light

    elif mode == "light_offset":  # Path 3: camera fixed, light moves sideways
        offset_max = 1.5
        s = -1.0 + 2.0 * t
        cam = n * D_cam
        light = n * D_light + u * (offset_max * s)

    elif mode == "ring":          # Path 4: ring sweep, cam=light
        phi = 2.0 * math.pi * t
        R = 0.3
        dir_vec = n + R * (math.cos(phi) * u + math.sin(phi) * v)
        dir_vec = dir_vec / np.linalg.norm(dir_vec)
        cam = dir_vec * D_cam
        light = dir_vec * D_light

    elif mode == "coaxial_large": # Path 5: large tilt up to ~45°, cam=light
        theta_min = math.radians(0.0)
        theta_max = math.radians(45.0)
        theta = theta_min + (theta_max - theta_min) * t
        dir_vec = n + math.tan(theta) * u
        dir_vec = dir_vec / np.linalg.norm(dir_vec)
        cam = dir_vec * D_cam
        light = dir_vec * D_light

    else:
        raise ValueError(f"Unknown mode {mode}")

    return cam, light


def render_path(scene, params, path_name, mode, n, u, v,
                num_frames=60,
                D_cam=5.0,
                D_light=6.2,
                out_root="output"):
    """
    Render a sequence for a given path/mode by updating camera & light params only.
    """
    out_dir = os.path.join(out_root, path_name)
    os.makedirs(out_dir, exist_ok=True)

    # Parameter keys (these are typical for Mitsuba 3; adjust if your build differs)
    sensor_key = 'sensor.to_world'
    light_pos_key = 'area_light.position'

    for frame in tqdm(range(num_frames)):
        t = frame / max(1, (num_frames - 1))  # t ∈ [0,1]
        cam, light = compute_cam_light(mode, t, n, u, v, D_cam, D_light)

        cam = np.array(cam, dtype=np.float64)
        light = np.array(light, dtype=np.float64)
        forward = np.array([0.0, 0.0, 0.0]) - cam
        up_vec = forward_to_up(forward, (0.0, 0.0, 1.0))

        # Update camera transform
        params[sensor_key] = mi.ScalarTransform4f.look_at(
            origin=tuple(cam.tolist()),
            target=(0.0, 0.0, 0.0),
            up=tuple(up_vec.tolist()),
        )

        # Update light position
        params[light_pos_key] = mi.ScalarPoint3f(
            light[0], light[1], light[2]
        )

        params.update()

        img = mi.render(scene)
        out_path = os.path.join(out_dir, f"{path_name}_{frame:03d}.png")
        mi.util.write_bitmap(out_path, img)
        # print(f"[{path_name}] Saved frame {frame:03d} -> {out_path}")

    save_folder_to_vid(out_dir)

if __name__ == "__main__":
    H = 0.45
    mesh_dir = "meshes"
    output_root = "output"
    
    # if previous rendered results exist, delete them
    if os.path.exists(output_root):
        import shutil
        shutil.rmtree(output_root)
    
    os.makedirs(output_root, exist_ok=True)

    n, u, v = compute_basis()

    D_cam = 5.0
    D_light = 5.0

    # Build scene ONCE
    scene = build_base_scene(
        H=H,
        sheet_size=3.0,
        D_cam=D_cam,
        D_light=D_light,
        light_strength=1.0,
        env_strength=3.0,
        img_res=640,
        spp=1024,
        fov=60.0
    )

    # render one image for reference
    img = mi.render(scene)
    ref_path = os.path.join(output_root, "reference.png")
    mi.util.write_bitmap(ref_path, img)
    print(f"Saved reference image to {ref_path}")
    
    # exit(0)

    params = mi.traverse(scene)

    # If you want to double-check the parameter keys once:
    # print(params.keys())

    # 1) Co-axial small tilt sweep (cam=light)
    render_path(
        scene=scene,
        params=params,
        path_name="path1_coaxial_small",
        mode="coaxial_small",
        n=n, u=u, v=v,
        num_frames=60,
        # num_frames=10,
        D_cam=D_cam,
        D_light=D_light,
        out_root=output_root
    )

    # 2) Camera offset sweep (light fixed)
    render_path(
        scene=scene,
        params=params,
        path_name="path2_camera_offset",
        mode="cam_offset",
        n=n, u=u, v=v,
        num_frames=60,
        D_cam=D_cam,
        D_light=D_light,
        out_root=output_root
    )

    # 3) Light offset sweep (camera fixed)
    render_path(
        scene=scene,
        params=params,
        path_name="path3_light_offset",
        mode="light_offset",
        n=n, u=u, v=v,
        num_frames=60,
        D_cam=D_cam,
        D_light=D_light,
        out_root=output_root
    )

    # 4) Ring sweep around normal (cam=light)
    render_path(
        scene=scene,
        params=params,
        path_name="path4_ring_sweep",
        mode="ring",
        n=n, u=u, v=v,
        num_frames=90,
        D_cam=D_cam,
        D_light=D_light,
        out_root=output_root
    )

    # 5) Large-angle co-axial sweep (cam=light)
    render_path(
        scene=scene,
        params=params,
        path_name="path5_coaxial_large",
        mode="coaxial_large",
        n=n, u=u, v=v,
        num_frames=60,
        D_cam=D_cam,
        D_light=D_light,
        out_root=output_root
    )

    print("All sequences rendered.")
