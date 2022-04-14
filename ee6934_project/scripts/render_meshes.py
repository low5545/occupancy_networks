import argparse
import os
import pathlib
import tqdm
import pytorch3d.io
import polyscope as ps


def main(args):
    p3d_io = pytorch3d.io.IO()
    ps.init()
    ps.set_ground_plane_mode("none")
    ps.set_autocenter_structures(True)
    ps.set_autoscale_structures(True)

    if not os.path.isdir(args.render_dir):
        pathlib.Path(args.render_dir).mkdir(parents=True)

    for mesh_file in tqdm.tqdm(os.listdir(args.mesh_dir)):
        mesh_path = os.path.join(args.mesh_dir, mesh_file)
        mesh = p3d_io.load_mesh(
            path=mesh_path,
            include_textures=False
        )
        vertices = mesh.verts_list()[0].numpy()
        faces = mesh.faces_list()[0].numpy()

        ps_mesh = ps.register_surface_mesh(
            name="mesh",
            vertices=vertices,
            faces=faces,
            back_face_policy="identical",
            color=(0.6, 0.6, 0.6)
        )
        ps.look_at((0.9, 0.5, 0.9), (0., 0., 0.))

        render_path = os.path.join(
            args.render_dir, mesh_file.split(".")[0] + ".png"
        )
        ps.set_screenshot_extension(".png")
        ps.screenshot(render_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mesh rendering script."
    )
    parser.add_argument(
        "mesh_dir", type=str,
        help="Mesh directory."
    )
    parser.add_argument(
        "render_dir", type=str,
        help="Mesh renders directory."
    )
    args = parser.parse_args()
    main(args)
