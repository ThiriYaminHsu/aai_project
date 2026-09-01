"""Bloch-sphere visualisation of a single qubit, for the dissertation Introduction
(motivating why a qubit is not just a classical bit, before ZX-diagrams and gates
are introduced in the Literature Review). Not tied to any experimental data -- a
standard textbook-style diagram (cf. Nielsen & Chuang Fig. 1.4), drawn fresh here
so it matches this document's figure style/resolution rather than being copied from
elsewhere.

Usage:
    python scripts/plot_bloch_sphere.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3D


def main() -> None:
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1, 1, 1])

    # -- translucent sphere surface --
    u, v = np.mgrid[0:2 * np.pi:60j, 0:np.pi:30j]
    xs = np.cos(u) * np.sin(v)
    ys = np.sin(u) * np.sin(v)
    zs = np.cos(v)
    ax.plot_surface(xs, ys, zs, color="steelblue", alpha=0.12, linewidth=0, shade=False)

    # -- great circles: equator (xy) and one vertical meridian (xz) --
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(t), np.sin(t), np.zeros_like(t), color="gray", lw=0.9, alpha=0.6)
    ax.plot(np.cos(t), np.zeros_like(t), np.sin(t), color="gray", lw=0.9, alpha=0.6)
    ax.plot(np.zeros_like(t), np.cos(t), np.sin(t), color="gray", lw=0.6, alpha=0.35)

    # -- axes through the sphere --
    axis_len = 1.35
    for vec in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
        v3 = np.array(vec) * axis_len
        ax.add_artist(Line3D(*zip(-v3, v3), color="black", lw=0.8, alpha=0.55))

    # -- basis-state labels --
    label_kw = dict(fontsize=12, ha="center", va="center")
    ax.text(0, 0, 1.22, r"$|0\rangle$", **label_kw)
    ax.text(0, 0, -1.22, r"$|1\rangle$", **label_kw)
    ax.text(1.28, 0, 0, r"$|+\rangle$", **label_kw)
    ax.text(-1.28, 0, 0, r"$|-\rangle$", **label_kw)
    ax.text(0, 1.28, 0, r"$|{+i}\rangle$", fontsize=11, ha="center", va="center", color="dimgray")

    # -- example state |psi> at (theta, phi) = (55 deg, 40 deg) --
    theta = np.deg2rad(55)
    phi = np.deg2rad(40)
    psi = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])

    ax.add_artist(Line3D(*zip([0, 0, 0], psi), color="crimson", lw=2.4))
    ax.scatter(*psi, color="crimson", s=28, depthshade=False)
    ax.text(psi[0] * 1.18, psi[1] * 1.18, psi[2] * 1.18 + 0.03,
            r"$|\psi\rangle$", color="crimson", fontsize=13, ha="center")

    # -- dashed helper lines showing theta (from +z) and phi (in xy-plane) --
    proj = np.array([psi[0], psi[1], 0])
    ax.add_artist(Line3D(*zip(psi, proj), color="crimson", lw=0.9, ls="--", alpha=0.6))
    ax.add_artist(Line3D(*zip([0, 0, 0], proj), color="crimson", lw=0.9, ls="--", alpha=0.6))
    ax.add_artist(Line3D(*zip([0, 0, 0], [0, 0, psi[2]]), color="gray", lw=0.7, ls=":", alpha=0.5))

    arc_t = np.linspace(0, theta, 30)
    arc_r = 0.32
    ax.plot(arc_r * np.sin(arc_t) * np.cos(phi), arc_r * np.sin(arc_t) * np.sin(phi),
            arc_r * np.cos(arc_t), color="dimgray", lw=1.0)
    ax.text(0.16 * np.cos(phi) + 0.02, 0.16 * np.sin(phi), 0.34, r"$\theta$",
            fontsize=11, color="dimgray")

    arc_p = np.linspace(0, phi, 30)
    ax.plot(0.55 * np.cos(arc_p), 0.55 * np.sin(arc_p), np.zeros_like(arc_p),
            color="dimgray", lw=1.0)
    ax.text(0.62 * np.cos(phi / 2), 0.62 * np.sin(phi / 2), -0.06, r"$\varphi$",
            fontsize=11, color="dimgray")

    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_zlim(-1.35, 1.35)
    ax.set_axis_off()
    ax.view_init(elev=18, azim=-55)

    fig.tight_layout()
    fig.savefig("results/figures/bloch_sphere.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    print("Saved results/figures/bloch_sphere.png")


if __name__ == "__main__":
    main()
