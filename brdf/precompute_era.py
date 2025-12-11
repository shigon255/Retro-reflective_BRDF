#!/usr/bin/env python3
import argparse, math
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Geometry & math utilities
# ---------------------------

N0 = np.array([1.0, 1.0, 1.0], dtype=float)  # (unnormalized) incident plane normal
NX = np.array([1.0, 0.0, 0.0], dtype=float)
NY = np.array([0.0, 1.0, 0.0], dtype=float)
NZ = np.array([0.0, 0.0, 1.0], dtype=float)

def build_basis_from_normal(n):
    n = n / np.linalg.norm(n)
    if abs(n[0]) < 0.9:
        a = np.array([1.0, 0.0, 0.0])
    else:
        a = np.array([0.0, 1.0, 0.0])
    t1 = np.cross(n, a); t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1); t2 /= np.linalg.norm(t2)
    return t1, t2, n

T1, T2, NHAT = build_basis_from_normal(N0)

def inward_direction(theta, phi):
    ct, st = math.cos(theta), math.sin(theta)
    cp, sp = math.cos(phi), math.sin(phi)
    d = -(ct * NHAT + st * (cp * T1 + sp * T2))
    return d / np.linalg.norm(d)

def reflect(d, n):
    return d - 2.0 * float(np.dot(d, n)) * n

def inside_incident_triangle(pt, eps=1e-6):
    x, y, z = pt
    if x < -eps or y < -eps or z < -eps:
        return False
    s = x + y + z
    return abs(s - 1.0) <= 1e-4

def sample_points_on_triangle(n):
    # Dirichlet(1,1,1): uniform on simplex x+y+z=1, x,y,z>=0
    a = np.random.exponential(scale=1.0, size=(n, 3))
    s = a.sum(axis=1, keepdims=True)
    return a / s

def simulate_ray_three_reflections(p0, d0, ang_tol_deg=0.5):
    p = p0.copy()
    d = d0.copy()
    hitX = hitY = hitZ = False

    for _ in range(3):
        candidates = []
        # distance to x=0 if heading toward negative x and currently x>0
        if not hitX and d[0] < 0 and p[0] > 0:
            tx = p[0] / (-d[0])
            if tx > 1e-12: candidates.append(('x', tx))
        if not hitY and d[1] < 0 and p[1] > 0:
            ty = p[1] / (-d[1])
            if ty > 1e-12: candidates.append(('y', ty))
        if not hitZ and d[2] < 0 and p[2] > 0:
            tz = p[2] / (-d[2])
            if tz > 1e-12: candidates.append(('z', tz))

        if not candidates:
            return False  # no more trihedral hits possible

        plane, tmin = min(candidates, key=lambda kv: kv[1])
        p = p + tmin * d
        if plane == 'x':
            d = reflect(d, NX); hitX = True
        elif plane == 'y':
            d = reflect(d, NY); hitY = True
        else:
            d = reflect(d, NZ); hitZ = True

    # Direction must match retro (-d0) within tolerance
    cosang = float(np.dot(d, -d0)) / (np.linalg.norm(d)*np.linalg.norm(d0))
    cosang = max(-1.0, min(1.0, cosang))
    ang = math.degrees(math.acos(cosang))
    if ang > ang_tol_deg:
        return False

    # Intersect exit with incident plane x+y+z=1
    denom = float(np.dot(N0, d))
    if denom <= 1e-12:
        return False
    t_exit = (1.0 - float(np.dot(N0, p))) / denom
    if t_exit <= 1e-12:
        return False
    p_exit = p + t_exit * d
    return inside_incident_triangle(p_exit)

def estimate_era(theta, phi, samples=4000, ang_tol_deg=0.5, seed=None):
    if seed is not None:
        np.random.seed(seed)
    d0 = inward_direction(theta, phi)
    pts = sample_points_on_triangle(samples)
    hits = 0
    for i in range(samples):
        if simulate_ray_three_reflections(pts[i], d0, ang_tol_deg):
            hits += 1
    return hits / samples

# ---------------------------
# Main: 1D ERA LUT
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="1D ERA LUT for cube-corner retroreflector (averaged over azimuth)")
    ap.add_argument("--theta-max", type=float, default=90.0,
                    help="Max tilt angle in degrees (from plane normal)")
    ap.add_argument("--n-theta", type=int, default=91,
                    help="Number of theta samples (inclusive of 0 and theta-max)")
    ap.add_argument("--n-phi", type=int, default=360,
                    help="Number of phi samples in [0, 360)")
    ap.add_argument("--samples", type=int, default=4000,
                    help="MC entry-point samples per (theta,phi)")
    ap.add_argument("--ang-tol", type=float, default=0.5,
                    help="Retro angular tolerance in degrees")
    ap.add_argument("--out-bin", type=str, default="era_lut.bin",
                    help="Output binary LUT file (float32)")
    args = ap.parse_args()

    # Theta & phi grids
    thetas = np.linspace(0.0, math.radians(args.theta_max), args.n_theta)
    phis = np.linspace(0.0, 2*math.pi, args.n_phi, endpoint=False)

    # ERA(theta, phi) and azimuthal average
    era_1d = np.zeros(args.n_theta, dtype=np.float64)

    for ti, th in enumerate(thetas):
        sum_era = 0.0
        for pj, ph in enumerate(phis):
            era = estimate_era(th, ph,
                               samples=args.samples,
                               ang_tol_deg=args.ang_tol,
                               seed=ti*100000 + pj)
            sum_era += era
        era_1d[ti] = sum_era / args.n_phi
        print(f"[ERA] theta={math.degrees(th):6.2f} deg -> ERA_mean={era_1d[ti]:.6f}", flush=True)

    # Save as float32 LUT: index i corresponds to theta_deg[i]
    lut = era_1d.astype(np.float32)
    lut.tofile(args.out_bin)

    # Also print a small summary
    theta_degs = np.linspace(0.0, args.theta_max, args.n_theta)
    print(f"\nWrote 1D ERA LUT to: {args.out_bin}")
    print("Index i corresponds to theta_deg = "
          "linear ramp from 0 to theta_max.")
    print("First few entries:")
    for i in range(min(5, len(lut))):
        print(f"  theta={theta_degs[i]:6.2f} deg -> ERA={lut[i]:.6f}")
    
    lut = np.fromfile(args.out_bin, dtype=np.float32)
    angles = np.linspace(0.0, args.theta_max, args.n_theta)
    plt.plot(angles, lut)
    plt.xlabel("Incident Angle (degrees)")
    plt.ylabel("Effective Retroreflective Area E(μₜ)")
    plt.title("ERA Lookup Table")
    plt.grid(True)
    plt.savefig("era_lut_plot.png")
    print("ERA LUT plot saved to era_lut_plot.png")

if __name__ == "__main__":
    main()
