"""
Diffuse Fresnel Precomputation Script

Computes the diffuse Fresnel reflectance F_d(eta_a/eta_p) using
Monte Carlo integration with cosine-weighted importance sampling.
Based on the retroreflective BRDF model from Guo et al., 2018.
"""

import numpy as np


# Parameters
ETA_A = 1.0  # Refractive index of air
ETA_P = 1.56  # Refractive index of prismatic sheet
NUM_SAMPLES = 1000000  # Number of MC samples
OUTPUT_FILE = "fd_value.txt"


def fresnel_exact(cos_theta_i, eta):
    """
    Compute the exact Fresnel reflectance for unpolarized light.

    Args:
        cos_theta_i: Cosine of the angle of incidence
        eta: Relative IOR (eta_incident / eta_transmitted)

    Returns:
        Fresnel reflectance value (0 to 1)
    """
    # Ensure cos_theta_i is positive
    cos_theta_i = abs(cos_theta_i)

    # Calculate sin^2(theta_i)
    sin2_theta_i = 1.0 - cos_theta_i * cos_theta_i

    # Calculate sin^2(theta_t) using Snell's law
    sin2_theta_t = sin2_theta_i / (eta * eta)

    # Check for total internal reflection
    if sin2_theta_t >= 1.0:
        return 1.0

    # Calculate cos(theta_t)
    cos_theta_t = np.sqrt(1.0 - sin2_theta_t)

    # Fresnel equations for s and p polarization
    # F_s = ((eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t))^2
    # F_p = ((cos_theta_i - eta * cos_theta_t) / (cos_theta_i + eta * cos_theta_t))^2

    term_s_num = eta * cos_theta_i - cos_theta_t
    term_s_den = eta * cos_theta_i + cos_theta_t
    F_s = (term_s_num / term_s_den) ** 2

    term_p_num = cos_theta_i - eta * cos_theta_t
    term_p_den = cos_theta_i + eta * cos_theta_t
    F_p = (term_p_num / term_p_den) ** 2

    # Average of s and p polarization
    return 0.5 * (F_s + F_p)


def sample_cosine_hemisphere():
    """
    Sample a direction from the cosine-weighted hemisphere.

    Returns:
        cos_theta: Cosine of the elevation angle
    """
    # Generate uniform random numbers
    r1 = np.random.random()

    # Cosine-weighted sampling
    # For p(theta, phi) = cos(theta) / pi
    # We have cos(theta) = sqrt(r1)
    cos_theta = np.sqrt(r1)

    return cos_theta


def compute_diffuse_fresnel(eta_a, eta_p, num_samples):
    """
    Compute the diffuse Fresnel reflectance using Monte Carlo integration.

    Args:
        eta_a: Refractive index of air
        eta_p: Refractive index of prismatic sheet
        num_samples: Number of Monte Carlo samples

    Returns:
        F_d: Diffuse Fresnel reflectance
    """
    # Relative IOR for internal reflection (from prism to air)
    eta = eta_a / eta_p

    F_sum = 0.0

    for i in range(num_samples):
        # Generate cosine-weighted direction
        cos_theta_i = sample_cosine_hemisphere()

        # Calculate Fresnel reflectance
        F = fresnel_exact(cos_theta_i, eta)

        # Accumulate (no division by PDF needed due to importance sampling)
        F_sum += F

        # Progress reporting
        if (i + 1) % 100000 == 0:
            print(f"Progress: {i + 1}/{num_samples} samples processed")

    # Average over all samples
    F_d = F_sum / num_samples

    return F_d


def main():
    """Main function to compute and save the diffuse Fresnel value."""
    print("Starting Diffuse Fresnel precomputation...")
    print(f"eta_a: {ETA_A}")
    print(f"eta_p: {ETA_P}")
    print(f"Number of samples: {NUM_SAMPLES}")

    # Compute F_d
    F_d = compute_diffuse_fresnel(ETA_A, ETA_P, NUM_SAMPLES)

    print(f"\nComputed F_d: {F_d}")

    # Save to file
    with open(OUTPUT_FILE, 'w') as f:
        f.write(f"{F_d}\n")

    print(f"F_d value saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
