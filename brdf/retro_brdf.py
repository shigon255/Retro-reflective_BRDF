"""
Retroreflective BRDF Mitsuba 3 Plugin

Implements the complete retroreflective BRDF model from Guo et al., 2018.
Combines surface reflection, retroreflection, and diffuse components.
"""

import mitsuba as mi
import drjit as dr
import numpy as np
from pathlib import Path 

class RetroBRDF(mi.BSDF):
    """
    Custom BSDF implementing the retroreflective BRDF model.

    Combines three components:
    1. Surface reflection (rho_sr)
    2. Retroreflection (rho_rr)
    3. Diffuse reflection (rho_d)
    """

    def __init__(self, props):
        """Initialize the BSDF with parameters and precomputed data."""
        mi.BSDF.__init__(self, props)

        # Handle k_s as either a texture or a spectrum
        k_s_prop = props.get('k_s')
        if k_s_prop is not None:
            # Check if it's a texture
            if hasattr(k_s_prop, 'eval') and hasattr(k_s_prop, 'eval_1'):
                # It's a texture object
                self.m_k_s_texture = k_s_prop
                self.m_k_s = None
            else:
                # It's a spectrum value
                self.m_k_s = mi.Spectrum(k_s_prop)
                self.m_k_s_texture = None
        else:
            # Default value
            self.m_k_s = mi.Spectrum(mi.Float(1.0))
            self.m_k_s_texture = None

        # Handle k_d as either a texture or a spectrum
        k_d_prop = props.get('k_d')
        if k_d_prop is not None:
            # Check if it's a texture
            if hasattr(k_d_prop, 'eval') and hasattr(k_d_prop, 'eval_1'):
                # It's a texture object
                self.m_k_d_texture = k_d_prop
                self.m_k_d = None
            else:
                # It's a spectrum value
                self.m_k_d = mi.Spectrum(k_d_prop)
                self.m_k_d_texture = None
        else:
            # Default value
            self.m_k_d = mi.Spectrum(mi.Float(0.0))
            self.m_k_d_texture = None

        # Handle alpha_m as either a texture or a scalar
        alpha_m_prop = props.get('alpha_m')
        if alpha_m_prop is not None:
            # Check if it's a texture
            if hasattr(alpha_m_prop, 'eval') and hasattr(alpha_m_prop, 'eval_1'):
                # It's a texture object
                self.m_alpha_m_texture = alpha_m_prop
                self.m_alpha_m = None
            else:
                # It's a scalar value
                self.m_alpha_m = mi.Float(alpha_m_prop)
                self.m_alpha_m_texture = None
        else:
            # Default value
            self.m_alpha_m = mi.Float(0.1)
            self.m_alpha_m_texture = None

        # Handle k_retro as either a texture or a scalar
        k_retro_prop = props.get('k_retro')
        if k_retro_prop is not None:
            # Check if it's a texture
            if hasattr(k_retro_prop, 'eval') and hasattr(k_retro_prop, 'eval_1'):
                # It's a texture object
                self.m_k_retro_texture = k_retro_prop
                self.m_k_retro = None
            else:
                # It's a scalar value
                self.m_k_retro = mi.Float(k_retro_prop)
                self.m_k_retro_texture = None
        else:
            # Default value (1.0 preserves original behavior)
            self.m_k_retro = mi.Float(1.0)
            self.m_k_retro_texture = None

        # Hardcoded IOR parameters
        self.m_eta_a = mi.Float(1.0)  # Air
        self.m_eta_p = mi.Float(1.56)  # Prismatic sheet

        # Load precomputed diffuse Fresnel value
        plugin_dir = Path(__file__).parent.resolve()
        fd_path = plugin_dir / 'fd_value.txt'
        era_path = plugin_dir / 'era_lut.bin'
        with open(str(fd_path), 'r') as f:
            self.m_fd_value = mi.Float(float(f.read().strip()))
        
        # Load precomputed ERA lookup table
        era_lut_np = np.fromfile(str(era_path), dtype=np.float32)
        self.m_era_lut_size = len(era_lut_np) # should be 91
        self.m_era_lut_data = mi.Float(era_lut_np)


        # Set BSDF flags
        self.m_components = [mi.BSDFFlags.GlossyReflection]
        self.m_flags = self.m_components[0]

    def eval(self, ctx, si, wo, active):
        """
        Evaluate the BRDF for given incident and outgoing directions.

        Args:
            ctx: BSDF context
            si: Surface interaction
            wo: Outgoing direction
            active: Mask for active lanes

        Returns:
            BRDF value as a spectrum
        """
        # Get directions
        wi = si.wi
        cos_theta_i = mi.Frame3f.cos_theta(wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)

        # Only evaluate for valid reflection directions
        active = active & (cos_theta_i > 0) & (cos_theta_o > 0)

        # Evaluate k_d (texture or constant)
        if self.m_k_d_texture is not None:
            k_d = self.m_k_d_texture.eval(si, active)
        else:
            k_d = self.m_k_d

        # Evaluate k_s (texture or constant)
        if self.m_k_s_texture is not None:
            k_s_scalar = self.m_k_s_texture.eval(si, active)  # Grayscale texture -> scalar
            k_s = mi.Spectrum(k_s_scalar)  # Expand scalar to RGB spectrum
        else:
            k_s = self.m_k_s

        # Evaluate alpha_m (texture or constant)
        if self.m_alpha_m_texture is not None:
            alpha_m = self.m_alpha_m_texture.eval(si, active)
        else:
            alpha_m = self.m_alpha_m

        # Evaluate k_retro (texture or constant)
        if self.m_k_retro_texture is not None:
            k_retro = self.m_k_retro_texture.eval(si, active)
        else:
            k_retro = self.m_k_retro

        # Compute the three BRDF components
        rho_sr = self._eval_surface_reflection(si, wi, wo, alpha_m, active)
        rho_rr = self._eval_retro_reflection(si, wi, wo, alpha_m, active)
        rho_d = self._eval_diffuse(si, wi, wo, k_d, active)

        # Combine components with new k_retro parameter
        fr = k_s * rho_sr + k_d * k_retro * mi.Spectrum(rho_rr) + rho_d
        # fr = k_s * rho_sr  + rho_d
        # fr = k_retro * mi.Spectrum(rho_rr)
        
        # NOTE: Multiply by cos_theta_o for correct lambertian scaling
        cos_theta = mi.Frame3f.cos_theta(wo)
        result = fr * dr.maximum(cos_theta, 0.0)
        
        return dr.select(active, result, mi.Spectrum(0.0))
    
    def sample(self, ctx, si, sample1, sample2, active):
        # 1. Warp sample to cosine-weighted hemisphere
        wo = mi.warp.square_to_cosine_hemisphere(sample2)
        
        # 2. Compute PDF
        pdf = mi.warp.square_to_cosine_hemisphere_pdf(wo)
        
        # 3. Evaluate your custom BRDF physics
        # Mask rays below horizon to avoid NaNs in your eval logic
        # Note: We use a small epsilon to avoid grazing angles
        valid_ray = active & (mi.Frame3f.cos_theta(wo) > 1e-4)
        value = self.eval(ctx, si, wo, valid_ray)
        
        # 4. Construct sample record with EXPLICIT Dr.Jit types
        bs = mi.BSDFSample3f()
        bs.wo = wo
        bs.pdf = pdf
        bs.sampled_type = mi.UInt32(+mi.BSDFFlags.GlossyReflection)
        
        # CRITICAL FIX: Explicit cast to mi.Float for AD graph tracking
        bs.eta = mi.Float(1.0) 
        bs.sampled_component = mi.UInt32(0)
        
        # 5. Return throughput (value / pdf)
        # Handle zero PDF to avoid division by zero (though unlikely with cosine sample)
        weight = value / dr.maximum(pdf, 1e-8)
        
        return (bs, dr.select(valid_ray, weight, mi.Spectrum(0.0)))
    
    def pdf(self, ctx, si, wo, active):
        """FIX 1: Corresponding PDF for the sampler above"""
        return mi.warp.square_to_cosine_hemisphere_pdf(wo)

    # =========================================================================
    # Helper Functions: BRDF Components
    # =========================================================================

    def _eval_surface_reflection(self, si, wi, wo, alpha_m, active):
        """
        Evaluate the surface reflection component rho_sr.

        Formula: rho_sr = (F * G * D) / (4 * |wi·n| * |wo·n|)

        Args:
            si: Surface interaction (for texture evaluation if needed)
            wi: Incident direction
            wo: Outgoing direction
            alpha_m: Surface roughness (can be texture-evaluated value)
            active: Active mask
        """
        n = mi.Vector3f(0, 0, 1)  # Surface normal in local frame

        # Halfway vector
        wm = dr.normalize(wi + wo)

        # Beckmann NDF
        D = self._beckmann_ndf(wm, n, alpha_m, active)

        # Fresnel (using Schlick approximation)
        F = self._fresnel_schlick(dr.dot(wi, wm), self.m_eta_p / self.m_eta_a)

        # Smith shadowing-masking
        G = self._smith_beckmann_rational(wi, wo, wm, n, alpha_m, active)

        # Denominator
        cos_theta_i = mi.Frame3f.cos_theta(wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)
        denom = 4.0 * dr.abs(cos_theta_i) * dr.abs(cos_theta_o)

        result = (D * F * G) / dr.maximum(denom, 1e-8)

        return dr.select(active, result, 0.0)

    def _compute_transmission_terms(self, wi, wo, active):
        """
        Computes the common terms for light transmitting into/out of the prism.
        This is shared by the retroreflection and diffuse components.

        Returns:
            mu_t: The normalized mean refracted direction
            E: The Effective Retroreflective Area (ERA) value
            F_hat: The Fresnel transmission probability (Eq. 13)
            valid_refract: Mask for valid refraction (no TIR)
        """
        # --- 1. Compute mu_t (Refracted Ray) ---
        cos_theta_i = mi.Frame3f.cos_theta(wi)
        eta_rel_in = self.m_eta_p / self.m_eta_a
        r_i, cos_theta_t, eta_it, eta_ti = mi.fresnel(cos_theta_i, eta_rel_in)
        
        valid_refract = r_i < 1.0
        
        mu_t_unnormalized = mi.refract(wi, cos_theta_t, eta_ti) 
        mu_t = dr.normalize(mu_t_unnormalized)
        
        # --- 2. Compute E (ERA Term) ---
        # NOTE: query using theta_i instead of theta_t
        # cos_theta_t_abs = dr.abs(mu_t.z) 
        # cos_theta_t_clamped = dr.clamp(cos_theta_t_abs, 0.0, 1.0)
        # theta_t_rad = dr.acos(cos_theta_t_clamped)
        # theta_t_deg = theta_t_rad * (180.0 / dr.pi)
        cos_theta_i_clamped = dr.clamp(cos_theta_i, 0.0, 1.0)
        theta_i_rad = dr.acos(cos_theta_i_clamped)
        theta_i_deg = theta_i_rad * (180.0 / dr.pi)

        # lut_max_index = mi.Float(self.m_era_lut_size - 1)
        # theta_clamped = dr.clamp(theta_t_deg, 0.0, lut_max_index - 1e-6)
        
        # idx_f = dr.floor(theta_clamped)
        # frac = theta_clamped - idx_f
        # idx0 = mi.UInt32(idx_f)
        # idx1 = idx0 + 1
        
        # v0 = dr.gather(mi.Float, self.m_era_lut_data, idx0, active & valid_refract)
        # v1 = dr.gather(mi.Float, self.m_era_lut_data, idx1, active & valid_refract)
        # E = dr.lerp(v0, v1, frac)
        
        lut_max_index = mi.Float(self.m_era_lut_size - 1)
        # Lookup using theta_i_deg
        theta_clamped = dr.clamp(theta_i_deg, 0.0, lut_max_index - 1e-6)
        
        idx_f = dr.floor(theta_clamped)
        frac = theta_clamped - idx_f
        idx0 = mi.UInt32(idx_f)
        idx1 = idx0 + 1
        
        v0 = dr.gather(mi.Float, self.m_era_lut_data, idx0, active)
        v1 = dr.gather(mi.Float, self.m_era_lut_data, idx1, active)
        E = dr.lerp(v0, v1, frac)
        
        # --- 3. Compute F_hat (Fresnel Term) ---
        trans_in = 1.0 - r_i
        cos_theta_o = mi.Frame3f.cos_theta(wo)
        r_o, _, _, _ = mi.fresnel(cos_theta_o, eta_it)
        trans_out = 1.0 - r_o
        F_hat = trans_in * trans_out
        
        return mu_t, E, F_hat, valid_refract

    def _eval_retro_reflection(self, si, wi, wo, alpha_m, active):
        """
        Evaluate the retroreflection component rho_rr.
        Formula: rho_rr = (E * F_hat * G_hat * D_o) / |wo·n|

        Args:
            si: Surface interaction (for texture evaluation if needed)
            wi: Incident direction
            wo: Outgoing direction
            alpha_m: Surface roughness (can be texture-evaluated value)
            active: Active mask
        """

        # --- 1. Compute Common Transmission Terms ---
        mu_t, E, F_hat, valid_refract = self._compute_transmission_terms(wi, wo, active)

        # Update active mask based on TIR
        active = active & valid_refract

        # --- 2. Compute G_hat (Retro-specific Shadowing) ---
        n = mi.Vector3f(0, 0, 1) # Normal

        # First term: G(wi, mu_t)
        eta_i_1 = self.m_eta_a
        eta_t_1 = self.m_eta_p
        wm_1 = -dr.normalize(eta_i_1 * wi + eta_t_1 * mu_t) # Your fixed normal
        G1 = self._smith_beckmann_rational(wi, mu_t, wm_1, n, alpha_m, active)

        # Second term: G(mu_t, wo)
        eta_i_2 = self.m_eta_p
        eta_t_2 = self.m_eta_a
        wm_2 = -dr.normalize(eta_i_2 * mu_t + eta_t_2 * wo) # Your fixed normal
        G2 = self._smith_beckmann_rational(mu_t, wo, wm_2, n, alpha_m, active)
        
        G_hat = G1 * G2
        
        # --- 3. Compute D_o (Retro-specific NDF) ---
        # mu_o = -wi
        mu_o = wi # coordinate issue

        # 3a. Get alpha_o (Warped Roughness)
        eta_slab = self.m_eta_p / self.m_eta_a
        alpha_m_2 = alpha_m * alpha_m

        # Get SIGNED cosines (z-components)
        cos_i = mi.Frame3f.cos_theta(wi)
        cos_t = mi.Frame3f.cos_theta(mu_t)
        cos_o = mi.Frame3f.cos_theta(mu_o)

        # --- J1 (Eq. 2) ---
        denom_1 = (eta_slab * cos_t) - cos_i
        denom_1_sq = dr.maximum(denom_1 * denom_1, 1e-8)
        J1 = dr.abs(cos_t) / denom_1_sq

        # --- J2 (Eq. 3) ---
        denom_2 = (-eta_slab * cos_t) + cos_o
        denom_2_sq = dr.maximum(denom_2 * denom_2, 1e-8)
        J2 = dr.abs(cos_o) / denom_2_sq

        active_J = active & (J1 > 1e-6) & (J2 > 1e-6)

        alpha_o_2 = (alpha_m_2 / dr.maximum(J1, 1e-8) +
                     alpha_m_2 / dr.maximum(J2, 1e-8))

        alpha_o = dr.sqrt(dr.maximum(alpha_o_2, 0.0))
        alpha_o = dr.select(active_J, alpha_o, 0.0)
            
        # 3b. Calculate the final NDF
        D_o = self._beckmann_ndf(wo, mu_o, alpha_o, active)
        
        # --- 4. Final Assembly (Eq. 12) ---
        cos_theta_o = mi.Frame3f.cos_theta(wo)
        denom = dr.maximum(dr.abs(cos_theta_o), 1e-8) # |wo·n|
        
        result = (E * F_hat * G_hat * D_o) / denom
        
        return dr.select(active, result, 0.0)

    def _eval_diffuse(self, si, wi, wo, k_d, active):
        """
        Evaluate the diffuse reflection component rho_d.
        Refactored to use _compute_transmission_terms.

        [cite_start]Formula: rho_d = rho_d_0 / (1 - k_d * F_d) [cite: 278, 290]
        [cite_start]where rho_d_0 = F_hat * (1 - E) * (eta_a/eta_p)^2 * k_d / pi [cite: 263]
        """

        # # --- 0. Evaluate k_d (texture or constant) ---
        # if self.m_k_d_texture is not None:
        #     k_d = self.m_k_d_texture.eval(si, active)
        # else:
        #     k_d = self.m_k_d

        # --- 1. Compute Common Transmission Terms ---
        # We get E, F_hat, and the validity mask from the new helper.
        # We don't need mu_t for this calculation, so we can use '_'
        _, E, F_hat, valid_refract = self._compute_transmission_terms(wi, wo, active)

        # Update active mask based on TIR
        active = active & valid_refract

        # --- 2. Compute rho_d_0 (Initial Diffuse Component) ---
        # This is Equation (15)

        # [cite_start]Spread term (eta_a / eta_p)^2 [cite: 265]
        spread_term = (self.m_eta_a / self.m_eta_p) * (self.m_eta_a / self.m_eta_p)

        # (1 - E) [cite_start]is the light that is *not* retroreflected [cite: 289]
        non_retro_E = mi.Float(1.0) - E

        # [cite_start]k_d / pi (Lambertian term from substrate [cite: 288])
        diffuse_term = k_d * dr.inv_pi

        # [cite_start]Assemble rho_d_0 [cite: 263]
        rho_d_0 = F_hat * non_retro_E * spread_term * diffuse_term

        # --- 3. Multiple Scattering Correction (Eq. 18) ---
        # Denominator: (1 - k_d * F_d)
        # [cite_start]self.m_fd_value is the precomputed F_d (diffuse Fresnel reflectance) [cite: 270, 273]
        denom_spec = mi.Spectrum(1.0) - k_d * self.m_fd_value
        denom = dr.maximum(denom_spec, mi.Spectrum(1e-8))

        # --- 4. Final Assembly ---
        result = rho_d_0 / denom

        return dr.select(active, result, mi.Spectrum(0.0))

    # =========================================================================
    # Helper Functions: Core Formulas
    # =========================================================================

    def _beckmann_ndf(self, m, n, alpha, active):
        """
        Beckmann Normal Distribution Function.

        MODIFIED: Removed (cos_theta_m > 0) check to support
        mean vectors (n) in the lower hemisphere, like mu_o.
        """
        cos_theta_m = dr.dot(m, n)
        
        # We only care about the angle, so use the absolute value
        # for the squared/fourth-power terms.
        cos_theta_m_abs = dr.abs(cos_theta_m)
        cos_theta_m_2 = cos_theta_m_abs * cos_theta_m_abs
        cos_theta_m_4 = cos_theta_m_2 * cos_theta_m_2
        
        # Clamp alpha to prevent division by zero
        alpha_2 = dr.maximum(alpha * alpha, 1e-8)

        # tan^2(theta) = (1 - cos^2(theta)) / cos^2(theta)
        tan_2_theta = (1.0 - cos_theta_m_2) / dr.maximum(cos_theta_m_2, 1e-8)

        num = dr.exp(-tan_2_theta / alpha_2)
        denom = dr.pi * alpha_2 * cos_theta_m_4

        result = num / dr.maximum(denom, 1e-8)
        
        # The NDF is only valid if alpha is non-zero
        active = active & (alpha > 0) 
        
        return dr.select(active, result, 0.0)

    def _fresnel_schlick(self, cos_theta, eta_rel):
        """
        Schlick's approximation for Fresnel reflectance.

        Formula: F = F0 + (1 - F0) * (1 - cos_theta)^5
        where F0 = ((eta_rel - 1) / (eta_rel + 1))^2
        """
        F0 = dr.power((eta_rel - 1.0) / (eta_rel + 1.0), 2.0)
        return F0 + (1.0 - F0) * dr.power(dr.maximum(1.0 - cos_theta, 0.0), 5.0)

    def _smith_beckmann_rational_g1(self, v, m, n, alpha_m, active):
        """
        Single-direction Smith shadowing-masking term (rational approximation).
        
        FIX: 
        1. Removed the 'active &= ...' check which killed rays with negative cosines.
        2. Uses dr.abs(cos_theta) so it works for both reflection (up) 
           and refraction (down) vectors used in the retro derivation.
        """
        # Calculate cosine with geometric normal
        cos_theta_v = dr.dot(v, n)
        
        # USE ABSOLUTE COSINE:
        # This allows the function to return valid probabilities even if the 
        # ray is technically pointing 'into' the surface (like mu_t).
        cos_theta_v_abs = dr.abs(cos_theta_v)
        cos_theta_v_2 = dr.maximum(cos_theta_v_abs * cos_theta_v_abs, 1e-8)
        
        # Calculate tan(theta) from cos(theta)
        # tan^2 = (1 - cos^2) / cos^2
        tan_2_theta = (1.0 - cos_theta_v_2) / cos_theta_v_2
        tan_theta = dr.sqrt(dr.maximum(tan_2_theta, 0.0))

        # Calculate 'a' parameter for Beckmann
        a = 1.0 / dr.maximum(alpha_m * tan_theta, 1e-8)
        a_sq = a * a

        # Rational approximation for the Smith function
        rational_term = (3.535 * a + 2.181 * a_sq) / (1.0 + 2.276 * a + 2.577 * a_sq)

        # The G1 term is 1.0 when 'a' is large (grazing angle is small)
        val = dr.select(a >= 1.6, 1.0, rational_term)
        
        # Only return 0 if the original active mask was false
        return dr.select(active, val, 0.0)
    
    def _smith_beckmann_rational(self, v1, v2, m, n, alpha_m, active):
        """
        Combined Smith shadowing-masking term.

        Formula: G = G1(v1) * G1(v2)
        """
        g1_v1 = self._smith_beckmann_rational_g1(v1, m, n, alpha_m, active)
        g1_v2 = self._smith_beckmann_rational_g1(v2, m, n, alpha_m, active)
        return g1_v1 * g1_v2

    def traverse(self, callback):
        """
        Expose BRDF parameters for traversal and optimization.

        This method is called by mi.traverse() to expose the BRDF's parameters
        so they can be accessed, modified, and optimized.
        """
        # Expose texture parameters by traversing the texture objects
        # This exposes their internal data arrays which can be optimized
        if self.m_k_d_texture is not None:
            callback.put_object('k_d', self.m_k_d_texture, mi.ParamFlags.Differentiable)

        if self.m_k_s_texture is not None:
            callback.put_object('k_s', self.m_k_s_texture, mi.ParamFlags.Differentiable)

        if self.m_alpha_m_texture is not None:
            callback.put_object('alpha_m', self.m_alpha_m_texture, mi.ParamFlags.Differentiable)

        if self.m_k_retro_texture is not None:
            callback.put_object('k_retro', self.m_k_retro_texture, mi.ParamFlags.Differentiable)

    def parameters_changed(self, keys=None):
        """
        Called when parameters are updated.

        This method is invoked after parameter updates to allow the BRDF
        to perform any necessary recomputations or cache invalidation.
        """
        # Call parent implementation to mark parameters as changed
        mi.BSDF.parameters_changed(self, keys)


    def to_string(self):
        """Return a string representation of the BSDF."""
        k_s_str = "texture" if self.m_k_s_texture is not None else str(self.m_k_s)
        k_d_str = "texture" if self.m_k_d_texture is not None else str(self.m_k_d)
        alpha_m_str = "texture" if self.m_alpha_m_texture is not None else str(self.m_alpha_m)
        k_retro_str = "texture" if self.m_k_retro_texture is not None else str(self.m_k_retro)

        return (f"RetroBRDF[\n"
                f"  k_s = {k_s_str},\n"
                f"  k_d = {k_d_str},\n"
                f"  alpha_m = {alpha_m_str},\n"
                f"  k_retro = {k_retro_str},\n"
                f"  eta_p = {self.m_eta_p}\n"
                f"]")
        
mi.register_bsdf("retro_brdf", lambda props: RetroBRDF(props))
