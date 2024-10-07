use crate::computing_cost::complexity_model::ComplexityModel;
use crate::config;
use crate::config::GpuPbsType;
use crate::global_parameters::{Range, DEFAUT_DOMAINS};

#[derive(Clone, Copy, Debug)]
pub struct NoiseBoundConfig {
    pub security_level: u64,
    pub maximum_acceptable_error_probability: f64,
    pub ciphertext_modulus_log: u32,
}

#[derive(Clone, Copy)]
pub struct Config<'a> {
    pub security_level: u64,
    pub maximum_acceptable_error_probability: f64,
    pub key_sharing: bool,
    pub ciphertext_modulus_log: u32,
    pub fft_precision: u32,
    pub complexity_model: &'a dyn ComplexityModel,
}

#[derive(Clone, Copy, Debug)]
pub struct ParameterRestrictions {
    pub log2_polynomial_size_min: u64,
    pub log2_polynomial_size_max: u64,
    pub glwe_dimension_min: u64,
    pub glwe_dimension_max: u64,
}

#[derive(Clone, Debug)]
pub struct SearchSpace {
    pub glwe_log_polynomial_sizes: Vec<u64>,
    pub glwe_dimensions: Vec<u64>,
    pub internal_lwe_dimensions: Vec<u64>,
    pub levelled_only_lwe_dimensions: Range,
}

impl SearchSpace {
    pub fn default_cpu(parameter_restrictions: ParameterRestrictions) -> Self {
        let mut glwe_log_polynomial_sizes_range = DEFAUT_DOMAINS.glwe_pbs_constrained.log2_polynomial_size;
        glwe_log_polynomial_sizes_range.start = std::cmp::max(glwe_log_polynomial_sizes_range.start,
            parameter_restrictions.log2_polynomial_size_min);
        glwe_log_polynomial_sizes_range.end = std::cmp::min(glwe_log_polynomial_sizes_range.end,
            parameter_restrictions.log2_polynomial_size_max);
        let glwe_log_polynomial_sizes: Vec<u64> = glwe_log_polynomial_sizes_range.as_vec();

        let mut glwe_dimensions_range = DEFAUT_DOMAINS.glwe_pbs_constrained.glwe_dimension;
        glwe_dimensions_range.start = std::cmp::max(glwe_dimensions_range.start,
            parameter_restrictions.glwe_dimension_min);
            glwe_dimensions_range.end = std::cmp::min(glwe_dimensions_range.end,
            parameter_restrictions.glwe_dimension_max);
        let glwe_dimensions: Vec<u64> = glwe_dimensions_range.as_vec();

        let internal_lwe_dimensions: Vec<u64> = DEFAUT_DOMAINS.free_glwe.glwe_dimension.as_vec();
        let levelled_only_lwe_dimensions = DEFAUT_DOMAINS.free_lwe;
        Self {
            glwe_log_polynomial_sizes,
            glwe_dimensions,
            internal_lwe_dimensions,
            levelled_only_lwe_dimensions,
        }
    }

    pub fn default_gpu_lowlat() -> Self {
        // See backends/concrete_cuda/implementation/src/bootstrap_low_latency.cu
        let glwe_log_polynomial_sizes: Vec<u64> = (8..=14).collect();

        let glwe_dimensions: Vec<u64> = DEFAUT_DOMAINS.glwe_pbs_constrained.glwe_dimension.as_vec();

        let internal_lwe_dimensions: Vec<u64> = DEFAUT_DOMAINS.free_glwe.glwe_dimension.as_vec();
        let levelled_only_lwe_dimensions = DEFAUT_DOMAINS.free_lwe;
        Self {
            glwe_log_polynomial_sizes,
            glwe_dimensions,
            internal_lwe_dimensions,
            levelled_only_lwe_dimensions,
        }
    }

    pub fn default_gpu_amortized() -> Self {
        // See backends/concrete_cuda/implementation/src/bootstrap_amortized.cu
        let glwe_log_polynomial_sizes: Vec<u64> = (8..=14).collect();

        let glwe_dimensions: Vec<u64> = DEFAUT_DOMAINS.glwe_pbs_constrained.glwe_dimension.as_vec();

        let internal_lwe_dimensions: Vec<u64> = DEFAUT_DOMAINS.free_glwe.glwe_dimension.as_vec();
        let levelled_only_lwe_dimensions = DEFAUT_DOMAINS.free_lwe;
        Self {
            glwe_log_polynomial_sizes,
            glwe_dimensions,
            internal_lwe_dimensions,
            levelled_only_lwe_dimensions,
        }
    }
    pub fn default(processing_unit: config::ProcessingUnit, parameter_restrictions: ParameterRestrictions) -> Self {
        match processing_unit {
            config::ProcessingUnit::Cpu => Self::default_cpu(parameter_restrictions),
            config::ProcessingUnit::Gpu {
                pbs_type: GpuPbsType::Amortized,
                ..
            } => Self::default_gpu_amortized(),
            config::ProcessingUnit::Gpu {
                pbs_type: GpuPbsType::Lowlat,
                ..
            } => Self::default_gpu_lowlat(),
        }
    }
}
