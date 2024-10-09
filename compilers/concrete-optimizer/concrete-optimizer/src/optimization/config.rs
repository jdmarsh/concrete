use crate::computing_cost::complexity_model::ComplexityModel;
use crate::config;
use crate::config::GpuPbsType;
use crate::global_parameters::{Range, ParameterDomains};

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
    pub fn default_cpu(parameter_domains: ParameterDomains) -> Self {
        let glwe_log_polynomial_sizes: Vec<u64> = parameter_domains.glwe_pbs_constrained_cpu.log2_polynomial_size.as_vec();
        let glwe_dimensions: Vec<u64> = parameter_domains.glwe_pbs_constrained_cpu.glwe_dimension.as_vec();
        let internal_lwe_dimensions: Vec<u64> = parameter_domains.free_glwe.glwe_dimension.as_vec();
        let levelled_only_lwe_dimensions = parameter_domains.free_lwe;
        Self {
            glwe_log_polynomial_sizes,
            glwe_dimensions,
            internal_lwe_dimensions,
            levelled_only_lwe_dimensions,
        }
    }

    pub fn default_gpu_lowlat(parameter_domains: ParameterDomains) -> Self {
        // See backends/concrete_cuda/implementation/src/bootstrap_low_latency.cu
        let glwe_log_polynomial_sizes: Vec<u64> = parameter_domains.glwe_pbs_constrained_gpu.log2_polynomial_size.as_vec();

        let glwe_dimensions: Vec<u64> = parameter_domains.glwe_pbs_constrained_gpu.glwe_dimension.as_vec();

        let internal_lwe_dimensions: Vec<u64> = parameter_domains.free_glwe.glwe_dimension.as_vec();
        let levelled_only_lwe_dimensions = parameter_domains.free_lwe;
        Self {
            glwe_log_polynomial_sizes,
            glwe_dimensions,
            internal_lwe_dimensions,
            levelled_only_lwe_dimensions,
        }
    }

    pub fn default_gpu_amortized(parameter_domains: ParameterDomains) -> Self {
        // See backends/concrete_cuda/implementation/src/bootstrap_amortized.cu
        let glwe_log_polynomial_sizes: Vec<u64> = parameter_domains.glwe_pbs_constrained_gpu.log2_polynomial_size.as_vec();

        let glwe_dimensions: Vec<u64> = parameter_domains.glwe_pbs_constrained_gpu.glwe_dimension.as_vec();

        let internal_lwe_dimensions: Vec<u64> = parameter_domains.free_glwe.glwe_dimension.as_vec();
        let levelled_only_lwe_dimensions = parameter_domains.free_lwe;
        Self {
            glwe_log_polynomial_sizes,
            glwe_dimensions,
            internal_lwe_dimensions,
            levelled_only_lwe_dimensions,
        }
    }
    pub fn default(processing_unit: config::ProcessingUnit, parameter_domains: ParameterDomains) -> Self {
        match processing_unit {
            config::ProcessingUnit::Cpu => Self::default_cpu(parameter_domains),
            config::ProcessingUnit::Gpu {
                pbs_type: GpuPbsType::Amortized,
                ..
            } => Self::default_gpu_amortized(parameter_domains),
            config::ProcessingUnit::Gpu {
                pbs_type: GpuPbsType::Lowlat,
                ..
            } => Self::default_gpu_lowlat(parameter_domains),
        }
    }
}
