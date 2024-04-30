from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, passes, llvm, cpu

from dataclasses import dataclass
import functools
from typing import Any, Tuple
import hashlib
import re
import tempfile
import signal
import os
import subprocess
from pathlib import Path


def _path_to_binary(binary: str):
    paths = [
        os.environ.get(f"TRITON_{binary.upper()}_PATH", ""),
        os.path.join(os.path.dirname(__file__), "bin", binary),
    ]

    for p in paths:
        bin = p.split(" ")[0]
        if os.path.exists(bin) and os.path.isfile(bin):
            result = subprocess.check_output([bin, "--version"], stderr=subprocess.STDOUT)
            if result is not None:
                version = re.search(r".*SPIRV-Tools v(\d+\.\d+).*", result.decode("utf-8"), flags=re.MULTILINE)
                if version is not None:
                    return p, version.group(1)
    raise RuntimeError(f"Cannot find {binary}")


@dataclass(frozen=True)
class CPUOptions:
    num_warps: int = 1
    num_ctas: int = 1
    num_stages: int = 1
    cluster_dims: tuple = (1, 1, 1)
    threads_per_warp: int = 1
    optimize_epilogue: bool = False
    enable_fp_fusion: bool = True
    default_dot_input_precision: str = "ieee"
    allowed_dot_input_precisions: Tuple[str] = ("ieee")
    allow_fp8e4nv: bool = False
    allow_fp8e4b15: bool = False
    max_num_imprecise_acc_default: int = 0
    extern_libs: dict = None
    debug: bool = False

    def hash(self):
        key = '_'.join([f'{name}-{val}' for name, val in self.__dict__.items()])
        return hashlib.md5(key.encode("utf-8")).hexdigest()


class CPUBackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'cpu'

    def __init__(self, target: tuple) -> None:
        super().__init__(target)
        self.properties = {}
        self.binary_ext = "bc"

    def _parse_target(self, tgt_prop) -> dict:
        dev_prop = {}
        dev_prop['name'] = tgt_prop.get('name', 'xpu')
        dev_prop['platform_name'] = tgt_prop.get('platform_name', None)
        dev_prop['vendor'] = tgt_prop.get('vendor', None)
        dev_prop['driver_version'] = tgt_prop.get('driver_version', None)
        dev_prop['version'] = tgt_prop.get('version', None)
        dev_prop['gpu_eu_count'] = tgt_prop.get('gpu_eu_count', None)
        dev_prop['gpu_subslice_count'] = tgt_prop.get('gpu_subslice_count', None)
        dev_prop['max_work_group_size'] = tgt_prop.get('max_work_group_size', None)
        dev_prop['max_num_sub_groups'] = tgt_prop.get('max_num_sub_groups', None)
        dev_prop['sub_group_sizes'] = tgt_prop.get('sub_group_sizes', None)
        dev_prop['has_fp64'] = tgt_prop.get('has_fp64', None)
        return dev_prop

    def parse_options(self, opts) -> Any:
        args = {k: opts[k] for k in CPUOptions.__dataclass_fields__.keys() if k in opts}
        args["allow_fp8e4nv"] = True
        return CPUOptions(**args)

    def pack_metadata(self, metadata):
        return metadata

    def load_dialects(self, ctx):
        cpu.load_dialects(ctx)

    @staticmethod
    def make_ttir(mod, metadata, opt):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_combine(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        passes.common.add_symbol_dce(pm)
        pm.run(mod)
        return mod

    @staticmethod
    def make_ttcir(mod, metadata, opt):
        # TTIR -> TTCIR
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        cpu.passes.ttc.add_triton_to_triton_cpu_pipeline(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.common.add_canonicalizer(pm)
        pm.run(mod)
        metadata["cluster_dims"] = (opt.cluster_dims[0], opt.cluster_dims[1], opt.cluster_dims[2])
        return mod

    @staticmethod
    def make_llir(src, metadata, options):
        # warp-specialization mutates num_warps
        num_warp_groups = src.get_int_attr("triton_gpu.num-warp-groups-per-cta")
        if num_warp_groups is not None:
            metadata["num_warps"] *= num_warp_groups
        metadata["threads_per_warp"] = 1
        mod = src
        # TritonGPU -> LLVM-IR (MLIR)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.convert.add_scf_to_cf(pm)
        passes.convert.add_index_to_llvmir(pm)
        cpu.passes.ttc.add_triton_cpu_to_llvmir_pipeline(pm)
        cpu.passes.add_vector_to_llvmir(pm)
        passes.convert.add_arith_to_llvmir(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        if os.environ.get("TRITON_DISABLE_LINE_INFO", "0") == "0":
            passes.llvmir.add_di_scope(pm)
        pm.run(mod)

        # Find kernel fn
        kernel_names = cpu.find_kernel_names(mod)
        assert len(kernel_names) == 1, f"expected exactly 1 kernel in a module, got {kernel_names}"

        # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
        llvm.init_targets()
        context = llvm.context()
        llvm_mod = llvm.to_cpu_module(mod, context)
        llvm.set_cpu_target_triple(llvm_mod)
        #if options.extern_libs:
        #    paths = [path for (name, path) in options.extern_libs]
        #   llvm.link_extern_libs(llvm_mod, paths)
        llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3)
        # Get some metadata
        metadata["shared"] = 0
        metadata["name"] = kernel_names[0]
        ret = str(llvm_mod)
        del llvm_mod
        del context
        return ret

    @staticmethod
    def make_bc(src, metadata):
        #asm_str = llvm.translate_to_asm(src, 'x86_64-unknown-linux-gnu', "znver4", '', [], True,
        #                                      False)
        #print("********** ASM Module **********")
        #print(asm_str)
        ret = llvm.translate_to_bc(src)
        return ret

    def add_stages(self, stages, options):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttcir"] = lambda src, metadata: self.make_ttcir(src, metadata, options)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)
        stages["bc"] = lambda src, metadata: self.make_bc(src, metadata)

    @functools.lru_cache()
    def hash(self):
        return f'{self.properties}'

    def get_codegen_implementation(self):
        codegen_fns = {}
        return codegen_fns
