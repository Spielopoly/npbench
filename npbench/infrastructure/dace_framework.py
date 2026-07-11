# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import importlib
import os
import pkg_resources
import traceback

import dace

from npbench.infrastructure import Benchmark, Framework, timeout_decorator as tout, utilities as util
from typing import Any, Callable, Sequence, Tuple


class DaceFramework(Framework):
    """ A class for reading and processing framework information. """

    def __init__(self, fname: str, save_strict: bool = False, load_strict: bool = False):
        """ Reads framework information.
        :param fname: The framework name.
        :param save_strict: If True, saves the simplified SDFG.
        :param load_strict: If True, loads the simplified SDFG.
        """

        self.save_strict = save_strict
        self.load_strict = load_strict

        import warnings
        warnings.filterwarnings("ignore")
        super().__init__(fname)

    def version(self) -> str:
        """ Return the framework version. """
        return pkg_resources.get_distribution("dace").version

    def copy_func(self) -> Callable:
        """ Returns the copy-method that should be used
        for copying the benchmark arguments. """
        if self.fname == "dace_cutile":
            # cuTile compiled SDFGs accept host arrays; copy states handle H2D/D2H
            return super().copy_func()
        if self.fname == "dace_gpu":
            import cupy

            def cp_copy_func(arr):
                darr = cupy.asarray(arr)
                cupy.cuda.stream.get_current_stream().synchronize()
                return darr

            return cp_copy_func
        return super().copy_func()

    def implementations(self, bench: Benchmark) -> Sequence[Tuple[Callable, str]]:
        """ Returns the framework's implementations for a particular benchmark.
        :param bench: A benchmark.
        :returns: A list of the benchmark implementations.
        """

        module_pypath = "npbench.benchmarks.{r}.{m}".format(r=bench.info["relative_path"].replace('/', '.'),
                                                            m=bench.info["module_name"])
        if "postfix" in self.info.keys():
            postfix = self.info["postfix"]
        else:
            postfix = self.fname
        module_str = "{m}_{p}".format(m=module_pypath, p=postfix)
        func_str = bench.info["func_name"]

        ldict = dict()
        # Import DaCe implementation
        try:
            import copy
            import dace
            import dace.data
            import dace.dtypes as dtypes
            from dace.transformation.optimizer import Optimizer
            from dace.transformation.dataflow import MapFusion, Vectorization, MapCollapse
            from dace.transformation.interstate import LoopToMap
            import dace.transformation.auto.auto_optimize as opt

            module = importlib.import_module(module_str)
            ct_impl = getattr(module, func_str)

        except Exception as e:
            if self.fname == "dace_cutile":
                print(f"DaCe cuTile: failed to import {module_str}: {e}")
                return []
            print("Failed to load the DaCe implementation.")
            raise (e)

        # ── cuTile branch ───────────────────────────────────────────────
        # VectorizeCuTile → compile, trying multiple width
        # configs.  Returns early — the standard DaCe pipeline below is
        # skipped for cuTile.
        if self.fname == "dace_cutile":
            return self._cutile_implementations(ct_impl, func_str)

        ##### Experimental: Load strict SDFG
        sdfg_loaded = False
        if self.load_strict:
            path = os.path.join(os.getcwd(), 'dace_sdfgs', f"{module_str}-{func_str}.sdfg")
            try:
                strict_sdfg = dace.SDFG.from_file(path)
                sdfg_loaded = True
            except Exception:
                pass

        if not sdfg_loaded:
            #########################################################
            # Prepare SDFGs
            try:
                with tout.time_limit(self.timeout, "DaCe parse + strict transformations"):
                    base_sdfg, parse_time = util.benchmark("__npb_result = ct_impl.to_sdfg(simplify=False)",
                                                           out_text="DaCe parsing time",
                                                           context=locals(),
                                                           output='__npb_result',
                                                           verbose=False)
                    strict_sdfg = copy.deepcopy(base_sdfg)
                    strict_sdfg._name = "strict"
                    ldict['strict_sdfg'] = strict_sdfg
                    _, strict_time = util.benchmark("strict_sdfg.apply_strict_transformations()",
                                                    out_text="DaCe Strict Transformations time",
                                                    context=locals(),
                                                    verbose=False)
            except tout.StageTimeout as e:
                # Everything below builds on strict_sdfg — nothing to salvage.
                print(e)
                return []
            sdfg_list = [strict_sdfg]
            time_list = [parse_time[0] + strict_time[0]]
        else:
            ldict['strict_sdfg'] = strict_sdfg
        parse_time = [0]
        sdfg_list = []
        time_list = []

        ##### Experimental: Saving strict SDFG
        if self.save_strict and not sdfg_loaded:
            path = os.path.join(os.getcwd(), 'dace_sdfgs')
            try:
                os.mkdir(path)
            except FileExistsError:
                pass
            path = os.path.join(os.getcwd(), 'dace_sdfgs', f"{module_str}-{func_str}.sdfg")
            strict_sdfg.save(path)

        ##########################################################

        try:
            with tout.time_limit(self.timeout, "DaCe fusion variant build"):
                fusion_sdfg = copy.deepcopy(strict_sdfg)
                fusion_sdfg._name = "fusion"
                ldict['fusion_sdfg'] = fusion_sdfg
                _, fusion_time1 = util.benchmark("fusion_sdfg.apply_transformations_repeated([MapFusion])",
                                                 out_text="DaCe MapFusion time",
                                                 context=locals(),
                                                 verbose=False)
                _, fusion_time2 = util.benchmark("fusion_sdfg.apply_strict_transformations()",
                                                 out_text="DaCe Strict Transformations time",
                                                 context=locals(),
                                                 verbose=False)
                sdfg_list.append(fusion_sdfg)
                # time_list.append(time_list[-1] + fusion_time1[0] + fusion_time2[0])
                time_list.append(parse_time[0] + fusion_time1[0] + fusion_time2[0])
        except (Exception, tout.StageTimeout) as e:
            print("DaCe MapFusion failed")
            print(e)
            fusion_sdfg = copy.deepcopy(strict_sdfg)
            ldict['fusion_sdfg'] = fusion_sdfg

        ###########################################################

        try:
            with tout.time_limit(self.timeout, "DaCe canonicalize variant build"):
                from dace.transformation.passes.canonicalize import canonicalize
                canon_sdfg = copy.deepcopy(strict_sdfg)
                canon_sdfg._name = "canonicalize"
                # target only picks knob presets; GPU scheduling still happens via
                # copy_to_gpu + apply_gpu_transformations below (arch-gated because
                # dace_cpu shares this code path).
                canon_target = 'gpu' if self.info["arch"] == "gpu" else 'cpu'
                ldict['canon_sdfg'] = canon_sdfg
                ldict['canon_target'] = canon_target
                _, canon_time = util.benchmark("canonicalize(canon_sdfg, target=canon_target)",
                                               out_text="DaCe Canonicalize time",
                                               context=locals(),
                                               verbose=False)
                sdfg_list.append(canon_sdfg)
                time_list.append(parse_time[0] + canon_time[0])
        except (Exception, tout.StageTimeout) as e:
            print("DaCe canonicalize failed")
            print(e)

        ###########################################################

        def parallelize(sdfg):
            from dace.sdfg import propagation
            try:
                strict_xforms = dace.transformation.simplification_transformations()
            except Exception:
                strict_xforms = None

            for sd in sdfg.all_sdfgs_recursive():
                propagation.propagate_states(sd)
            if strict_xforms:
                sdfg.apply_transformations_repeated([LoopToMap, MapCollapse] + strict_xforms)
            else:
                num = 1
                while num > 0:
                    num = sdfg.apply_transformations_repeated([LoopToMap, MapCollapse])
                    sdfg.simplify()

        try:
            with tout.time_limit(self.timeout, "DaCe parallel variant build"):
                parallel_sdfg = copy.deepcopy(fusion_sdfg)
                parallel_sdfg._name = "parallel"
                ldict['parallel_sdfg'] = parallel_sdfg
                _, ptime1 = util.benchmark("parallelize(parallel_sdfg)",
                                           out_text="DaCe LoopToMap time1",
                                           context=locals(),
                                           verbose=False)
                _, ptime2 = util.benchmark("parallel_sdfg.apply_transformations_repeated([MapFusion])",
                                           out_text="DaCe LoopToMap time2",
                                           context=locals(),
                                           verbose=False)
                # sdfg_list.append(parallel_sdfg)
                # time_list.append(time_list[-1] + ptime1[0] + ptime2[0])

        except (Exception, tout.StageTimeout) as e:
            print("DaCe LoopToMap failed")
            print(e)
            parallel_sdfg = copy.deepcopy(fusion_sdfg)
            ldict['parallel_sdfg'] = parallel_sdfg

        ###########################################################
        ###### Standalone Test Auto - Opt after strict transformation
        try:

            def autoopt(sdfg, device, symbols):  #, nofuse):
                # # Mark arrays as on the GPU
                # if device == dtypes.DeviceType.GPU:
                #     for k, v in sdfg.arrays.items():
                #         if not v.transient and type(v) == dace.data.Array:
                #             v.storage = dace.dtypes.StorageType.GPU_Global

                # Auto-optimize SDFG
                opt.auto_optimize(auto_opt_sdfg, device, symbols=symbols, use_gpu_storage=True)

            with tout.time_limit(self.timeout, "DaCe auto_opt variant build"):
                auto_opt_sdfg = copy.deepcopy(strict_sdfg)
                auto_opt_sdfg._name = 'auto_opt'
                ldict['auto_opt_sdfg'] = auto_opt_sdfg
                device = dtypes.DeviceType.GPU if self.info["arch"] == "gpu" else dtypes.DeviceType.CPU

                _, auto_time = util.benchmark(f"autoopt(auto_opt_sdfg, device, symbols = locals())",
                                              out_text="DaCe Auto - Opt",
                                              context=locals(),
                                              verbose=False)

                sdfg_list.append(auto_opt_sdfg)
                time_list.append(time_list[-1] + auto_time[0])

        except (Exception, tout.StageTimeout) as e:
            print("DaCe autoopt failed")
            # print(e)
            # traceback.print_exc()
            auto_opt_sdfg = copy.deepcopy(strict_sdfg)
            ldict['auto_opt_sdfg'] = auto_opt_sdfg

        ###########################################################
        # Tile-op vectorization tracks (GPU only): the K-dim masked tile-op
        # vectorizer (the VectorizeCuTile core) lowered through the C++ CUDA
        # backend with 'pure' tile expansions. Two tracks: plain vectorize
        # (vec_<w>) and canonicalize-then-vectorize (canon_vec_<w>), at the 1-D
        # cutile widths. Vectorization happens here at seed time; the GPU block
        # below then rides copy_to_gpu + GPU transforms + MapFusion +
        # set_fast_implementations like every other variant.

        def vectorize_tileops(sdfg: dace.SDFG, width: int) -> int:
            """ Vectorize a seed SDFG with the K-dim masked tile-op track.

            :param sdfg: SDFG to vectorize in place.
            :param width: 1-D cutile tile width (power of 2).
            :returns: number of tile nodes created (0 = nothing vectorized).
            """
            # K-dim masked tile-op vectorizer, lowered for the C++ CUDA backend:
            # 'pure' tile expansions are plain per-lane C++ loops that compile as
            # device code. ISA impls ('scalar'/'avx*') call host-only headers and
            # must not be used; target_isa='SCALAR' + the explicit 'pure'
            # overwrite below is the verified-safe combination.
            # _TILE_NODE_TYPES must cover every node type VectorizeCPUMultiDim can
            # emit; a missing type would stay 'scalar'-stamped and fail nvcc.
            from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import (
                VectorizeCPUMultiDim, _TILE_NODE_TYPES)
            from dace.transformation.passes.vectorization.remove_unused_per_lane_symbols import (
                RemoveUnusedPerLaneSymbols)
            VectorizeCPUMultiDim(widths=(width, ), target_isa="SCALAR",
                                 expand_tile_nodes=False).apply_pass(sdfg, {})
            tiles = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, _TILE_NODE_TYPES)]
            if not tiles:
                return 0
            for t in tiles:
                t.implementation = 'pure'
            # Expand ONLY tile nodes: lifted Einsum/Reduce/MatMul stay unexpanded
            # so the GPU block's set_fast_implementations re-stamps them for GPU
            # (the vectorizer finalized them for CPU).
            sdfg.expand_library_nodes(predicate=lambda n: isinstance(n, _TILE_NODE_TYPES))
            RemoveUnusedPerLaneSymbols().apply_pass(sdfg, {})
            return len(tiles)

        if self.info["arch"] == "gpu":
            # vectorize_tileops mutates its SDFG in place and is NOT idempotent,
            # so it must be called directly — never via util.benchmark with
            # output=, which execs the statement twice (once to time, once for
            # the result) and would double-vectorize (yielding broken
            # `auto N = N` codegen). Timing here is print-only (never reaches the
            # DB), so a plain timer.
            import time as _time
            vec_widths = [512, 256, 128, 64, 32]  # the 1-D cutile sweep
            for w in vec_widths:
                try:
                    with tout.time_limit(self.timeout, f"DaCe vec_{w} seed (vectorize)"):
                        vec_sdfg = copy.deepcopy(strict_sdfg)
                        vec_sdfg._name = f"vec_{w}"
                        _t0 = _time.perf_counter()
                        ntiles = vectorize_tileops(vec_sdfg, w)
                        vec_elapsed = _time.perf_counter() - _t0
                        print(f"DaCe vec_{w} vectorize time: {vec_elapsed * 1000:.0f}ms")
                        if ntiles == 0:
                            print(f"DaCe vec_{w}: nothing vectorized, skipping")
                            continue
                        sdfg_list.append(vec_sdfg)
                        time_list.append(parse_time[0] + vec_elapsed)
                except (Exception, tout.StageTimeout) as e:
                    print(f"DaCe vec_{w} failed: {e}")

            cv_base = None
            cv_time = 0.0
            try:
                from dace.transformation.passes.vectorization import VectorizeCuTile
                # canonicalize_for_cutile, NOT a copy of canon_sdfg: the default
                # assumption_guard trips C++ codegen after vectorization ("use of
                # 'M' before deduction of 'auto'"), and this knob row is the one
                # the vectorizer is tested against. Width-independent -> run once,
                # deepcopy per width. Called directly (mutates in place).
                _t0 = _time.perf_counter()
                with tout.time_limit(self.timeout, "DaCe canon_vec canonicalize"):
                    cv_base = copy.deepcopy(strict_sdfg)
                    VectorizeCuTile.canonicalize_for_cutile(cv_base)
                cv_time = _time.perf_counter() - _t0
                print(f"DaCe canon_vec canonicalize time: {cv_time * 1000:.0f}ms")
            except (Exception, tout.StageTimeout) as e:
                cv_base = None
                print(f"DaCe canon_vec canonicalize failed, skipping all canon_vec variants: {e}")
            if cv_base is not None:
                for w in vec_widths:
                    try:
                        with tout.time_limit(self.timeout, f"DaCe canon_vec_{w} seed (vectorize)"):
                            cv_sdfg = copy.deepcopy(cv_base)
                            cv_sdfg._name = f"canon_vec_{w}"
                            _t0 = _time.perf_counter()
                            ntiles = vectorize_tileops(cv_sdfg, w)
                            vec_elapsed = _time.perf_counter() - _t0
                            print(f"DaCe canon_vec_{w} vectorize time: {vec_elapsed * 1000:.0f}ms")
                            if ntiles == 0:
                                print(f"DaCe canon_vec_{w}: nothing vectorized, skipping")
                                continue
                            sdfg_list.append(cv_sdfg)
                            time_list.append(parse_time[0] + cv_time + vec_elapsed)
                    except (Exception, tout.StageTimeout) as e:
                        print(f"DaCe canon_vec_{w} failed: {e}")

        ###########################################################

        def vectorize(sdfg, vec_len=None):
            matches = []
            for xform in Optimizer(sdfg).get_pattern_matches(patterns=[Vectorization]):
                matches.append(xform)
            for xform in matches:
                if vec_len:
                    xform.vector_len = vec_len
                xform.apply(sdfg)

        if self.info["arch"] == "gpu":
            def_impl = dace.Config.get('library', 'blas', 'default_implementation')
            if def_impl != "pure":
                dace.Config.set('library', 'blas', 'default_implementation', value='cuBLAS')

        def copy_to_gpu(sdfg):
            opt.apply_gpu_storage(sdfg)
            # for k, v in sdfg.arrays.items():
            #     if not v.transient and isinstance(v, dace.data.Array):
            #         v.storage = dace.dtypes.StorageType.GPU_Global

        if self.info["arch"] == "gpu":
            import cupy as cp

        implementations = []
        for sdfg, t in zip(sdfg_list, time_list):
            ldict['sdfg'] = sdfg
            fe_time = t
            try:
                with tout.time_limit(self.timeout, f"DaCe {sdfg._name} GPU transformations + compile"):
                    if sdfg._name != 'auto_opt':
                        device = dtypes.DeviceType.GPU if self.info["arch"] == "gpu" else dtypes.DeviceType.CPU
                        # if self.info["arch"] == "cpu":
                        #     # GPUTransform will set GPU schedules by itself
                        opt.set_fast_implementations(sdfg, device)
                    if self.info["arch"] == "gpu":
                        if sdfg._name in ['strict', 'parallel', 'fusion', 'canonicalize'] \
                                or sdfg._name.startswith(('vec_', 'canon_vec_')):
                            _, gpu_time1 = util.benchmark("copy_to_gpu(sdfg)",
                                                          out_text="DaCe GPU transformation time1",
                                                          context=locals(),
                                                          verbose=False)

                            _, gpu_time2 = util.benchmark("sdfg.apply_gpu_transformations()",
                                                          out_text="DaCe GPU transformation time2",
                                                          context=locals(),
                                                          verbose=False)
                            _, gpu_time3 = util.benchmark("sdfg.simplify()",
                                                          out_text="DaCe GPU transformation time3",
                                                          context=locals(),
                                                          verbose=False)
                            # NOTE: to be fair, allow one additional greedy MapFusion after GPU trafos
                            _, gpu_time4 = util.benchmark("sdfg.apply_transformations_repeated(MapFusion)",
                                                          out_text="DaCe GPU transformation time4",
                                                          context=locals(),
                                                          verbose=False)
                            fe_time += gpu_time2[0] + gpu_time3[0] + gpu_time4[0]
                            opt.set_fast_implementations(sdfg, device)
                        else:
                            gpu_time1 = [0]
                        fe_time += gpu_time1[0]
                    dc_exec, compile_time = util.benchmark("__npb_result = sdfg.compile()",
                                                           out_text="DaCe compilation time",
                                                           context=locals(),
                                                           output='__npb_result',
                                                           verbose=False)
                    implementations.append((dc_exec, sdfg._name))
            except tout.StageTimeout as e:
                print(e)
                continue
            except Exception as e:
                print("Failed to compile DaCe {a} {s} implementation.".format(a=self.info["arch"], s=sdfg._name))
                print(e)
                traceback.print_exc()
                print("Traceback")
                continue

            fe_time += compile_time[0]

        return implementations

    # ── cuTile helpers ────────────────────────────────────────────────

    def _cutile_implementations(
        self, func: Callable, func_str: str
    ) -> Sequence[Tuple[Callable, str]]:
        """Build cuTile variants across four front-end tracks, each followed by
        the same VectorizeCuTile width sweep -> compile.

        Tracks (front-end preprocessing before VectorizeCuTile):
          - ``canon``:       ``VectorizeCuTile.canonicalize_for_cutile`` (the
                             proven pipeline; recorded as ``cutile_canon_<w>``,
                             replacing the old ``cutile_<w>`` names — DB migrated).
          - ``parallel``:    simplify + repeated LoopToMap/MapCollapse + MapFusion,
                             recorded as ``cutile_parallel_<w>``.
          - ``autoopt_cpu``: ``auto_optimize(DeviceType.CPU, expand=False)``,
                             recorded as ``cutile_autoopt_cpu_<w>``.
          - ``autoopt_gpu``: ``auto_optimize(DeviceType.GPU, expand=False)``,
                             recorded as ``cutile_autoopt_gpu_<w>`` (speculative;
                             see prep_autoopt_gpu).

        Each track is wrapped in its own try/except so one failing front-end
        does not kill the others. Within a track, the per-width try/except still
        skips individual widths. Width configs whose dimensionality exceeds the
        kernel's are skipped because the cuTile runtime can SIGABRT (not
        catchable) on shape mismatches.

        :param func: The ``@dace.program`` function from the shared _dace module.
        :param func_str: The function name (for diagnostics).
        :returns: List of (compiled_sdfg, variant_name) tuples.
        """
        import copy
        import dace.data
        import dace.dtypes as dtypes
        import dace.transformation.auto.auto_optimize as opt
        from dace.sdfg import propagation
        from dace.transformation.dataflow import MapFusion, MapCollapse
        from dace.transformation.interstate import LoopToMap
        from dace.transformation.passes.vectorization import VectorizeCuTile

        try:
            with tout.time_limit(self.timeout, f"[dace_cutile] to_sdfg for {func_str}"):
                base_sdfg = func.to_sdfg(simplify=False)
        except (Exception, tout.StageTimeout) as e:
            print(f"  [dace_cutile] to_sdfg failed for {func_str}: {e}")
            return []

        # Kernel dimensionality from non-transient arrays (unchanged by any
        # front-end) — used to skip width configs that would cause a SIGABRT.
        max_ndim = max(
            (len(arr.shape) for arr in base_sdfg.arrays.values()
             if not arr.transient and isinstance(arr, dace.data.Array)),
            default=1
        )

        width_configs = [
            (512,), (256,), (128,), (64,), (32,),  # (16,), (8,), (4,), (2,), (1,),
            (32, 32), (16, 32), (16, 16), (8, 8), (8, 4),
            (2, 4, 4), (8,8,8),
            # (32,)
            ]

        def prep_canon(sdfg: dace.SDFG) -> None:
            VectorizeCuTile.canonicalize_for_cutile(sdfg)

        def prep_parallel(sdfg: dace.SDFG) -> None:
            # Simplify first: base is built with simplify=False and LoopToMap
            # works on simplified control flow. Then repeatedly parallelize
            # loops and collapse maps to fixpoint, and greedily fuse.
            sdfg.simplify()
            for sd in sdfg.all_sdfgs_recursive():
                propagation.propagate_states(sd)
            num = 1
            while num > 0:
                num = sdfg.apply_transformations_repeated([LoopToMap, MapCollapse])
                sdfg.simplify()
            sdfg.apply_transformations_repeated([MapFusion])
            sdfg.simplify()

        def prep_autoopt_cpu(sdfg: dace.SDFG) -> None:
            # CPU device: VectorizeCuTile owns GPU lowering (its step 3), so
            # hand the vectorizer CPU-form input. expand=False leaves library
            # nodes unexpanded for CuTileSetLibraryImplementations to select.
            opt.auto_optimize(sdfg, dtypes.DeviceType.CPU, expand=False)

        def prep_autoopt_gpu(sdfg: dace.SDFG) -> None:
            # Speculative track (user-requested). auto_optimize(GPU) already
            # GPU-schedules and inserts host<->device copies; VectorizeCuTile
            # step 3 re-runs GPUTransformSDFG, which is near-idempotent.
            # The vectorizer was designed for CPU-form input and may not fire on
            # GPU-form input — a failed/empty track is an accepted data point.
            opt.auto_optimize(sdfg, dtypes.DeviceType.GPU, expand=False)

        tracks = [
            ("canon", prep_canon),
            ("parallel", prep_parallel),
            ("autoopt_cpu", prep_autoopt_cpu),
            ("autoopt_gpu", prep_autoopt_gpu),
        ]

        results = []
        for label, prep in tracks:
            try:
                with tout.time_limit(self.timeout, f"[dace_cutile] {label} front-end"):
                    prepped = copy.deepcopy(base_sdfg)
                    prepped._name = f"{prepped.name}_{label}"  # distinct build dirs
                    prep(prepped)
            except (Exception, tout.StageTimeout) as e:
                print(f"  [dace_cutile] {label} front-end failed for {func_str}: {e}")
                continue
            for widths in width_configs:
                if len(widths) > max_ndim:
                    continue  # dimensionality mismatch -> skip
                try:
                    with tout.time_limit(self.timeout, f"[dace_cutile] {label} widths={widths}"):
                        sdfg_copy = copy.deepcopy(prepped)
                        self._lower_cutile(sdfg_copy, widths)
                        width_str = "x".join(str(w) for w in widths)
                        # Distinct build dir per (track, width): prepped.name already
                        # carries the track label, width_str ('x'-joined digits) keeps
                        # the name a valid SDFG identifier (e.g. atax_canon_32x32).
                        sdfg_copy._name = f"{prepped.name}_{width_str}"
                        csdfg = sdfg_copy.compile()
                        results.append((csdfg, f"cutile_{label}_{width_str}"))
                except (Exception, tout.StageTimeout) as e:
                    print(f"  [dace_cutile] {label} widths={widths} failed for {func_str}: {e}")

        return results

    @staticmethod
    def _lower_cutile(sdfg: dace.SDFG, widths: Tuple[int, ...]) -> None:
        """cuTile-lower an SDFG with a fixed tile-width configuration.

        Canonicalization is intentionally skipped (``run_canonicalize=False``):
        each track in ``_cutile_implementations`` does its own front-end prep
        before calling this, so the built-in canonicalize step must not re-run.

        :param sdfg: The SDFG to lower (modified in place).
        :param widths: Tile widths (must be powers of 2).
        """
        from dace.transformation.passes.vectorization import VectorizeCuTile

        VectorizeCuTile(widths=widths, run_canonicalize=False).apply_pass(sdfg, {})

    def params(self, bench: Benchmark, impl: Callable = None):
        return [p for p in bench.info["parameters"]['L'].keys() if p not in bench.info["input_args"]]

    def arg_str(self, bench: Benchmark, impl: Callable = None):
        """ Generates the argument-string that should be used for calling
        the benchmark implementation.
        :param bench: A benchmark.
        :param impl: A benchmark implementation.
        """

        input_args = self.args(bench, impl)
        params = self.params(bench, impl)
        input_args_str = ", ".join(["{b}={a}".format(a=a, b=b) for a, b in zip(input_args, bench.info["input_args"])])
        params_str = ", ".join(["{a}={a}".format(a=a) for a in params])
        return ", ".join((input_args_str, params_str))

    def param_str(self, bench: Benchmark, impl: Callable = None):
        """ Generates the parameter-string that should be used for calling
        the benchmark implementation.
        :param bench: A benchmark.
        :param impl: A benchmark implementation.
        """

        input_params = self.params(bench, impl)
        return ", ".join(["{p}={p}".format(p=p) for p in input_params])
