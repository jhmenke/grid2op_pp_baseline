import logging

import numpy as np
import pandas as pd

import pandapower as pp


def find_value_for_pp_bus(grid: pp.pandapowerNet,
                          voltages_from: np.ndarray,
                          voltages_to: np.ndarray,
                          bus: int):
    # map line extremity voltage readings to the correct pandapower bus
    bus_vn_kv = grid.bus.loc[bus, "vn_kv"]
    for idx, line in enumerate(grid.line.itertuples()):
        if line.from_bus == bus:
            return voltages_from[idx] / bus_vn_kv
        elif line.to_bus == bus:
            return voltages_to[idx] / bus_vn_kv
    for idx, trafo in enumerate(grid.trafo.itertuples()):
        if trafo.hv_bus == bus:
            return voltages_from[len(grid.line) + idx] / bus_vn_kv
        elif trafo.lv_bus == bus:
            return voltages_to[len(grid.line) + idx] / bus_vn_kv
    return None


def calc_losses(grid: pp.pandapowerNet):
    # calculate grid losses
    return grid.res_gen.p_mw.sum() + grid.res_ext_grid.p_mw.sum() + grid.res_sgen.p_mw.sum() - grid.load.p_mw.sum()


def make_zero_idx(elements: pd.DataFrame,
                  pd_indices,
                  offset: int = 0):
    # translate arbitrary pandas indices to 0-based numpy array indices
    return [elements.index.get_loc(lidx) + offset for lidx in pd_indices]


def run_opf(grid: pp.pandapowerNet,
            min_loss_reduction_mwt: float,
            acceptable_loading: float,
            opf_type: str,
            asset: str = "",
            logger=logging.getLogger()):
    loss_before = calc_losses(grid)
    line_loading_before = grid.res_line.loading_percent.max()
    trafo_loading_before = (grid.res_trafo.i_hv_ka / grid.trafo.max_i_ka * 100.).max()
    try:
        if opf_type == "pypower":
            pp.runopp(grid)  # pypower OPF
        elif opf_type == "powermodels":
            grid.line["in_service"] = True
            grid.trafo["in_service"] = True
            pp.runpm_ots(grid, pm_nl_solver="ipopt", pm_model="ACPPowerModel")  # PowerModels.jl OPF
            grid.line.loc[:, "in_service"] = grid.res_line.loc[:, "in_service"].values.astype(bool)
            grid.trafo.loc[:, "in_service"] = grid.res_trafo.loc[:, "in_service"].values.astype(bool)
        generation = np.array(grid.res_gen.p_mw.tolist() + grid.res_ext_grid.p_mw.tolist())
        losses_avoided = loss_before - calc_losses(grid)
        line_loading_after = grid.res_line.loading_percent.max()
        trafo_loading_after = (grid.res_trafo.i_hv_ka / grid.trafo.max_i_ka * 100.).max()
        use_opf_results = False
        # three conditions for using the OPF results:
        # 1. transformer was overloaded before and is now less overloaded
        # 2. line was overloaded before and is now less overloaded
        # 3. losses are minimized at least by "min_loss_reduction_mwt" while all loadings are acceptably low
        if trafo_loading_before > acceptable_loading and trafo_loading_after < trafo_loading_before:
            use_opf_results = True
        if line_loading_before > acceptable_loading and line_loading_after < line_loading_before:
            use_opf_results = True
        if losses_avoided > min_loss_reduction_mwt and line_loading_after < acceptable_loading and trafo_loading_after < acceptable_loading:
            use_opf_results = True
        if use_opf_results:
            logger.info(f"Losses avoided: {losses_avoided:.3f} MW (max. loading: {line_loading_after:.1f}% [{line_loading_after - line_loading_before:.1f}%] "
                        f"/ Trafo: {trafo_loading_after:.1f}% [{trafo_loading_after - trafo_loading_before:.1f}%])")
            return generation, grid.line.in_service, grid.trafo.in_service, True
        return np.zeros(len(grid.gen)), grid.line.in_service, grid.trafo.in_service, False
    except pp.optimal_powerflow.OPFNotConverged:
        asset_str = f"if asset #{asset} is out of service" if len(asset) else ""
        logger.info(f"OPF failed {asset_str}")
        return np.zeros(len(grid.gen)), grid.line.in_service, grid.trafo.in_service, False
