import logging
from copy import deepcopy

import numpy as np

import grid2op
import l2rpn_baselines.PandapowerOPFAgent.pp_functions as ppf
import pandapower as pp
from grid2op.Agent import BaseAgent


def make_logger():
    with open("agent_logs.txt", "w") as f:
        f.write("")
    _logger = logging.getLogger(__name__)
    _logger.addHandler(logging.FileHandler("result.txt"))
    _logger.addHandler(logging.StreamHandler())
    _logger.setLevel(logging.INFO)
    return _logger


class PandapowerOPFAgent(BaseAgent):
    """
    The purpose of this agent is to supply a usable base for power system calculations based on the grid2op backend.
    Incoming observations are automatically parsed into a pandapower grid. The grid can then be used for any
    algorithms based on it, e.g. optimal power flow.
    We show
    """
    timestep = 0
    last_opf_ts = 0
    grid = None
    baseline_losses = None
    total_losses = 0.
    gen_max_p = None
    gen_min_p = None
    logger = make_logger()

    def __init__(self, action_space, grid_path: str,
                 acceptable_loading_pct: float = 99.9,
                 min_loss_reduction_mwt: float = 0.5,
                 opf_type: str = "pypower",
                 line_auto_reconnect: bool = False):
        """
        Initialize agent
        :param action_space: the Grid2Op action space
        :param grid_path: path to the pandapower grid as a json file
        :param acceptable_loading_pct: maximum line/transformer loading allowed before OPF engages
        :param min_loss_reduction_mwt: minimum loss (as MW * timestep unit t) that needs to be reduced for the OPF to engage
        :param opf_type: which OPF to use (pypower or powermodels)
        :param line_auto_reconnect: reconnect disconnected lines automatically as soon as possible
        """
        BaseAgent.__init__(self, action_space)
        self.do_nothing_action = action_space({})
        self.grid = pp.from_json(grid_path)
        self.acceptable_loading_pct = acceptable_loading_pct
        self.min_loss_reduction_mwt = min_loss_reduction_mwt
        self.opf_type = opf_type
        self.line_auto_reconnect = line_auto_reconnect
        assert self.opf_type.lower() in ("pypower", "powermodels"), "choose either pypower or powermodels as the opf type"

    def parse_observation_to_grid(self, obs: grid2op.Observation):
        if self.timestep == 0:
            self.grid.trafo["max_loading_percent"] = self.acceptable_loading_pct
            self.grid.line["max_loading_percent"] = self.acceptable_loading_pct
            self.grid.poly_cost.drop(self.grid.poly_cost.index, inplace=True)
            assert len(self.grid.ext_grid) == 1 and len(obs.gen_type) == len(self.grid.gen) + len(self.grid.ext_grid)
            self.grid.bus.min_vm_pu = 0.9
            self.grid.bus.max_vm_pu = 1.2
            # gen
            self.grid.gen.type = obs.gen_type[:-1]
            self.grid.gen.controllable = obs.gen_redispatchable[:-1]
            self.grid.gen.min_p_mw = obs.gen_pmin[:-1]
            self.grid.gen.max_p_mw = obs.gen_pmax[:-1]
            self.gen_min_p = obs.gen_pmin
            self.gen_max_p = obs.gen_pmax
            # ext grid (last gen)
            self.grid.gen.min_q_mvar = -obs.gen_pmax[:-1]
            self.grid.gen.max_q_mvar = obs.gen_pmax[:-1]
            self.grid.ext_grid.type = obs.gen_type[-1]
            assert obs.gen_redispatchable[-1]
            self.grid.ext_grid.min_p_mw = obs.gen_pmin[-1]
            self.grid.ext_grid.max_p_mw = obs.gen_pmax[-1]
            self.grid.ext_grid.min_q_mvar = -obs.gen_pmax[-1]
            self.grid.ext_grid.max_q_mvar = obs.gen_pmax[-1]
            for row in self.grid.gen[self.grid.gen.controllable].itertuples():
                pp.create_poly_cost(self.grid, row.Index, "gen", cp1_eur_per_mw=1)  # perform loss minimization
            pp.create_poly_cost(self.grid, self.grid.ext_grid.index[0], "ext_grid", cp1_eur_per_mw=1)
        self.refresh_gen_values(obs)
        self.grid.load["p_mw"] = obs.load_p
        self.grid.load["q_mvar"] = obs.load_q
        self.grid.line["in_service"] = obs.line_status[:len(self.grid.line)]
        self.grid.trafo["in_service"] = obs.line_status[-len(self.grid.trafo):]
        pp.runpp(self.grid, init="results")
        self.timestep += 1
        self.baseline_losses = ppf.calc_losses(self.grid)
        self.total_losses += self.baseline_losses
        if (self.grid.res_line.loading_percent > self.acceptable_loading_pct).any() \
                or (self.grid.res_trafo.loading_percent > self.acceptable_loading_pct).any():
            self.logger.info(f"OVERLOADING: line @ {self.grid.res_line.loading_percent.max():.1f} % "
                             f"/ trafo @ {(self.grid.res_trafo.loading_percent / self.grid.trafo.max_loading_percent).max() * 100.:.1f} %")

    def refresh_gen_values(self, obs: grid2op.Observation):
        # update pandapower gen values with observation
        for row in self.grid.ext_grid.itertuples():
            self.grid.ext_grid.loc[row.Index, "vm_pu"] = ppf.find_value_for_pp_bus(self.grid, obs.v_or, obs.v_ex, row.bus)
        for row in self.grid.gen.itertuples():
            self.grid.gen.loc[row.Index, "vm_pu"] = ppf.find_value_for_pp_bus(self.grid, obs.v_or, obs.v_ex, row.bus)
        self.grid.gen["p_mw"] = obs.prod_p[:-1]
        self.grid.gen["res_q_mvar"] = obs.prod_q[:-1]
        ramp_down, ramp_up = obs.gen_max_ramp_down + 1e-6, obs.gen_max_ramp_up - 1e-6
        abs_min, abs_max = np.minimum(self.grid.gen.p_mw, obs.gen_pmin[:-1]), np.maximum(self.grid.gen.p_mw, obs.gen_pmax[:-1])
        self.grid.gen["max_p_mw"] = (self.grid.gen.p_mw + ramp_up[:-1]).clip(abs_min, abs_max)
        self.grid.gen["min_p_mw"] = (self.grid.gen.p_mw - ramp_down[:-1]).clip(abs_min, abs_max)
        self.grid.ext_grid["p_mw"] = obs.prod_p[-1]
        self.grid.ext_grid["max_p_mw"] = (obs.prod_p[-1] + ramp_up[-1]).clip(obs.gen_pmin[-1], obs.gen_pmax[-1])
        self.grid.ext_grid["min_p_mw"] = (obs.prod_p[-1] + ramp_up[-1]).clip(obs.gen_pmin[-1], obs.gen_pmax[-1])

    def act(self, observation: grid2op.Observation, reward, done=False):
        # 1. Parse observations into pandapower grid
        self.parse_observation_to_grid(observation)
        # 2. Check if any lines are out of service in the observed grid and mark them (so they are not changed with an action)
        #    Lines that were out of service but can be reconnected will automatically be connected again with an action
        #    Grid2Op lines can both be pandapower lines or pandapower transformers
        opf_grid = deepcopy(self.grid)
        lines_to_be_connected, lines_to_disconnect, line_failures = [], [], []
        if not self.grid.line.in_service.all():
            for line_oos in self.grid.line.loc[~self.grid.line.in_service].index:
                line_zero_idx = self.grid.line.index.get_loc(line_oos)
                line_state = observation.state_of(line_id=line_zero_idx)
                if line_state["indisponibility"] > 0:
                    opf_grid.line.drop(line_oos, inplace=True)  # drop line, otherwise it would be used in the OPF
                    self.logger.info(f"Observation: Line {line_oos} out of service!")
                    line_failures.append(line_zero_idx)
                elif self.line_auto_reconnect:
                    lines_to_be_connected.append(line_zero_idx)  # automatically reconnect lines if possible
                    self.grid.line.loc[line_oos, "in_service"] = True
        if not self.grid.trafo.in_service.all():
            for trafo_oos in self.grid.trafo.loc[~self.grid.trafo.in_service].index:
                trafo_zero_idx = self.grid.trafo.index.get_loc(trafo_oos) + len(self.grid.line)  # offset pp lines
                trafo_state = observation.state_of(line_id=trafo_zero_idx)
                if trafo_state["indisponibility"] > 0:
                    opf_grid.trafo.drop(trafo_oos, inplace=True)  # drop trafo, otherwise it would be used in the OPF
                    self.logger.info(f"Observation: trafo {trafo_oos} out of service!")
                    line_failures.append(trafo_zero_idx)
                elif self.line_auto_reconnect:
                    lines_to_be_connected.append(trafo_zero_idx)  # automatically reconnect trafos if possible
                    self.grid.trafo.loc[trafo_oos, "in_service"] = True
        # 3. Perform OPF
        p_dispatched_current = np.array(self.grid.gen.p_mw.tolist() + self.grid.ext_grid.p_mw.tolist())
        p_redispatched = p_dispatched_current
        generation, opf_line_status, opf_trafo_status, make_changes = ppf.run_opf(opf_grid, self.min_loss_reduction_mwt, self.acceptable_loading_pct,
                                                                                  self.opf_type.lower(), logger=self.logger)
        if make_changes:
            lines_to_disconnect = ppf.make_zero_idx(self.grid.line, opf_line_status.index[~opf_line_status])
            lines_to_disconnect.extend(ppf.make_zero_idx(self.grid.trafo, opf_trafo_status.index[~opf_trafo_status], offset=len(self.grid.line)))
            lines_to_be_connected = ppf.make_zero_idx(self.grid.line, set(self.grid.line.index[~self.grid.line.in_service])
                                                       & set(opf_line_status.index[opf_line_status]))
            lines_to_be_connected.extend(ppf.make_zero_idx(self.grid.trafo, set(self.grid.trafo.index[~self.grid.trafo.in_service])
                                                            & set(opf_trafo_status.index[opf_trafo_status]), offset=len(self.grid.line)))
            new_connections = sorted(list(set(self.grid.line.index[~self.grid.line.in_service]) & set(opf_line_status.index[opf_line_status])))
            self.logger.info(f"OPF wants to disconnect: pp_{opf_line_status.index[~opf_line_status].tolist()} "
                             f"/ connect: pp_{self.grid.line.index[new_connections].tolist()} ")
            p_redispatched = generation
        # 4. Convert to actions
        action_space = {}
        # Redispatch action
        gen_p_redispatched = ~np.isclose(p_dispatched_current, p_redispatched)
        if np.any(gen_p_redispatched):
            p_redispatched -= p_dispatched_current  # make absolute power relative
            p_redispatched[abs(p_redispatched) < 0.5] = 0.
            redispatch_sum = p_redispatched[gen_p_redispatched].sum()  # this sum needs to be 0. for a valid redispatch
            if redispatch_sum > 0.:
                power_up = np.where(p_redispatched > 0)[0]
                p_redispatched[power_up] -= redispatch_sum / len(power_up)
            elif redispatch_sum < 0.:
                power_down = np.where(p_redispatched < 0)[0]
                p_redispatched[power_down] -= redispatch_sum / len(power_down)
            assert abs(p_redispatched[gen_p_redispatched].sum()) < 1e-14
            action_space["redispatch"] = [(idx, p) for idx, p in zip(np.arange(len(self.grid.gen) + 1)[gen_p_redispatched],
                                                                     p_redispatched[gen_p_redispatched]) if abs(p) >= 1e-3]
            # self.logger.info(action_space["redispatch"])
        # Set lines action
        new_line_status_array = np.zeros(observation.rho.shape)
        bus_idx = {"lines_or_id": [], "lines_ex_id": []}
        assert len(set(lines_to_disconnect) & set(lines_to_be_connected)) == 0
        if len(lines_to_disconnect) > 0:
            # new_line_status_array[np.random.choice(lines_to_disconnect)] = -1  # can only open 1 line per time step (select randomly)
            new_line_status_array[lines_to_disconnect[-1]] = -1  # can only open 1 line per time step (select transformers first)
        for line_idx in lines_to_be_connected:
            line_state = observation.state_of(line_id=line_idx)
            if line_state["cooldown_time"] > 0:
                self.logger.error(f"Cannot reconnect line {line_idx}, it is still in cooldown!")
                continue
            new_line_status_array[line_idx] = 1
            bus_idx["lines_or_id"].append((line_idx, 1))
            bus_idx["lines_ex_id"].append((line_idx, 1))
        if np.count_nonzero(new_line_status_array) > 0:
            action_space["set_line_status"] = new_line_status_array
            action_space["set_bus"] = bus_idx
        action = self.action_space(action_space)
        # 5. Simulate action to see whether it makes sense, otherwise do nothing
        # if action != self.do_nothing_action:
        #     simul_obs, simul_reward, simul_has_error, simul_info = observation.simulate(action)
        #     if simul_has_error:
        #         action = self.do_nothing_action
        #     else:
        #         self.logger.info(action)
        #         _, dnr, _, _ = observation.simulate(self.do_nothing_action)
        #         self.logger.info(f"Expected reward: {simul_reward:.2f} (do nothing: {dnr:.2f})")
        # 6. Send action, done
        # return self.do_nothing_action
        denied, reason = action.is_ambiguous()
        assert not denied, f"{reason} {action}"
        return action
