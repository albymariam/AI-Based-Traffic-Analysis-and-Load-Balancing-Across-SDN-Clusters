"""
Mininet OpenFlow Controller Adapter Module
Academic Research Project RM-T11 (Phase P2)

Integrates M1 Predictive Routing Policy with Mininet Open vSwitch (OVS) Switches:
  - Generates OpenFlow OFPFlowMod rule installation specifications.
  - Interfaces telemetry port stats polling (OFPPortStatsRequest).
  - Pushes dynamic path rerouting rules to switches.
"""

import numpy as np

class MininetM1ControllerAdapter:
    """
    OpenFlow Controller Adapter for M1 Forecast-Driven Flow Rerouting.
    """
    def __init__(self, topology_graph, hysteresis_eta=0.08):
        self.topo = topology_graph
        self.eta = hysteresis_eta
        self.installed_rules = {}  # Flow_ID -> Active Path
        self.rule_counter = 0

    def install_openflow_flow_mod(self, flow_id, new_path):
        """
        Simulates pushing OpenFlow OFPFlowMod rule messages to switches along new_path.
        Specifies match fields (in_port, eth_type, ipv4_src, ipv4_dst) and actions (output_port).
        """
        switches_updated = len(new_path) - 1
        self.installed_rules[flow_id] = new_path
        self.rule_counter += switches_updated
        
        flow_spec = {
            'flow_id': flow_id,
            'path': new_path,
            'switches_modified': switches_updated,
            'openflow_version': 'OpenFlow 1.3',
            'action': 'OFPFlowMod (MODIFY)'
        }
        return flow_spec

    def process_telemetry_and_reroute(self, flow_id, candidate_paths, real_lstm_predictions):
        """
        Executes complete M1 control loop:
          1. Evaluates path bottlenecks using LSTM predictions (Eq 4).
          2. Selects target path p* (Eq 5).
          3. Evaluates hysteresis rule B(p_curr) - B(p*) > eta (Eq 6).
          4. Installs OpenFlow FlowMod rules if rerouting is triggered.
        """
        from controller.routing_policy import select_optimal_path_with_hysteresis
        
        current_path = self.installed_rules.get(flow_id, candidate_paths[0])
        selected_path, should_reroute, util_savings = select_optimal_path_with_hysteresis(
            candidate_paths, current_path, real_lstm_predictions, eta=self.eta
        )
        
        if should_reroute:
            flow_mod = self.install_openflow_flow_mod(flow_id, selected_path)
            status = 'REROUTED'
        else:
            status = 'STAY (Hysteresis Active)'
            flow_mod = None
            
        return {
            'flow_id': flow_id,
            'status': status,
            'current_path': current_path,
            'selected_path': selected_path,
            'util_savings': util_savings,
            'flow_mod': flow_mod
        }
