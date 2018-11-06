from __future__ import annotations
from collections import defaultdict
from enum import Enum
from numbers import Number
from typing import TypeVar, Dict, List, Optional

import numpy as np

from arbor.common import RelationType
from arbor.arbor import Arbor, BaseArbor


class StringEnum(Enum):
    def __str__(self):
        return self.value


class SkeletonUrl(StringEnum):
    COMPACT_SKELETON = "compact-skeleton"
    COMPACT_DETAIL = "compact-skeleton"
    COMPACT_ARBOR = "compact-arbor"


ValidSkeletonUrl = TypeVar("ValidSkeletonUrl", str, SkeletonUrl)
Jso = TypeVar("Jso", str, Number, Dict, List)


class ArborParser:
    arbor_class = Arbor

    def __init__(
        self, url: Optional[ValidSkeletonUrl] = None, json: Optional[Jso] = None
    ):
        self.arbor: Optional[BaseArbor] = None
        self.inputs = None
        self.outputs = None

        self.n_outputs = None
        self.n_inputs = None

        self.n_presynaptic_sites = None
        self.n_postsynaptic_sites = None

        self.input_partners = None
        self.output_partners = None

        self.n_input_connectors = None
        self.n_output_connectors = None

        self.positions: Dict[int, np.ndarray] = None

        if url is not None and json is not None:
            self.init(url, json)

    def to_dict(self):
        d = {
            key: getattr(self, key, None)
            for key in [
                "inputs",
                "outputs",
                "n_outputs",
                "n_inputs",
                "n_presynaptic_sites",
                "n_postsynaptic_sites",
                "input_partners",
                "output_partners",
                "n_input_connectors",
                "n_output_connectors",
            ]
        }
        d["arbor"] = self.arbor.to_dict()
        d["positions"] = {key: list(value) for key, value in self.positions.items()}
        return d

    def init(self, url: ValidSkeletonUrl, json: Jso) -> ArborParser:
        self.tree(json[0])
        url = SkeletonUrl(url)
        if url == SkeletonUrl.COMPACT_SKELETON:
            self.connectors(json[1])
        elif url == SkeletonUrl.COMPACT_ARBOR:
            self.synapses(json[1])
        return self

    def tree(self, rows: List[List]) -> ArborParser:
        """Parse skeleton from either response"""
        self.arbor = self.arbor_class()
        self.positions = dict()
        edges = []
        for node, proximal, pos1, pos2, pos3, *_ in rows:
            self.positions[node] = np.array([pos1, pos2, pos3])
            if proximal:
                edges.extend([node, proximal])
            else:
                self.arbor.root = node
        self.arbor.add_edges(edges)
        return self

    def connectors(self, rows: List[List]) -> ArborParser:
        """Parse connectors from compact-skeleton response"""
        # presynaptic and postsynaptic
        outputs_inputs = [defaultdict(lambda: 0), defaultdict(lambda: 0)]
        outputs_inputs_totals = [0, 0]

        for row in rows:
            try:
                rel_type = RelationType(row[2])
            except ValueError:
                continue

            if rel_type not in RelationType.synaptic():
                continue
            syn_dict = outputs_inputs[rel_type.value]

            syn_dict[row[0]] += 1
            outputs_inputs_totals[rel_type.value] += 1

        self.n_presynaptic_sites, self.n_postsynaptic_sites = outputs_inputs_totals
        self.outputs, self.inputs = [dict(d) for d in outputs_inputs]
        return self

    def synapses(self, rows: List[List]) -> ArborParser:
        """Parse connectors from compact-arbor response"""
        outputs_inputs = [defaultdict(lambda: 0), defaultdict(lambda: 0)]
        outputs_inputs_details = [
            {"partners": {}, "count": 0, "connectors": {}},
            {"partners": {}, "count": 0, "connectors": {}},
        ]

        for row in rows:
            try:
                rel_type = RelationType(row[6])
            except ValueError:
                continue

            if rel_type not in RelationType.synaptic():
                continue

            syn_dict = outputs_inputs[rel_type.value]
            details_dict = outputs_inputs_details[rel_type.value]
            node = row[0]

            syn_dict[node] += 1
            details_dict["count"] += 1
            details_dict["partners"][row[5]] = True
            details_dict["connectors"][row[2]] = True

        self.n_outputs = outputs_inputs_details[0]["count"]
        self.n_inputs = outputs_inputs_details[1]["count"]
        self.output_partners = outputs_inputs_details[0]["partners"]
        self.input_partners = outputs_inputs_details[1]["partners"]
        self.n_output_connectors = len(list(outputs_inputs_details[0]["connectors"]))
        self.n_input_connectors = len(list(outputs_inputs_details[1]["connectors"]))

        self.outputs, self.inputs = [dict(d) for d in outputs_inputs]
        return self

    def create_synapse_map(self) -> Dict[int, int]:
        """Get treenode: n_synapses mapping"""
        out = defaultdict(lambda: 0, self.outputs)
        for node, n_inputs in self.inputs.items():
            out[node] += n_inputs
        return dict(out)

    def collapse_artifactual_branches(self):
        raise NotImplementedError()
