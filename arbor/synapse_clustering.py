import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import networkx as nx
import numpy as np

from arbor.arbor import BaseArbor, ArborNX, FlowCentrality
from arbor.arborparser import ArborParser


def id_generator(start=0):
    while True:
        yield start
        start += 1


@dataclass
class ArborRegions:
    above: List[int]
    plateau: List[int]
    zeros: List[int]

    def to_dict(self):
        return {
            "above": sorted(self.above),
            "plateau": sorted(self.plateau),
            "zeros": sorted(self.zeros),
        }


class SynapseClustering:
    """
    A fairly honest reimplementation of synapse_clustering.js, see
    https://github.com/catmaid/CATMAID/blob/master/django/applications/catmaid/static/libs/catmaid/synapse_clustering.js

    Could possibly be refactored to make better use of numpy
    """

    def __init__(
        self,
        arbor: ArborNX,
        locations: Dict[int, np.ndarray],
        synapses: Dict[int, int],
        lambda_: float,
    ):
        """

        Parameters
        ----------
        arbor : BaseArbor
        locations : dict
            treenode ID to 3-length np.ndarray
        synapses : dict
            treenode ID to number of connector nodes associated with the treenode
        lambda_
        """
        self.arbor = arbor
        self.locations = locations
        self.synapses = synapses
        self.lambda_ = lambda_

        # list of lists of treenode IDs, where the last contains the root
        self.partitions = list(arbor.partition())[::-1]

        # dict of treenode ID vs cable length to the root node
        self.distances_to_root = arbor.nodes_distance_to(
            location_dict=self.locations
        ).distances

        self._distances = None

    def to_dict(self):
        return {
            "arbor": self.arbor.to_dict(),
            "locations": {k: list(v) for k, v in self.locations.items()},
            "synapses": self.synapses,
            "lambda": self.lambda_,
            "partitions": self.partitions,
            "distancesToRoot": self.distances_to_root,
            "Ds": self.distances,
        }

    @property
    def distances(self):
        if self._distances is None:
            self._distances = self.distance_map()
        return self._distances

    def distance_map(self):
        """
        Compute a distance map, where each skeleton treenode is a key, and its value
        is an array of calibrated cable distances to all arbor synapses.

        Operates in O((3+2+4+2+2+1)n) + n*log(n) + n*m, where n is number of treenodes
        and m is number of synapses.

        The networkx implementation is really, really slow.

        Returns
        -------
        dict
            mapping of treenode ID to list of cable lengths to all arbor synapses
        """

        if isinstance(self.arbor, ArborNX):
            all_distances = self.arbor._all_distances(location_dict=self.locations)
            sorted_tns = sorted(self.locations)
            d = defaultdict(list)
            for syn_tn, tn_dists in all_distances:
                if not self.synapses.get(syn_tn):
                    continue
                for tn in sorted_tns:
                    d[tn].append(tn_dists[tn])

            return dict(d)
        else:
            return self._distance_map_classic()

    def _distance_map_classic(self):
        """WARNING: really slow"""
        # treenode ID -> list of distances to treenodes with synapses
        distances = defaultdict(list)

        # branch point treenode ID -> list of treenode IDs upstream of it
        # Items are removed once the branch has been visited as part of a partition,
        # if it is not the last treenode of the partition
        seen_downstream_nodes = dict()

        max_distance = 3 * self.lambda_

        for partition in self.partitions:
            # update synapses for the previous node, and upstream nodes for the current node
            downstream_nodes = []
            treenode_iter = iter(partition)
            prev_treenode_id = next(treenode_iter)

            for treenode_id in treenode_iter:
                downstream_nodes.append(prev_treenode_id)
                prev_distances = distances[prev_treenode_id]

                synapses_for_tn = self.synapses.get(prev_treenode_id, 0)
                if synapses_for_tn:
                    d = self.distances_to_root[prev_treenode_id]
                    for child_id in downstream_nodes:
                        ds = distances[child_id]
                        distance_child_to_synapse = self.distances_to_root[child_id] - d
                        if distance_child_to_synapse > max_distance:
                            continue

                        for _ in range(synapses_for_tn):
                            ds.append(distance_child_to_synapse)

                seen = seen_downstream_nodes.get(treenode_id)
                distance_to_root = self.distances_to_root[treenode_id]
                distance_prev_to_current = (
                    self.distances_to_root[prev_treenode_id] - distance_to_root
                )

                if seen:
                    current_distances = distances[treenode_id]
                    prev_distances = prev_distances.copy()

                    for child_id in downstream_nodes:
                        child_distances = distances[child_id]
                        distance = self.distances_to_root[child_id] - distance_to_root
                        if distance > max_distance:
                            continue
                        for current_distance in current_distances:
                            child_distances.append(current_distance + distance)

                    for child_id in seen:
                        child_distances = distances[child_id]
                        distance = (
                            self.distances_to_root[child_id]
                            + distance_prev_to_current
                            - distance_to_root
                        )
                        if distance > max_distance:
                            continue
                        for prev_distance in prev_distances:
                            child_distances.append(prev_distance + distance)

                        downstream_nodes.append(child_id)
                        try:
                            del seen_downstream_nodes[treenode_id]
                        except KeyError:
                            pass

                translated_prev_distances = []

                for prev_distance in prev_distances:
                    distance = prev_distance + distance_prev_to_current
                    if distance < max_distance:
                        translated_prev_distances.append(distance)

                current_distances = distances.get(treenode_id)
                if current_distances:
                    distances[treenode_id] = (
                        current_distances + translated_prev_distances
                    )
                else:
                    distances[treenode_id] = translated_prev_distances

            seen_downstream_nodes[partition[-1]] = downstream_nodes

        synapses_at_root = self.synapses.get(self.arbor.root, 0)
        if synapses_at_root:
            for treenode_id in distances:
                these_distances = distances[treenode_id]

                for _ in range(synapses_at_root):
                    distance = self.distances_to_root[treenode_id]
                    if distance < max_distance:
                        these_distances.append(distance)

        return distances

    def cluster_sizes(self, density_hill_map) -> Dict[int, int]:
        """Density hill ID to number of treenodes labelled"""
        return {
            key: len(value) for key, value in self.clusters(density_hill_map).items()
        }

    def cluster_maps(self, density_hill_map) -> Dict[int, Dict[int, bool]]:
        """Density hill ID to dict of treenode ID to True"""
        return {
            key: {ikey: True for ikey in value}
            for key, value in self.clusters(density_hill_map).items()
        }

    def clusters(self, density_hill_map: Dict[int, int]) -> Dict[int, List[int]]:
        """Density hill ID to list of treenode IDs"""
        clusters = defaultdict(list)
        for treenode_id, hill_id in density_hill_map.items():
            clusters[hill_id].append(treenode_id)
        return dict(clusters)

    def density_hill_map(self) -> Dict[int, int]:
        """
        Calculate which density hill each treenode belongs to.

        Returns
        -------
        Mapping from treenode IDs to hill IDs
        """
        if isinstance(self.arbor, ArborNX):
            return self._density_hill_map_nx()
        else:
            return self._density_hill_map_classic()

    def _density_hill_map_nx(self):
        graph = self.arbor._proximo_distal
        densities = calculate_densities(self.distances, self.lambda_ ** 2)
        all_neighbours = self.arbor.all_neighbours()

        node_ids = iter(nx.topological_sort(graph))
        root = next(node_ids)
        assert root == self.arbor.root

        hill_id_gen = id_generator()
        density_hill_map = {root: next(hill_id_gen)}

        for node_id in node_ids:
            if density_hill_map.get(node_id) is not None:
                # node's hill has already been assigned, because it is immediately uphill from a valley
                continue

            pred_id = list(graph.predecessors(node_id)).pop()
            pred_hill_id = density_hill_map[pred_id]
            neighbour_ids = sorted(all_neighbours[node_id])
            self_density = densities[node_id]
            delta_densities = {
                n_id: densities[n_id] - self_density for n_id in neighbour_ids
            }

            if sum(v > 0 for v in delta_densities.values()) <= 1:
                # node is not a density valley
                density_hill_map[node_id] = pred_hill_id
                continue

            for neighbour_id, delta_density in delta_densities.items():
                if neighbour_id == pred_id or delta_density < 0:
                    continue
                # new hill for each successor up a hill
                density_hill_map[neighbour_id] = next(hill_id_gen)

            distance_to_current = self.distances_to_root[node_id]

            # this node's hill is the same as hill of the neighbour up the steepest hill
            density_hill_map[node_id] = density_hill_map[
                max(
                    neighbour_ids,
                    key=lambda n: delta_densities[n]
                    / abs(self.distances_to_root[n] - distance_to_current),
                )
            ]

        return density_hill_map

    def _density_hill_map_classic(self) -> Dict[int, int]:
        raise NotImplementedError("implementation not correct")
        edges = self.arbor.edges
        density_hill_map = dict()
        density = calculate_densities(self.distances, self.lambda_ ** 2)

        max_density_index = 0
        all_neighbours = self.arbor.all_neighbours()

        # root belongs to cluster 0
        density_hill_map[self.arbor.root] = 0

        # longest to shortest ensures working on each density hill from one direction
        for partition in reversed(self.partitions):
            idx = len(partition) - 1
            # branches' children will already have an index; root's will not
            try:
                density_hill_index = density_hill_map[partition[idx - 1]]
            except KeyError:
                density_hill_index = density_hill_map[partition[idx]]

            for idx in range(idx, 0, -1):
                treenode_id = partition[idx]
                density_hill_map[treenode_id] = density_hill_index
                neighbours = all_neighbours[treenode_id]
                if len(neighbours) == 1:
                    # leaf node, is a maximum
                    continue

                self_density = density[treenode_id]
                n_over_zero = 0
                delta_density = dict()

                for neighbour in neighbours:
                    d = density[neighbour] - self_density
                    if d > 0:
                        n_over_zero += 1
                    delta_density[neighbour] = d

                if n_over_zero <= 1:
                    # current node is not a valley between density hills
                    continue

                parent_id = edges[treenode_id]
                for neighbour in neighbours:
                    if parent_id == neighbour or delta_density[neighbour] < 0:
                        # looking at parent, which already has a hill, or is part of the same hill
                        continue

                    max_density_index += 1
                    density_hill_map[neighbour] = max_density_index

                # change in density divided by change in location
                distance_to_current = self.distances_to_root[treenode_id]
                steepest_neighbour = max(
                    neighbours,
                    key=lambda n: delta_density[n]
                    / abs(self.distances_to_root[n] - distance_to_current),
                )

                steepest = density_hill_map[steepest_neighbour]
                density_hill_map[treenode_id] = steepest
                for neighbour in neighbours:
                    if delta_density[neighbour] < 0:
                        density_hill_map[neighbour] = steepest

                density_hill_index = density_hill_map[partition[idx - 1]]

        return density_hill_map

    def segregation_index(
        self,
        clusters: Dict[int, List[int]],
        outputs: Dict[int, int],
        inputs: Dict[int, int],
    ):
        """
        Global measure.

        Sum the entropy of each cluster, measured as a deviation from uniformity
        (same number of inputs and outputs per cluster),
        relative to the entropy of the arbor as a whole.

        Parameters
        ----------
        clusters : dict of density hill IDs to list of treenodes in that hill
        outputs : dict of treenode IDs to how many connectors it is presynaptic to
        inputs : dict of treenode IDs to how many connectors it is postsynaptic to

        Returns
        -------
        float
        """
        n_inputs = sum(inputs.values())
        n_outputs = sum(outputs.values())
        n_synapses = n_inputs + n_outputs

        S = 0  # entropy

        for cluster_id, treenode_ids in clusters.items():
            cluster_n_inputs = sum(inputs.get(tn, 0) for tn in treenode_ids)
            cluster_n_outputs = sum(outputs.get(tn, 0) for tn in treenode_ids)
            cluster_n_synapses = cluster_n_inputs + cluster_n_outputs

            cluster_S = entropy(n_inputs, n_outputs)

            S += cluster_n_synapses * cluster_S

        if S == 0 or n_inputs == 0 or n_outputs == 0:
            return 1

        S /= n_synapses

        return 1 - S / entropy(n_inputs, n_outputs)  # normalise by arbor entropy

    def find_arbor_regions(
        self,
        flow_centrality: Dict[int, FlowCentrality],
        fraction: float,
        arbor: BaseArbor = None,
    ) -> Optional[ArborRegions]:
        """
        Find which nodes have flow centralities above a threshold fraction of the arbor's
        maximum centrifugal flow centrality, which have 0, and which are equal to the max.

        Parameters
        ----------
        flow_centrality : dict of node ID to its flow centrality
        fraction : what proportion of the max centrifugal flow centrality to use as a threshold
        arbor : arbor instance (default self.arbor)

        Returns
        -------

        """
        arbor = arbor or self.arbor
        max_centrifugal = max(fc.centrifugal for fc in flow_centrality.values())
        if max_centrifugal == 0:
            return None

        above = []
        plateau = []
        zeros = []
        threshold = fraction * max_centrifugal

        for node_id in arbor.nodes_list():
            this_flow_centrality = flow_centrality[node_id]
            this_centrifugal = this_flow_centrality.centrifugal
            if this_centrifugal > threshold:
                above.append(node_id)
                if this_centrifugal == max_centrifugal:
                    plateau.append(node_id)
            elif this_flow_centrality.sum == 0:
                zeros.append(node_id)

        return ArborRegions(above, plateau, zeros)

    def find_axon_cut(
        self,
        outputs: Dict[int, Any],
        above: List[int],
        positions: Dict[int, np.ndarray],
        arbor: BaseArbor = None,
    ) -> Optional[int]:
        """
        Find a node ID at which is its optimal to cut an arbor so that the downstream
        sub-arbor is the axon and the rest is the dendrites.

        The heuristic is fidgety: finds the lowest-order node (relative to root)
        that contains an output synapse or is a branch where more than one of the downstream branches
        contains output synapses and is on the lower 50% of the cable for the flow centrality plateau
        (the "above" array).

        Parameters
        ----------
        outputs : dict of node ID to object which is truthy if output synapses exist
        above : list of nodes with supra-threshold centrifugal flow centrality
        positions : dict of node ID to 3-array of its location
        arbor : arbor (default self.arbor)

        Returns
        -------

        """
        arbor = arbor or self.arbor
        if len(above) == 1:
            return above[1]
        if len(above) == 0:
            return None

        orders = arbor.nodes_order_from()
        successors = arbor.all_successors()
        sorted_above = sorted(above, key=orders.get, reverse=True)
        closest_to_root = sorted_above[-1]
        is_above = defaultdict(lambda: False, {node: True for node in above})

        # spatial distances of all nodes to the node in above which is closest to the root
        all_dists_to_closest_above = arbor.nodes_distance_to(
            closest_to_root, location_dict=positions
        )
        dist_to_closest_above = {
            key: all_dists_to_closest_above.distances[key] for key in above
        }
        max_dist_to_closest_above = max(dist_to_closest_above.values())

        # nodes whose distance from the closest-to-root above node is in the top 50%
        threshold = max_dist_to_closest_above / 2
        beyond = [n for n in above if dist_to_closest_above.get(n, 0) > threshold]

        branches, _ = arbor.find_branch_and_end_nodes()
        lowest = sorted_above[0]
        lowest_order = math.inf

        for node in beyond:
            order = orders[node]
            if outputs.get(node):
                if order < lowest_order:
                    lowest = node
                    lowest_order = order
            elif branches.get(node):
                # exclude branch points whose parent is sub-threshold
                if order < lowest_order and is_above[arbor.edges[node]]:
                    # check if more than one branch has downstream inputs
                    these_successors = successors[node]
                    count = 0
                    for child in these_successors:
                        if is_above[child] or any(
                            outputs.get(n) for n in arbor.sub_arbor(child).nodes_list()
                        ):
                            count += 1
                            if count > 1:
                                lowest = node
                                lowest_order = order
                                break

        return lowest

    def find_axon(
        self,
        arbor_parser: ArborParser,
        fraction: float,
        positions: Dict[int, np.ndarray],
    ) -> Optional[BaseArbor]:
        fc = arbor_parser.arbor.flow_centrality(
            arbor_parser.outputs, arbor_parser.inputs
        )
        if not fc:
            return None
        regions = self.find_arbor_regions(fc, fraction, arbor=arbor_parser.arbor)
        if regions is None:
            return None
        cut = self.find_axon_cut(
            arbor_parser.outputs, regions.above, positions, arbor=arbor_parser.arbor
        )
        if cut is None:
            return None
        axon = arbor_parser.arbor.sub_arbor(cut)
        axon.fc_max_plateau = regions.plateau
        axon.fc_zeros = regions.zeros
        return axon


def entropy(*state_counts):
    total = 0
    total_count = sum(state_counts)
    for count in state_counts:
        if count == 0 or count == total_count:
            continue
        ppn = count / total_count
        total -= ppn * math.log(ppn)
    return total


def calculate_densities(distances, lambda_sq):
    density = dict()
    for treenode_id, these_distances in distances.items():
        density[treenode_id] = sum(
            math.exp(-math.pow(distance, 2) / lambda_sq) for distance in these_distances
        )
    return density
