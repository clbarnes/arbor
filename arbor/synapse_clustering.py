import math
from collections import defaultdict
from enum import IntEnum

import networkx as nx
from typing import Dict, List, Tuple, TypeVar

import numpy as np

from arbor.arbor import BaseArbor, ArborNX, ArborClassic
from arbor.common import euclidean_distance


class RelationType(IntEnum):
    PRESYNAPTIC = 0
    POSTSYNAPTIC = 1


ValidRelType = TypeVar('ValidRelType', int, RelationType)


def id_generator(start=0):
    while True:
        yield start
        start += 1


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
            synapses: Dict[int, List[Tuple[ValidRelType, int]]],
            lambda_: float
    ):
        """

        Parameters
        ----------
        arbor : BaseArbor
        locations : dict
            treenode ID to 3-length np.ndarray
        synapses : dict  # todo: is this true? or is it {tnid: n_synapses}
            treenode ID to list of (type, connector ID) tuples,
            where type is 0 for presynaptic and 1 for postsynaptic
        lambda_
        """
        self.arbor = arbor
        self.locations = locations
        self.synapses = synapses
        self.lambda_ = lambda_

        # list of lists of treenode IDs, sorted from smaller to larger lists
        self.partitions = arbor.partition_sorted()

        # dict of treenode ID vs cable length to the root node
        self.distances_to_root = arbor.nodes_distance_to(location_dict=self.locations).distances

        self.distances = self.distance_map()

    def distance_map(self):
        """
        Compute a distance map, where each skeleton treenode is a key, and its value
        is an array of calibrated cable distances to all arbor synapses.

        Operates in O((3+2+4+2+2+1)n) + n*log(n) + n*m, where n is number of treenodes
        and m is number of synapses.

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
                if syn_tn not in self.synapses:
                    continue
                for tn in sorted_tns:
                    d[tn].append(tn_dists[tn])

            return dict(d)
        else:
            return self._distance_map_classic()

    def _distance_map_classic(self):
        raise NotImplementedError("Implementation is not correct")
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

                synapses_for_tn = self.synapses.get(prev_treenode_id, [])
                if synapses_for_tn:
                    d = self.distances_to_root[prev_treenode_id]
                    for child_id in downstream_nodes:
                        ds = distances[child_id]
                        distance_child_to_synapse = self.distances_to_root[child_id] - d
                        if distance_child_to_synapse > max_distance:
                            continue
                        for _ in synapses_for_tn:
                            ds.append(distance_child_to_synapse)

                seen = seen_downstream_nodes.get(treenode_id)
                distance_to_root = self.distances_to_root[treenode_id]
                distance_prev_to_current = self.distances_to_root[prev_treenode_id] - distance_to_root

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
                        distance = self.distances_to_root[child_id] + distance_prev_to_current - distance_to_root
                        if distance > max_distance:
                            continue
                        for prev_distance in prev_distances:
                            child_distances.append(prev_distance + distance)

                        downstream_nodes.append(child_id)
                        del seen_downstream_nodes[treenode_id]

                translated_prev_distances = []

                for prev_distance in prev_distances:
                    distance = prev_distance + distance_prev_to_current
                    if distance < max_distance:
                        translated_prev_distances.append(distance)

                current_distances = distances.get(treenode_id)
                if current_distances:
                    distances[treenode_id] = current_distances + translated_prev_distances
                else:
                    distances[treenode_id] = translated_prev_distances

            seen_downstream_nodes[partition[-1]] = downstream_nodes

        synapses_at_root = self.synapses.get(self.arbor.root, [])
        if synapses_at_root:
            for treenode_id in distances:
                these_distances = distances[treenode_id]
                for _ in synapses_at_root:
                    distance = self.distances_to_root[treenode_id]
                    if distance < max_distance:
                        these_distances.append(distance)

        return distances

    def cluster_sizes(self, density_hill_map) -> Dict[int, int]:
        """Density hill ID to number of treenodes labelled"""
        return {key: len(value) for key, value in self.clusters(density_hill_map).items()}

    def cluster_maps(self, density_hill_map) -> Dict[int, Dict[int, bool]]:
        """Density hill ID to dict of treenode ID to True"""
        return {key: {ikey: True for ikey in value} for key, value in self.clusters(density_hill_map).items()}

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
        densities = calculate_densities(self.distances, self.lambda_**2)
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
            delta_densities = {n_id: densities[n_id] - self_density for n_id in neighbour_ids}

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
            density_hill_map[node_id] = density_hill_map[max(
                neighbour_ids,
                key=lambda n: delta_densities[n] / abs(self.distances_to_root[n] - distance_to_current)
            )]

        return density_hill_map

    def _density_hill_map_classic(self) -> Dict[int, int]:
        raise NotImplementedError('implementation not correct')
        edges = self.arbor.edges
        density_hill_map = dict()
        density = calculate_densities(self.distances, self.lambda_**2)

        max_density_index = 0
        all_neighbours = self.arbor.all_neighbours()

        # root belongs to cluster 0
        density_hill_map[self.arbor.root] = 0

        # longest to shortest ensures working on each density hill from one direction
        for partition in reversed(self.partitions):
            idx = len(partition) - 1
            # branches' children will already have an index; root's will not
            try:
                density_hill_index = density_hill_map[partition[idx-1]]
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
                    key=lambda n: delta_density[n] / abs(self.distances_to_root[n] - distance_to_current)
                )

                steepest = density_hill_map[steepest_neighbour]
                density_hill_map[treenode_id] = steepest
                for neighbour in neighbours:
                    if delta_density[neighbour] < 0:
                        density_hill_map[neighbour] = steepest

                density_hill_index = density_hill_map[partition[idx-1]]

        return density_hill_map

    def segregation_index(self, clusters: Dict[int, List[int]], outputs: Dict[int, int], inputs: Dict[int, int]):
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
            math.exp(-math.pow(distance, 2) / lambda_sq)
            for distance in these_distances
        )
    return density