from collections import defaultdict
from enum import IntEnum
from typing import Dict, List, Tuple, TypeVar

import numpy as np

from arbor.arbor import Arbor
from arbor.common import euclidean_distance


class RelationType(IntEnum):
    PRESYNAPTIC = 0
    POSTSYNAPTIC = 1


ValidRelType = TypeVar('ValidRelType', int, RelationType)


class SynapseClustering:
    """
    A fairly honest reimplementation of synapse_clustering.js, see
    https://github.com/catmaid/CATMAID/blob/master/django/applications/catmaid/static/libs/catmaid/synapse_clustering.js

    Could possibly be refactored to make better use of numpy
    """

    def __init__(
            self,
            arbor: Arbor,
            locations: Dict[int, np.ndarray],
            synapses: Dict[int, List[Tuple[ValidRelType, int]]],
            lambda_: float
    ):
        """

        Parameters
        ----------
        arbor : Arbor
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


