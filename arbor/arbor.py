from __future__ import annotations
from collections import defaultdict
from functools import lru_cache
from itertools import tee
from typing import (
    Optional, Callable, Dict, Sequence, List, NamedTuple, Iterable, Tuple, Set
)

import numpy as np


from arbor.common import euclidean_distance


class NodesDistanceTo(NamedTuple):
    distances: Dict[int, float]

    @property
    @lru_cache(1)
    def max(self):
        return max(self.distances.values())


class BranchAndEndNodes(NamedTuple):
    branches: Dict[int, int]
    ends: Set[int]

    @property
    def n_branches(self):
        return len(self.branches)


class Arbor:
    """
    Fairly honest reimplementation of Arbor.js, see
    https://github.com/catmaid/CATMAID/blob/master/django/applications/catmaid/static/libs/catmaid/Arbor.js

    Should be refactored to use networkx.
    """

    def __init__(self):
        self.root = None

        # mapping of child ID to parent ID
        self.edges = dict()

    def find_root(self) -> int:
        for parent_id in self.edges.values():
            if parent_id not in self.edges:
                return parent_id

    def add_edges(self, edges: Iterable[int], accessor: Optional[Callable[[int, int], int]]=None) -> Arbor:
        """
        Add a flattened sequence of edges to the arbor.

        Parameters
        ----------
        edges : sequence
            A 1D sequence of nodes which is a flattened list of (child, parent) pairs
        accessor : callable
            Function which takes 2 arguments; node ID and index in the given edges sequence

        Returns
        -------
        self
        """
        if not accessor:
            def accessor(node_id, idx):
                return node_id

        return self.add_edge_pairs(*(
            [accessor(*idx_id_pair[::-1]) for idx_id_pair in child_parent_idx_id]
            for child_parent_idx_id in zip(*[iter(enumerate(edges))] * 2)
        ))

    def add_edge_pairs(self, *child_parent_ids: Tuple[int, int]) -> Arbor:
        """
        Add any number of edge pairs to the arbor.

        Parameters
        ----------
        child_parent_ids : (child_id, parent_id) pairs

        Returns
        -------
        self
        """
        for child_id, parent_id in child_parent_ids:
            self.edges[child_id] = parent_id

        self.root = self.find_root()

        return self

    def add_path(self, path: Sequence[int]) -> Arbor:
        """
        Add path of nodes to the arbor from root towards leaf.

        Assumes new path intersects with existing nodes.

        Reroots the arbor at the start of the path unless that node already has a parent.

        Parameters
        ----------
        path : sequence
            A 1D sequence of node IDs where every node is the parent of its successor

        Returns
        -------
        self
        """
        parent_iter, child_iter = tee(path)
        possible_root = next(child_iter, None)

        has_rerooted = False

        for parent_id, child_id in zip(parent_iter, child_iter):
            if child_id in self.edges and not has_rerooted:
                self.reroot(child_id)
                has_rerooted = True
            self.edges[child_id] = parent_id

        if possible_root not in self.edges:
            self.root = possible_root

        return self

    def reroot(self, new_root: int) -> Arbor:
        if self.root is None:
            self.root = new_root
        if new_root == self.root:
            return self

        path = [new_root]
        parent_id = self.edges.get(new_root)

        while parent_id:
            del self.edges[path[-1]]
            path.append(parent_id)
            parent_id = self.edges.get(parent_id)

        if len(path) > 1:
            return self.add_path(path)
        else:
            return self

    def nodes_distance_to(
            self,
            root: Optional[int]=None,
            distance_fn: Optional[Callable]=None,
            location_dict: Optional[Dict[int, np.ndarray]]=None
    ) -> NodesDistanceTo:
        """

        Parameters
        ----------
        root : int (optional)
            Root node ID. If none, use existing root.
        distance_fn : callable (optional)
            Function which takes two node IDs and returns the distance between them.
            Only ever called on adjacent nodes.
            If ``distance_fn`` is ``None``, and ``location_dict`` is not, the distance function is euclidean distance
            between locations given in ``location_dict``.
            If both are ``None``, unweighted path length is used.
        location_dict : dict (optional)
            Dict of node ID to location (3-array)

        Returns
        -------
        NodesDistanceTo
            keys are "distances", a dict of node IDs to their weighted path length from the root,
            and "max", the longest of those weighted path lengths
        """
        if root is None:
            root = self.root

        if distance_fn is None:
            if location_dict is None:
                def distance_fn(node1, node2):
                    return 1
            else:
                def distance_fn(node1, node2):
                    return euclidean_distance(location_dict[node1], location_dict[node2])

        distances = dict()

        if root is None:
            return NodesDistanceTo(distances)

        successors = self.all_successors()
        open_ = [(root, 0)]

        while open_:
            parent_id, dist = open_.pop(0)
            distances[parent_id] = dist
            successor_ids = successors[parent_id]

            while len(successor_ids) == 1:
                child_id = successor_ids[0]
                dist += distance_fn(child_id, parent_id)
                distances[child_id] = dist
                parent_id = child_id
                successor_ids = successors[parent_id]

            if successor_ids:
                for successor_id in successor_ids:
                    open_.append((successor_id, dist + distance_fn(successor_id, parent_id)))

        return NodesDistanceTo(distances)

    def all_successors(self) -> Dict[int, List[int]]:
        """
        Returns
        -------
        dict
            Dict of parent node IDs to a list of their children (end nodes have an empty list)
        """
        successors = defaultdict(list)
        for child_id, parent_id in self.edges.items():
            successors[parent_id].append(child_id)

            # ensure end nodes have empty lists
            if child_id not in successors:
                successors[child_id] = []

        return dict(successors)

    def partition(self) -> List[List[int]]:
        """
        Return a list of paths where each path starts with a leaf node and ends with either the root node,
        or a node which exists in another path.
        Nodes which are not junctions should only appear once across all paths.

        Returns
        -------
        List of lists of node IDs
        """
        branches, ends = self.find_branch_and_end_nodes()
        partitions = []
        junctions = dict()

        # sort for deterministic result
        open_ = [[n] for n in sorted(ends)]

        while open_:
            seq = open_.pop(0)
            node_id = seq[-1]
            n_successors = None
            parent_id = None

            while n_successors is None:
                parent_id = self.edges.get(node_id)
                if parent_id is None:
                    break
                seq.append(parent_id)
                n_successors = branches.get(parent_id)
                node_id = parent_id

            if parent_id is None:
                partitions.append(seq)
            else:
                junction = junctions.get(node_id)
                if junction is None:
                    junctions[node_id] = [seq]
                else:
                    junction.append(seq)
                    if len(junction) == n_successors:
                        longest_idx, longest_item = max(
                            enumerate(junction),
                            key=lambda x: len(x[1])
                        )
                        for idx, item in enumerate(junction):
                            if idx == longest_idx:
                                open_.append(item)
                            else:
                                partitions.append(item)

        assert len(partitions) == len(ends)
        return partitions

    def children_list(self) -> List[int]:
        """Return list of non-root nodes"""
        return list(self.edges)

    def find_branch_and_end_nodes(self) -> BranchAndEndNodes:
        """

        Returns
        -------
        BranchAndEndNodes
            branches: dict of branch node ID -> count of leaf nodes downstream of this node
            ends: list of leaf node IDs
        """
        children = self.children_list()
        parents = set()
        branches = dict()

        for child_id in children:
            parent_id = self.edges[child_id]
            if parent_id in parents:
                count = branches.get(parent_id)
                if count is None:
                    branches[parent_id] = 2
                else:
                    branches[parent_id] = count + 1
            else:
                parents.add(parent_id)

        ends = {child_id for child_id in children if child_id not in parents}
        if not children and (self.root is not None):
            ends.add(self.root)

        return BranchAndEndNodes(branches, ends)

    def partition_sorted(self):
        return sorted(self.partition(), key=len)
