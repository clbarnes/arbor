from __future__ import annotations

from dataclasses import dataclass
from numbers import Number

import itertools
from abc import ABCMeta, abstractmethod
from collections import defaultdict, Counter
from functools import lru_cache
from itertools import tee
from typing import (
    Optional,
    Callable,
    Dict,
    Sequence,
    List,
    Iterable,
    Tuple,
    Set,
    NamedTuple,
)

import networkx as nx
import numpy as np


from arbor.common import euclidean_distance


@dataclass
class NodesDistanceTo:
    distances: Dict[int, float]

    @property
    @lru_cache(1)
    def max(self):
        return max(self.distances.values())

    def to_dict(self):
        return {
            "distances": self.distances,
            "max": self.max,
        }


class BranchAndEndNodes(NamedTuple):
    branches: Dict[int, int]
    ends: Set[int]

    @property
    def n_branches(self):
        return len(self.branches)

    def to_dict(self):
        return {
            "branches": self.branches,
            "ends": self.ends,
            "n_branches": self.n_branches
        }


@dataclass
class FlowCentrality:
    centrifugal: int
    centripetal: int

    @property
    @lru_cache(1)
    def sum(self):
        return self.centrifugal + self.centripetal

    def to_dict(self):
        return {
            "centrifugal": self.centrifugal,
            "centripetal": self.centripetal,
            "sum": self.sum
        }


class BaseArbor(metaclass=ABCMeta):
    def to_dict(self):
        return {'root': self.root, 'edges': self.edges}

    def _is_valid(self):
        if self.edges:
            assert self.root is not None, 'Edges exist but root not set'

        proximo_distal = nx.DiGraph()
        proximo_distal.add_node(self.root)
        for distal, proximal in self.edges.items():
            assert isinstance(distal, int), f'Node {distal} is not an integer'
            assert isinstance(proximal, int), f'Node {proximal} is not an integer'
            proximo_distal.add_edge(proximal, distal)

        assert nx.is_directed_acyclic_graph(proximo_distal), "Arbor is not a DAG"

        in_degree = dict(proximo_distal.in_degree)

        roots = [n for n, d in in_degree.items() if d == 0]
        assert len(roots) == 1, f"Arbor has {len(roots)} roots: {roots}"
        assert self.root == roots[0], f"Explicit root ({self.root}) is not implicit root ({roots[0]})"

        assert all(d <= 1 for d in in_degree.values()), f"Some nodes have more than one proximal neighbour"

    @abstractmethod
    def find_root(self) -> Optional[int]:
        pass

    @abstractmethod
    def add_edges(
        self, edges: Iterable[int], accessor: Optional[Callable[[int, int], int]] = None
    ) -> BaseArbor:
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
        pass

    @abstractmethod
    def add_edge_pairs(self, *child_parent_ids: Tuple[int, int]) -> BaseArbor:
        """
        Add any number of edge pairs to the arbor.

        Parameters
        ----------
        child_parent_ids : (child_id, parent_id) pairs

        Returns
        -------
        self
        """
        pass

    @abstractmethod
    def add_path(self, path: Sequence[int]) -> BaseArbor:
        """
        Add path of nodes to the arbor from root towards leaf.

        Assumes new path intersects with existing nodes at exactly one point.

        Reroots the arbor at the start of the path unless that node already has a parent.

        Parameters
        ----------
        path : sequence
            A 1D sequence of node IDs where every node is the parent of its successor

        Returns
        -------
        self
        """
        pass

    @abstractmethod
    def reroot(self, new_root: int) -> BaseArbor:
        pass

    @abstractmethod
    def nodes_distance_to(
        self,
        root: Optional[int] = None,
        distance_fn: Optional[Callable] = None,
        location_dict: Optional[Dict[int, np.ndarray]] = None,
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
        pass

    @abstractmethod
    def all_successors(self) -> Dict[int, List[int]]:
        """
        Returns
        -------
        dict
            Dict of parent node IDs to a list of their children (end nodes have an empty list)
        """
        pass

    @abstractmethod
    def partition(self) -> List[List[int]]:
        """
        Return a list of paths where each path starts with a leaf node and ends with either the root node,
        or a node which exists in another path.
        Nodes which are not junctions should only appear once across all paths.

        Returns
        -------
        List of lists of node IDs
        """
        pass

    @abstractmethod
    def children_list(self) -> List[int]:
        """Return list of non-root nodes"""
        pass

    @abstractmethod
    def find_branch_and_end_nodes(self) -> BranchAndEndNodes:
        """

        Returns
        -------
        BranchAndEndNodes
            branches: dict of branch node ID -> count of leaf nodes downstream of this node
            ends: list of leaf node IDs
        """
        pass

    def partition_sorted(self):
        return sorted(self.partition(), key=len)

    @abstractmethod
    def all_neighbours(self) -> Dict[int, List[int]]:
        pass

    def flow_centrality(
        self, targets: Dict[int, int], sources: Dict[int, int]
    ) -> Optional[Dict[int, FlowCentrality]]:
        """
        Calculate the flow centrality for each treenode:
        i.e. the number of paths from all input, output pairs which go through it

        Parameters
        ----------
        targets : dict of treenode ID to number of synapses it is presynaptic to
        sources : dict of treenode ID to number of synapses it is postsynaptic to

        Returns
        -------
        dict of treenode ID to its flow centrality
        """
        total_sources = sum(sources.values())
        total_targets = sum(targets.values())
        if total_sources == 0 or total_targets == 0:
            return None

        partitions = self.partition_sorted()
        counts = dict()  # node: {"seen_src": int, "seen_tgt": int}
        centrality = dict()

        for partition in partitions:
            seen_src = 0
            seen_tgt = 0

            for idx, node_id in enumerate(partition):
                if node_id == 3:
                    a = 1 + 1
                these_counts = counts.get(node_id)
                if these_counts is None:
                    seen_src += sources.get(node_id, 0)
                    seen_tgt += targets.get(node_id, 0)
                    if idx == len(partition) - 1:
                        # node is the root, or a branch being addressed for the first time
                        counts[node_id] = {"seen_src": seen_src, "seen_tgt": seen_tgt}
                else:
                    # node is a branch point which has been addressed before
                    seen_src += these_counts["seen_src"]
                    seen_tgt += these_counts["seen_tgt"]
                    these_counts["seen_src"] = seen_src
                    these_counts["seen_tgt"] = seen_tgt

                centrality[node_id] = FlowCentrality(
                    centrifugal=seen_src * (total_targets - seen_tgt),
                    centripetal=seen_tgt * (total_sources - seen_src),
                )

        return centrality

    def nodes_list(self) -> List[int]:
        """List all nodes present"""
        out = list(self.edges)
        if self.root:
            out.append(self.root)
        return out

    def nodes_order_from(self, root: Optional[int] = None) -> Dict[int, Number]:
        """Find graph topological distance from all nodes to given node (default root)"""
        return self.nodes_distance_to(root or self.root).distances

    def sub_arbor(self, new_root: int) -> BaseArbor:
        """Return a new arbor which is a shallow copy of this arbor, starting at the given node"""
        pass


class ArborNX(BaseArbor):
    def __init__(self):
        self._root = None

        # DiGraph with edges pointing towards root
        self._disto_proximal = nx.OrderedDiGraph()

        # views of disto_proximal
        self._undirected = self._disto_proximal.to_undirected(as_view=True)
        self._proximo_distal = self._disto_proximal.reverse(copy=False)

        self._length_key = "_length"

    def _invalidate_cache(self):
        self._edges = None
        self._undirected = None

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, value):
        self._disto_proximal.add_node(value)
        self._root = value

    @property
    def edges(self):
        # generator is necessary due to networkx weirdness
        return dict(kv for kv in self._disto_proximal.edges())

    def find_root(self) -> Optional[int]:
        for node_id, degree in self._disto_proximal.out_degree:
            if degree == 0:
                return node_id

    def add_edges(
        self, edges: Iterable[int], accessor: Optional[Callable[[int, int], int]] = None
    ):
        if not accessor:

            def accessor(node_id, idx):
                return node_id

        return self.add_edge_pairs(
            *(
                [accessor(*idx_id_pair[::-1]) for idx_id_pair in child_parent_idx_id]
                for child_parent_idx_id in zip(*[iter(enumerate(edges))] * 2)
            )
        )

    def add_edge_pairs(self, *distal_proximal_ids: Tuple[int, int]) -> BaseArbor:
        self._disto_proximal.add_edges_from(distal_proximal_ids)
        self.root = self.find_root()
        return self

    def add_path(self, path: Sequence[int]) -> BaseArbor:
        distal_iter, proximal_iter = tee(path)
        possible_root = next(distal_iter, None)

        self._disto_proximal.add_edges_from(zip(distal_iter, proximal_iter))

        if self._disto_proximal.out_degree(possible_root) == 0:
            self.reroot(possible_root)

        return self

    def reroot(self, new_root: int) -> BaseArbor:
        if self.root is not None:
            srcs, tgts = tee(nx.shortest_path(self._undirected, self.root, new_root))
            next(tgts)
            for src, tgt in zip(srcs, tgts):
                try:
                    self._disto_proximal.remove_edge(tgt, src)
                    self._disto_proximal.add_edge(src, tgt)
                except nx.NetworkXError:
                    break
        self.root = new_root
        return self

    def _populate_edge_length(
        self,
        distance_fn: Optional[Callable] = None,
        location_dict: Optional[Dict[int, np.ndarray]] = None,
    ):
        if distance_fn is None:
            if location_dict is None:

                def distance_fn(node1, node2):
                    return 1

            else:

                def distance_fn(node1, node2):
                    return euclidean_distance(
                        location_dict[node1], location_dict[node2]
                    )

        for src, tgt in self._disto_proximal.edges:
            self._disto_proximal.edges[src, tgt][self._length_key] = distance_fn(
                src, tgt
            )

    def nodes_distance_to(
        self,
        root: Optional[int] = None,
        distance_fn: Optional[Callable] = None,
        location_dict: Optional[Dict[int, np.ndarray]] = None,
    ) -> NodesDistanceTo:
        # todo: may need to constrain to only look at disto-proximal paths
        if root is None:
            root = self.root

        self._populate_edge_length(distance_fn, location_dict)

        return NodesDistanceTo(
            nx.shortest_path_length(self._undirected, target=root, weight="_length")
        )

    def _all_distances(
        self,
        distance_fn: Optional[Callable] = None,
        location_dict: Optional[Dict[int, np.ndarray]] = None,
    ) -> Dict[int, Dict[int, float]]:
        self._populate_edge_length(distance_fn, location_dict)
        return nx.shortest_path_length(self._undirected, weight=self._length_key)

    def all_successors(self) -> Dict[int, List[int]]:
        return {
            n: list(self._proximo_distal.successors(n))
            for n in self._proximo_distal.nodes
        }

    def partition(self) -> List[List[int]]:
        branches, ends, paths_to_root = self._branches_ends_paths()
        visited = set()
        partitions = []
        for end in sorted(ends):
            partition = []
            for node in paths_to_root[end]:
                partition.append(node)
                if node in visited:
                    break
                visited.add(node)
            partitions.append(partition)
        return partitions

    def children_list(self) -> List[int]:
        return [n for n in self._disto_proximal.nodes if n != self.root]

    def _branches_ends_paths(self):
        branches = set()
        ends = set()
        for node_id, degree in self._disto_proximal.in_degree:
            if degree == 0:
                ends.add(node_id)
            elif degree > 1:
                branches.add(node_id)

        paths_to_root = nx.single_target_shortest_path(self._disto_proximal, self.root)

        return branches, ends, paths_to_root

    def find_branch_and_end_nodes(self) -> BranchAndEndNodes:
        branches, ends, paths_to_root = self._branches_ends_paths()
        visitations = Counter(
            itertools.chain.from_iterable(paths_to_root[end] for end in ends)
        )
        branches = {branch: visitations[branch] for branch in branches}
        return BranchAndEndNodes(branches, ends)

    def all_neighbours(self):
        out = defaultdict(list)
        if self.root:
            out[self.root] = []
        for src, tgt in self._undirected.edges:
            out[src].append(tgt)
            out[tgt].append(src)
        return dict(out)

    def nodes_list(self):
        return list(self._disto_proximal.nodes)

    def sub_arbor(self, new_root: int) -> BaseArbor:
        sub = ArborNX()
        sub.root = new_root
        to_visit = [new_root]
        edge_pairs = []
        while to_visit:
            proximal = to_visit.pop()
            for distal in self._proximo_distal.successors(proximal):
                edge_pairs.append((distal, proximal))
                to_visit.append(distal)
        sub.add_edge_pairs(*edge_pairs)
        return sub


class ArborClassic(BaseArbor):
    """
    Fairly honest reimplementation of Arbor.js, see
    https://github.com/catmaid/CATMAID/blob/master/django/applications/catmaid/static/libs/catmaid/Arbor.js

    Should be refactored to use networkx.
    """

    def all_neighbours(self) -> Dict[int, List[int]]:
        raise NotImplementedError()

    def __init__(self):
        self.root = None

        # mapping of child ID to parent ID
        self.edges = dict()

    def find_root(self) -> int:
        for parent_id in self.edges.values():
            if parent_id not in self.edges:
                return parent_id

    def add_edges(
        self, edges: Iterable[int], accessor: Optional[Callable[[int, int], int]] = None
    ):
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

        return self.add_edge_pairs(
            *(
                [accessor(*idx_id_pair[::-1]) for idx_id_pair in child_parent_idx_id]
                for child_parent_idx_id in zip(*[iter(enumerate(edges))] * 2)
            )
        )

    def add_edge_pairs(self, *child_parent_ids: Tuple[int, int]):
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

    def add_path(self, path: Sequence[int]):
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
        root: Optional[int] = None,
        distance_fn: Optional[Callable] = None,
        location_dict: Optional[Dict[int, np.ndarray]] = None,
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
                    return euclidean_distance(
                        location_dict[node1], location_dict[node2]
                    )

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
                    open_.append(
                        (successor_id, dist + distance_fn(successor_id, parent_id))
                    )

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
        branches_ends = self.find_branch_and_end_nodes()
        partitions = []
        junctions = dict()

        # sort for deterministic result
        open_ = [[n] for n in sorted(branches_ends.ends)]

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
                n_successors = branches_ends.branches.get(parent_id)
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
                            enumerate(junction), key=lambda x: len(x[1])
                        )
                        for idx, item in enumerate(junction):
                            if idx == longest_idx:
                                open_.append(item)
                            else:
                                partitions.append(item)

        assert len(partitions) == len(branches_ends.ends)
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


Arbor = ArborNX
