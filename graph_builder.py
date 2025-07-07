#!/usr/bin/env python3
import json
from typing import Dict, List, Any, Optional, Set


class Vertex:
    """
    Represents a vertex in the graph with properties from the JSON data.
    """

    def __init__(
        self,
        counter: int,
        node_type: str,
        connections: List[int],
        arguments: List[Any] = None,
        params: Dict[str, Any] = None,
    ):
        self.id = counter
        self.node_type = node_type
        self.connections = sorted(list(set(connections)))
        self.arguments = arguments or []
        self.params = params or {}

        # For traversal
        self.stacking_level = 0

        if len(connections) != len(set(connections)):
            print(f"Duplicate connections found for vertex {counter}: {connections}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the vertex back to a dictionary representation."""
        return {
            "index": self.id,
            "name": self.params.get("name"),
            # "node_type": self.node_type,
            "children": self.connections,
            "arguments": self.arguments,
            # "params": self.params,
        }

    def __str__(self) -> str:
        """String representation of the vertex."""
        # print id, type, name and stacking level
        return (
            f"Vertex(id={self.id}, type={self.node_type}, name={self.params.get('name')}, level={self.stacking_level})"
        )
        # params_str = f", params: {self.params}" if self.params else ""
        # return f"Vertex(id={self.id}, type={self.node_type}{params_str})"

    def __repr__(self) -> str:
        return self.__str__()


class ClusterVertex:
    """
    Represents a cluster of vertices.
    """

    def __init__(self, cluster_idx: int):
        self.cluster_idx = cluster_idx
        self.vertices = []
        self.connections = []
        self.cluster_connections = []

    def add_vertex(self, vertex: Vertex) -> None:
        self.vertices.append(vertex)
        self.connections = sorted(list(set(self.connections + vertex.connections)))

    def add_cluster_connection(self, cluster_idx: int) -> None:
        if cluster_idx == self.cluster_idx:
            return
        self.cluster_connections = sorted(list(set(self.cluster_connections + [cluster_idx])))

    def __str__(self) -> str:
        return f"ClusterVertex(vertices={self.vertices}, connections={self.connections})"

    def __repr__(self) -> str:
        return self.__str__()


class Graph:
    """
    Represents a graph structure built from the JSON data.
    """

    def __init__(self):
        self.vertices: Dict[int, Vertex] = {}
        self.cluster_vertices: List[ClusterVertex] = []

    def add_vertex(self, vertex_data: Dict[str, Any]) -> Vertex:
        """
        Create and add a vertex to the graph from vertex data.
        """
        counter = vertex_data.get("counter")
        node_type = vertex_data.get("node_type")
        connections = vertex_data.get("connections", [])
        arguments = vertex_data.get("arguments", [])
        params = vertex_data.get("params", {})

        vertex = Vertex(counter, node_type, connections, arguments, params)
        self.vertices[counter] = vertex

        return vertex

    def add_cluster_vertex(self, cluster_vertex: ClusterVertex) -> None:
        self.cluster_vertices.append(cluster_vertex)

    def process_stacking_levels(self) -> None:
        vertices = sorted(self.vertices.values(), key=lambda v: v.id)

        curr_stacking_level = 1
        for vertex in vertices:
            if vertex.node_type == "function_end":
                curr_stacking_level -= 1

            vertex.stacking_level = curr_stacking_level

            if vertex.node_type == "function_start":
                curr_stacking_level += 1

        assert curr_stacking_level == 1

    def get_vertex(self, vertex_id: int) -> Optional[Vertex]:
        """Get a vertex by its ID."""
        return self.vertices.get(vertex_id)

    def get_children_vertices(self, vertex_id: int) -> List[Vertex]:
        """Get all vertices that this vertex connects to."""
        vertex = self.get_vertex(vertex_id)
        return [self.vertices[conn_id] for conn_id in vertex.connections]

    def get_parent_vertices(self, vertex_id: int) -> List[Vertex]:
        """Get all vertices that connect to this vertex."""
        return [v for v in self.vertices.values() if vertex_id in v.connections]

    def merge_vertex(self, vertex: Vertex) -> None:
        parents = self.get_parent_vertices(vertex.id)

        # Update parent pointers
        for parent in parents:
            parent_to_child_connections = parent.connections
            # remove vertex.id from parent_to_child_connections
            parent_to_child_connections.remove(vertex.id)
            parent_to_child_connections.extend(vertex.connections)
            parent_to_child_connections = list(set(parent_to_child_connections))
            parent.connections = sorted(parent_to_child_connections)

        self.vertices.pop(vertex.id)
        del vertex

    def clusterize(self) -> None:
        # iterate graph vertices with index
        cluster_idx = -1
        curr_cluster_vertex = None
        for idx, vertex in enumerate(self.vertices.values()):
            assert idx == vertex.id

            if vertex.node_type == "function_start" and vertex.stacking_level == 1:
                cluster_idx += 1
                curr_cluster_vertex = ClusterVertex(cluster_idx)
                curr_cluster_vertex.add_vertex(vertex)
            elif vertex.node_type == "function_end" and vertex.stacking_level == 1:
                curr_cluster_vertex.add_vertex(vertex)
                self.add_cluster_vertex(curr_cluster_vertex)
                curr_cluster_vertex = None
            elif vertex.stacking_level > 1:
                curr_cluster_vertex.add_vertex(vertex)
            else:
                # skip these
                assert (
                    (vertex.stacking_level == 1 and vertex.node_type == "tensor")
                    or (vertex.stacking_level == 1 and vertex.node_type == "buffer")
                    or (vertex.stacking_level == 1 and vertex.node_type == "capture_start")
                    or (vertex.stacking_level == 1 and vertex.node_type == "buffer_deallocate")
                    or (vertex.stacking_level == 1 and vertex.node_type == "capture_end")
                )

    def unify_clusters(self) -> None:
        vertex_to_cluster_map: Dict[int, ClusterVertex] = {}
        for cluster in self.cluster_vertices:
            for vertex in cluster.vertices:
                vertex_to_cluster_map[vertex.id] = cluster

        # Iterate clusters and update connections to point to clusters instead of vertices
        for cluster in self.cluster_vertices:
            for conn_id in cluster.connections:
                if conn_id not in vertex_to_cluster_map:
                    print(f"Connection {conn_id} not found in vertex_to_cluster_map")
                    continue
                cluster.add_cluster_connection(vertex_to_cluster_map[conn_id].cluster_idx)

    def get_clusters(self) -> List[ClusterVertex]:
        return self.cluster_vertices

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Convert the graph back to a list of dictionaries."""
        return [vertex.to_dict() for vertex in self.vertices.values()]

    def fix_backward_links(self, names: List[str]) -> None:
        # Remove all connections from nodes to previous nodes
        for vertex in self.vertices.values():
            if vertex.params["name"] not in names:
                continue
            if len(vertex.connections) == 0:
                continue

            conns = [conn for conn in vertex.connections if conn >= vertex.id]
            vertex.connections = conns

    def fix_links_to_ones(self) -> None:
        for vertex in self.vertices.values():
            for child in self.get_children_vertices(vertex.id):
                if child.params["name"] == "ttnn::ones":
                    vertex.connections.remove(child.id)

    def fix_tensors(self) -> None:
        # Replace each argument starting with "Tensor(" with index of incoming node
        for vertex in self.vertices.values():
            cnt_tensor_args = sum(1 for arg in vertex.arguments if arg.startswith("Tensor("))
            if cnt_tensor_args == 0:
                continue

            parent_vertices = self.get_parent_vertices(vertex.id)
            assert len(parent_vertices) == cnt_tensor_args

            parent_index = 0
            for i, arg in enumerate(vertex.arguments):
                if arg.startswith("Tensor("):
                    vertex.arguments[i] = f"tensor: {parent_vertices[parent_index].id}"
                    parent_index += 1


def load_graph_from_json(json_file_path: str) -> Graph:
    """
    Load a graph from a JSON file.
    """
    with open(json_file_path, "r") as f:
        data = json.load(f)

    graph = Graph()
    for vertex_data in data:
        graph.add_vertex(vertex_data)

    # graph.build_connections()

    graph.process_stacking_levels()

    for vertex in graph.vertices.values():
        # assert vertex.stacking_level > -1
        assert len(vertex.connections) == len(set(vertex.connections))
        # assert len(vertex.connections) > 0

    return graph


def create_simplified_graph_from_clusterized(graph: Graph) -> Graph:
    new_graph = Graph()

    for cluster_vertex in graph.get_clusters():
        first_vertex = cluster_vertex.vertices[0]
        added_vertex = new_graph.add_vertex(
            {
                "counter": cluster_vertex.cluster_idx,
                "node_type": first_vertex.node_type,
                "connections": cluster_vertex.cluster_connections,
                "arguments": first_vertex.arguments,
                "params": first_vertex.params,
            }
        )

        # For some reason, ttnn::ones calls somehow always point to next node in the graph + the correct node where its resulting tensor is consumed - fixing this by taking only the consumption node
        if added_vertex.params["name"] == "ttnn::ones":
            if len(added_vertex.connections) == 2:
                added_vertex.connections = [added_vertex.connections[-1]]
            else:
                # assert False
                pass

    return new_graph


def main():
    import sys

    # Check if input file is provided as argument
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = "dump_formatted.txt"

    # Load graph from JSON
    graph = load_graph_from_json(input_file)

    # Print some basic information about the graph
    print(f"Loaded graph with {len(graph.vertices)} vertices")

    # Clusterize
    graph.clusterize()
    graph.unify_clusters()
    clusters = graph.get_clusters()

    simplified_graph = create_simplified_graph_from_clusterized(graph)

    simplified_graph.fix_backward_links(["ttnn::conv2d", "ttnn::to_memory_config"])
    simplified_graph.fix_links_to_ones()
    simplified_graph.fix_tensors()

    # Output simplified graph if requested
    output_file = "simplified_graph.json"
    with open(output_file, "w") as f:
        json.dump(simplified_graph.to_dict_list(), f, indent=2)
    print(f"Simplified graph written to {output_file}")


if __name__ == "__main__":
    main()
