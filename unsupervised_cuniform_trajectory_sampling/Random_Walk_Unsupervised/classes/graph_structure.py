#################################################### Define the Node and Edge Structure ####################################################
class Node:
    _id_counter = 1  # Class-level variable to track the number of nodes created
    def __init__(self, level_set, point):
        """
        Represents a node in the graph.

        Args:
            level_set (int): The level set index e.g. (0 to 15).
            point (tuple): The state of the reachable point (e.g., (x, y, theta)).
        """
        self.level_set = level_set
        self.point = point  # Tuple (x, y, theta)
        self.id = Node._id_counter  # Unique integer ID for the node
        Node._id_counter += 1  # Increment the global counter for the next node
        self.edges = []  # Edges to other nodes 
    def __repr__(self):
        return f"Node(ID={self.id}, Level={self.level_set}, Point={self.point})"

class Edge:
    def __init__(self, from_node, to_node, action_index=None):
        """
        Represents an edge in the graph.

        Args:
            from_node (Node): The starting node of the edge.
            to_node (Node): The ending node of the edge.
            action_index (int): The index of the action that led to this transition.
        """
        self.from_node = from_node
        self.to_node = to_node
        self.flow = 0  # Flow through the edge (initialized to 0)
        self.action_index = action_index # the action_index that lead the from_node to the to_node

    def __repr__(self):
        to_node_id = self.to_node.id if self.to_node is not None else 'None'
        return f"Edge(From={self.from_node.id}, To={to_node_id}, Flow={self.flow}, Action_index={self.action_index})"
