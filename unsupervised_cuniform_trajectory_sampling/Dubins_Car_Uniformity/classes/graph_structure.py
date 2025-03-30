#################################################### Define the Node and Edge Structure ####################################################
class Node:
    __slots__ = ('level_set', 'point', 'id', 'outgoing_arcs')  # Define slots to save memory
    _id_counter = 1  # Class-level variable to track the number of nodes created
    def __init__(self, level_set, point, outgoing_arcs=None):
        """
        Represents a node in the graph.
        Args:
            level_set (int): The level set index e.g. (0 to 15).
            point (tuple): The state of the reachable point (e.g., (x, y, theta)).
            outgoing_arcs (list, optional): A list of arcs (to_node_id, flow) connecting this node to others.
        """
        self.level_set = level_set
        self.point = point  # Tuple (x, y, theta)
        self.id = Node._id_counter  # Unique integer ID for the node
        Node._id_counter += 1  # Increment the global counter for the next node

        # connected nodes to the next level set, list of tuples (to_node_id, flow) 
        self.outgoing_arcs = outgoing_arcs if outgoing_arcs is not None else []  # Initialize to a new list if None is passed

    def __repr__(self):
        return f"Node(ID={self.id}, Level={self.level_set}, Point={self.point})"