import numpy as np
import networkx as nx
import heapq
from sklearn.metrics.pairwise import euclidean_distances
import random

class HNSW:
    """
    Hierarchical Navigable Small World (HNSW) implementation.
    """
    
    def __init__(self, n_layers=3, k_values=None, m_l=1.0, ef_construction=50, random_seed=None):
        """
        Initialize the HNSW structure.
        
        Parameters:
        -----------
        n_layers : int
            Number of layers in the hierarchical graph
        k_values : list or None
            List of k values for each layer (how many neighbors to connect to)
            If None, defaults to [8, 4, 2] for bottom to top layers
        m_l : float
            Normalization factor for layer selection probability
        ef_construction : int
            Size of the dynamic candidate list during construction
        random_seed : int or None
            Random seed for reproducibility
        """
        self.n_layers = n_layers
        self.k_values = k_values if k_values is not None else [8, 4, 2][:n_layers]
        self.k_values = self.k_values[:n_layers]  # Ensure we have the right number of k values
        self.m_l = m_l
        self.ef_construction = ef_construction
        
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        
        # Create empty graphs for each layer
        self.graphs = [nx.Graph() for _ in range(n_layers)]
        
        # Store points for distance calculations
        self.points = []
        
        # For callbacks during construction (useful for visualization)
        self.callbacks = {}
    
    def distance(self, idx1, idx2):
        """Calculate Euclidean distance between two points by their indices."""
        return np.linalg.norm(self.points[idx1] - self.points[idx2])
    
    def distance_to_query(self, query_point, idx):
        """Calculate Euclidean distance from a query point to a point in the dataset."""
        return np.linalg.norm(query_point - self.points[idx])
    
    def register_callback(self, event_name, callback_func):
        """Register a callback function for a specific event."""
        self.callbacks[event_name] = callback_func
    
    def _trigger_callback(self, event_name, **kwargs):
        """Trigger a registered callback with the given arguments."""
        if event_name in self.callbacks:
            self.callbacks[event_name](**kwargs)
    
    def _select_top_layer(self):
        """Select the top layer for a new point based on a probabilistic distribution."""
        return int(np.clip(
            np.floor(-np.log(np.random.uniform(0, 1)) * self.m_l),
            a_min=0,
            a_max=self.n_layers - 1
        ))
    
    def _get_neighbors_from_layer_below(self, node_idx, layer):
        """Get all neighbors of a node from the layer below."""
        if layer <= 0 or layer >= self.n_layers:
            return []
        
        neighbors = []
        # Get all neighbors from the layer below
        lower_graph = self.graphs[layer - 1]
        if node_idx in lower_graph:
            neighbors = list(lower_graph.neighbors(node_idx))
        
        return neighbors
    
    def search_layer(self, G, query_point, entry_points, ef, layer=None):
        """
        Search for nearest neighbors in a single layer using beam search.
        Implementation based on the provided approach.
        
        Parameters:
        -----------
        G : networkx.Graph
            The graph representing the current layer
        query_point : numpy.ndarray
            The query point coordinates
        entry_points : list
            List of entry points to start the search from
        ef : int
            Size of the dynamic candidate list
        layer : int or None
            Current layer index (for visualization callbacks)
            
        Returns:
        --------
        list
            List of nearest neighbors sorted by distance
        """
        # Trigger search_started callback
        self._trigger_callback('search_started', 
                              query_point=query_point, 
                              entry_points=entry_points,
                              layer=layer)
        
        # Set of visited elements
        v = set(entry_points)
        
        # Priority queue of candidates (min heap by distance)
        C = [(self.distance_to_query(query_point, ep), ep) for ep in entry_points]
        heapq.heapify(C)
        
        # Dynamic list of found nearest neighbors (max heap by negative distance)
        W = [(-self.distance_to_query(query_point, ep), ep) for ep in entry_points]
        heapq.heapify(W)
        
        # Trigger search_state callback for initial state
        self._trigger_callback('search_state',
                              query_point=query_point,
                              candidates=C.copy(),
                              nearest=W.copy(),
                              visited=v.copy(),
                              layer=layer)
        
        # Main search loop
        while C:
            # Get closest candidate
            c_dist, c = heapq.heappop(C)
            
            # Get furthest distance in our nearest neighbors list
            f_dist = -W[0][0] if len(W) > 0 else float('inf')
            
            # Uncommented early stopping condition (can be turned off if needed)
            if c_dist > f_dist and len(W) >= ef:
                break  # all elements in W are evaluated
            
            # Check all neighbors of the current node
            for e in G.neighbors(c):
                if e not in v:
                    v.add(e)
                    e_dist = self.distance_to_query(query_point, e)
                    f_dist = -W[0][0] if len(W) > 0 else float('inf')
                    
                    # If we haven't found enough nearest neighbors yet, or if this neighbor
                    # is closer than the furthest nearest neighbor found so far
                    if e_dist < f_dist or len(W) < ef:
                        # Add to candidates
                        heapq.heappush(C, (e_dist, e))
                        
                        # Add to nearest neighbors
                        heapq.heappush(W, (-e_dist, e))
                        
                        # If we have more than ef nearest neighbors, remove the furthest
                        if len(W) > ef:
                            heapq.heappop(W)
                        
                    # Trigger search_state callback for the updated state
                    self._trigger_callback('search_state',
                                          query_point=query_point,
                                          candidates=C.copy(),
                                          nearest=W.copy(),
                                          visited=v.copy(),
                                          current_node=c,
                                          neighbor=e,
                                          layer=layer)
        
        # Convert nearest to a list sorted by distance
        nearest_neighbors = [idx for _, idx in sorted(
            [(self.distance_to_query(query_point, node), node) for _, node in W],
            key=lambda x: x[0]
        )]
        
        # Trigger search_completed callback
        self._trigger_callback('search_completed',
                              query_point=query_point,
                              nearest_neighbors=nearest_neighbors,
                              layer=layer)
        
        return nearest_neighbors
    
    def find_nearest_neighbors(self, query_point, k=1,
                               ef_search=None, entry_point=None):
        """
        Find k nearest neighbors to a query point using the HNSW structure.
        
        Parameters:
        -----------
        query_point : numpy.ndarray
            The query point coordinates
        k : int
            Number of nearest neighbors to find
        ef_search : int or None
            Size of the dynamic candidate list during search
            If None, uses ef_construction
        
        Returns:
        --------
        list
            List of k nearest neighbor indices
        """
        if len(self.points) == 0:
            return []
        
        ef_search = ef_search if ef_search is not None else self.ef_construction
        ef_search = max(ef_search, k)
        
        top_layer = self.n_layers - 1
        while top_layer >= 0 and len(self.graphs[top_layer].nodes()) == 0:
            top_layer -= 1
        
        if top_layer < 0:
            return []
        
        # Select entry point - either user-specified or a random node
        if entry_point is not None:
            # Find the highest layer containing the entry point
            entry_layer = top_layer
            while entry_layer >= 0 and entry_point not in self.graphs[entry_layer].nodes():
                entry_layer -= 1
            
            if entry_layer >= 0:
                # Use specified entry point if it exists in the graph
                top_layer = entry_layer
                entry_points = [entry_point]
            else:
                # Fall back to random entry point if specified one doesn't exist
                entry_points = [random.choice(list(self.graphs[top_layer].nodes()))]
        else:
            # Use random entry point (original behavior)
            entry_points = [random.choice(list(self.graphs[top_layer].nodes()))]
        
        # Trigger search_process_started callback
        self._trigger_callback('search_process_started',
                              query_point=query_point,
                              top_layer=top_layer,
                              entry_points=entry_points)
        
        # Search from top to bottom layer
        for layer in range(top_layer, -1, -1):
            G = self.graphs[layer]
            ef = ef_search
            entry_points = self.search_layer(G, query_point, entry_points, ef, layer)
        
        # Trigger search_process_completed callback
        self._trigger_callback('search_process_completed',
                              query_point=query_point,
                              nearest_neighbors=entry_points[:k])
        
        # Return the k nearest neighbors from the bottom layer
        return entry_points[:k]
    
    def add_point(self, point):
        """
        Add a new point to the HNSW structure.
        
        Parameters:
        -----------
        point : numpy.ndarray
            The point coordinates to add
        
        Returns:
        --------
        int
            Index of the added point
        """
        # Add point to our list of points
        point_idx = len(self.points)
        self.points.append(point)
        
        # For the first point (point_idx = 0), add to all layers
        # Otherwise, select a top layer probabilistically
        if point_idx == 0:
            top_layer = self.n_layers - 1  # Add to all layers
        else:
            top_layer = self._select_top_layer()
        
        # Trigger point_adding callback
        self._trigger_callback('point_adding',
                              point_idx=point_idx,
                              point=point,
                              top_layer=top_layer)
        
        # Add point to each layer up to the top layer
        for layer in range(min(top_layer + 1, self.n_layers)):
            G = self.graphs[layer]
            
            # Add node to this layer
            G.add_node(point_idx, pos=point)
            
            # Find neighbors for this point in the current layer
            neighbors = []
            if point_idx > 0:
                k = min(self.k_values[layer], point_idx)
                
                # Find entry point if we're not the first point in this layer
                if len(G.nodes()) > 1:
                    # Start from a random existing node
                    existing_nodes = [n for n in G.nodes() if n != point_idx]
                    entry_point = random.choice(existing_nodes)
                    
                    # If we have connections in higher layers, use them as entry points
                    if layer < self.n_layers - 1 and layer > 0:
                        higher_neighbors = self._get_neighbors_from_layer_below(point_idx, layer + 1)
                        if higher_neighbors:
                            entry_point = higher_neighbors[0]
                    
                    # Find nearest neighbors using search_layer
                    neighbors = self.search_layer(G, point, [entry_point], self.ef_construction)[:k]
                elif point_idx > 0:
                    # If we're the second point in this layer, connect to the first point
                    neighbors = [0]  # Connect to the first point
                
                # Connect the new point to its neighbors
                for neighbor in neighbors:
                    G.add_edge(point_idx, neighbor)
            
            # Trigger layer_point_added callback
            self._trigger_callback('layer_point_added',
                                  point_idx=point_idx,
                                  layer=layer,
                                  neighbors=neighbors)
        
        # Trigger point_added callback
        self._trigger_callback('point_added',
                              point_idx=point_idx,
                              top_layer=top_layer)
        
        return point_idx
    
    def build_from_points(self, points):
        """
        Build the HNSW structure from a collection of points.
        
        Parameters:
        -----------
        points : numpy.ndarray
            Array of shape (n, d) containing n points in d-dimensional space
        """
        # Initialize with empty list of points
        self.points = []
        
        # Clear existing graphs
        self.graphs = [nx.Graph() for _ in range(self.n_layers)]
        
        # Trigger build_started callback
        self._trigger_callback('build_started',
                              n_points=len(points),
                              n_layers=self.n_layers)
        
        # Add each point one by one
        for i, point in enumerate(points):
            self.add_point(point)
        
        # Trigger build_completed callback
        self._trigger_callback('build_completed',
                              n_points=len(self.points),
                              n_layers=self.n_layers)
    
    def get_graph(self, layer=0):
        """Get the graph at a specific layer."""
        if layer < 0 or layer >= self.n_layers:
            raise ValueError(f"Layer {layer} does not exist")
        return self.graphs[layer]
    
    def get_graphs(self):
        """Get all graph layers."""
        return self.graphs
    
    def get_points(self):
        """Get all points."""
        return np.array(self.points)

# Example of NSW as a special case of HNSW with a single layer
class NSW(HNSW):
    """
    Navigable Small World (NSW) implementation.
    This is a special case of HNSW with a single layer.
    """
    
    def __init__(self, k=8, ef_construction=50, random_seed=None):
        """
        Initialize the NSW structure.
        
        Parameters:
        -----------
        k : int
            Number of neighbors to connect to each point
        ef_construction : int
            Size of the dynamic candidate list during construction
        random_seed : int or None
            Random seed for reproducibility
        """
        super().__init__(
            n_layers=1, 
            k_values=[k], 
            ef_construction=ef_construction,
            random_seed=random_seed
        )
