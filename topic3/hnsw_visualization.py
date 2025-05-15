import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from IPython.display import clear_output
import time
from matplotlib.animation import FuncAnimation
from IPython.display import Image, display

class HNSWVisualizer:
    """
    Class for visualizing HNSW graph construction and search processes.
    """
    
    def __init__(self, hnsw, figsize=None, pause_time=1.0, node_size=100,
                 node_color='blue', edge_color='gray', label_offset=0.025):
        """
        Initialize the HNSW visualizer.
        
        Parameters:
        -----------
        hnsw : HNSW
            The HNSW structure to visualize
        figsize : tuple or None
            Figure size for visualization (default adjusts based on number of layers)
        pause_time : float
            Time to pause between visualization steps
        node_size : int
            Size of nodes in visualization
        node_color : str
            Color of nodes
        edge_color : str
            Color of edges
        label_offset : float
            Offset for node labels from the node positions
        """
        self.hnsw = hnsw
        
        # Adjust figure size based on number of layers if not specified
        if figsize is None:
            n_layers = len(hnsw.get_graphs())
            width_per_layer = 6  # Width per subplot
            self.figsize = (width_per_layer * n_layers, 6)
        else:
            self.figsize = figsize
            
        self.pause_time = pause_time
        self.node_size = node_size
        self.node_color = node_color
        self.edge_color = edge_color
        self.label_offset = label_offset
        
        # For tracking current state
        self.current_step = -1
        
        # For search visualization
        self.current_search_query = None
        self.current_search_nearest = []
        
        # For final search results
        self.search_result = None
        
        # For tracking search process
        self.search_states = []
        
        # Register callbacks with the HNSW structure for search tracking
        self.hnsw.register_callback('search_process_started', self.on_search_started)
        self.hnsw.register_callback('search_state', self.on_search_state)
        self.hnsw.register_callback('search_completed', self.on_search_layer_completed)
        self.hnsw.register_callback('search_process_completed', self.on_search_completed)
    
    def on_point_added(self, point_idx, top_layer):
        """Callback for when a point is added to the HNSW structure."""
        self.current_step += 1
        # No visualization here
    
    def on_search_started(self, query_point, entry_points, top_layer=None, layer=None):
        """
        Callback for when a search process starts.
        
        Parameters:
        -----------
        query_point : numpy.ndarray
            The query point coordinates
        entry_points : list
            List of entry points to start the search from
        top_layer : int or None
            The top layer in the search process (from search_process_started)
        layer : int or None
            Current layer index (from search_started)
        """
        self.current_search_query = query_point
        self.current_search_nearest = entry_points
        self.search_states = []  # Reset search states
        
        # Use the appropriate layer value
        current_layer = top_layer if layer is None else layer
        
        # Record initial state
        self.search_states.append({
            'layer': current_layer,
            'query_point': query_point,
            'candidates': [(0, ep) for ep in entry_points],  # Dummy distances
            'nearest': [(-0, ep) for ep in entry_points],    # Dummy distances
            'visited': set(entry_points),
            'current_node': None,
            'neighbor': None,
            'type': 'start'
        })
    
    
    def on_search_state(self, query_point, candidates, nearest, visited,
                       current_node=None, neighbor=None, layer=None):
        """Callback for each search state update."""
        # Record this state
        self.search_states.append({
            'layer': layer,
            'query_point': query_point,
            'candidates': candidates.copy() if candidates else [],
            'nearest': nearest.copy() if nearest else [],
            'visited': visited.copy() if visited else set(),
            'current_node': current_node,
            'neighbor': neighbor,
            'type': 'step'
        })
    
    def on_search_layer_completed(self, query_point, nearest_neighbors, layer):
        """Callback for when a search in a layer is completed."""
        self.current_search_nearest = nearest_neighbors
        
        # Record layer completion
        self.search_states.append({
            'layer': layer,
            'query_point': query_point,
            'candidates': [],
            'nearest': [(-0, n) for n in nearest_neighbors],  # Dummy negative distances
            'visited': set(),
            'current_node': None,
            'neighbor': None,
            'type': 'layer_complete'
        })
    
    def on_search_completed(self, query_point, nearest_neighbors):
        """Callback for when a search process is completed."""
        self.search_result = nearest_neighbors
        
        # Record final state
        self.search_states.append({
            'layer': 0,  # Bottom layer
            'query_point': query_point,
            'candidates': [],
            'nearest': [(-0, n) for n in nearest_neighbors],  # Dummy negative distances
            'visited': set(),
            'current_node': None,
            'neighbor': None,
            'type': 'complete'
        })
    
    def _get_node_positions(self, graph):
        """
        Get node positions for a graph, handling missing positions properly.
        
        Parameters:
        -----------
        graph : networkx.Graph
            The graph to get positions for
            
        Returns:
        --------
        dict
            Dictionary mapping node IDs to their positions
        list
            List of nodes that have valid positions
        """
        # First try to get positions from node attributes
        pos = nx.get_node_attributes(graph, 'pos')
        
        # If we don't have positions from attributes, use the points array
        if not pos:
            points = self.hnsw.get_points()
            # Only include nodes that have a valid index in the points array
            pos = {}
            for node in graph.nodes():
                if isinstance(node, int) and node >= 0 and node < len(points):
                    pos[node] = points[node]
        
        # Get list of nodes that have valid positions
        valid_nodes = list(pos.keys())
        
        return pos, valid_nodes
    
    def create_animation(self, filename='hnsw_construction.gif', fps=2, dpi=100):
        """
        Create an animation from the current state of the HNSW structure.
        
        Parameters:
        -----------
        filename : str
            Output filename for the GIF
        fps : int
            Frames per second in the output GIF
        dpi : int
            Resolution of the output GIF
        """
        # Get current state of the HNSW structure
        graphs = self.hnsw.get_graphs()
        points = self.hnsw.get_points()
        
        if len(points) == 0:
            print("No points in the HNSW structure.")
            return
        
        # Create a new figure for the animation
        n_layers = len(graphs)
        fig, axs = plt.subplots(1, n_layers, figsize=self.figsize, dpi=dpi)
        if n_layers == 1:
            axs = [axs]
        
        for ax in axs:
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True)
        
        # Function to update the animation for each frame
        def update(frame):
            for ax in axs:
                ax.clear()
                ax.set_xlim(-0.05, 1.05)
                ax.set_ylim(-0.05, 1.05)
                ax.grid(True)
            
            fig.suptitle(f"HNSW Graph Construction - Step {frame}")
            
            # For each layer in the graph
            for layer, (G, ax) in enumerate(zip(graphs, axs)):
                # Get nodes up to this frame
                nodes = [n for n in G.nodes() if n <= frame]
                
                if not nodes:
                    ax.set_title(f"Layer {layer}")
                    continue
                
                # Create a subgraph with just these nodes
                subgraph = G.subgraph(nodes)
                
                # Get positions from the points array
                pos = {}
                for node in nodes:
                    if node < len(points):  # Safety check
                        pos[node] = points[node]
                
                if not pos:
                    ax.set_title(f"Layer {layer}")
                    continue
                
                # Draw nodes that have positions
                visible_nodes = list(pos.keys())
                if visible_nodes:
                    nx.draw_networkx_nodes(
                        subgraph.subgraph(visible_nodes), 
                        pos, 
                        ax=ax, 
                        node_color=self.node_color, 
                        node_size=self.node_size
                    )
                
                # Draw edges between nodes that have positions
                edge_list = [(u, v) for u, v in subgraph.edges() if u in pos and v in pos]
                if edge_list:
                    edge_graph = subgraph.edge_subgraph(edge_list)
                    nx.draw_networkx_edges(
                        edge_graph, 
                        pos, 
                        ax=ax, 
                        edge_color=self.edge_color, 
                        alpha=0.7
                    )
                
                # Draw labels for nodes that have positions
                if visible_nodes:
                    labels = {node: str(node) for node in visible_nodes}
                    label_pos = {i: (pos[i] + self.label_offset) for i in visible_nodes}
                    nx.draw_networkx_labels(
                        subgraph.subgraph(visible_nodes), 
                        label_pos, 
                        labels, 
                        ax=ax, 
                        font_size=10
                    )
                
                # If this is the latest added node, highlight it
                if frame in visible_nodes:
                    nx.draw_networkx_nodes(
                        subgraph.subgraph([frame]), 
                        {frame: pos[frame]}, 
                        ax=ax,
                        node_color='red', 
                        node_size=self.node_size*1.5
                    )
                    
                    # Highlight connections to the latest node
                    frame_neighbors = [n for n in G.neighbors(frame) if n in visible_nodes]
                    if frame_neighbors:
                        highlight_edges = [(frame, n) for n in frame_neighbors]
                        highlight_graph = subgraph.edge_subgraph(highlight_edges)
                        nx.draw_networkx_edges(
                            highlight_graph, 
                            pos, 
                            ax=ax, 
                            edge_color='red', 
                            width=2.0, 
                            alpha=1.0
                        )
                
                ax.set_title(f"Layer {layer}")
            
            return axs
        
        # Create the animation
        ani = FuncAnimation(fig, update, frames=len(points), interval=1000//fps, blit=False)
        
        # Save the animation as a GIF
        ani.save(filename, writer='pillow', fps=fps)
        
        # Close the figure to avoid displaying it twice in notebooks
        plt.close(fig)
        
        print(f"Animation saved to {filename}")
        
        # Display the animation in the notebook
        display(Image(filename))
        
        return filename
    
    def create_search_animation(self, filename='hnsw_search.gif', fps=2, dpi=100):
        """
        Create an animation of the search process.
        
        Parameters:
        -----------
        filename : str
            Output filename for the GIF
        fps : int
            Frames per second in the output GIF
        dpi : int
            Resolution of the output GIF
            
        Returns:
        --------
        str
            Filename of the created animation
        """
        if not self.search_states:
            print("No search data available. Run perform_search_demo first.")
            return None
        
        # Get current state of the HNSW structure
        graphs = self.hnsw.get_graphs()
        n_layers = len(graphs)
        
        # Create a new figure for the animation
        fig, axs = plt.subplots(1, n_layers, figsize=self.figsize, dpi=dpi)
        if n_layers == 1:
            axs = [axs]
        
        for ax in axs:
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True)
        
        # Get query point from search states
        query_point = self.search_states[0]['query_point']
        
        # Function to update the animation for each search state
        def update(frame_idx):
            state = self.search_states[frame_idx]
            current_layer = state['layer']
            
            # Clear all axes
            for ax in axs:
                ax.clear()
                ax.set_xlim(-0.05, 1.05)
                ax.set_ylim(-0.05, 1.05)
                ax.grid(True)
            
            # Set overall title based on state type
            if state['type'] == 'start':
                fig.suptitle(f"HNSW Search - Starting at Layer {current_layer}")
            elif state['type'] == 'layer_complete':
                fig.suptitle(f"HNSW Search - Layer {current_layer} Completed")
            elif state['type'] == 'complete':
                fig.suptitle(f"HNSW Search - Search Completed")
            else:
                fig.suptitle(f"HNSW Search - Processing Layer {current_layer}")
            
            # Draw each layer
            for layer_idx, (G, ax) in enumerate(zip(graphs, axs)):
                # Get positions and valid nodes
                pos, valid_nodes = self._get_node_positions(G)
                
                if not valid_nodes:
                    ax.set_title(f"Layer {layer_idx}")
                    continue
                
                # Create a subgraph with only valid nodes
                valid_subgraph = G.subgraph(valid_nodes)
                
                # Draw all nodes
                nx.draw_networkx_nodes(
                    valid_subgraph, 
                    pos, 
                    ax=ax, 
                    node_color=self.node_color, 
                    node_size=self.node_size
                )
                
                # Draw all edges
                nx.draw_networkx_edges(
                    valid_subgraph, 
                    pos, 
                    ax=ax, 
                    edge_color=self.edge_color, 
                    alpha=0.5
                )
                
                # Draw labels
                labels = {node: str(node) for node in valid_nodes}
                label_pos = {i: (pos[i] + self.label_offset) for i in valid_nodes}
                nx.draw_networkx_labels(
                    valid_subgraph, 
                    label_pos, 
                    labels, 
                    ax=ax, 
                    font_size=10
                )
                
                # Draw query point if it's 2D
                if len(query_point) == 2:
                    ax.scatter(
                        query_point[0], 
                        query_point[1], 
                        color='green', 
                        s=200, 
                        zorder=20, 
                        marker='*', 
                        label='Query'
                    )
                
                # If this is the current layer being processed
                if layer_idx == current_layer:
                    # Highlight visited nodes
                    visited_nodes = [n for n in state['visited'] if n in valid_nodes]
                    if visited_nodes:
                        nx.draw_networkx_nodes(
                            valid_subgraph.subgraph(visited_nodes), 
                            pos, 
                            ax=ax, 
                            node_color='lightblue', 
                            node_size=self.node_size,
                            alpha=0.7
                        )
                    
                    # Highlight nearest neighbors found so far
                    nearest_nodes = [node for _, node in state['nearest'] if node in valid_nodes]
                    if nearest_nodes:
                        nx.draw_networkx_nodes(
                            valid_subgraph.subgraph(nearest_nodes), 
                            pos, 
                            ax=ax, 
                            node_color='orange', 
                            node_size=self.node_size,
                            alpha=0.9
                        )
                    
                    # Highlight current node being processed
                    if state['current_node'] is not None and state['current_node'] in valid_nodes:
                        nx.draw_networkx_nodes(
                            valid_subgraph.subgraph([state['current_node']]), 
                            pos, 
                            ax=ax, 
                            node_color='red', 
                            node_size=self.node_size*1.5,
                            alpha=1.0
                        )
                    
                    # Highlight current neighbor being evaluated
                    if state['neighbor'] is not None and state['neighbor'] in valid_nodes:
                        nx.draw_networkx_nodes(
                            valid_subgraph.subgraph([state['neighbor']]), 
                            pos, 
                            ax=ax, 
                            node_color='purple', 
                            node_size=self.node_size*1.2,
                            alpha=1.0
                        )
                        
                        # Draw edge between current node and neighbor
                        if (state['current_node'] is not None and 
                            state['current_node'] in valid_nodes):
                            edge = (state['current_node'], state['neighbor'])
                            # Check if edge exists in graph
                            if valid_subgraph.has_edge(*edge):
                                nx.draw_networkx_edges(
                                    valid_subgraph.edge_subgraph([edge]), 
                                    pos, 
                                    ax=ax, 
                                    edge_color='purple', 
                                    width=2.0, 
                                    alpha=1.0
                                )
                
                # Highlight final results for completed searches
                if state['type'] in ['layer_complete', 'complete'] and layer_idx <= current_layer:
                    result_nodes = [node for _, node in state['nearest'] if node in valid_nodes]
                    if result_nodes:
                        nx.draw_networkx_nodes(
                            valid_subgraph.subgraph(result_nodes), 
                            pos, 
                            ax=ax, 
                            node_color='red', 
                            node_size=self.node_size*1.5,
                            alpha=1.0
                        )
                
                # Set title to indicate active layer
                title = f"Layer {layer_idx}"
                if layer_idx == current_layer:
                    title += " (Active)"
                ax.set_title(title)
                
                # Add legend
                handles = []
                labels = []
                
                if len(query_point) == 2:
                    ax.legend(loc='upper right')
            
            return axs
        
        # Create the animation
        ani = FuncAnimation(
            fig, 
            update, 
            frames=len(self.search_states), 
            interval=1000//fps, 
            blit=False
        )
        
        # Save the animation as a GIF
        ani.save(filename, writer='pillow', fps=fps)
        
        # Close the figure to avoid displaying it twice in notebooks
        plt.close(fig)
        
        print(f"Search animation saved to {filename}")
        
        # Display the animation in the notebook
        display(Image(filename))
        
        return filename

    def perform_construction_demo(self, points=None, incremental=True, filename='hnsw_construction.gif', fps=1, dpi=100):
        """
        Build the HNSW graph with the provided points and create an animation of the construction process.
        
        Parameters:
        -----------
        points : numpy.ndarray or None
            Array of points to add to the graph. If None, uses existing points if any.
        incremental : bool
            If True, builds the graph incrementally. If False, uses batch construction.
        filename : str
            Filename for the animation
        fps : int
            Frames per second in the output GIF
        dpi : int
            Resolution of the output GIF
        
        Returns:
        --------
        str
            Filename of the created animation
        """
        # Check if we have points to add
        if points is None:
            if len(self.hnsw.get_points()) == 0:
                raise ValueError("No points provided and HNSW structure is empty.")
            # If points already exist in the HNSW structure, just create the animation
            return self.create_animation(filename=filename, fps=fps, dpi=dpi)
        
        # Clear existing HNSW structure if we're adding new points
        if len(self.hnsw.get_points()) > 0:
            # Reset HNSW structure (reinitialize the graphs)
            self.hnsw.graphs = [nx.Graph() for _ in range(self.hnsw.n_layers)]
            self.hnsw.points = []
        
        # Build the graph with the provided points
        if incremental:
            # Build incrementally (add one point at a time)
            for point in points:
                self.hnsw.add_point(point)
        else:
            # Build all at once
            self.hnsw.build_from_points(points)
        
        # Create and return the animation
        return self.create_animation(filename=filename, fps=fps, dpi=dpi)

    def perform_search_demo(self, query_point, k=3, ef_search=None, entry_point=None,
                            create_animation=False, filename="hnsw_search.gif"):
        """
        Perform a search and visualize the result.
        
        Parameters:
        -----------
        query_point : numpy.ndarray
            The query point coordinates
        k : int
            Number of nearest neighbors to find
        ef_search : int or None
            Size of the dynamic candidate list during search
        create_animation : bool
            Whether to create an animation of the search process
        
        Returns:
        --------
        list
            List of k nearest neighbor indices
        """
        # Reset search states
        self.search_states = []
        
        # Call the search method from the HNSW structure
        nearest_neighbors = self.hnsw.find_nearest_neighbors(
            query_point=query_point, k=k, 
            ef_search=ef_search, entry_point=entry_point
            )
        
        # Store the result
        self.current_search_query = query_point
        self.search_result = nearest_neighbors
        
        # Create search animation if requested
        if create_animation and self.search_states:
            self.create_search_animation(filename=filename)
        
        return nearest_neighbors
