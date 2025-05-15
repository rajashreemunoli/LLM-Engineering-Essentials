import re
import json
import os
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import numpy as np
from collections import defaultdict

# Extract the think section content from the solution
def extract_think_section(text):
    match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text  # Return original text if no think tags

# Improved function to split the solution into individual thought blocks
def split_into_thoughts(solution_text, min_split_size=120):
    """
    Split solution text into thought blocks based on:
    1. Paragraphs (with minimum size requirement)
    2. Explicit branching indicators ("But", "Wait", "Alternatively")
    
    Args:
        solution_text (str): The text to split
        min_split_size (int): Minimum character length for a split to be considered separate
        
    Returns:
        list: List of thought blocks
    """
    # Define branching indicators
    branching_indicators = ["But", "Wait", "Alternatively"]
    branching_pattern = r'\b({})\b'.format("|".join(branching_indicators))
    
    # First, split by paragraphs
    paragraphs = re.split(r'\n\s*\n', solution_text)
    
    # Clean paragraphs (remove excessive whitespace)
    paragraphs = [re.sub(r'\s+', ' ', para).strip() for para in paragraphs if para.strip()]
    
    thoughts = []
    current_thought = ""
    
    for para in paragraphs:
        # Check if paragraph starts with a branching indicator
        branch_match = re.match(branching_pattern, para)
        
        if branch_match or not current_thought:
            # Start a new thought if it's a branching indicator or this is the first paragraph
            if current_thought and len(current_thought) >= min_split_size:
                thoughts.append(current_thought.strip())
            current_thought = para
        else:
            # Check if adding this paragraph would make the thought too large
            # If so, start a new thought unless the current paragraph is too small
            if len(current_thought) >= min_split_size and len(para) >= min_split_size:
                thoughts.append(current_thought.strip())
                current_thought = para
            else:
                # Handle LaTeX formulas - don't split if this is just a formula line
                is_latex_formula = bool(re.match(r'\s*\\[\(\[]|^\s*\$\$', para))
                
                if is_latex_formula:
                    current_thought += "\n\n" + para
                else:
                    # Check if the new paragraph can stand on its own
                    if len(para) >= min_split_size:
                        if len(current_thought) >= min_split_size:
                            thoughts.append(current_thought.strip())
                            current_thought = para
                        else:
                            current_thought += "\n\n" + para
                    else:
                        current_thought += "\n\n" + para
    
    # Add the final thought if it exists and meets minimum size
    if current_thought.strip() and len(current_thought) >= min_split_size:
        thoughts.append(current_thought.strip())
        
    return thoughts

# Count tokens for each thought
def count_tokens(thoughts, tokenizer):
    token_counts = []
    for thought in thoughts:
        # Different tokenizers have different interfaces
        try:
            # Try the transformers library style tokenizer first
            tokens = tokenizer.encode(thought)
            token_counts.append(len(tokens))
        except AttributeError:
            try:
                # Try the tiktoken style tokenizer
                tokens = tokenizer.encode(thought)
                token_counts.append(len(tokens))
            except:
                # If all else fails, use a simple approximation
                token_counts.append(len(thought.split()))
                print("Warning: Using word count as approximation for tokens")

    return token_counts

# Function to query LLM API to find connections between thoughts
def find_thought_connections(thoughts, client, model, token_counts=None):
    connections = []
    
    # Add a root thought as the starting point (index 0)
    connections.append({
        "id": 0,
        "text": "ROOT",
        "connects_to": None,
        "token_count": 1  # Just a placeholder token count
    })
    
    # For each thought, determine which previous thought it connects to
    for i in tqdm(range(len(thoughts))):
        thought_id = i + 1  # Actual thoughts start from index 1
        
        # Check if this thought starts with a branching indicator
        is_branching = any(thoughts[i].startswith(indicator) for indicator in ["But", "Wait", "Alternatively"])
        
        # If it's a branching thought, we need to query the LLM to find connections
        if is_branching:
            # Prepare prompt for the API
            prompt = f"""You are given a sequence of mathematical solution steps, each expressed as a thought. Your goal is to determine which **one** previous thought each thought most directly continues or references to.

**Guidelines:**  
1. **General Continuation:** Most thoughts naturally follow the immediately preceding one. Identify the closest logical predecessor.  
2. **Special Cases:**  
   - Thoughts beginning with 'Alternatively', 'Wait', or 'But wait' reference an earlier thought rather than the immediately previous one. Look for a past thought they are continuing, not the immediately previous one. 
3. **Root Thought:**  
   - There is a special ROOT thought with ID 0, representing the start of the solution.  
   - If a thought starts a completely new line of reasoning or does not clearly continue any prior thought, connect it to ROOT (ID 0). For example, if the tought starts with 'Alternatively, maybe there is another way to approach this problem' 

**Output:**  
For each thought, identify the **one** previous thought (by ID) that it most directly continues. If it does not clearly follow any prior thought, assign it to **ROOT (ID 0).**

Previous thoughts:
{json.dumps([{"id": -1, "text": "ROOT"}] + [{"id": j, "text": thoughts[j]} for j in range(i)])}

Current thought (ID {i}):
{thoughts[i]}

Answer in the format
#ID: <id>
where <id> is the ID number of the one previous thought this directly connects to (can be 0 for ROOT):
"""

            try:
                # Call the API
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10
                )

                # Extract the predicted connection
                print(response.choices[0].message.content)
                response_text = response.choices[0].message.content.strip()
                try:
                    # Try to extract the ID using regex to be more robust
                    id_match = re.search(r'#ID:\s*(\d+)', response_text)
                    if id_match:
                        connection_id = int(id_match.group(1))
                    else:
                        # Fallback to the old method
                        connection_id = int(response_text.strip(". ").split(" ")[-1].strip("#"))
                    
                    # Validate the connection ID
                    if connection_id < 0 or connection_id > thought_id - 1:
                        print(f"Warning: Invalid connection ID {connection_id} for thought {thought_id}. Setting to 0 (ROOT).")
                        connection_id = 0
                except ValueError:
                    print(f"Warning: Could not parse connection ID from response: {response_text}. Setting to 0 (ROOT).")
                    connection_id = 0

            except Exception as e:
                print(f"Error processing thought {thought_id}: {e}")
                # Default to connecting to the previous thought if there's an error
                connection_id = thought_id - 1
        else:
            # For non-branching thoughts, we assume it continues from the previous thought
            connection_id = thought_id - 1
        
        connections.append({
            "id": thought_id,
            "text": thoughts[i],
            "connects_to": connection_id,
            "token_count": token_counts[i] if token_counts else None
        })

    return connections

# Create a visualization of the thought tree
def visualize_thought_tree(connections, output_dir="."):
    """
    Create a PNG visualization of the thought tree.
    
    Args:
        connections: List of connection objects
        output_dir: Directory to save the output file
        
    Returns:
        Path to the generated PNG file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output path
    png_path = os.path.join(output_dir, "thought_tree.png")
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes
    for conn in connections:
        G.add_node(conn['id'])
    
    # Add edges
    for conn in connections:
        if conn.get('connects_to') is not None:
            G.add_edge(conn['connects_to'], conn['id'])
    
    # Create a dictionary to count incoming connections for each node
    incoming_connections = {}
    for conn in connections:
        if conn.get('connects_to') is not None:
            if conn['connects_to'] not in incoming_connections:
                incoming_connections[conn['connects_to']] = 0
            incoming_connections[conn['connects_to']] += 1
    
    # Find the last thought (highest ID)
    last_thought_id = max([conn['id'] for conn in connections if conn['id'] != 0], default=None)
    
    # Node color mapping
    node_color_dict = {}
    for node in G.nodes():
        if node == 0:
            # Root node is pink
            node_color_dict[node] = '#f9f'
        elif node == last_thought_id:
            # The very last thought is green (final answer)
            node_color_dict[node] = 'limegreen'
        elif node in incoming_connections and incoming_connections[node] > 1:
            # Branch nodes (nodes with multiple connections TO them) are blue
            node_color_dict[node] = 'skyblue'
        else:
            # Regular nodes are light gray
            node_color_dict[node] = 'lightgray'
    
    # Extract colors in the right order
    node_colors = [node_color_dict[node] for node in G.nodes()]
    
    # Set figure size based on number of nodes
    node_count = len(G.nodes())
    figsize = (12, 8)  # Default size
    if node_count > 20:
        figsize = (16, 10)
    if node_count > 50:
        figsize = (20, 12)
    
    plt.figure(figsize=figsize)
    
    # Use a hierarchical layout
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    except:
        try:
            pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='dot')
        except:
            # Fall back to spring layout if graphviz-based layouts are not available
            pos = nx.spring_layout(G, k=1.5/max(1, (node_count ** 0.5)), iterations=50)
    
    # Add labels for node IDs
    labels = {0: "ROOT"}
    for node in G.nodes():
        if node != 0:
            # For thought nodes, use a shorter version of the text if available
            for conn in connections:
                if conn['id'] == node:
                    # Add token count to label if available
                    if conn.get('token_count') is not None:
                        labels[node] = f"T{node} [{conn['token_count']}]"
                    else:
                        labels[node] = f"T{node}"
    
    # Draw the graph
    nx.draw(G, pos, with_labels=True, labels=labels, node_color=node_colors,
            node_size=1200, font_size=10, font_weight='bold',
            edge_color='gray', arrowsize=15, width=1.5,
            arrows=True, connectionstyle='arc3,rad=0.1')
    
    # Add a legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#f9f', markersize=15, label='Root'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', markersize=15, label='Regular'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue', markersize=15, label='Branch (Multiple Connections)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='limegreen', markersize=15, label='Final Answer')
    ]
    plt.legend(handles=legend_elements, loc='best')
    
    # Add a title
    plt.title("Thought Tree Visualization", fontsize=16, pad=20)
    
    # Save the figure
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {png_path}")
    return png_path

# Create a summary of the thought tree
def create_summary(connections, output_dir="."):
    """
    Create a text summary of the thought tree.
    
    Args:
        connections: List of connection objects
        output_dir: Directory to save the output file
        
    Returns:
        Path to the generated summary file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output path
    summary_path = os.path.join(output_dir, "thought_summary.txt")
    
    # Count types of thoughts
    total_thoughts = len(connections)
    root_thoughts = sum(1 for conn in connections if conn['id'] == 0)
    
    # Count branching nodes (nodes with multiple incoming connections)
    incoming_connections = {}
    for conn in connections:
        if conn.get('connects_to') is not None:
            if conn['connects_to'] not in incoming_connections:
                incoming_connections[conn['connects_to']] = 0
            incoming_connections[conn['connects_to']] += 1
    
    branch_nodes = [node for node, count in incoming_connections.items() if count > 1]
    
    # Find the last thought ID
    last_thought_id = max([conn['id'] for conn in connections if conn['id'] != 0], default=None)
    
    # Calculate total tokens if available
    total_tokens = 0
    thoughts_with_tokens = 0
    for conn in connections:
        if conn.get('token_count') is not None:
            total_tokens += conn['token_count']
            thoughts_with_tokens += 1
    
    # Create summary text
    summary = f"""
Thought Tree Summary:
---------------------
Total thoughts: {total_thoughts}
Root thought: {root_thoughts}
Branch nodes (with multiple connections): {len(branch_nodes)}
Regular thoughts: {total_thoughts - root_thoughts - len(branch_nodes) - 1}
Final answer thought ID: {last_thought_id}
"""

    if thoughts_with_tokens > 0:
        summary += f"\nToken Statistics:\n-----------------\n"
        summary += f"Total tokens: {total_tokens}\n"
        summary += f"Average tokens per thought: {total_tokens / (thoughts_with_tokens - 1):.1f}\n"
    
    
    # Save the summary
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"Summary saved to {summary_path}")
    print(summary)
    
    return summary_path

# Generate token count visualizations
def visualize_token_counts(connections, output_dir="."):
    """
    Create visualizations of token counts.
    
    Args:
        connections: List of connection objects
        output_dir: Directory to save the output files
        
    Returns:
        Dictionary with paths to the generated visualization files
    """
    # Check if token counts are available
    if not all('token_count' in conn and conn['token_count'] is not None for conn in connections):
        print("Token count information missing for some thoughts. Skipping token visualizations.")
        return {}
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract token counts and IDs
    token_counts = [conn['token_count'] for conn in connections if conn['id'] != 0]  # Skip ROOT
    thought_ids = [conn['id'] for conn in connections if conn['id'] != 0]  # Skip ROOT
    
    # Token count distribution
    plt.figure(figsize=(12, 6))
    bars = plt.bar(thought_ids, token_counts)
    
    # Color the bars based on branching
    incoming_connections = {}
    for conn in connections:
        if conn.get('connects_to') is not None:
            if conn['connects_to'] not in incoming_connections:
                incoming_connections[conn['connects_to']] = 0
            incoming_connections[conn['connects_to']] += 1
            
    for i, thought_id in enumerate(thought_ids):
        if thought_id in incoming_connections and incoming_connections[thought_id] > 1:
            bars[i].set_color('skyblue')
    
    # Highlight the last thought
    last_thought_id = max(thought_ids, default=None)
    if last_thought_id is not None:
        last_idx = thought_ids.index(last_thought_id)
        bars[last_idx].set_color('limegreen')
    
    plt.xlabel('Thought ID')
    plt.ylabel('Token Count')
    plt.title('Token Count Distribution Across Thoughts')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    token_dist_path = os.path.join(output_dir, "token_distribution.png")
    plt.savefig(token_dist_path, dpi=300)
    plt.close()
    
    # Cumulative token count
    plt.figure(figsize=(12, 6))
    cumulative_tokens = np.cumsum(token_counts)
    plt.plot(thought_ids, cumulative_tokens, marker='o', linestyle='-')
    
    # Add a horizontal line for the total
    plt.axhline(y=cumulative_tokens[-1], color='r', linestyle='--', alpha=0.5)
    plt.text(thought_ids[-1], cumulative_tokens[-1], f' Total: {cumulative_tokens[-1]} tokens', verticalalignment='bottom')
    
    plt.xlabel('Thought ID')
    plt.ylabel('Cumulative Token Count')
    plt.title('Cumulative Token Count')
    plt.grid(True, alpha=0.7)
    plt.tight_layout()
    
    cumulative_path = os.path.join(output_dir, "cumulative_tokens.png")
    plt.savefig(cumulative_path, dpi=300)
    plt.close()
    
    print(f"Token visualizations saved to {output_dir}")
    
    return {
        "distribution": token_dist_path,
        "cumulative": cumulative_path
    }

# Main function to run the analysis
def analyze_solution_thoughts(solution_text, client, model, tokenizer=None, output_dir=".", min_split_size=120):
    """
    Analyze the solution text and split it into thought fragments.
    
    Args:
        file_path (str): Path to the solution file
        client: API client for LLM 
        model (str): Model identifier for LLM
        tokenizer: Tokenizer to count tokens
        output_dir (str): Directory to save outputs
        min_split_size (int): Minimum character length for a thought split
        
    Returns:
        tuple: (connections, visualization_path, summary_path)
    """
    
    # Read and process the solution
    think_content = extract_think_section(solution_text)
    thoughts = split_into_thoughts(think_content, min_split_size=min_split_size)
    
    print(f"Split solution into {len(thoughts)} thought fragments")
    
    # Save the initial thought fragments
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "raw_thoughts.txt"), 'w', encoding='utf-8') as f:
        for i, thought in enumerate(thoughts):
            f.write(f"Thought {i+1}:\n")
            f.write(f"{thought}\n\n")
            f.write("-" * 80 + "\n\n")
    
    # Count tokens if tokenizer is provided
    token_counts = None
    if tokenizer:
        token_counts = count_tokens(thoughts, tokenizer)
        print(f"Calculated token counts for all thoughts")
    
    # Find connections between thoughts
    print(f"Finding connections between thoughts...")
    connections = find_thought_connections(thoughts, client, model, token_counts)
    
    # Save connections as JSON
    connections_path = os.path.join(output_dir, "thought_connections.json")
    with open(connections_path, 'w', encoding='utf-8') as f:
        json.dump(connections, f, indent=2)
    
    # Create visualizations
    print(f"Creating visualizations...")
    viz_path = visualize_thought_tree(connections, output_dir)
    
    # Create summary
    summary_path = create_summary(connections, output_dir)
    
    # Create token visualizations if token counts are available
    if token_counts:
        visualize_token_counts(connections, output_dir)
    
    print(f"Analysis complete! Results saved to {output_dir}")
    return connections, viz_path, summary_path
