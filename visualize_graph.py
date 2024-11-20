from langchain_core.runnables.graph import MermaidDrawMethod
from agent import workflow

# Assuming 'graph' is your compiled StateGraph object
# If it's not in scope, you'll need to import or recreate it here


def generate_mermaid_diagram(builder):
    # Generate the Mermaid diagram definition
    graph = builder.compile()
    mermaid_definition = graph.get_graph().draw_mermaid_png(
        draw_method=MermaidDrawMethod.API,
    )

    # Save the Mermaid diagram definition to a file
    file_name = "graph_diagram.png"
    with open(file_name, "wb") as f:  # Change "w" to "wb" for binary write mode
        f.write(mermaid_definition)  # Write bytes directly

    print(f"Mermaid diagram saved to {file_name}")

    # Open the file (platform-independent)
    import os
    import subprocess

    if os.name == "nt":  # For Windows
        os.startfile(file_name)
    elif os.name == "posix":  # For macOS and Linux
        subprocess.call(("open", file_name))


# Call the function to generate and save the diagram
generate_mermaid_diagram(workflow)
