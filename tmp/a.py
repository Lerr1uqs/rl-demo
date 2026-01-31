from rich.console import Console
from rich.bar import Bar
from rich.table import Table
import numpy as np

def rich_histogram(data, bins=10):
    console = Console()
    counts, edges = np.histogram(data, bins=bins)
    percentages = counts / len(data) * 100
    
    table = Table(title="概率分布", show_lines=True)
    table.add_column("区间", style="cyan")
    table.add_column("分布")
    table.add_column("占比", style="green")
    
    for i in range(bins):
        bar = "█" * int(percentages[i] * 2)
        table.add_row(
            f"{edges[i]:.2f} ~ {edges[i+1]:.2f}",
            f"[blue]{bar}[/blue]",
            f"{percentages[i]:.1f}%"
        )
    
    console.print(table)

data = np.random.normal(0, 1, 1000)
rich_histogram(data)