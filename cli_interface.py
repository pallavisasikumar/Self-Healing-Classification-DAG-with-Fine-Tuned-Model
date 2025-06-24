from dag_nodes import inference_node, confidence_check_node, fallback_node
from rich.console import Console
from rich.table import Table
from rich import box

def run_cli():
    print("🤖 Self-Healing Sentiment Classifier (Type 'exit' to quit)\n")

    # Track stats
    confidence_values = []
    fallback_count = 0
    console = Console()

    while True:
        user_input = input("Enter a sentence: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        label, confidence = inference_node(user_input)
        confidence_values.append(confidence)
        print(f"[InferenceNode] Prediction: {label} | Confidence: {int(confidence * 100)}%")

        if confidence_check_node(label, confidence):
            print(f"[Final Decision] Accepted: {label}")
        else:
            fallback_count += 1
            corrected_label = fallback_node(label, confidence)
            print(f"[Final Decision] Corrected to: {corrected_label}")

    # Session summary
    print("\n📊 Session Summary")

    # Confidence curve table
    table = Table(title="Confidence Per Input", box=box.ROUNDED)
    table.add_column("Input #", style="cyan")
    table.add_column("Confidence %", style="magenta")

    for i, conf in enumerate(confidence_values, 1):
        table.add_row(str(i), f"{int(conf * 100)}%")

    console.print(table)

    # Fallback frequency
    fallback_percent = (fallback_count / len(confidence_values)) * 100 if confidence_values else 0
    console.print(f"\n[bold red]Fallbacks triggered:[/bold red] {fallback_count} times out of {len(confidence_values)} inputs.")
    console.print(f"[bold green]Fallback rate:[/bold green] {fallback_percent:.1f}%\n")

if __name__ == "__main__":
    run_cli()
