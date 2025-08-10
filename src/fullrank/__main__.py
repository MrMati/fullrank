import json
from pathlib import Path
import sys
import concurrent.futures
import numpy as np
import typer
from rich import print
from rich.progress import track

import fullrank
from fullrank import posterior_stats
from fullrank.comparison_tui import ComparisonApp

app = typer.Typer()


@app.command()
def compare(
    items_file: Path = typer.Argument(
        ..., help="File containing the items to rank, seperated by newlines"
    ),
    output_file: Path = typer.Argument(
        ..., help="Output JSON file"
    ),
    shuffle_items: bool = False,
    prior_var: float = typer.Option(1.0, help="The variance of the prior"),
):
    """
    Compare items and write the comparisons to file in JSON format for inference.
    """

    items = [line.rstrip("\n") for line in items_file.read_text(encoding="utf-8").splitlines()]

    if shuffle_items:
        np.random.shuffle(items)

    comparisons = ComparisonApp(items, prior_var=prior_var).run()
    if comparisons is None:
        print("[bold red]No comparisons were made.[/bold red]", file=sys.stderr)
        return

    print(
        f"[bold green]Finished {len(comparisons)} comparisons of {len(items)} items.[/bold green]",
        file=sys.stderr,
    )

    output_file.write_text(
        json.dumps(
            {"items": items, "prior_var": prior_var, "comparisons": comparisons},
            indent="\t", ensure_ascii=False
        ), encoding="utf-8"
    )


@app.command()
def compare_cont(
    comp_file: Path = typer.Argument(
        ..., help="Existing comparison JSON file"
    ),
    output_file: Path = typer.Argument(
        None, help="Optional comparison output JSON file"
    )
):
    """
    Load a comparison session and continue it. Results can be saved to input file or a new one.
    """

    compare_result = json.loads(comp_file.read_text(encoding="utf-8"))
    items = compare_result["items"]
    curr_comparisons = compare_result["comparisons"]
    len_curr_comparisons = len(curr_comparisons)
    prior_var = compare_result["prior_var"]

    comparisons = ComparisonApp(items, comparisons=curr_comparisons, prior_var=prior_var).run()
    if len(comparisons) - len_curr_comparisons == 0:
        print("[bold red]No changes to comparisons were made.[/bold red]", file=sys.stderr)
        return

    print(
        f"[bold green]Finished {len(comparisons)} comparisons of {len(items)} items.[/bold green]",
        file=sys.stderr,
    )

    if output_file is None:
        output_file = comp_file

    output_file.write_text(
        json.dumps(
            {"items": items, "prior_var": prior_var, "comparisons": comparisons},
            indent="\t", ensure_ascii=False
        ), encoding="utf-8"
    )


@app.command()
def raw_sample(
    n: int = typer.Argument(100_000, help="The number of samples to draw"),
    output_file: Path = typer.Argument(..., help="The file to write the samples to"),
    batch_size: int = typer.Option(
        1000, help="The number of samples to draw per batch"
    ),
):
    """
    Sample from the posterior distribution and write to a file in JSONL format.
    """
    compare_result = json.loads(sys.stdin.read())
    posterior = fullrank.infer(
        np.zeros(len(compare_result["items"])),
        compare_result["prior_var"] * np.eye(len(compare_result["items"])),
        compare_result["comparisons"],
    )
    print("[bold green]Finished inferring posterior.[/bold green]", file=sys.stderr)

    print(f"[bold]Output File:[/bold]", output_file, file=sys.stderr)

    with open(output_file, "w") as f:
        for _ in track(range(0, n, batch_size), description="[blue]Sampling...[/blue]"):
            for sample in posterior.sample(batch_size).T:
                f.write(json.dumps(sample.tolist()) + "\n")


@app.command()
def infer_sun():
    """
    Infer the posterior unified skew-normal distribution, and print its parameters.
    """
    compare_result = json.loads(sys.stdin.read())
    items = compare_result["items"]
    posterior = fullrank.infer(
        np.zeros(len(items)),
        compare_result["prior_var"] * np.eye(len(items)),
        compare_result["comparisons"],
    )
    print("[bold green]Finished inferring posterior.[/bold green]", file=sys.stderr)

    print("[bold]xi:[/bold]", posterior.xi, sep="\n")
    print("[bold]Omega:[/bold]", posterior.prior_cov, sep="\n")
    print("[bold]Delta:[/bold]", posterior.Delta, sep="\n")
    print("[bold]tau:[/bold]", posterior.tau, sep="\n")
    print("[bold]Gamma:[/bold]", posterior.Gamma, sep="\n")


@app.command()
def stats(
    comp_file: Path = typer.Argument(
        ..., help="Comparison JSON file"
    ),
    n: int = typer.Argument(25_000, help="The number of samples to draw"),
    entropy: bool = typer.Option(False, flag_value=True, help="Compute entropy"),
):
    """
    Compute statistics from a posterior distribution.
    """

    compare_result = json.loads(comp_file.read_text(encoding="utf-8"))
    items = compare_result["items"]
    comparisons = compare_result["comparisons"]
    prior_var = compare_result["prior_var"]

    posterior = fullrank.infer(
        np.zeros(len(items)),
        compare_result["prior_var"] * np.eye(len(items)),
        compare_result["comparisons"],
    )
    print("[bold green]Finished inferring posterior.[/bold green]", file=sys.stderr)

    batch_size = 1000

    num_batches = (n + batch_size - 1) // batch_size

    def sample_batch():
        _posterior = fullrank.infer(
            np.zeros(len(items)),
            prior_var * np.eye(len(items)),
            comparisons,
        )
        return _posterior.sample(batch_size)

    batches = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(sample_batch) for _ in range(num_batches)]
        batches = []
        for future in track(futures, total=num_batches, description="[blue]Sampling...[/blue]"):
            batches.append(future.result())
    samples = np.concatenate(batches, axis=1)
    del batches

    print("[bold]Mean:[/bold] ", samples.mean(axis=1))

    sorted_indices = np.argsort(samples, axis=0)
    sorted_indices.sort(axis=1)
    print(
        "[bold]Ranking Deciles:[/bold]",
        sorted_indices[:, np.arange(0, n, n // 10)],
        sep="\n",
    )
    print(
        "[bold]Ranking Probabilities (rows are items, columns are rankings):[/bold]",
        np.stack([np.bincount(row, minlength=len(items)) for row in sorted_indices])
        / n,
        sep="\n",
    )

    if entropy:
        print(
            "[bold]Entropy:[/bold] ",
            posterior_stats.lddp(posterior, samples=samples),
        )

    # Print human-readable ranking
    mean_scores = samples.mean(axis=1)
    ranking = np.argsort(-mean_scores)  # descending order
    print("\n[bold]Approximated ranking:[/bold]")
    for i, idx in enumerate(ranking, 1):
        print(f"{i}. {items[idx]}")


if __name__ == "__main__":
    app()
