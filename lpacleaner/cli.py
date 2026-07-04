"""Click CLI: analyze, run, inspect, review, cleanup commands."""

import click

from lpacleaner import __version__


@click.group()
@click.version_option(version=__version__)
def main():
    """LPA Cleaner -- process photographed book pages into searchable PDFs."""


@main.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.option("-o", "--output-dir", type=click.Path(file_okay=False), default=None)
@click.option("--profile", type=click.Choice(["full", "geometry", "clean", "quick"]), default="full")
@click.option("--preview", type=int, default=0, help="Process only N images")
@click.option("--skip-dewarp", is_flag=True)
@click.option("--skip-deskew", is_flag=True)
@click.option("--skip-enhance", is_flag=True)
@click.option("--skip-normalize", is_flag=True)
@click.option("--skip-ocr", is_flag=True)
@click.option("--skip-content-area", is_flag=True)
@click.option("--ai-dewarp", is_flag=True)
@click.option("--binarize", is_flag=True)
@click.option("--cleanup", is_flag=True, help="Delete intermediate checkpoints after success")
@click.option("--on-error", type=click.Choice(["skip", "stop", "force"]), default="skip")
@click.option("--verbose", is_flag=True)
@click.option("--quiet", is_flag=True)
def run(input_dir, output_dir, profile, preview, skip_dewarp, skip_deskew,
        skip_enhance, skip_normalize, skip_ocr, skip_content_area,
        ai_dewarp, binarize, cleanup, on_error, verbose, quiet):
    """Process book page photos into a searchable PDF."""
    click.echo(f"Processing {input_dir}...")


@main.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.option("-o", "--output-dir", type=click.Path(file_okay=False), default=None)
@click.option("--samples", type=int, default=15)
def analyze(input_dir, output_dir, samples):
    """Analyze book photos and generate book.toml configuration."""
    click.echo(f"Analyzing {input_dir}...")


@main.command()
@click.argument("image_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--config", type=click.Path(exists=True, dir_okay=False), default=None)
def inspect(image_path, config):
    """Inspect a single image with diagnostic output."""
    click.echo(f"Inspecting {image_path}...")


@main.command()
@click.argument("output_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--stage", type=str, default=None)
def review(output_dir, stage):
    """Review processed pages and generate contact sheet."""
    click.echo(f"Reviewing {output_dir}...")


@main.command(name="cleanup")
@click.argument("output_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--keep", type=str, default=None, help="Comma-separated stage numbers to keep")
def cleanup_cmd(output_dir, keep):
    """Delete intermediate checkpoint directories."""
    click.echo(f"Cleaning up {output_dir}...")
