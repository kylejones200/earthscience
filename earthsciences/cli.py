"""
Command-line interface for earthsciences package.
"""

import click
from pathlib import Path
import sys


@click.group()
@click.version_option()
def cli():
    """
    EarthSciences - Config-driven geostatistical analysis.
    
    Run complete geostatistical workflows from YAML configuration files.
    """
    pass


@cli.command()
@click.option('--config', '-c', required=True, type=click.Path(exists=True),
              help='Path to YAML configuration file')
@click.option('--output', '-o', type=click.Path(),
              help='Override output directory from config')
@click.option('--verbose/--quiet', '-v/-q', default=True,
              help='Verbose output (default: verbose)')
def run(config, output, verbose):
    """
    Run geostatistical analysis from config file.
    
    Example:
        earthsciences run --config my_analysis.yaml
    """
    try:
        from earthsciences.workflows.runner import ConfigRunner
        
        # Load and run
        runner = ConfigRunner(config, verbose=verbose)
        
        # Override output directory if provided
        if output:
            runner.config.project.output_dir = str(Path(output).expanduser())
        
        results = runner.run()
        
        click.secho("\n✓ Analysis complete!", fg='green', bold=True)
        click.echo(f"Results saved to: {results['output_dir']}")
        
        if 'validation' in results:
            click.echo(f"\nValidation Metrics:")
            click.echo(f"  RMSE: {results['validation']['rmse']:.4f}")
            click.echo(f"  MAE:  {results['validation']['mae']:.4f}")
            click.echo(f"  R²:   {results['validation']['r2']:.4f}")
        
    except Exception as e:
        click.secho(f"\n✗ Error: {e}", fg='red', err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option('--output', '-o', default='analysis_config.yaml',
              type=click.Path(), help='Output file path')
@click.option('--template', '-t', 
              type=click.Choice(['full', 'minimal', 'kriging', 'validation']),
              default='full', help='Template type')
def init(output, template):
    """
    Generate template configuration file.
    
    Templates:
        full       - Complete config with all options
        minimal    - Minimal config with required fields only  
        kriging    - Focused on kriging parameters
        validation - Focused on validation and assessment
    
    Example:
        earthsciences init --output my_config.yaml --template minimal
    """
    try:
        from earthsciences.config.templates import generate_template
        
        generate_template(output, template_type=template)
        click.secho(f"✓ Template created: {output}", fg='green')
        click.echo(f"\nNext steps:")
        click.echo(f"  1. Edit {output} with your data paths and parameters")
        click.echo(f"  2. Run: earthsciences run --config {output}")
        
    except Exception as e:
        click.secho(f"✗ Error: {e}", fg='red', err=True)
        sys.exit(1)


@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
def validate(config_file):
    """
    Validate configuration file.
    
    Example:
        earthsciences validate my_analysis.yaml
    """
    from earthsciences.config.parser import validate_config
    
    is_valid, message = validate_config(config_file)
    
    if is_valid:
        click.secho(message, fg='green')
        sys.exit(0)
    else:
        click.secho(message, fg='red', err=True)
        sys.exit(1)


@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
def show(config_file):
    """
    Display configuration file contents.
    
    Example:
        earthsciences show my_analysis.yaml
    """
    try:
        from earthsciences.config.parser import load_config, config_to_dict
        import yaml
        
        config = load_config(config_file)
        config_dict = config_to_dict(config)
        
        click.echo(f"\nConfiguration: {config_file}")
        click.echo("=" * 70)
        click.echo(yaml.dump(config_dict, default_flow_style=False, sort_keys=False))
        
    except Exception as e:
        click.secho(f"✗ Error: {e}", fg='red', err=True)
        sys.exit(1)


@cli.command()
def templates():
    """
    List available configuration templates.
    """
    from earthsciences.config.templates import list_templates
    
    template_list = list_templates()
    
    click.echo("\nAvailable templates:")
    click.echo("-" * 40)
    for template in template_list:
        click.echo(f"  • {template}")
    click.echo("\nUse: earthsciences init --template <name>")


@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--output', '-o', default='config_summary.txt',
              help='Output summary file')
def summarize(config_file, output):
    """
    Generate summary of configuration.
    
    Example:
        earthsciences summarize my_analysis.yaml
    """
    try:
        from earthsciences.config.parser import load_config
        
        config = load_config(config_file)
        
        with open(output, 'w') as f:
            f.write("="*70 + "\n")
            f.write("CONFIGURATION SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Project: {config.project.name}\n")
            f.write(f"Output Directory: {config.project.output_dir}\n\n")
            
            f.write("Data:\n")
            f.write(f"  Input file: {config.data.input_file}\n")
            f.write(f"  Columns: X={config.data.x_column}, Y={config.data.y_column}, Z={config.data.z_column}\n\n")
            
            if config.preprocessing:
                f.write("Preprocessing:\n")
                f.write(f"  Remove outliers: {config.preprocessing.remove_outliers}\n")
                f.write(f"  Transform: {config.preprocessing.transform}\n\n")
            
            f.write("Variogram:\n")
            f.write(f"  Models: {', '.join(config.variogram.models)}\n")
            f.write(f"  Lags: {config.variogram.n_lags}\n")
            f.write(f"  Auto-fit: {config.variogram.auto_fit}\n\n")
            
            f.write("Kriging:\n")
            f.write(f"  Method: {config.kriging.method}\n")
            f.write(f"  Max neighbors: {config.kriging.neighborhood.max_neighbors}\n")
            grid = config.kriging.grid
            f.write(f"  Grid: [{grid.x_min}, {grid.x_max}] × [{grid.y_min}, {grid.y_max}]\n")
            f.write(f"  Resolution: {grid.resolution}\n\n")
            
            if config.validation:
                f.write("Validation:\n")
                f.write(f"  Method: {config.validation.method}\n")
                f.write(f"  Folds: {config.validation.n_folds}\n")
                f.write(f"  Metrics: {', '.join(config.validation.metrics)}\n\n")
        
        click.secho(f"✓ Summary saved to: {output}", fg='green')
        
    except Exception as e:
        click.secho(f"✗ Error: {e}", fg='red', err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()
