import typer
from schema.schema import SynthMethods, get_input_validation_schema, validate_input_json
from pathlib import Path
import webbrowser

APP_NAME = "QUiPP"

def app_config():
    """Create an application directory for QUiPP
    """

    app_dir = Path(typer.get_app_dir(APP_NAME))
    if not app_dir.is_dir():
        typer.echo("Creating QUiPP application directory")
        app_dir.mkdir()


app = typer.Typer(help="QUiPP Pipeline CLI", callback=app_config)


@app.command(help="Get the JSON Schema for an available synth methods")
def schema(synth_method: SynthMethods):

    method = get_input_validation_schema(synth_method)
    typer.echo(method.schema_json(indent=2))

@app.command( help="Check an input schema is valid")
def validate(path: Path):
    
    try:
        validate_input_json(path.resolve())
        typer.echo(f"{path.name} is valid")
    except ValueError as error:
        typer.echo(f"{error}")
        typer.Abort()

@app.command(help="Open a browser and create a json input file")
def create(synth_method: SynthMethods):

    from jinja2 import Environment, FileSystemLoader
    from tempfile import NamedTemporaryFile

    # Get app directory
    app_dir = Path(typer.get_app_dir(APP_NAME))

    # Template html file
    search_path = Path(__file__).parent / "frontend"
    
    if not search_path.exists():
        typer.echo("Internal error. Could not find templates")
        typer.Abort()
    
    templateLoader = FileSystemLoader(searchpath=search_path)
    templateEnv = Environment(loader=templateLoader)
    template_file = templateEnv.get_template("input_validation.html")

    # Get schema
    method = get_input_validation_schema(synth_method)
    input_schema = method.schema_json(indent=2)

    # Template html file and write to temporary file
    templated_file_path = app_dir / (synth_method.value + '.html')

    with open(templated_file_path, 'w') as f:
        f.write(template_file.render({'input_schema': input_schema}))

    webbrowser.open("file://" + str(templated_file_path.resolve()))

if __name__ == "__main__":
    app()