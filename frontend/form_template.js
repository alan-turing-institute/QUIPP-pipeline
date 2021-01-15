const Form = JSONSchemaForm.default;

const schema = {{ input_schema }}

// const log = (type) => console.log.bind(console, type);

function download(filename, text) {
  var element = document.createElement('a');
  element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text));
  element.setAttribute('download', filename);

  element.style.display = 'none';
  document.body.appendChild(element);

  element.click();

  document.body.removeChild(element);
}

ReactDOM.render(
    React.createElement(Form, {
        schema: schema,
        onSubmit: (input) =>
            download("example.json", JSON.stringify(input.formData, null, 4)),
    }),
    document.getElementById("app"));
