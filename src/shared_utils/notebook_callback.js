// connects to the Python kernel for the notebook in use,
// and runs the function function_name("data").
// Only works in Google Colab and Jupyter Notebook environments.
export function execute(function_name, data) {
    if (typeof Jupyter !== 'undefined') {
        let call_statement = function_name + '("' + data + '")';
        Jupyter.notebook.kernel.execute(call_statement);
    } else if (typeof google !== 'undefined') {
        google.colab.kernel.invokeFunction(function_name, [data], {});
    }
}