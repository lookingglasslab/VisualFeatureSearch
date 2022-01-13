export function execute(function_name, data) {
    if (typeof Jupyter !== 'undefined') {
        // TODO: add support for Colab as well
        let call_statement = function_name + '("' + data + '")';
        Jupyter.notebook.kernel.execute(call_statement);
    }
}