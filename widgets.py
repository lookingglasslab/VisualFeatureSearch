CALLBACK_JS = '''
<script>
    function isCallbackSupported() {
        return (typeof Jupyter !== 'undefined') || (typeof google !== 'undefined');
    }

    function executeCallback(function_name, data) {
        if (typeof Jupyter !== 'undefined') {
            let call_statement = function_name + '("' + data + '")';
            Jupyter.notebook.kernel.execute(call_statement);
        } else if (typeof google !== 'undefined') {
            google.colab.kernel.invokeFunction(function_name, [data], {});
        }
    }
</script>
'''

HIGHLIGHT_HTML = '''
<div class="container" style="width: {sz}px; height: {sz}px;">
    <div class="image" style="background-image: url('{url}'); width: {sz}px; height: {sz}px;"></div>
    <canvas 
        id="drawCanvas"
        class="overlay" 
        style="z-index: 10; width: {sz}px; height: {sz}px;"
    ></canvas>
</div>

<br>
<button id="resetBtn">Reset</button>

<style>
    .container {{
        position: relative;
        display: inline-block;
    }}

    .image, .overlay {{
        position: absolute;
        left: 0px;
        top: 0px;
    }}

    .overlay {{
        opacity: 0.5;
        transition: opacity .1s ease-in-out;
    }}

    .overlay:hover {{
        opacity: 0.7;
    }}
</style>

<script>
    (function() {{
    let resetBtn = document.getElementById('resetBtn');
    let drawCanvas = document.getElementById('drawCanvas');

    drawCanvas.width = {sz};
    drawCanvas.height = {sz};
    let ctx = drawCanvas.getContext('2d');

    let drawing = false;

    function mouseDown() {{
        drawing = true;
    }}

    function mouseUp() {{
        drawing = false;
        if('{callName}' !== 'None') {{
            executeCallback('{callName}', drawCanvas.toDataURL());
        }}
    }}

    function handleMove(e) {{
        if(!drawing) return;
        ctx.fillStyle = '#fff';
        ctx.beginPath();
        ctx.arc(e.offsetX, e.offsetY, {hlghtRad}, 0, 2 * Math.PI);
        ctx.fill();
    }}

    function reset() {{
        ctx.clearRect(0, 0, {sz}, {sz});
        if('{callName}' !== 'None') {{
            executeCallback('{callName}', drawCanvas.toDataURL());
        }}
    }}

    drawCanvas.onmousedown = mouseDown;
    drawCanvas.onmouseup = mouseUp;
    drawCanvas.onmousemove = handleMove;

    resetBtn.onclick = reset;
    }})();
</script>

'''

class HighlightWidget:
    def __init__(self, img_url, callback_name=None, size=224, highlight_radius=20):
        self._img_url = img_url
        self._callback_name = callback_name
        self._size = size
        self._highlight_radius = highlight_radius

    def _repr_html_(self):
        return CALLBACK_JS + HIGHLIGHT_HTML.format(
            sz = self._size, 
            url = self._img_url, 
            hlghtRad = self._highlight_radius,
            callName = self._callback_name
        )