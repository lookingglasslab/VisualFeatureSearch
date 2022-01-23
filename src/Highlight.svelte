<script>
    import { onMount } from "svelte";
    import * as callback from "./shared_utils/notebook_callback";

    export let img_url;
    export let size = 256;
    export let highlightRadius = 15;
    export let callback_name;

    let canvas;
    let ctx;

    let drawing = false;

    onMount(() => {
        canvas.width = size;
        canvas.height = size;
        ctx = canvas.getContext('2d');
    });

    function mouseDown() {
        drawing = true;
    }

    function mouseUp() {
        drawing = false;
        if(callback_name) {
            callback.execute(callback_name, canvas.toDataURL());
        }
    }

    function handleMove(e) {
        if(!drawing) return;
        ctx.fillStyle = '#fff';
        ctx.beginPath();
        ctx.arc(e.offsetX, e.offsetY, highlightRadius, 0, 2 * Math.PI);
        ctx.fill();
    }

    function reset() {
        ctx.clearRect(0, 0, size, size);
        if(callback_name) {
            callback.execute(callback_name, canvas.toDataURL());
        }
    }
</script>

<div class="container" style="width: {size}px; height: {size}px;">
    <div class="image" style="background-image: url('{img_url}'); width: {size}px; height: {size}px;"></div>

    <canvas 
        bind:this={canvas}
        on:mousemove={handleMove} 
        on:mousedown={mouseDown}
        on:mouseup={mouseUp}
        class="overlay" 
        style="z-index: 10; width: {size}px; height: {size}px;"
    ></canvas>
</div>

<br>
<button on:click={reset}>Reset</button>

{#if !callback.is_supported()}
    <p style="border: 1px solid red;">
        Warning: this environment does not support callbacks between JavaScript and Python,
        and this widget will not work as a result. Consider using Google Colab or a native
        Jupyter Notebook setup instead.
    </p>
{/if}

<style>
    .container {
        position: relative;
        display: inline-block;
    }

    .image, .overlay {
        position: absolute;
        left: 0px;
        top: 0px;
    }

    .overlay {
        opacity: 0;
        transition: opacity .1s ease-in-out;
    }

    .overlay:hover {
        opacity: 0.7;
    }
</style>