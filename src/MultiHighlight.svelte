<script>
    import { onMount } from "svelte";
    import * as callback from "./shared_utils/notebook_callback";

    export let all_urls;
    export let size = 224;
    export let highlightRadius = 20;
    export let callback_name;

    let canvas;
    let ctx;

    let drawing = false;

    let selected = 0;
    let curr_url;

    onMount(() => {
        canvas.width = size;
        canvas.height = size;
        ctx = canvas.getContext('2d');
        curr_url = all_urls[selected];
    });

    function mouseDown() {
        drawing = true;
    }

    function mouseUp() {
        drawing = false;
        if(callback_name) {
            callback.execute(callback_name, [canvas.toDataURL(), selected]);
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
            callback.execute(callback_name, [canvas.toDataURL(), selected]);
        }
    }

    function photoChange(e) {
        let idVal = e.target.id;
        let index = idVal.split('_')[1];
        if(selected != index) {
            selected = index;
            curr_url = all_urls[selected];
            reset();
        }
    }
</script>

<div class="photos" style="width: {size}px">
    {#each all_urls as url, idx}
        <img 
            class="photoBtn" 
            id="photoBtn_{idx}" 
            src={url}
            on:click={photoChange}
        />
    {/each}
</div>

<div class="container" style="width: {size}px; height: {size}px;">
    <div class="image" style="background-image: url('{curr_url}'); width: {size}px; height: {size}px;"></div>

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

    .photos {
        height: 20px;
        margin-bottom: 2px;
        display: flex;
        justify-content: center;
    }

    .photoBtn {
        width: 20px;
        height: 20px;
        margin-right: 2px;
        border: 1px #00000000;
    }

    .photoBtn:hover {
        border: 1px white;
    }

</style>